# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import time
from datetime import timedelta
from typing import Any, Generator, Iterable, Optional

import torch
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.components.ft as ft
import torchtitan.protocols.train_spec as train_spec_module

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.config_manager import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)
from torchtitan.experiments.vem.utils import get_rank, init_logger, dump_config

from torchtitan.experiments.vem.ema import ShardedEMA

class Trainer(torch.distributed.checkpoint.stateful.Stateful):
    job_config: JobConfig
    gc_handler: utils.GarbageCollection

    parallel_dims: ParallelDims
    train_spec: train_spec_module.TrainSpec
    world_mesh: torch.distributed.DeviceMesh

    dataloader: train_spec_module.BaseDataLoader
    metrics_processor: train_spec_module.MetricsProcessor
    checkpointer: CheckpointManager
    train_context: Generator[None, None, None]

    model_parts: list[torch.nn.Module]
    optimizers: train_spec_module.OptimizersContainer
    lr_schedulers: train_spec_module.LRSchedulersContainer

    pp_has_first_stage: bool
    pp_has_last_stage: bool

    device: torch.device

    # states
    step: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config

        logger.info(f"Starting job: {job_config.job.description}")

        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        if job_config.job.print_args:
            logger.info(f"Running with args: {job_config.to_dict()}")

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not parallelism_config.disable_loss_parallel,
        )
        dist_utils.init_distributed(job_config)

        # build meshes
        self.world_mesh = world_mesh = parallel_dims.build_mesh(device_type=device_type)
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        self.ft_manager = ft.init_ft_manager(job_config)
        # If TorchFT is enabled, the dp_rank and dp_degree, which are used for
        # dataloader must be changed.
        if self.ft_manager.enabled:
            dp_degree, dp_rank = self.ft_manager.get_dp_info(dp_degree, dp_rank)

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
            distinct_seed_mesh_dim="dp_shard",
        )
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)

        # build dataloader

        self.dataloader = self.train_spec.build_dataloader_fn(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            job_config=job_config,
        )

        # build model (using meta init)
        model_cls = self.train_spec.cls
        model_args = self.train_spec.config[job_config.model.flavor]

        # auto set some model args based on tokenizer 
        # also padding the embedding layer to multiple of world size

        logger.info(
            f"Building {self.train_spec.name} {job_config.model.flavor} with {model_args}"
        )
        with torch.device("meta"):
            model = model_cls.from_model_args(model_args)

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        # metrics logging
        build_metrics_processor_fn = (
            build_metrics_processor
            if self.train_spec.build_metrics_processor_fn is None
            else self.train_spec.build_metrics_processor_fn
        )
        self.metrics_processor = build_metrics_processor_fn(
            job_config, parallel_dims, model_args
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        model_param_count = model_args.get_nparams(model)

        logger.info(
            f"{color.blue}Model {self.train_spec.name} {job_config.model.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {self.train_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                world_mesh,
                parallel_dims,
                job_config,
                self.device,
                model_args,
                self.train_spec.parallelize_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(parallel_dims, job_config, color)
        else:
            # IMPORTANT NOTE:
            # since model is initialized in the meta device and filled with uninitialized data with to_empty
            # we need to make sure EVERY parameter is properly initialized in model.init_weights
            # Update: add .reset_parameters() for default initialization

            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.train_spec.parallelize_fn(
                model, world_mesh, parallel_dims, job_config
            )
            model.to_empty(device=init_device)
            with torch.no_grad():
                # apply default weight initialization first
                # model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
                model.init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        if self.ft_manager.enabled:
            self.ft_manager.set_all_reduce_hook(self.model_parts)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = self.train_spec.build_optimizers_fn(
            self.model_parts, job_config, parallel_dims, self.ft_manager
        )
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, job_config
        )
        if job_config.training.mixed_precision_param == "float16":
            # from torch.amp.grad_scaler import GradScaler
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler as GradScaler
            self.scaler = GradScaler()
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        self.metrics_processor.optimizers = self.optimizers

        if self.job_config.ema.enabled:
            ema = self.job_config.ema
            self.ema = ShardedEMA(
                model=self.model_parts[0],  # EMA currently only supports non-PP model
                beta=ema.beta,
                update_after_step=ema.update_after_step,
                update_every=ema.update_every,
                inv_gamma=ema.inv_gamma,
                power=ema.power,
                min_value=ema.min_value,
            )
            logger.info(
                f"Initialized Sharded EMA with {self.job_config.ema}"
            )

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0

        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self, "ema": self.ema} if self.job_config.ema.enabled else {"train_state": self},
            job_config=job_config,
            ft_manager=self.ft_manager,
        )

        self.train_context = dist_utils.get_train_context(
            parallel_dims.loss_parallel_enabled,
            parallelism_config.enable_compiled_autograd,
        )

        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder, CLIP text encoder).
        # We cast the encoders and it's input/output to this dtype.  If FSDP with mixed precision training is not used,
        # the dtype for encoders is torch.float32.
        # Otherwise, we use the same dtype as mixed precision training process.
        self._dtype = (
            TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        self.job_config = job_config

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.batch_size}, "
            f"global batch size {job_config.training.batch_size * dp_degree}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})."
        )

        # self.instance_id_set = set()
        # self.total_instances = 0

    def next_batch(
        self, data_iterator: Iterable
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        data_load_start = time.perf_counter()
        input_dict = next(data_iterator)
        self.metrics_processor.data_loading_times.append(
            time.perf_counter() - data_load_start
        )

        device_type = utils.device_type
        for k, _ in input_dict.items():
            if isinstance(input_dict[k], torch.Tensor):
                input_dict[k] = input_dict[k].to(device_type)
        
        return input_dict

    # def forward_model(self, model, input_dict):
    #     pred = model(
    #         pos=input_dict['vertices'],
    #         edge_index=input_dict['edges'].permute(1, 0),
    #         offsets=input_dict['offsets'],
    #     )
    #     loss, log_dict = model.loss_fn(pred, input_dict)
    #     return loss, log_dict, {
    #         "num_tokens": input_dict['edges'].shape[0],
    #         "num_flops": 100,
    #     }

    def train_step(self, input_dict: dict[str, torch.Tensor]):
        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        model_parts = self.model_parts
        world_mesh = self.world_mesh
        parallel_dims = self.parallel_dims

        if parallel_dims.pp_enabled:
            raise NotImplementedError
        
        assert len(self.model_parts) == 1
        model = self.model_parts[0]

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        assert not parallel_dims.cp_enabled # TODO
        optional_context_parallel_ctx = None

        # Non-PP forward / backward
        with self.train_context(optional_context_parallel_ctx):
            assert len(model_parts) == 1
            loss, log_dict, metrics = model.train_step(input_dict) # self.forward_model(model, input_dict)
            # TODO
            self.metrics_processor.ntokens_since_last_log += metrics.get("num_tokens", 0)
            self.metrics_processor.num_flops_since_last_log += metrics.get("num_flops", 0)

            if self.job_config.training.mixed_precision_param == "bfloat16":
                loss.backward()
            elif self.job_config.training.mixed_precision_param == "float16":
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizers)
            else:
                loss.backward()

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=self.world_mesh["pp"] if parallel_dims.pp_enabled else None,
        )
        self.checkpointer.maybe_wait_for_staging()

        if self.job_config.training.mixed_precision_param == "bfloat16":
            self.optimizers.step()
        elif self.job_config.training.mixed_precision_param == "float16":
            self.scaler.step(self.optimizers)
            self.scaler.update()
        else:
            self.optimizers.step()

        self.lr_schedulers.step()

        if self.job_config.ema.enabled:
            self.ema.update(model)

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        if (
            parallel_dims.dp_replicate_enabled
            or parallel_dims.dp_shard_enabled
            or parallel_dims.cp_enabled
            or self.ft_manager.enabled
        ):
            loss = loss.detach()
            ft_pg = self.ft_manager.replicate_pg if self.ft_manager.enabled else None
            global_avg_loss, global_max_loss = (
                dist_utils.dist_mean(loss, world_mesh["dp_cp"], ft_pg),
                dist_utils.dist_max(loss, world_mesh["dp_cp"], ft_pg),
            )
            for k, v in log_dict.items():
                log_dict[k] = dist_utils.dist_mean(v, world_mesh["dp_cp"], ft_pg)
        else:
            global_avg_loss = global_max_loss = loss.detach().item()
            for k, v in log_dict.items():
                log_dict[k] = v.detach().item()
        
        self.metrics_processor.log(
            self.step, 
            global_avg_loss, 
            global_max_loss, 
            extra_metrics={
                "grad_norm": grad_norm.item(), 
                'lr': lr,
                **log_dict,
            },
        )
    @record
    def train(self):
        job_config = self.job_config

        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}.")
        if "ema" in job_config.checkpoint.exclude_from_loading:
            logger.info("EMA is excluded from loading. Try to directly copy from model.")
            self.ema.copy_from(self.model_parts[0])

        with maybe_enable_profiling(
            job_config, global_step=self.step
        ) as torch_profiler, maybe_enable_memory_snapshot(
            job_config, global_step=self.step
        ) as memory_profiler, ft.maybe_semi_sync_training(
            job_config,
            ft_manager=self.ft_manager,
            model=self.model_parts[0],
            optimizer=self.optimizers,
            sync_every=job_config.fault_tolerance.sync_steps,
        ):
            data_iterator = iter(self.dataloader)
            while self.step < job_config.training.steps:
                self.step += 1
                self.gc_handler.run(self.step)
                inputs = self.next_batch(data_iterator)
                self.train_step(inputs)
                self.checkpointer.save(
                    self.step, force=(self.step == job_config.training.steps)
                )
                # # get model checkpoint
                # if self.checkpointer._should_save(self.step, force=(self.step == job_config.training.steps)):
                #     from torch.distributed.checkpoint.state_dict import get_model_state_dict
                #     import torch.distributed as dist
                #     from torchtitan.experiments.vem.debug import compare_state_dict
                #     # temp_state_dict = get_model_state_dict(self.model_parts[0])
                #     self.checkpointer.save(
                #         self.step, force=(self.step == job_config.training.steps)
                #     )
                #     self.checkpointer.load(self.step)
                #     # get model checkpoint
                #     # temp_state_dict2 = get_model_state_dict(self.model_parts[0])
                #     # dist.barrier()
                #     # mismatch_count = compare_state_dict(
                #     #     temp_state_dict,
                #     #     temp_state_dict2,
                #     # )

                #     # if dist.get_rank() == 0:
                #     #     print(f"\nTotal mismatches: {mismatch_count}")
                #     #     breakpoint()

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(
                            seconds=job_config.comm.train_timeout_seconds
                        ),
                        world_mesh=self.world_mesh,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        self.metrics_processor.close()
        logger.info("Training completed")

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]

    def close(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()


if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Optional[Trainer] = None

    init_logger(log_file=os.path.join(config.job.dump_folder, "logs", f"rank{get_rank()}.log"))
    dump_config(config, os.path.join(config.job.dump_folder, "config.toml"))

    try:
        trainer = Trainer(config)

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable_checkpoint
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, force=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    finally:
        if trainer:
            trainer.close()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Process group destroyed.")
