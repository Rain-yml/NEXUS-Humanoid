# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import time
from datetime import timedelta
from typing import Any, Generator, Iterable, Optional, Tuple, Dict, Union
import math

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
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)
from torchtitan.experiments.vem.extra_args_smgen import Scheduler
from torchtitan.experiments.vem.utils import get_rank, init_logger, dump_config
from torchtitan.experiments.vem.datasets.octree_utils import OctreeBatch
from torchtitan.experiments.vem.image_encoder import (
    DINOv2ImageEncoder, 
    DINOv3ImageEncoder, 
    DINOv2ImageEncoderWithoutPooler, 
    SigLIP2ImageEncoder,
)

from torchtitan.experiments.vem.ema import ShardedEMA


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    logit_mean: float = None,
    logit_std: float = None,
    uniform_ratio: float = None,
    uniform_power: float = None,
    uniform_min: float = None,
    uniform_max: float = None,
    dtype: torch.dtype = torch.float32,
    device: Union[torch.device, str] = "cpu",
    generator: Optional[torch.Generator] = None,
    batch_size: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    assert batch_size is not None or cu_seqlens is not None
    if batch_size is None:
        batch_size = cu_seqlens.shape[0] - 1
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), dtype=torch.float32, device=device, generator=generator)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mixed_logit_normal":
        u_ln = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), dtype=torch.float32, device=device, generator=generator)
        u_ln = torch.nn.functional.sigmoid(u_ln)
        u_uf = torch.rand(size=(batch_size,), dtype=torch.float32, device=device, generator=generator) ** uniform_power
        u = torch.where(torch.rand(size=(batch_size,), dtype=torch.float32, device=device, generator=generator) < uniform_ratio, u_uf, u_ln)
    elif weighting_scheme == "uniform":
        u = torch.rand(size=(batch_size,), dtype=torch.float32, device=device, generator=generator)
        u = u * (uniform_max - uniform_min) + uniform_min
    else:
        raise NotImplementedError
    if cu_seqlens is not None:
        seq_len = cu_seqlens[1:] - cu_seqlens[:-1]
        u = torch.repeat_interleave(u, seq_len)
    return u.to(dtype=dtype)


class DiffusionTrainer(torch.distributed.checkpoint.stateful.Stateful):
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
                model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
                model.init_weights(buffer_device=buffer_device)
            model.train()

            # DEBUG mup
            # print({name: param.shape for name, param in model.named_parameters()})
            # from mup import set_base_shapes
            # set_base_shapes(model, "/mnt/pfs/users/guoyuanchen/torchtitan/.cache/400m-base-shapes")

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
        # the dtype for encoders is torch.float32 (default dtype for Flux Model).
        # Otherwise, we use the same dtype as mixed precision training process.
        self._dtype = (
            TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        self.prepare_scheduler(job_config.scheduler)
        self.prepare_extra_modules(job_config)
        self.job_config = job_config

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.batch_size}, "
            f"global batch size {job_config.training.batch_size * dp_degree}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})."
        )
    
    def prepare_scheduler(self, config: Scheduler):
        self.scheduler_config = config

    def get_sigmas(self, timesteps, n_dim, dtype, device):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def next_batch(
        self, data_iterator: Iterable
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        data_load_start = time.perf_counter()
        input_dict = next(data_iterator)
        self.metrics_processor.data_loading_times.append(
            time.perf_counter() - data_load_start
        )

        device_type = utils.device_type
        input_dict = input_dict.to(device_type)
        # for k, _ in input_dict.items():
        #     if isinstance(input_dict[k], torch.Tensor):
        #         input_dict[k] = input_dict[k].to(device_type)
        return input_dict

    def prepare_encoder(self, job_config: JobConfig):
        model_cls = self.train_spec.encoder_cls
        model_args = self.train_spec.encoder_config[job_config.training.encoder_flavor]
        logger.info(
            f"Building {self.train_spec.name} encoder {job_config.training.encoder_flavor} with {model_args}"
        )
        model = model_cls.from_model_args(model_args).to(device=self.device)
        if job_config.training.encoder_pretrain_path:
            state_dict = torch.load(job_config.training.encoder_pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict['model'], strict=True)
            logger.info(f"Loaded encoder from {job_config.training.encoder_pretrain_path}")
        
        model.requires_grad_(False)
        model.eval()

        return model
    
    def prepare_image_encoder(self, image_encoder_type: str, image_encoder_model: str, image_encoder_return_the_nth_hidden_states: int):
        if image_encoder_type == "dinov2":
            image_encoder = DINOv2ImageEncoder(
                model_name=image_encoder_model,
                return_the_nth_hidden_states=image_encoder_return_the_nth_hidden_states,
            ).to(device=self.device, dtype=self._dtype)
        elif image_encoder_type == "dinov3":
            image_encoder = DINOv3ImageEncoder(
                model_name=image_encoder_model,
                return_the_nth_hidden_states=image_encoder_return_the_nth_hidden_states,
            ).to(device=self.device, dtype=self._dtype)
        elif image_encoder_type == "dinov2_without_pooler":
            image_encoder = DINOv2ImageEncoderWithoutPooler(
                model_name=image_encoder_model,
                return_the_nth_hidden_states=image_encoder_return_the_nth_hidden_states,
            ).to(device=self.device, dtype=self._dtype)
        elif image_encoder_type == "siglip2":
            image_encoder = SigLIP2ImageEncoder(
                model_name=image_encoder_model,
                return_the_nth_hidden_states=image_encoder_return_the_nth_hidden_states,
            ).to(device=self.device, dtype=self._dtype)
        else:
            raise ValueError(f"Invalid image encoder type: {image_encoder_type}")

        return image_encoder

    def prepare_extra_modules(self, job_config: JobConfig):
        logger.info("Preparing extra modules...")

        logger.info("Prepare image encoder")
        self.image_encoder = self.prepare_image_encoder(
            job_config.training.image_encoder_type,
            job_config.training.image_encoder_model,
            job_config.training.image_encoder_return_the_nth_hidden_states,
        )
        logger.info("Extra modules are prepared.")
    
    @torch.no_grad()
    def _select_foreground_condition_tokens(
        self,
        encoder_hidden_states: torch.Tensor,
        image_masks: torch.Tensor,
        dilation: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = encoder_hidden_states.shape
        _, height, width = image_masks.shape
        if dilation < 0:
            raise ValueError("foreground_condition_token_dilation must be non-negative")
        patch_size = getattr(self.image_encoder.model.config, "patch_size", None)
        if patch_size is None:
            patch_size = getattr(getattr(self.image_encoder.model.config, "vision_config", None), "patch_size", None)
        if patch_size is None:
            raise ValueError("Foreground condition tokens require image encoder config.patch_size")

        grid_h = height // patch_size
        grid_w = width // patch_size
        num_patch_tokens = grid_h * grid_w
        num_special_tokens = seq_len - num_patch_tokens
        if num_special_tokens < 0:
            raise ValueError(
                f"Image mask grid {grid_h}x{grid_w} has more patch tokens than encoder sequence length {seq_len}."
            )

        image_masks = image_masks[:, :grid_h * patch_size, :grid_w * patch_size]
        patch_masks = F.max_pool2d(
            image_masks.to(dtype=torch.float32).unsqueeze(1),
            kernel_size=patch_size,
            stride=patch_size,
        ).squeeze(1) > 0
        if dilation > 0:
            kernel_size = dilation * 2 + 1
            patch_masks = F.max_pool2d(
                patch_masks.to(dtype=torch.float32).unsqueeze(1),
                kernel_size=kernel_size,
                stride=1,
                padding=dilation,
            ).squeeze(1) > 0

        token_masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=encoder_hidden_states.device)
        if num_patch_tokens > 0:
            token_masks[:, num_special_tokens:] = patch_masks.reshape(batch_size, num_patch_tokens)

        selected_lens = token_masks.sum(dim=1, dtype=torch.int32)
        condition_cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=encoder_hidden_states.device)
        condition_cu_seqlens[1:] = torch.cumsum(selected_lens, dim=0)
        encoder_hidden_states = encoder_hidden_states[token_masks].reshape(-1, hidden_size)
        return encoder_hidden_states, condition_cu_seqlens

    @torch.no_grad()
    def prepare_conditions(self, batch: dict[str, torch.Tensor | Any], job_config: JobConfig) -> dict[str, torch.Tensor | Any]:
        cond_images = batch.images
        cond_images_processed = self.image_encoder.preprocess(cond_images)
        encoder_hidden_states = self.image_encoder(cond_images_processed)
        cond = {
            "encoder_hidden_states": encoder_hidden_states,
        }
        if job_config.training.use_foreground_condition_tokens:
            if batch.image_masks is None:
                raise ValueError("use_foreground_condition_tokens=True requires batch.image_masks")
            encoder_hidden_states, condition_cu_seqlens = self._select_foreground_condition_tokens(
                encoder_hidden_states,
                batch.image_masks,
                job_config.training.foreground_condition_token_dilation,
            )
            cond["encoder_hidden_states"] = encoder_hidden_states
            cond["condition_cu_seqlens"] = condition_cu_seqlens
        if batch.view_indices is not None:
            cond["view_indices"] = batch.view_indices
        if batch.mv_cu_seqlens is not None:
            cond["mv_cu_seqlens"] = batch.mv_cu_seqlens
        return cond

    def train_step(self, input_dict: OctreeBatch):
        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        model_parts = self.model_parts
        world_mesh = self.world_mesh
        parallel_dims = self.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage

        if parallel_dims.pp_enabled:
            raise NotImplementedError
        
        assert len(self.model_parts) == 1
        model = self.model_parts[0]

        latents = input_dict.layer_occupancy_flat
        cu_seqlens = input_dict.cu_seqlens
        conditions = self.prepare_conditions(input_dict, self.job_config)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents) * self.scheduler_config.noise_std

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        sigmas = compute_density_for_timestep_sampling(
            weighting_scheme=self.scheduler_config.t_sampling_scheme,
            batch_size=None if cu_seqlens is not None else latents.shape[0],
            cu_seqlens=cu_seqlens,
            logit_mean=self.scheduler_config.logit_mean,
            logit_std=self.scheduler_config.logit_std,
            uniform_ratio=self.scheduler_config.uniform_ratio,
            uniform_power=self.scheduler_config.uniform_power,
            uniform_min=self.scheduler_config.uniform_min,
            uniform_max=self.scheduler_config.uniform_max,
            dtype=latents.dtype,
            device=latents.device,
        )
        timesteps = sigmas * self.scheduler_config.num_train_timesteps
        while len(sigmas.shape) < latents.ndim:
            sigmas = sigmas.unsqueeze(-1)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=world_mesh["cp"],
                cp_buffers=[noise, latents, noisy_latents] + [model.pos_embed],
                cp_seq_dims=[2, 2, 2] + [-2],
                cp_no_restore_buffers={noise, latents, noisy_latents},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
                cp_enable_load_balance=False # must be False for full attention
            )
            if parallel_dims.cp_enabled
            else None
        )   
        
        model_pred, metrics = model.train_step(
            noisy_latents,
            timesteps,
            cu_seqlens=cu_seqlens,
            conditions=conditions,
            input_dict=input_dict,
            dtype=self._dtype,
        )

        self.metrics_processor.ntokens_since_last_log += metrics.get("num_tokens", 0)
        self.metrics_processor.num_flops_since_last_log += metrics.get("num_flops", 0)

        # in case we need loss weighting
        # weighting = metrics.get("weighting", None) or input_dict.get("weighting", None)
        weighting = torch.ones_like(sigmas)

        # flow matching loss from jit (x-pred and v-loss)
        if self.job_config.training.loss_type == "xpred-vloss":
            v_pred = (noisy_latents - model_pred) / torch.clamp(sigmas, min=0.05)
            target = (noisy_latents - latents) / torch.clamp(sigmas, min=0.05)
        elif self.job_config.training.loss_type == "vpred-vloss":
            v_pred = model_pred
            target = (noise - latents).detach()
        else:
            raise NotImplementedError("Loss type {} not implemented".format(self.job_config.training.loss_type))

        # origin flow matching loss
        # target = (noise - latents).detach()
        loss = torch.mean((weighting.float() * (v_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1)
        loss = loss.mean()
        loss.backward()

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=self.world_mesh["pp"] if parallel_dims.pp_enabled else None,
        )
        self.checkpointer.maybe_wait_for_staging()
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
        else:
            global_avg_loss = global_max_loss = loss.detach().item()
        
        self.metrics_processor.log(self.step, global_avg_loss, global_max_loss, extra_metrics={"grad_norm": grad_norm.item(), 'lr': lr})

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
                # logger.info(f"Step {self.step}")
                self.checkpointer.save(
                    self.step, force=(self.step == job_config.training.steps)
                )

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
    trainer: Optional[DiffusionTrainer] = None

    init_logger(log_file=os.path.join(config.job.dump_folder, "logs", f"rank{get_rank()}.log"))
    dump_config(config, os.path.join(config.job.dump_folder, "config.toml"))
    
    try:
        trainer = DiffusionTrainer(config)
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
