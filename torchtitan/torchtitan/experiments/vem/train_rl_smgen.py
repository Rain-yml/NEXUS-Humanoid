# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""RL finetuning entrypoint for VEM stage-2 smart mesh generation."""

from __future__ import annotations

import contextlib
import copy
import os
import time
import types
from datetime import timedelta
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.experiments.vem  # noqa: F401 - registers train specs.
import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.config_manager import ConfigManager
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.vem.ema import ShardedEMA
from torchtitan.experiments.vem.rollout_debug import (
    resolve_rollout_log_dir,
    save_rollout_meshes,
)
from torchtitan.experiments.vem.rewards import build_reward
from torchtitan.experiments.vem.pipelines.v2f import V2FPipeline
from torchtitan.experiments.vem.rl_samplers import build_rollout_scheduler
from torchtitan.experiments.vem.train_sm_gen import DiffusionTrainer
from torchtitan.experiments.vem.utils import dump_config, get_rank, init_logger
from torchtitan.tools.logging import logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)
import torchtitan.components.ft as ft

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:  # PEFT is optional until RL LoRA is enabled.
    LoraConfig = None
    PeftModel = None
    get_peft_model = None


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def _tensor_item(x: torch.Tensor | float | int) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().item())
    return float(x)


class RLTrainState(Stateful):
    def __init__(self, trainer: "RLSMGenTrainer") -> None:
        self.trainer = trainer

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.trainer.step,
            "rl_epoch": self.trainer.rl_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.trainer.step = state_dict["step"]
        self.trainer.rl_epoch = state_dict.get("rl_epoch", 0)


class RLSMGenTrainer(DiffusionTrainer):
    """DiffusionNFT-style x0-space RL finetuning for packed VEM meshes."""

    @record
    def __init__(self, job_config):
        if not job_config.rl.enabled:
            raise ValueError("train_rl_smgen.py requires [rl].enabled = true")
        self._install_lora_parallelize_hook(job_config)
        self._lora_ema_pending = False
        try:
            super().__init__(job_config)
        finally:
            self._restore_parallelize_hook()
        self.rl_epoch = 0
        self.rl_scheduler = build_rollout_scheduler(
            sampler_type=job_config.rl.sampler_type,
            num_train_timesteps=self.scheduler_config.num_train_timesteps,
            shift=job_config.rl.scheduler_shift,
        )
        self.rl_pipeline = V2FPipeline(
            sm_dit=self.model_parts[0],
            image_encoder=self.image_encoder,
            scheduler=self.rl_scheduler,
            latent_mean=self.latent_mean,
            latent_std=self.latent_std,
        )
        self.reward_fn = self._build_reward_fn(job_config.rl)
        self.rollout_log_dir = resolve_rollout_log_dir(
            job_config.job.dump_folder,
        )
        self._copy_default_adapter_to_old()

        states = {"train_state": RLTrainState(self)}
        if job_config.rl.lora_ema_enabled:
            self.lora_ema = ShardedEMA(
                model=self.model_parts[0],
                beta=job_config.rl.lora_ema_beta,
                update_after_step=job_config.rl.lora_ema_update_after_step,
                update_every=job_config.rl.lora_ema_update_every,
            )
            states["lora_ema"] = self.lora_ema
            logger.info("Initialized LoRA EMA for RL finetuning")
        else:
            self.lora_ema = None

        if self.job_config.ema.enabled and hasattr(self, "ema"):
            states["ema"] = self.ema
        if hasattr(self.checkpointer, "states"):
            self.checkpointer.states.update(states)

        if self.parallel_dims.pp_enabled:
            raise NotImplementedError("RL finetuning currently supports non-PP models only")

    def _build_reward_fn(self, rl):
        reward_kwargs = {
            "min_loop_len": rl.reward_min_loop_len,
            "workers": rl.reward_workers,
        }
        if rl.reward_names:
            reward_weights = rl.reward_weights if rl.reward_weights else None
            return build_reward(
                "weighted",
                reward_names=rl.reward_names,
                reward_weights=reward_weights,
                reward_weighting_mode=rl.reward_weighting_mode,
                **reward_kwargs,
            )
        return build_reward(
            rl.reward_name,
            **reward_kwargs,
        )

    def _prepare_model_args(self, model_args, job_config):
        if not job_config.rl.pretrained_path:
            return model_args
        model_args = copy.copy(model_args)
        if hasattr(model_args, "pretrained_path"):
            model_args.pretrained_path = None
        return model_args

    def _after_model_init(self, model, model_args, job_config) -> None:
        del model_args
        if not job_config.rl.pretrained_path:
            return
        self._load_rl_pretrained_checkpoint(model, job_config.rl.pretrained_path)

    def _extract_pretrained_model_state(self, checkpoint: Any) -> dict[str, Any]:
        if (
            isinstance(checkpoint, dict)
            and isinstance(checkpoint.get("ema"), dict)
            and isinstance(checkpoint["ema"].get("model"), dict)
        ):
            return checkpoint["ema"]["model"]
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
            return checkpoint["model"]
        if isinstance(checkpoint, dict):
            return checkpoint
        raise TypeError(
            "RL pretrained checkpoint must be a state_dict or contain model/ema.model weights"
        )

    def _patch_pretrained_state_for_model(
        self,
        model: torch.nn.Module,
        state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        state_dict = dict(state_dict)
        current_state = model.state_dict()
        source_norm_keys = {
            self._normalize_lora_state_key(key) for key in state_dict.keys()
        }
        for key, value in current_state.items():
            if (
                self._normalize_lora_state_key(key) == "view_embed.weight"
                and "view_embed.weight" not in source_norm_keys
            ):
                state_dict[key] = torch.zeros_like(value)
        return state_dict

    def _load_rl_pretrained_checkpoint(
        self,
        model: torch.nn.Module,
        pretrained_path: str,
    ) -> None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        logger.info(f"Loading RL pretrained checkpoint from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = self._extract_pretrained_model_state(checkpoint)
        state_dict = self._patch_pretrained_state_for_model(model, state_dict)
        self._remap_lora_state_dict(model, state_dict)
        set_model_state_dict(
            model,
            model_state_dict=state_dict,
            options=StateDictOptions(full_state_dict=True, strict=False),
        )
        logger.info("Loaded RL pretrained checkpoint into policy model")

    def _install_lora_parallelize_hook(self, job_config):
        train_spec = train_spec_module.get_train_spec(job_config.model.name)
        base_parallelize_fn = train_spec.parallelize_fn
        trainer_ref = self

        def parallelize_with_lora(model, world_mesh, parallel_dims, cfg):
            model = trainer_ref._setup_lora(model, cfg)
            return base_parallelize_fn(model, world_mesh, parallel_dims, cfg)

        self._original_parallelize_fn = base_parallelize_fn
        self._patched_train_spec = train_spec
        train_spec.parallelize_fn = parallelize_with_lora

    def _restore_parallelize_hook(self):
        train_spec = getattr(self, "_patched_train_spec", None)
        original = getattr(self, "_original_parallelize_fn", None)
        if train_spec is not None and original is not None:
            train_spec.parallelize_fn = original

    def _setup_lora(self, model: torch.nn.Module, job_config):
        if get_peft_model is None or LoraConfig is None:
            raise ImportError(
                "PEFT is required for RL LoRA finetuning but is not installed in this environment"
            )

        rl = job_config.rl
        model.requires_grad_(False)
        lora_config = LoraConfig(
            r=rl.lora_rank,
            lora_alpha=rl.lora_alpha,
            lora_dropout=rl.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=list(rl.lora_target_modules),
        )
        if rl.lora_path:
            if PeftModel is None:
                raise ImportError("PEFT PeftModel is required to load rl.lora_path")
            peft_model = PeftModel.from_pretrained(model, rl.lora_path, is_trainable=True)
        else:
            peft_model = get_peft_model(model, lora_config)
        peft_model.add_adapter("old", lora_config)
        peft_model.set_adapter("default")
        self._set_adapter_trainability(peft_model, trainable_adapter="default")
        self._patch_lora_state_loading(peft_model, model)
        peft_model.train()
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        logger.info(f"Applied LoRA to RL policy with {trainable:,} trainable parameters")
        return peft_model

    def _normalize_lora_state_key(self, key: str) -> str:
        parts = [part for part in key.split(".") if part != "_orig_mod"]
        key = ".".join(parts)
        for prefix in ("base_model.model.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        return key.replace(".base_layer.", ".")

    def _remap_lora_state_dict(self, module: torch.nn.Module, state_dict: dict[str, Any]):
        current_state = module.state_dict()
        target_by_norm = {}
        for target_key in current_state:
            if "lora_" in target_key:
                continue
            target_by_norm.setdefault(self._normalize_lora_state_key(target_key), target_key)

        for source_key in list(state_dict.keys()):
            target_key = target_by_norm.get(self._normalize_lora_state_key(source_key))
            if target_key is None or target_key in state_dict:
                continue
            source_value = state_dict[source_key]
            target_value = current_state[target_key]
            if hasattr(source_value, "shape") and hasattr(target_value, "shape"):
                if tuple(source_value.shape) != tuple(target_value.shape):
                    continue
            state_dict[target_key] = source_value

    def _patch_lora_state_loading(self, peft_model: torch.nn.Module, base_model: torch.nn.Module):
        def pre_hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            del prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            self._remap_lora_state_dict(module, state_dict)

        peft_model.register_load_state_dict_pre_hook(pre_hook)

        if hasattr(base_model, "_load_pretrained_to_model"):
            def load_pretrained_with_lora(base_self, model, state_dict, model_name):
                from torch.distributed.checkpoint.state_dict import (
                    StateDictOptions,
                    set_model_state_dict,
                )

                mapped_state_dict = dict(state_dict)
                self._remap_lora_state_dict(model, mapped_state_dict)
                set_model_state_dict(
                    model,
                    model_state_dict=mapped_state_dict,
                    options=StateDictOptions(full_state_dict=True, strict=False),
                )
                logger.info(f"{model_name}: loaded pretrained weights with LoRA key remapping")

            base_model._load_pretrained_to_model = types.MethodType(
                load_pretrained_with_lora,
                base_model,
            )

    def _set_adapter_trainability(self, model: torch.nn.Module, trainable_adapter: str):
        marker = f".{trainable_adapter}."
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(marker in name)

    def _adapter_param_map(self, model: torch.nn.Module, adapter: str):
        marker = f".{adapter}."
        return {
            name.replace(marker, ".{adapter}."): param
            for name, param in model.named_parameters()
            if marker in name
        }

    def _base_model(self):
        return _unwrap_model(self.model_parts[0])

    @contextlib.contextmanager
    def _adapter(self, name: Optional[str]):
        model = self._base_model()
        if name is None:
            if not hasattr(model, "disable_adapter"):
                yield
                return
            with model.disable_adapter():
                yield
            if hasattr(model, "set_adapter"):
                model.set_adapter("default")
                self._set_adapter_trainability(model, trainable_adapter="default")
            return
        if hasattr(model, "set_adapter"):
            model.set_adapter(name)
        yield
        if name != "default" and hasattr(model, "set_adapter"):
            model.set_adapter("default")
        if hasattr(model, "set_adapter"):
            self._set_adapter_trainability(model, trainable_adapter="default")

    def _copy_default_adapter_to_old(self):
        model = self._base_model()
        if not hasattr(model, "set_adapter"):
            return
        default_params = self._adapter_param_map(model, "default")
        old_params = self._adapter_param_map(model, "old")
        if set(default_params) != set(old_params):
            raise RuntimeError(
                f"Default/old LoRA adapter parameter count mismatch: "
                f"{len(default_params)} vs {len(old_params)}"
            )
        with torch.no_grad():
            for key, src in default_params.items():
                dst = old_params[key]
                dst.copy_(src.detach())
        self._set_adapter_trainability(model, trainable_adapter="default")
        model.set_adapter("default")

    @torch.no_grad()
    def _update_old_adapter(self):
        rl = self.job_config.rl
        if rl.old_policy_decay_type == "none":
            return
        decay = rl.old_policy_decay
        model = self._base_model()
        if not hasattr(model, "set_adapter"):
            return

        default_params = self._adapter_param_map(model, "default")
        old_params = self._adapter_param_map(model, "old")
        if set(default_params) != set(old_params):
            raise RuntimeError("Default/old LoRA adapter parameter names do not match")
        for key, cur_p in default_params.items():
            old_p = old_params[key]
            old_p.mul_(decay).add_(cur_p.detach(), alpha=1.0 - decay)
        self._set_adapter_trainability(model, trainable_adapter="default")
        model.set_adapter("default")

    def _model_pred_to_x0(self, model_pred, x_t, sigmas):
        if self.job_config.training.loss_type == "xpred-vloss":
            return model_pred
        if self.job_config.training.loss_type == "vpred-vloss":
            return x_t - sigmas * model_pred
        raise NotImplementedError(f"Loss type {self.job_config.training.loss_type} not supported")

    def _model_forward_x0(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
        conditions: dict[str, Any],
        adapter: Optional[str],
        requires_grad: bool,
    ) -> torch.Tensor:
        model = self.model_parts[0]
        ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()
        with self._adapter(adapter), ctx:
            pred = model(
                hidden_states=x_t.to(self._dtype),
                timesteps=timesteps,
                cu_seqlens=cu_seqlens,
                encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
                hidden_states_position=positions,
                cu_seqlens_encoder=None,
                view_indices=conditions.get("view_indices"),
                mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
            )
        sigma = (timesteps / self.scheduler_config.num_train_timesteps).to(dtype=x_t.dtype)
        while len(sigma.shape) < x_t.ndim:
            sigma = sigma.unsqueeze(-1)
        return self._model_pred_to_x0(pred, x_t, sigma)

    def _repeat_packed_batch_for_rollouts(self, batch: dict[str, Any], k: int) -> dict[str, Any]:
        if k == 1:
            if "group_ids" not in batch:
                batch["group_ids"] = torch.arange(
                    batch["cu_seqlens"].shape[0] - 1,
                    device=batch["vertices"].device,
                )
            if "input_images" not in batch:
                batch["input_images"] = batch["image"]
            if "input_instance_ids" not in batch:
                batch["input_instance_ids"] = batch.get(
                    "instance_ids",
                    [str(i) for i in range(batch["cu_seqlens"].shape[0] - 1)],
                )
            return batch

        cu_vertices = batch["cu_seqlens"]
        vertex_lengths = cu_vertices[1:] - cu_vertices[:-1]
        vertices: list[torch.Tensor] = []
        images: list[torch.Tensor] = []
        instance_ids: list[str] = []
        vertex_offsets = [0]
        decoder_position: list[torch.Tensor] = []

        for sample_idx, seq_len in enumerate(vertex_lengths.tolist()):
            vertex_start = int(cu_vertices[sample_idx].item())
            vertex_end = int(cu_vertices[sample_idx + 1].item())

            for sample_rep in range(k):
                vertices.append(batch["vertices"][vertex_start:vertex_end])
                images.append(batch["image"][sample_idx])
                vertex_offsets.append(vertex_offsets[-1] + seq_len)
                instance_id = batch.get("instance_ids", [str(sample_idx)])[sample_idx]
                instance_ids.append(f"{instance_id}#rl{sample_rep}")
                if "decoder_position" in batch:
                    decoder_position.append(batch["decoder_position"][vertex_start:vertex_end])

        repeated = {
            "vertices": torch.cat(vertices, dim=0),
            "image": torch.stack(images, dim=0),
            "cu_seqlens": torch.tensor(vertex_offsets, dtype=torch.int32, device=batch["vertices"].device),
            "instance_ids": instance_ids,
            "group_ids": torch.arange(len(vertex_lengths), device=batch["vertices"].device).repeat_interleave(k),
            "input_images": batch["image"],
            "input_instance_ids": batch.get(
                "instance_ids",
                [str(i) for i in range(len(vertex_lengths))],
            ),
        }
        if decoder_position:
            repeated["decoder_position"] = torch.cat(decoder_position, dim=0)
        return repeated

    def _reward_generated(
        self,
        x0_generated: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        instance_ids: list[str],
        decoder_positions: Optional[torch.Tensor] = None,
    ):
        if self.encoder is None:
            raise RuntimeError(
                "RL mesh recovery requires a prepared VAE encoder/decoder."
            )
        reward_timeout = getattr(self.job_config.rl, "reward_timeout", 60)
        start_time = time.time()

        x0_dec = x0_generated * self.latent_std.to(x0_generated.dtype) + self.latent_mean.to(x0_generated.dtype)
        meshes = self.encoder.decode_face(
            vertex_latents=x0_dec,
            vertex_positions=positions,
            cu_seqlens=cu_seqlens,
            mode=self.job_config.rl.quad_decoder_mode,
            decoder_positions=decoder_positions,
        )

        elapsed = time.time() - start_time
        if elapsed > reward_timeout:
            rank = dist.get_rank() if dist.is_initialized() else 0
            n_samples = cu_seqlens.shape[0] - 1
            logger.warning(
                f"[Rank {rank}] Mesh recovery took {elapsed:.1f}s "
                f"(timeout={reward_timeout}s), returning zero rewards"
            )
            from torchtitan.experiments.vem.rewards.base import MeshRewardOutput
            return meshes, MeshRewardOutput(
                rewards=torch.zeros(n_samples, device=x0_generated.device),
                metrics={},
            )

        metadata = [{"instance_id": iid} for iid in instance_ids]
        reward_out = self.reward_fn(meshes, metadata=metadata, device=x0_generated.device)

        total_elapsed = time.time() - start_time
        if total_elapsed > reward_timeout:
            rank = dist.get_rank() if dist.is_initialized() else 0
            n_samples = cu_seqlens.shape[0] - 1
            logger.warning(
                f"[Rank {rank}] Reward computation took {total_elapsed:.1f}s "
                f"(timeout={reward_timeout}s), returning zero rewards"
            )
            return meshes, MeshRewardOutput(
                rewards=torch.zeros(n_samples, device=x0_generated.device),
                metrics={},
            )

        return meshes, reward_out

    def _collect_rollouts(self, data_iterator: Iterable):
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(f"[Rank {rank}] Starting _collect_rollouts at step {self.step}")

        batches = []
        rewards = []
        reward_metrics: dict[str, list[float]] = {}
        rl = self.job_config.rl
        group_offset = 0

        for batch_idx in range(rl.num_batches_per_epoch):
            batch_start_time = time.time()
            logger.debug(f"[Rank {rank}] Processing batch {batch_idx}/{rl.num_batches_per_epoch}")
            batch = self.next_batch(data_iterator)
            batch = self._repeat_packed_batch_for_rollouts(batch, rl.num_samples_per_input)
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            if "group_ids" not in batch:
                batch["group_ids"] = torch.arange(
                    batch["cu_seqlens"].shape[0] - 1,
                    device=self.device,
                )
            batch["group_ids"] = batch["group_ids"] + group_offset
            group_offset = int(batch["group_ids"].max().item()) + 1
            conditions = self.prepare_conditions(batch, self.job_config)
            positions = batch["vertices"]
            cu_seqlens = batch["cu_seqlens"].to(dtype=torch.int32)
            token_dim = self.latent_mean.shape[-1]
            noise = torch.randn(
                (positions.shape[0], token_dim),
                device=positions.device,
                dtype=self.latent_mean.dtype,
            )

            if self.job_config.training.loss_type == "xpred-vloss":
                prediction = "x"
            elif self.job_config.training.loss_type == "vpred-vloss":
                prediction = "v"
            else:
                raise NotImplementedError(
                    f"Loss type {self.job_config.training.loss_type} not supported"
                )

            with self._adapter("old"):
                rollout = self.rl_pipeline.rollout(
                    vertices=positions,
                    cu_seqlens=cu_seqlens,
                    prediction=prediction,
                    conditions=conditions,
                    guidance_scale=1.0,
                    num_inference_steps=rl.num_rollout_steps,
                    device=self.device,
                    dtype=self._dtype,
                    scheduler=self.rl_scheduler,
                    noise=noise,
                    noise_std=self.scheduler_config.noise_std,
                    return_latents_every_step=True,
                )

            instance_ids = batch.get("instance_ids", [])
            if not instance_ids:
                instance_ids = [str(i) for i in range(cu_seqlens.shape[0] - 1)]

            reward_start_time = time.time()
            logger.debug(f"[Rank {rank}] Starting mesh recovery for batch {batch_idx}")
            meshes, reward_out = self._reward_generated(
                rollout.x0,
                positions,
                cu_seqlens,
                instance_ids,
                decoder_positions=batch.get("decoder_position"),
            )
            reward_elapsed = time.time() - reward_start_time
            logger.debug(f"[Rank {rank}] Completed mesh recovery for batch {batch_idx} in {reward_elapsed:.2f}s")
            if rl.save_rollout_interval > 0 and (self.step % rl.save_rollout_interval == 0 or self.step == 1):
                save_rollout_meshes(
                    self.rollout_log_dir,
                    step=self.step,
                    rank=dist.get_rank() if dist.is_initialized() else get_rank(),
                    batch_idx=batch_idx,
                    meshes=meshes,
                    instance_ids=instance_ids,
                    input_images=batch.get("input_images"),
                    input_instance_ids=batch.get("input_instance_ids"),
                )
            rewards.append(reward_out.rewards.detach())
            for key, value in reward_out.metrics.items():
                reward_metrics.setdefault(key, []).append(_tensor_item(value))

            batch_elapsed = time.time() - batch_start_time
            logger.debug(f"[Rank {rank}] Completed batch {batch_idx} in {batch_elapsed:.2f}s")
            batches.append(
                {
                    "x0": rollout.x0.detach(),
                    "noise": noise.detach(),
                    "positions": positions,
                    "cu_seqlens": cu_seqlens,
                    "conditions": {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in conditions.items()},
                    "group_ids": batch.get(
                        "group_ids",
                        torch.arange(cu_seqlens.shape[0] - 1, device=positions.device),
                    ),
                    "rewards": reward_out.rewards.detach(),
                    "instance_ids": instance_ids,
                }
            )

        logger.info(f"[Rank {rank}] Completed all batches, computing advantages")
        rewards_tensor = torch.cat(rewards, dim=0)
        adv_start_time = time.time()
        advantages, adv_metrics = self._compute_advantages(
            rewards_tensor,
            torch.cat([b["group_ids"] for b in batches], dim=0),
        )
        adv_elapsed = time.time() - adv_start_time
        logger.info(f"[Rank {rank}] Completed advantages computation in {adv_elapsed:.2f}s")
        offset = 0
        for batch in batches:
            n = batch["rewards"].shape[0]
            batch["advantages"] = advantages[offset : offset + n].to(batch["x0"].device)
            offset += n

        metrics = self._reward_summary(rewards_tensor)
        metrics.update(
            {
                key: float(sum(values) / max(len(values), 1))
                for key, values in reward_metrics.items()
            }
        )
        metrics.update(adv_metrics)
        logger.info(f"[Rank {rank}] Completed _collect_rollouts")
        return batches, metrics

    def _all_gather_variable_1d(self, tensor: torch.Tensor, group=None):
        if not dist.is_initialized():
            return tensor, [tensor.numel()], 0

        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return tensor, [tensor.numel()], 0

        rank = dist.get_rank(group=group)
        length = torch.tensor([tensor.numel()], dtype=torch.long, device=tensor.device)
        lengths = [torch.zeros_like(length) for _ in range(world_size)]
        dist.all_gather(lengths, length, group=group)
        lengths_list = [int(x.item()) for x in lengths]
        max_len = max(lengths_list)
        padded = torch.zeros(max_len, dtype=tensor.dtype, device=tensor.device)
        padded[: tensor.numel()] = tensor
        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded, group=group)
        gathered_tensor = torch.cat(
            [part[:lengths_list[idx]] for idx, part in enumerate(gathered)],
            dim=0,
        )
        return gathered_tensor, lengths_list, rank

    def _normalize_advantages(self, rewards: torch.Tensor, group_ids: torch.Tensor):
        rl = self.job_config.rl
        rewards = rewards.float()
        group_ids = group_ids.to(device=rewards.device)
        if not rl.use_per_input_advantage_norm:
            mean = rewards.mean()
            std = rewards.std(unbiased=False)
            advantages = (rewards - mean) / (std + 1e-4)
            zero_std_ratio = float(std <= 1e-8)
        else:
            advantages = torch.zeros_like(rewards)
            zero_std = 0
            groups = torch.unique(group_ids)
            for gid in groups:
                mask = group_ids == gid
                group_rewards = rewards[mask]
                mean = group_rewards.mean()
                std = group_rewards.std(unbiased=False)
                if std <= 1e-8:
                    zero_std += 1
                advantages[mask] = (group_rewards - mean) / (std + 1e-4)
            zero_std_ratio = zero_std / max(int(groups.numel()), 1)
        advantages = torch.clamp(advantages, -rl.adv_clip_max, rl.adv_clip_max)
        return advantages, {
            "rl/advantages_mean": advantages.mean().item(),
            "rl/advantages_std": advantages.std(unbiased=False).item(),
            "rl/zero_std_ratio": zero_std_ratio,
        }

    def _compute_advantages(self, rewards: torch.Tensor, group_ids: torch.Tensor):
        if not dist.is_initialized():
            return self._normalize_advantages(rewards, group_ids)

        if (
            not self.parallel_dims.dp_replicate_enabled
            and not self.parallel_dims.dp_shard_enabled
            and not self.parallel_dims.cp_enabled
        ):
            return self._normalize_advantages(rewards, group_ids)

        group = self.world_mesh["dp_cp"].get_group()
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return self._normalize_advantages(rewards, group_ids)

        rank = dist.get_rank(group=group)
        local_group_count = (
            int(group_ids.max().item()) + 1 if group_ids.numel() > 0 else 0
        )
        group_count_tensor = torch.tensor(
            [local_group_count],
            dtype=torch.long,
            device=group_ids.device,
        )
        group_counts = [torch.zeros_like(group_count_tensor) for _ in range(world_size)]
        dist.all_gather(group_counts, group_count_tensor, group=group)
        group_offsets = [0]
        for count in group_counts[:-1]:
            group_offsets.append(group_offsets[-1] + int(count.item()))
        global_group_ids = group_ids.to(dtype=torch.long) + group_offsets[rank]

        global_rewards, reward_lengths, rank = self._all_gather_variable_1d(
            rewards.float(),
            group=group,
        )
        global_group_ids, _group_lengths, _ = self._all_gather_variable_1d(
            global_group_ids,
            group=group,
        )
        global_advantages, metrics = self._normalize_advantages(
            global_rewards,
            global_group_ids,
        )
        local_start = sum(reward_lengths[:rank])
        local_end = local_start + rewards.numel()
        return global_advantages[local_start:local_end], metrics

    def _reward_summary(self, rewards: torch.Tensor) -> dict[str, float]:
        rewards = rewards.float()
        if rewards.numel() == 0:
            return {}
        return {
            "reward/mean": rewards.mean(),
            "reward/std": rewards.std(unbiased=False),
            "reward/min": rewards.min(),
            "reward/max": rewards.max(),
        }

    def _sample_per_token_timestep(self, cu_seqlens: torch.Tensor, dtype: torch.dtype, device):
        batch_size = cu_seqlens.shape[0] - 1
        sigma_per_sample = torch.rand(batch_size, dtype=dtype, device=device)
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        sigmas = torch.repeat_interleave(sigma_per_sample, lengths)
        timesteps = sigmas * self.scheduler_config.num_train_timesteps
        while len(sigmas.shape) < 2:
            sigmas = sigmas.unsqueeze(-1)
        return timesteps, sigmas

    def _rl_inner_train(self, rollout_batches: list[dict[str, Any]]):
        metrics_accum: dict[str, list[float]] = {}
        last_loss = torch.tensor(0.0, device=self.device)
        grad_norm = torch.tensor(0.0, device=self.device)
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        for _ in range(self.job_config.rl.num_inner_epochs):
            for rollout in rollout_batches:
                self.optimizers.zero_grad()
                loss, step_metrics = self._rl_loss_step(rollout)
                last_loss = loss.detach()
                loss.backward()
                grad_norm = dist_utils.clip_grad_norm_(
                    [p for m in self.model_parts for p in m.parameters()],
                    self.job_config.training.max_norm,
                    foreach=True,
                    pp_mesh=None,
                )
                self.checkpointer.maybe_wait_for_staging()
                self.optimizers.step()
                self.lr_schedulers.step()
                if self.lora_ema is not None:
                    self.lora_ema.update(self.model_parts[0])

                for key, value in step_metrics.items():
                    metrics_accum.setdefault(key, []).append(_tensor_item(value))

        metrics = {key: float(sum(values) / max(len(values), 1)) for key, values in metrics_accum.items()}
        metrics["grad_norm"] = _tensor_item(grad_norm)
        metrics["lr"] = lr
        return last_loss, metrics

    def _rl_loss_step(self, rollout: dict[str, Any]):
        x0 = rollout["x0"]
        cu_seqlens = rollout["cu_seqlens"]
        positions = rollout["positions"]
        conditions = rollout["conditions"]
        advantages = rollout["advantages"]

        timesteps, sigmas = self._sample_per_token_timestep(
            cu_seqlens,
            dtype=x0.dtype,
            device=x0.device,
        )
        noise = torch.randn_like(x0.float()).to(dtype=x0.dtype)
        x_t = (1.0 - sigmas) * x0 + sigmas * noise

        old_x0 = self._model_forward_x0(
            x_t,
            timesteps,
            cu_seqlens,
            positions,
            conditions,
            adapter="old",
            requires_grad=False,
        )
        current_x0 = self._model_forward_x0(
            x_t,
            timesteps,
            cu_seqlens,
            positions,
            conditions,
            adapter="default",
            requires_grad=True,
        )
        ref_x0 = self._model_forward_x0(
            x_t,
            timesteps,
            cu_seqlens,
            positions,
            conditions,
            adapter=None,
            requires_grad=False,
        )

        rl = self.job_config.rl
        beta = rl.beta
        if beta <= 0:
            raise ValueError("rl.beta must be > 0 for the DiffusionNFT policy loss")
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        token_adv = torch.repeat_interleave(advantages.to(x0.device), seq_lens).to(dtype=x0.dtype)
        r = torch.clamp((token_adv / rl.adv_clip_max) / 2.0 + 0.5, 0.0, 1.0).unsqueeze(-1)

        positive_x0 = beta * current_x0 + (1.0 - beta) * old_x0.detach()
        negative_x0 = (1.0 + beta) * old_x0.detach() - beta * current_x0

        with torch.no_grad():
            positive_weight = (positive_x0.float() - x0.float()).abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
            negative_weight = (negative_x0.float() - x0.float()).abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)

        positive_loss = ((positive_x0.float() - x0.float()) ** 2 / positive_weight).mean(dim=-1)
        negative_loss = ((negative_x0.float() - x0.float()) ** 2 / negative_weight).mean(dim=-1)
        policy_loss_tokens = (
            r.squeeze(-1).float() * positive_loss / beta
            + (1.0 - r.squeeze(-1).float()) * negative_loss / beta
        )
        policy_loss = policy_loss_tokens.mean() * rl.adv_clip_max

        kl_loss = ((current_x0.float() - ref_x0.float()) ** 2).mean()
        total_loss = policy_loss + rl.kl_coeff * kl_loss

        self.metrics_processor.ntokens_since_last_log += int(x0.shape[0])
        if conditions["encoder_hidden_states"].dim() == 3:
            s_enc = conditions["encoder_hidden_states"].shape[1]
        else:
            s_enc = (conditions["encoder_hidden_states"].shape[0] // (cu_seqlens.shape[0] - 1))
        if getattr(self.model_parts[0], "mv_mode", False) and conditions.get("mv_cu_seqlens") is not None:
            mv_cu_seqlens = conditions["mv_cu_seqlens"]
            num_views = mv_cu_seqlens[1:] - mv_cu_seqlens[:-1]
            s_enc = (conditions["encoder_hidden_states"].shape[1] * num_views).tolist()
        self.metrics_processor.num_flops_since_last_log += self.model_parts[0].flops(cu_seqlens, s_enc)

        return total_loss, {
            "rl/policy_loss": policy_loss.detach(),
            "rl/kl_loss": kl_loss.detach(),
            "rl/total_loss": total_loss.detach(),
            "rl/old_deviate": ((current_x0.detach().float() - old_x0.float()) ** 2).mean(),
            "rl/ref_kl": kl_loss.detach(),
            "rl/lora_ema_decay": self.lora_ema.get_current_decay() if self.lora_ema is not None else 0.0,
        }

    def _log_rl_metrics(self, loss, metrics):
        if not self.metrics_processor.should_log(self.step):
            return

        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
            or self.ft_manager.enabled
        ):
            loss_tensor = loss.detach()
            ft_pg = self.ft_manager.replicate_pg if self.ft_manager.enabled else None
            global_avg_loss, global_max_loss = (
                dist_utils.dist_mean(loss_tensor, self.world_mesh["dp_cp"], ft_pg),
                dist_utils.dist_max(loss_tensor, self.world_mesh["dp_cp"], ft_pg),
            )
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = dist_utils.dist_mean(value.detach(), self.world_mesh["dp_cp"], ft_pg)
                else:
                    metrics[key] = dist_utils.dist_mean(torch.tensor(value, device=self.device), self.world_mesh["dp_cp"], ft_pg)
        else:
            global_avg_loss = global_max_loss = _tensor_item(loss)
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            extra_metrics=metrics,
        )

    @record
    def train(self):
        job_config = self.job_config
        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"RL finetuning starts at step {self.step + 1}.")

        with maybe_enable_profiling(
            job_config,
            global_step=self.step,
        ) as torch_profiler, maybe_enable_memory_snapshot(
            job_config,
            global_step=self.step,
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
                self.rl_epoch += 1
                self.gc_handler.run(self.step)

                rollout_batches, rollout_metrics = self._collect_rollouts(data_iterator)
                loss, train_metrics = self._rl_inner_train(rollout_batches)
                metrics = {**rollout_metrics, **train_metrics}
                self._log_rl_metrics(loss, metrics)

                if self.step % job_config.rl.old_policy_update_every == 0:
                    self._update_old_adapter()

                self.checkpointer.save(
                    self.step,
                    force=(self.step == job_config.training.steps),
                )

                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                        world_mesh=self.world_mesh,
                    )

        if dist.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        self.metrics_processor.close()
        logger.info("RL finetuning completed")


if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Optional[RLSMGenTrainer] = None

    init_logger(log_file=os.path.join(config.job.dump_folder, "logs", f"rank{get_rank()}.log"))
    dump_config(config, os.path.join(config.job.dump_folder, "config.toml"))

    try:
        trainer = RLSMGenTrainer(config)
        if config.checkpoint.create_seed_checkpoint:
            assert int(os.environ["WORLD_SIZE"]) == 1
            assert config.checkpoint.enable_checkpoint
            trainer.checkpointer.save(curr_step=0, force=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    finally:
        if trainer:
            trainer.close()

        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Process group destroyed.")
