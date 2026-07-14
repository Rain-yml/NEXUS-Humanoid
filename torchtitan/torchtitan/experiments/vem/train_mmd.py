# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.experiments.vem.train_octdiff import (
    DiffusionTrainer,
    compute_density_for_timestep_sampling,
)
from torchtitan.config_manager import ConfigManager
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools.logging import logger
from torchtitan.experiments.vem.utils import get_rank, init_logger, dump_config


# ---------------------------------------------------------------------------
# DDPM posterior helper
# ---------------------------------------------------------------------------

def _q_posterior_step(z_t, x0_hat, sigma_t, sigma_s, eps=1e-8):
    """Sample z_s ~ q(z_s | z_t, x0_hat) using the Gaussian posterior.

    Flow-matching noise schedule: z_t = (1-t)*x0 + t*eps,
    so alpha(t) = 1 - t, sigma(t) = t.

    Args:
        z_t:      Noised data at noise level sigma_t. Shape ``(N, D)``.
        x0_hat:   Predicted clean data. Shape ``(N, D)``.
        sigma_t:  Current noise level, per-token broadcastable ``(N, 1, ...)``.
        sigma_s:  Target noise level (must be < sigma_t), same shape as sigma_t.
        eps:      Numerical stability constant.

    Returns:
        z_s: Posterior sample at noise level sigma_s.
    """
    sigma_t_f = sigma_t.to(torch.float32)
    sigma_s_f = sigma_s.to(torch.float32)

    # Ensure s < t for valid posterior (defensive clamp)
    sigma_s_f = torch.minimum(sigma_s_f, sigma_t_f - eps)
    sigma_s_f = sigma_s_f.clamp_min(eps)

    alpha_t = 1.0 - sigma_t_f
    alpha_s = 1.0 - sigma_s_f

    # Transition coefficients: z_t = alpha_{t|s} * z_s + sigma_{t|s} * noise
    alpha_t_given_s = alpha_t / alpha_s.clamp_min(eps)
    sigma_t_given_s2 = (sigma_t_f.square() - alpha_t_given_s.square() * sigma_s_f.square()).clamp_min(eps)

    # Posterior variance
    inv_sigma_s2 = 1.0 / sigma_s_f.square().clamp_min(eps)
    inv_sigma_t_given_s2 = 1.0 / sigma_t_given_s2
    posterior_var = 1.0 / (inv_sigma_s2 + alpha_t_given_s.square() * inv_sigma_t_given_s2)

    # Posterior mean
    z_t_f = z_t.to(torch.float32)
    x0_f = x0_hat.to(torch.float32)
    term1 = (alpha_t_given_s * inv_sigma_t_given_s2) * z_t_f
    term2 = (alpha_s * inv_sigma_s2) * x0_f
    mu = posterior_var * (term1 + term2)

    noise = torch.randn_like(z_t_f)
    z_s = mu + posterior_var.sqrt() * noise
    return z_s.to(z_t.dtype)


# ---------------------------------------------------------------------------
# Optimizer wrapper (unchanged from DMD)
# ---------------------------------------------------------------------------

class MMDOptimizersWrapper:
    """
    Wrapper for optimizers that bypasses DCP's get_optimizer_state_dict.

    DCP's get_optimizer_state_dict doesn't work well with nested FSDP modules
    (like MMDModel with teacher/student/fake_score). This wrapper uses native
    PyTorch optimizer state_dict instead, which works but loses some DCP features.
    """

    def __init__(self, optimizers_container):
        self._container = optimizers_container

    def __getattr__(self, name):
        return getattr(self._container, name)

    def state_dict(self):
        """Return native optimizer state_dicts instead of using DCP's function."""
        return {
            f"optimizer_{i}": opt.state_dict()
            for i, opt in enumerate(self._container.optimizers)
        }

    def load_state_dict(self, state_dict):
        """Load native optimizer state_dicts."""
        for i, opt in enumerate(self._container.optimizers):
            key = f"optimizer_{i}"
            if key in state_dict:
                opt.load_state_dict(state_dict[key])


# ---------------------------------------------------------------------------
# MMD Trainer
# ---------------------------------------------------------------------------

class MMDTrainer(DiffusionTrainer):
    """
    MMD (Modified Distribution Matching Distillation) Trainer.

    Differences from DMDTrainer:
    - Multi-step VSD uses DDPM posterior q(x_s | x_t, x0_hat) instead of
      independent noise for perturbation.
    - Fake score step has optional teacher regularization loss.

    Alternates between:
    - Student update: VSD loss (match teacher distribution)
    - Fake score update: DSM loss (learn generated data distribution)

    Supports both xpred-vloss and vpred-vloss by converting to x0 space.
    """

    @record
    def __init__(self, job_config):
        super().__init__(job_config)
        self._last_vsd_loss = 0.0
        self._last_dsm_loss = 0.0

        wrapped_optimizers = MMDOptimizersWrapper(self.optimizers)
        self.checkpointer.states["optimizer"] = wrapped_optimizers

    def _model_pred_to_x0(self, model_pred, x_t, sigmas):
        """
        Convert model prediction to x0 based on loss_type.

        For flow matching: x_t = (1 - sigma) * x_0 + sigma * eps
        - xpred-vloss: model outputs x_0 directly
        - vpred-vloss: model outputs v = eps - x_0, so x_0 = x_t - sigma * v
        """
        if self.job_config.training.loss_type == "xpred-vloss":
            return model_pred
        elif self.job_config.training.loss_type == "vpred-vloss":
            return x_t - sigmas * model_pred
        else:
            raise NotImplementedError(f"Loss type {self.job_config.training.loss_type} not supported")

    def _x0_to_model_target(self, x0, x_t, sigmas, eps):
        """
        Convert x0 target to appropriate model target based on loss_type.

        - xpred-vloss: target is x0
        - vpred-vloss: target is v = eps - x0
        """
        if self.job_config.training.loss_type == "xpred-vloss":
            return x0
        elif self.job_config.training.loss_type == "vpred-vloss":
            return eps - x0
        else:
            raise NotImplementedError(f"Loss type {self.job_config.training.loss_type} not supported")

    @torch.no_grad()
    def prepare_conditions(self, batch, job_config):
        """
        Override to include unconditional embeddings for CFG.

        Returns:
            dict with:
            - encoder_hidden_states: conditional embeddings
            - encoder_hidden_states_uncond: unconditional embeddings (for CFG)
        """
        cond_images = batch.images
        cond_images_processed = self.image_encoder.preprocess(cond_images)
        encoder_hidden_states = self.image_encoder(cond_images_processed)

        uncond_images = batch.uncond_images
        uncond_images_processed = self.image_encoder.preprocess(uncond_images)
        encoder_hidden_states_uncond = self.image_encoder(uncond_images_processed)

        cond = {
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_uncond": encoder_hidden_states_uncond,
        }

        if batch.view_indices is not None:
            cond["view_indices"] = batch.view_indices
        if batch.mv_cu_seqlens is not None:
            cond["mv_cu_seqlens"] = batch.mv_cu_seqlens

        return cond

    def _generate_student_input(self, latents, cu_seqlens):
        """
        Generate student input and corresponding timestep/sigma.

        For single-step (student_sample_steps == 1):
            Returns pure noise at t_max with sigma=1.0 (original behavior).
            Extra info dict is None.

        For multi-step (student_sample_steps > 1):
            Computes uniform base sigmas [N/N, (N-1)/N, ..., 1/N], applies timestep
            shift: shifted = shift * t / (1 + (shift - 1) * t), then randomly samples
            one shifted sigma per batch element.

            Returns extra info dict with per-sample sigma values for posterior.

        Returns:
            input_student: Input tensor to the student network
            t_student: Timestep tensor (per-token, expanded via cu_seqlens)
            sigma_student: Sigma tensor (per-token, with trailing dims for broadcasting)
            extra_info: None (single-step) or dict with:
                - sigma_t_per_sample: (B,) per-sample sigma_t
                - sigma_t_next_per_sample: (B,) per-sample sigma_t_next
        """
        student_sample_steps = self.job_config.distill.student_sample_steps

        if student_sample_steps <= 1:
            # Single-step: pure noise at t_max
            noise_student = torch.randn_like(latents) * self.scheduler_config.noise_std
            t_student = torch.full(
                (noise_student.shape[0],),
                float(self.scheduler_config.num_train_timesteps),
                device=latents.device,
                dtype=latents.dtype,
            )
            sigma_student = torch.ones_like(t_student)
            while len(sigma_student.shape) < latents.ndim:
                sigma_student = sigma_student.unsqueeze(-1)
            input_student = noise_student
            return input_student, t_student, sigma_student, None
        else:
            # Multi-step: compute uniform base sigmas and apply shift
            N = student_sample_steps
            shift = self.job_config.distill.student_t_shift

            # Uniform base sigmas: [N/N, (N-1)/N, ..., 1/N] = [1.0, 0.75, 0.5, 0.25] for N=4
            base_sigmas = torch.tensor(
                [(N - i) / N for i in range(N)],
                device=latents.device, dtype=latents.dtype,
            )
            # Apply timestep shift: shifted = shift * t / (1 + (shift - 1) * t)
            shifted_sigmas = shift * base_sigmas / (1.0 + (shift - 1.0) * base_sigmas)

            eps_student = torch.randn_like(latents) * self.scheduler_config.noise_std

            # Randomly sample a shifted sigma for each sample in the batch
            batch_size = cu_seqlens.shape[0] - 1
            idx = torch.randint(0, N, (batch_size,), device=latents.device)
            sigma_per_sample = shifted_sigmas[idx]

            # Compute sigma_t_next: the next (smaller) sigma in the schedule
            next_idx = torch.clamp(idx + 1, max=N - 1)
            sigma_t_next_per_sample = shifted_sigmas[next_idx].clone()
            sigma_t_next_per_sample[idx == N - 1] = 0.0

            # Expand per-sample sigma to per-token sigma via cu_seqlens
            seq_len = cu_seqlens[1:] - cu_seqlens[:-1]
            sigma_student = torch.repeat_interleave(sigma_per_sample, seq_len)
            t_student = sigma_student * self.scheduler_config.num_train_timesteps

            while len(sigma_student.shape) < latents.ndim:
                sigma_student = sigma_student.unsqueeze(-1)

            # Forward process: x_t = (1 - sigma) * x_0 + sigma * eps
            input_student = (1.0 - sigma_student) * latents + sigma_student * eps_student

            extra_info = {
                "sigma_t_per_sample": sigma_per_sample,
                "sigma_t_next_per_sample": sigma_t_next_per_sample,
            }
            return input_student, t_student, sigma_student, extra_info

    def train_step(self, input_dict):
        """Override train_step with MMD training logic."""

        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        model = self.model_parts[0]
        student_update_freq = self.job_config.distill.student_update_freq

        # Check if within CFG distillation warmup phase (updates student every iteration)
        student_cfg_steps = self.job_config.distill.student_cfg_steps
        is_cfg_warmup = student_cfg_steps > 0 and self.step < student_cfg_steps

        # Determine if this is a student update step
        is_student_step = (self.step % student_update_freq == 0)

        # Prepare data
        latents = input_dict.layer_occupancy_flat
        cu_seqlens = input_dict.cu_seqlens
        conditions = self.prepare_conditions(input_dict, self.job_config)

        if is_cfg_warmup:
            model.setup_grad_requirements(update_student=True)
            loss, metrics = self._student_cfg_update_step(
                latents, cu_seqlens, conditions, input_dict, self.step
            )
            self._last_vsd_loss = metrics.get("vsd_loss", 0.0)
            is_student_step = True
        elif is_student_step:
            model.setup_grad_requirements(update_student=True)
            loss, metrics = self._student_update_step(
                latents, cu_seqlens, conditions, input_dict, self.step
            )
            self._last_vsd_loss = metrics.get("vsd_loss", 0.0)
        else:
            model.setup_grad_requirements(update_student=False)
            loss, metrics = self._fake_score_update_step(
                latents, cu_seqlens, conditions, input_dict, self.step
            )
            self._last_dsm_loss = metrics.get("dsm_loss", 0.0)

        metrics["vsd_loss"] = self._last_vsd_loss
        metrics["dsm_loss"] = self._last_dsm_loss

        self.metrics_processor.ntokens_since_last_log += metrics.get("num_tokens", latents.shape[0])
        self.metrics_processor.num_flops_since_last_log += metrics.get("num_flops", 100)

        loss.backward()

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=self.world_mesh["pp"] if self.parallel_dims.pp_enabled else None,
        )

        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        if self.job_config.ema.enabled and is_student_step:
            self.ema.update(model)

        self._log_metrics(loss, metrics, grad_norm, lr, is_student_step)

    def _student_update_step(self, latents, cu_seqlens, conditions, input_dict, step):
        """
        Student update via VSD loss.

        1. Generate from student (single-step or multi-step)
        2. Perturb generated data:
           - Single-step: independent noise (original behavior)
           - Multi-step: DDPM posterior q(x_s | x_t, x0_hat)
        3. Get teacher & fake_score predictions (no grad), convert to x0
        4. Compute VSD loss in x0 space
        """
        model = self.model_parts[0]

        # === Step 1: Student generates x0 ===
        input_student, t_student, sigma_student, extra_info = self._generate_student_input(latents, cu_seqlens)

        student_pred = model.forward(
            x_t=input_student.to(self._dtype),
            t=t_student,
            centers=input_dict.layer_parent_centers_flat,
            depths=input_dict.layer_depths_flat,
            cu_seqlens_q=cu_seqlens,
            num_layers_per_mesh=input_dict.num_layers_per_mesh,
            encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
            num_vertices=input_dict.num_vertices,
            quad_ratios=input_dict.quad_ratios,
            view_indices=conditions.get("view_indices"),
            mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
            use_model="student",
        )

        gen_data = self._model_pred_to_x0(student_pred, input_student, sigma_student)

        # === Step 2: Perturb generated data for VSD ===
        if extra_info is not None:
            # Multi-step: use DDPM posterior q(x_s | x_t, x0_hat)
            batch_size = cu_seqlens.shape[0] - 1
            seq_len = cu_seqlens[1:] - cu_seqlens[:-1]

            # Uniformly sample sigma_s in (sigma_t_next, sigma_t) per sample
            sigma_t_ps = extra_info["sigma_t_per_sample"]
            sigma_t_next_ps = extra_info["sigma_t_next_per_sample"]
            u = torch.rand(batch_size, device=latents.device, dtype=latents.dtype)
            sigma_s_per_sample = sigma_t_next_ps + u * (sigma_t_ps - sigma_t_next_ps)

            # Expand to per-token
            sigmas = torch.repeat_interleave(sigma_s_per_sample, seq_len)
            t = sigmas * self.scheduler_config.num_train_timesteps
            while len(sigmas.shape) < gen_data.ndim:
                sigmas = sigmas.unsqueeze(-1)

            # DDPM posterior: z_s ~ q(z_s | z_t=input_student, x0_hat=gen_data)
            perturbed_data = _q_posterior_step(
                z_t=input_student,
                x0_hat=gen_data,
                sigma_t=sigma_student,
                sigma_s=sigmas,
            )
        else:
            # Single-step: independent noise (original behavior)
            eps = torch.randn_like(gen_data)
            sigmas = compute_density_for_timestep_sampling(
                weighting_scheme=self.scheduler_config.t_sampling_scheme,
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
            t = sigmas * self.scheduler_config.num_train_timesteps
            while len(sigmas.shape) < gen_data.ndim:
                sigmas = sigmas.unsqueeze(-1)

            perturbed_data = (1.0 - sigmas) * gen_data + sigmas * eps

        # === Step 3: Teacher and fake_score predictions (no grad) ===
        with torch.no_grad():
            teacher_pred = model.forward(
                x_t=perturbed_data.to(self._dtype),
                t=t,
                centers=input_dict.layer_parent_centers_flat,
                depths=input_dict.layer_depths_flat,
                cu_seqlens_q=cu_seqlens,
                num_layers_per_mesh=input_dict.num_layers_per_mesh,
                encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
                num_vertices=input_dict.num_vertices,
                quad_ratios=input_dict.quad_ratios,
                view_indices=conditions.get("view_indices"),
                mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                use_model="teacher",
            )
            teacher_x0 = self._model_pred_to_x0(teacher_pred, perturbed_data, sigmas)

            # Apply CFG if enabled
            if self.job_config.distill.guidance_scale is not None:
                teacher_pred_uncond = model.forward(
                    x_t=perturbed_data.to(self._dtype),
                    t=t,
                    centers=input_dict.layer_parent_centers_flat,
                    depths=input_dict.layer_depths_flat,
                    cu_seqlens_q=cu_seqlens,
                    num_layers_per_mesh=input_dict.num_layers_per_mesh,
                    encoder_hidden_states=conditions["encoder_hidden_states_uncond"].to(self._dtype),
                    num_vertices=input_dict.num_vertices,
                    quad_ratios=input_dict.quad_ratios,
                    view_indices=conditions.get("view_indices"),
                    mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                    use_model="teacher",
                )
                teacher_x0_uncond = self._model_pred_to_x0(teacher_pred_uncond, perturbed_data, sigmas)
                guidance_scale = self.job_config.distill.guidance_scale
                teacher_x0 = teacher_x0_uncond + guidance_scale * (teacher_x0 - teacher_x0_uncond)

            teacher_x0 = torch.clamp(teacher_x0, -1, 1)

            fake_score_pred = model.forward(
                x_t=perturbed_data.to(self._dtype),
                t=t,
                centers=input_dict.layer_parent_centers_flat,
                depths=input_dict.layer_depths_flat,
                cu_seqlens_q=cu_seqlens,
                num_layers_per_mesh=input_dict.num_layers_per_mesh,
                encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
                num_vertices=input_dict.num_vertices,
                quad_ratios=input_dict.quad_ratios,
                view_indices=conditions.get("view_indices"),
                mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                use_model="fake_score",
            )
            fake_score_x0 = self._model_pred_to_x0(fake_score_pred, perturbed_data, sigmas)

            # === Step 4: VSD Loss (in x0 space) ===
            diff_abs_mean = (gen_data.float() - teacher_x0.float()).abs().mean(dim=-1, keepdim=True)
            w = (1.0 / (diff_abs_mean + 1e-6)).to(gen_data.dtype)

            vsd_grad = (fake_score_x0 - teacher_x0) * w
            vsd_target = gen_data - vsd_grad

        vsd_loss = F.mse_loss(gen_data.float(), vsd_target.float())

        return vsd_loss, {"vsd_loss": vsd_loss.item(), "step_type": "student"}

    def _student_cfg_update_step(self, latents, cu_seqlens, conditions, input_dict, step):
        """
        Student CFG distillation step (warmup phase).

        Directly trains student to match teacher's CFG-guided predictions.
        No fake_score involvement, no other losses. Uses v-loss for proper weighting.
        """
        model = self.model_parts[0]

        eps = torch.randn_like(latents)
        sigmas = compute_density_for_timestep_sampling(
            weighting_scheme=self.scheduler_config.t_sampling_scheme,
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
        t = sigmas * self.scheduler_config.num_train_timesteps
        while len(sigmas.shape) < latents.ndim:
            sigmas = sigmas.unsqueeze(-1)

        x_t = (1.0 - sigmas) * latents + sigmas * eps

        with torch.no_grad():
            teacher_pred_cond = model.forward(
                x_t=x_t.to(self._dtype),
                t=t,
                centers=input_dict.layer_parent_centers_flat,
                depths=input_dict.layer_depths_flat,
                cu_seqlens_q=cu_seqlens,
                num_layers_per_mesh=input_dict.num_layers_per_mesh,
                encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
                num_vertices=input_dict.num_vertices,
                quad_ratios=input_dict.quad_ratios,
                view_indices=conditions.get("view_indices"),
                mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                use_model="teacher",
            )
            teacher_x0_cond = self._model_pred_to_x0(teacher_pred_cond, x_t, sigmas)

            teacher_pred_uncond = model.forward(
                x_t=x_t.to(self._dtype),
                t=t,
                centers=input_dict.layer_parent_centers_flat,
                depths=input_dict.layer_depths_flat,
                cu_seqlens_q=cu_seqlens,
                num_layers_per_mesh=input_dict.num_layers_per_mesh,
                encoder_hidden_states=conditions["encoder_hidden_states_uncond"].to(self._dtype),
                num_vertices=input_dict.num_vertices,
                quad_ratios=input_dict.quad_ratios,
                view_indices=conditions.get("view_indices"),
                mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                use_model="teacher",
            )
            teacher_x0_uncond = self._model_pred_to_x0(teacher_pred_uncond, x_t, sigmas)

            guidance_scale = self.job_config.distill.guidance_scale
            if guidance_scale is not None:
                teacher_x0_cfg = teacher_x0_uncond + guidance_scale * (teacher_x0_cond - teacher_x0_uncond)
            else:
                teacher_x0_cfg = teacher_x0_cond
            teacher_x0_cfg = torch.clamp(teacher_x0_cfg, -1, 1)

        student_pred = model.forward(
            x_t=x_t.to(self._dtype),
            t=t,
            centers=input_dict.layer_parent_centers_flat,
            depths=input_dict.layer_depths_flat,
            cu_seqlens_q=cu_seqlens,
            num_layers_per_mesh=input_dict.num_layers_per_mesh,
            encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
            num_vertices=input_dict.num_vertices,
            quad_ratios=input_dict.quad_ratios,
            view_indices=conditions.get("view_indices"),
            mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
            use_model="student",
        )
        student_x0 = self._model_pred_to_x0(student_pred, x_t, sigmas)

        if self.job_config.training.loss_type == "xpred-vloss":
            v_pred = (x_t - student_x0) / torch.clamp(sigmas, min=0.05)
            v_target = (x_t - teacher_x0_cfg) / torch.clamp(sigmas, min=0.05)
            cfg_loss = F.mse_loss(v_pred.float(), v_target.float())
        elif self.job_config.training.loss_type == "vpred-vloss":
            v_target = eps - teacher_x0_cfg
            cfg_loss = F.mse_loss(student_pred.float(), v_target.float())
        else:
            raise NotImplementedError(f"Loss type {self.job_config.training.loss_type} not supported for CFG distill")

        return cfg_loss, {"vsd_loss": cfg_loss.item(), "step_type": "student_cfg"}

    def _fake_score_update_step(self, latents, cu_seqlens, conditions, input_dict, step):
        """
        Fake score update via denoising score matching.

        1. Generate from student (no grad) - single-step or multi-step
        2. Perturb generated data (independent noise, unchanged)
        3. Train fake_score to denoise (in model's native prediction type)
        4. Optional: teacher regularization loss (fake_score ≈ teacher)
        """
        model = self.model_parts[0]

        # === Step 1: Generate from student (no grad) ===
        with torch.no_grad():
            input_student, t_student, sigma_student, _extra = self._generate_student_input(latents, cu_seqlens)

            student_pred = model.forward(
                x_t=input_student.to(self._dtype),
                t=t_student,
                centers=input_dict.layer_parent_centers_flat,
                depths=input_dict.layer_depths_flat,
                cu_seqlens_q=cu_seqlens,
                num_layers_per_mesh=input_dict.num_layers_per_mesh,
                encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
                num_vertices=input_dict.num_vertices,
                quad_ratios=input_dict.quad_ratios,
                view_indices=conditions.get("view_indices"),
                mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                use_model="student",
            )
            gen_data = self._model_pred_to_x0(student_pred, input_student, sigma_student)
            gen_data = torch.clamp(gen_data, -1, 1)

        # === Step 2: Perturb generated data (independent noise) ===
        eps = torch.randn_like(gen_data)
        sigmas = compute_density_for_timestep_sampling(
            weighting_scheme=self.scheduler_config.t_sampling_scheme,
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
        t = sigmas * self.scheduler_config.num_train_timesteps
        while len(sigmas.shape) < gen_data.ndim:
            sigmas = sigmas.unsqueeze(-1)

        perturbed_data = (1.0 - sigmas) * gen_data + sigmas * eps

        # === Step 3: Fake score prediction ===
        fake_score_pred = model.forward(
            x_t=perturbed_data.to(self._dtype),
            t=t,
            centers=input_dict.layer_parent_centers_flat,
            depths=input_dict.layer_depths_flat,
            cu_seqlens_q=cu_seqlens,
            num_layers_per_mesh=input_dict.num_layers_per_mesh,
            encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
            num_vertices=input_dict.num_vertices,
            quad_ratios=input_dict.quad_ratios,
            view_indices=conditions.get("view_indices"),
            mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
            use_model="fake_score",
        )

        # === Step 4: DSM Loss ===
        if self.job_config.training.loss_type == "xpred-vloss":
            v_pred = (perturbed_data - fake_score_pred) / torch.clamp(sigmas, min=0.05)
            v_target = (perturbed_data - gen_data.detach()) / torch.clamp(sigmas, min=0.05)
            dsm_loss = F.mse_loss(v_pred.float(), v_target.float())
        elif self.job_config.training.loss_type == "vpred-vloss":
            v_target = (eps - gen_data.detach())
            dsm_loss = F.mse_loss(fake_score_pred.float(), v_target.float())
        else:
            raise NotImplementedError(f"Loss type {self.job_config.training.loss_type} not supported for DSM")

        # === Step 5: Optional teacher regularization loss ===
        fake_score_reg = getattr(self.job_config.distill, "fake_score_reg", False)
        if fake_score_reg:
            with torch.no_grad():
                teacher_pred = model.forward(
                    x_t=perturbed_data.to(self._dtype),
                    t=t,
                    centers=input_dict.layer_parent_centers_flat,
                    depths=input_dict.layer_depths_flat,
                    cu_seqlens_q=cu_seqlens,
                    num_layers_per_mesh=input_dict.num_layers_per_mesh,
                    encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
                    num_vertices=input_dict.num_vertices,
                    quad_ratios=input_dict.quad_ratios,
                    view_indices=conditions.get("view_indices"),
                    mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                    use_model="teacher",
                )

            if self.job_config.training.loss_type == "xpred-vloss":
                v_fake = (perturbed_data - fake_score_pred) / torch.clamp(sigmas, min=0.05)
                v_teacher = (perturbed_data - teacher_pred) / torch.clamp(sigmas, min=0.05)
                reg_loss = F.mse_loss(v_fake.float(), v_teacher.float())
            elif self.job_config.training.loss_type == "vpred-vloss":
                reg_loss = F.mse_loss(fake_score_pred.float(), teacher_pred.float())
            else:
                raise NotImplementedError(f"Loss type {self.job_config.training.loss_type} not supported for reg")

            dsm_loss = dsm_loss + reg_loss

        return dsm_loss, {"dsm_loss": dsm_loss.item(), "step_type": "fake_score"}

    def _log_metrics(self, loss, metrics, grad_norm, lr, is_student_step):
        """Log training metrics including both student and fake_score losses."""
        if not self.metrics_processor.should_log(self.step):
            return

        if is_student_step:
            current_loss = metrics.get("vsd_loss", 0.0)
        else:
            current_loss = metrics.get("dsm_loss", 0.0)

        if (self.parallel_dims.dp_replicate_enabled or
            self.parallel_dims.dp_shard_enabled or
            self.parallel_dims.cp_enabled or
            self.ft_manager.enabled):
            loss_tensor = torch.tensor(current_loss, device=loss.device)
            ft_pg = self.ft_manager.replicate_pg if self.ft_manager.enabled else None
            global_avg_loss, global_max_loss = (
                dist_utils.dist_mean(loss_tensor, self.world_mesh["dp_cp"], ft_pg),
                dist_utils.dist_max(loss_tensor, self.world_mesh["dp_cp"], ft_pg),
            )
        else:
            global_avg_loss = global_max_loss = current_loss

        step_type = "student" if is_student_step else "fake_score"
        extra_metrics = {
            "grad_norm": grad_norm.item(),
            "lr": lr,
            "vsd_loss": metrics.get("vsd_loss", 0.0),
            "dsm_loss": metrics.get("dsm_loss", 0.0),
            "is_student_step": 1.0 if is_student_step else 0.0,
        }
        self.metrics_processor.log(self.step, global_avg_loss, global_max_loss, extra_metrics=extra_metrics)

        logger.info(f"[MMD] step_type: {step_type}  vsd_loss: {self._last_vsd_loss:.6f}  dsm_loss: {self._last_dsm_loss:.6f}")


if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Optional[MMDTrainer] = None

    init_logger(log_file=os.path.join(config.job.dump_folder, "logs", f"rank{get_rank()}.log"))
    dump_config(config, os.path.join(config.job.dump_folder, "config.toml"))

    try:
        trainer = MMDTrainer(config)
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
