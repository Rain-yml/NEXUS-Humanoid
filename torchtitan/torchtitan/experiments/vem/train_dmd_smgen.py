# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DMD Distillation Training Script for Octree Diffusion"""

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.experiments.vem.train_sm_gen import (
    DiffusionTrainer,
    compute_density_for_timestep_sampling,
)
from torchtitan.config_manager import ConfigManager
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools.logging import logger
from torchtitan.experiments.vem.utils import get_rank, init_logger, dump_config


class DMDOptimizersWrapper:
    """
    Wrapper for optimizers that bypasses DCP's get_optimizer_state_dict.
    
    DCP's get_optimizer_state_dict doesn't work well with nested FSDP modules
    (like DMDModel with teacher/student/fake_score). This wrapper uses native
    PyTorch optimizer state_dict instead, which works but loses some DCP features.
    
    For DMD training, we trade off DCP-compatible optimizer checkpointing for
    working checkpoints. Model weights are still saved/loaded correctly.
    """
    
    def __init__(self, optimizers_container):
        self._container = optimizers_container
    
    def __getattr__(self, name):
        # Delegate all other attributes to the underlying container
        return getattr(self._container, name)
    
    def state_dict(self):
        """Return native optimizer state_dicts instead of using DCP's function."""
        # Use native PyTorch optimizer state_dict
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


class DMDTrainer(DiffusionTrainer):
    """
    DMD Distillation Trainer for Octree Diffusion.
    
    Alternates between:
    - Student update: VSD loss (match teacher distribution)
    - Fake score update: DSM loss (learn generated data distribution)
    
    Supports both xpred-vloss and vpred-vloss by converting to x0 space.
    """
    
    @record
    def __init__(self, job_config):
        super().__init__(job_config)
        # Track recent losses for logging both
        self._last_vsd_loss = 0.0
        self._last_dsm_loss = 0.0
        
        # Wrap optimizers to bypass DCP's get_optimizer_state_dict which doesn't
        # work with nested FSDP modules (DMDModel has teacher/student/fake_score)
        wrapped_optimizers = DMDOptimizersWrapper(self.optimizers)
        # Update checkpointer's states to use wrapped optimizers
        self.checkpointer.states["optimizer"] = wrapped_optimizers
    
    def _model_pred_to_x0(self, model_pred, x_t, sigmas):
        """
        Convert model prediction to x0 based on loss_type.
        
        For flow matching: x_t = (1 - sigma) * x_0 + sigma * eps
        - xpred-vloss: model outputs x_0 directly
        - vpred-vloss: model outputs v = eps - x_0, so x_0 = x_t - sigma * v
        """
        if self.job_config.training.loss_type == "xpred-vloss":
            # Model directly predicts x0
            return model_pred
        elif self.job_config.training.loss_type == "vpred-vloss":
            # Model predicts v, convert to x0: x_0 = x_t - sigma * v
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
        # Conditional embeddings (same as parent)
        cond_images = batch["image"]
        cond_images_processed = self.image_encoder.preprocess(cond_images)
        encoder_hidden_states = self.image_encoder(cond_images_processed)
        
        # Unconditional embeddings for CFG
        uncond_images = torch.full_like(cond_images, 0.5)
        uncond_images_processed = self.image_encoder.preprocess(uncond_images)
        encoder_hidden_states_uncond = self.image_encoder(uncond_images_processed)
        
        cond = {
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_uncond": encoder_hidden_states_uncond,
        }

        if 'view_indices' in batch:
            cond['view_indices'] = batch['view_indices']
        if 'mv_cu_seqlens' in batch:
            cond['mv_cu_seqlens'] = batch['mv_cu_seqlens']

        return cond
    
    def _generate_student_input(self, latents, cu_seqlens):
        """
        Generate student input and corresponding timestep/sigma.
        
        For single-step (student_sample_steps == 1):
            Returns pure noise at t_max with sigma=1.0 (original behavior).
        
        For multi-step (student_sample_steps > 1):
            Computes uniform base sigmas [N/N, (N-1)/N, ..., 1/N], applies timestep
            shift: shifted = shift * t / (1 + (shift - 1) * t), then randomly samples
            one shifted sigma per batch element. The student learns to predict x0
            from each noise level independently.
        
        Returns:
            input_student: Input tensor to the student network
            t_student: Timestep tensor (per-token, expanded via cu_seqlens)
            sigma_student: Sigma tensor (per-token, with trailing dims for broadcasting)
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
            
            # Expand per-sample sigma to per-token sigma via cu_seqlens
            seq_len = cu_seqlens[1:] - cu_seqlens[:-1]
            sigma_student = torch.repeat_interleave(sigma_per_sample, seq_len)
            t_student = sigma_student * self.scheduler_config.num_train_timesteps
            
            while len(sigma_student.shape) < latents.ndim:
                sigma_student = sigma_student.unsqueeze(-1)
            
            # Forward process: x_t = (1 - sigma) * x_0 + sigma * eps
            input_student = (1.0 - sigma_student) * latents + sigma_student * eps_student
        
        return input_student, t_student, sigma_student
    
    def train_step(self, input_dict):
        """Override train_step with DMD training logic."""

        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        
        model = self.model_parts[0]
        student_update_freq = self.job_config.distill.student_update_freq
        
        # Check if within CFG distillation warmup phase (updates student every iteration)
        student_cfg_steps = self.job_config.distill.student_cfg_steps
        is_cfg_warmup = student_cfg_steps > 0 and self.step < student_cfg_steps
        
        # Determine if this is a student update step
        # step % freq == 0 → student update (following reference script convention)
        is_student_step = (self.step % student_update_freq == 0)
        
        # Prepare data
        latents, cu_seqlens = model.get_latents(self.encoder, input_dict, mean=self.latent_mean, std=self.latent_std)
        conditions = self.prepare_conditions(input_dict, self.job_config)
        
        if is_cfg_warmup:
            # CFG distillation: update student every iteration, ignore is_student_step
            model.setup_grad_requirements(update_student=True)
            loss, metrics = self._student_cfg_update_step(
                latents, cu_seqlens, conditions, input_dict, self.step
            )
            self._last_vsd_loss = metrics.get("vsd_loss", 0.0)
            is_student_step = True  # For EMA and logging
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
        
        # Add both losses to metrics for logging
        metrics["vsd_loss"] = self._last_vsd_loss
        metrics["dsm_loss"] = self._last_dsm_loss
        
        # Update metrics processor with token/flops counts for logging
        self.metrics_processor.ntokens_since_last_log += metrics.get("num_tokens", latents.shape[0])
        self.metrics_processor.num_flops_since_last_log += metrics.get("num_flops", 100)
        
        loss.backward()
        
        # Gradient clipping
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=self.world_mesh["pp"] if self.parallel_dims.pp_enabled else None,
        )
        
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()
        
        # EMA update (tracks student + fake_score automatically)
        if self.job_config.ema.enabled and is_student_step:
            self.ema.update(model)
        
        # Logging
        self._log_metrics(loss, metrics, grad_norm, lr, is_student_step)
    
    def _student_update_step(self, latents, cu_seqlens, conditions, input_dict, step):
        """
        Student update via VSD loss.
        
        1. Generate from student (single-step from pure noise, or multi-step from noised data)
        2. Perturb generated data at random t
        3. Get teacher & fake_score predictions (no grad), convert to x0
        4. Compute VSD loss in x0 space
        """
        model = self.model_parts[0]
        
        # === Step 1: Student generates x0 ===
        input_student, t_student, sigma_student = self._generate_student_input(latents, cu_seqlens)
        
        # Student forward (outputs in model's native prediction type)
        student_pred = model.forward(
            hidden_states=input_student.to(self._dtype),
            timesteps=t_student,
            cu_seqlens=cu_seqlens,
            encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
            hidden_states_position=input_dict["vertices"],
            cu_seqlens_encoder=None,
            view_indices=conditions.get("view_indices"),
            mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
            quad_ratios=input_dict.get("quad_ratios"),
            use_model="student",
        )
        # Convert to x0 (gen_data)
        gen_data = self._model_pred_to_x0(student_pred, input_student, sigma_student)
        
        # === Step 2: Perturb generated data for VSD ===
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
        
        # Forward process: x_t = (1 - sigma) * x_0 + sigma * eps
        perturbed_data = (1.0 - sigmas) * gen_data + sigmas * eps
        
        # === Step 3: Teacher and fake_score predictions (no grad) ===
        with torch.no_grad():

            # Conditional teacher prediction
            teacher_pred = model.forward(
                hidden_states=perturbed_data.to(self._dtype),
                timesteps=t,
                cu_seqlens=cu_seqlens,
                encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
                hidden_states_position=input_dict["vertices"],
                cu_seqlens_encoder=None,
                view_indices=conditions.get("view_indices"),
                mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                quad_ratios=input_dict.get("quad_ratios"),
                use_model="teacher",
            )
            teacher_x0 = self._model_pred_to_x0(teacher_pred, perturbed_data, sigmas)
            
            # Apply CFG if enabled
            if self.job_config.distill.guidance_scale is not None:

                # Unconditional teacher prediction for CFG
                teacher_pred_uncond = model.forward(
                    hidden_states=perturbed_data.to(self._dtype),
                    timesteps=t,
                    cu_seqlens=cu_seqlens,
                    encoder_hidden_states=conditions["encoder_hidden_states_uncond"].to(self._dtype),
                    hidden_states_position=input_dict["vertices"],
                    cu_seqlens_encoder=None,
                    view_indices=conditions.get("view_indices"),
                    mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                    quad_ratios=input_dict.get("quad_ratios"),
                    use_model="teacher",
                )
                teacher_x0_uncond = self._model_pred_to_x0(teacher_pred_uncond, perturbed_data, sigmas)
                guidance_scale = self.job_config.distill.guidance_scale
                teacher_x0 = teacher_x0_uncond + guidance_scale * (teacher_x0 - teacher_x0_uncond)
            
            # Fake score prediction
            fake_score_pred = model.forward(
                hidden_states=perturbed_data.to(self._dtype),
                timesteps=t,
                cu_seqlens=cu_seqlens,
                encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
                hidden_states_position=input_dict["vertices"],
                cu_seqlens_encoder=None,
                view_indices=conditions.get("view_indices"),
                mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                quad_ratios=input_dict.get("quad_ratios"),
                use_model="fake_score",
            )
            fake_score_x0 = self._model_pred_to_x0(fake_score_pred, perturbed_data, sigmas)
        
            # === Step 4: VSD Loss (in x0 space) ===
            # VSD gradient direction: (fake_score_x0 - teacher_x0)
            # This pushes gen_data toward teacher_x0 via SGD update
            
            # Compute adaptive weight in fp32 for numerical stability
            # Weight is inversely proportional to prediction error:
            # - Large error (early training) → small weight → damped gradients
            # - Small error (late training) → large weight → amplified gradients
            diff_abs_mean = (gen_data.float() - teacher_x0.float()).abs().mean(dim=-1, keepdim=True)
            w = 1.0 / (diff_abs_mean + 1e-6)
            
            # Corrected VSD loss with adaptive scaling
            vsd_grad = (fake_score_x0.float() - teacher_x0.float()) * w
            vsd_target = gen_data.float() - vsd_grad

        vsd_loss = F.mse_loss(gen_data.float(), vsd_target.float())
        
        return vsd_loss, {"vsd_loss": vsd_loss.item(), "step_type": "student"}
    
    def _student_cfg_update_step(self, latents, cu_seqlens, conditions, input_dict, step):
        """
        Student CFG distillation step (warmup phase).
        
        Directly trains student to match teacher's CFG-guided predictions.
        No fake_score involvement, no other losses. Uses v-loss for proper weighting.
        
        1. Sample noise and perturb real data at random timestep
        2. Get teacher prediction with CFG (cond + uncond -> CFG combo)
        3. Get student prediction
        4. Loss in velocity space (v-loss) for proper timestep weighting
        """
        model = self.model_parts[0]
        
        # === Step 1: Sample timesteps and perturb real data ===
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
        
        # Perturb real data: x_t = (1 - sigma) * x_0 + sigma * eps
        x_t = (1.0 - sigmas) * latents + sigmas * eps
        
        # === Step 2: Teacher prediction with CFG (no grad) ===
        with torch.no_grad():

            # Conditional teacher prediction
            teacher_pred_cond = model.forward(
                hidden_states=x_t.to(self._dtype),
                timesteps=t,
                cu_seqlens=cu_seqlens,
                encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
                hidden_states_position=input_dict["vertices"],
                cu_seqlens_encoder=None,
                view_indices=conditions.get("view_indices"),
                mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                quad_ratios=input_dict.get("quad_ratios"),
                use_model="teacher",
            )
            teacher_x0_cond = self._model_pred_to_x0(teacher_pred_cond, x_t, sigmas)
            
            # Unconditional teacher prediction for CFG
            teacher_pred_uncond = model.forward(
                hidden_states=x_t.to(self._dtype),
                timesteps=t,
                cu_seqlens=cu_seqlens,
                encoder_hidden_states=conditions["encoder_hidden_states_uncond"].to(self._dtype),
                hidden_states_position=input_dict["vertices"],
                cu_seqlens_encoder=None,
                view_indices=conditions.get("view_indices"),
                mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                quad_ratios=input_dict.get("quad_ratios"),
                use_model="teacher",
            )
            teacher_x0_uncond = self._model_pred_to_x0(teacher_pred_uncond, x_t, sigmas)
            
            # Apply CFG
            guidance_scale = self.job_config.distill.guidance_scale
            if guidance_scale is not None:
                teacher_x0_cfg = teacher_x0_uncond + guidance_scale * (teacher_x0_cond - teacher_x0_uncond)
            else:
                teacher_x0_cfg = teacher_x0_cond
        
        # === Step 3: Student prediction (with grad) ===
        student_pred = model.forward(
            hidden_states=x_t.to(self._dtype),
            timesteps=t,
            cu_seqlens=cu_seqlens,
            encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
            hidden_states_position=input_dict["vertices"],
            cu_seqlens_encoder=None,
            view_indices=conditions.get("view_indices"),
            mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
            quad_ratios=input_dict.get("quad_ratios"),
            use_model="student",
        )
        student_x0 = self._model_pred_to_x0(student_pred, x_t, sigmas)

        # === Step 4: V-loss for proper timestep weighting ===
        if self.job_config.training.loss_type == "xpred-vloss":
            # Convert x0 predictions to velocity space
            v_pred = (x_t - student_x0) / torch.clamp(sigmas, min=0.05)
            v_target = (x_t - teacher_x0_cfg) / torch.clamp(sigmas, min=0.05)
            cfg_loss = F.mse_loss(v_pred.float(), v_target.float())
        elif self.job_config.training.loss_type == "vpred-vloss":
            # Model predicts v directly, compute target v from teacher x0
            v_target = eps - teacher_x0_cfg
            cfg_loss = F.mse_loss(student_pred.float(), v_target.float())
        else:
            raise NotImplementedError(f"Loss type {self.job_config.training.loss_type} not supported for CFG distill")
        
        return cfg_loss, {"vsd_loss": cfg_loss.item(), "step_type": "student_cfg"}
    
    def _fake_score_update_step(self, latents, cu_seqlens, conditions, input_dict, step):
        """
        Fake score update via denoising score matching.
        
        1. Generate from student (no grad) - single-step or multi-step
        2. Perturb generated data
        3. Train fake_score to denoise (in model's native prediction type)
        """
        model = self.model_parts[0]
        
        # === Step 1: Generate from student (no grad) ===
        with torch.no_grad():
            input_student, t_student, sigma_student = self._generate_student_input(latents, cu_seqlens)
            
            student_pred = model.forward(
                hidden_states=input_student.to(self._dtype),
                timesteps=t_student,
                cu_seqlens=cu_seqlens,
                encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
                hidden_states_position=input_dict["vertices"],
                cu_seqlens_encoder=None,
                view_indices=conditions.get("view_indices"),
                mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
                quad_ratios=input_dict.get("quad_ratios"),
                use_model="student",
            )
            gen_data = self._model_pred_to_x0(student_pred, input_student, sigma_student)
        
        # === Step 2: Perturb generated data ===
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
            hidden_states=perturbed_data.to(self._dtype),
            timesteps=t,
            cu_seqlens=cu_seqlens,
            encoder_hidden_states=conditions["encoder_hidden_states"].to(self._dtype),
            hidden_states_position=input_dict["vertices"],
            cu_seqlens_encoder=None,
            view_indices=conditions.get("view_indices"),
            mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
            quad_ratios=input_dict.get("quad_ratios"),
            use_model="fake_score",
        )
        
        # === Step 4: DSM Loss ===
        # Compute loss in velocity space (v-loss) for proper timestep weighting
        # Dividing by sigma normalizes gradients across different noise levels
        if self.job_config.training.loss_type == "xpred-vloss":
            # Model predicts x0, convert to velocity for v-loss
            v_pred = (perturbed_data.float() - fake_score_pred.float()) / torch.clamp(sigmas.float(), min=0.05)
            v_target = (perturbed_data.float() - gen_data.detach().float()) / torch.clamp(sigmas.float(), min=0.05)
            dsm_loss = F.mse_loss(v_pred, v_target)
        elif self.job_config.training.loss_type == "vpred-vloss":
            # Model predicts v directly
            v_target = (eps.float() - gen_data.detach().float())
            dsm_loss = F.mse_loss(fake_score_pred.float(), v_target.float())
        else:
            raise NotImplementedError(f"Loss type {self.job_config.training.loss_type} not supported for DSM")
        
        return dsm_loss, {"dsm_loss": dsm_loss.item(), "step_type": "fake_score"}
    
    def _log_metrics(self, loss, metrics, grad_norm, lr, is_student_step):
        """
        Log training metrics including both student and fake_score losses.
        
        Always logs:
        - vsd_loss: Student's VSD loss (most recent value)
        - dsm_loss: Fake score's DSM loss (most recent value)
        - step_type: "student" or "fake_score"
        """
        if not self.metrics_processor.should_log(self.step):
            return
        
        # Use the actual loss value from metrics (vsd_loss or dsm_loss)
        # This is more accurate than the backward loss
        if is_student_step:
            current_loss = metrics.get("vsd_loss", 0.0)
        else:
            current_loss = metrics.get("dsm_loss", 0.0)
        
        # Aggregate loss across ranks if needed
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
        
        # Print step type and both losses for clarity
        logger.info(f"[DMD] step_type: {step_type}  vsd_loss: {self._last_vsd_loss:.6f}  dsm_loss: {self._last_dsm_loss:.6f}")


if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Optional[DMDTrainer] = None

    init_logger(log_file=os.path.join(config.job.dump_folder, "logs", f"rank{get_rank()}.log"))
    dump_config(config, os.path.join(config.job.dump_folder, "config.toml"))
    
    try:
        trainer = DMDTrainer(config)
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
