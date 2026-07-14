from typing import Optional, Union, List, Any, Dict, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import trimesh
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torchtitan.experiments.vem.mesh_tokenizers import TokenizerSpec
import math


@dataclass
class V2FPipelineOutput(BaseOutput):
    vertices: Optional[np.array] = None
    vertices_embed: Optional[torch.Tensor] = None
    vertices_embed_every_step: Optional[torch.Tensor] = None

@dataclass
class V2FRolloutPipelineOutput(BaseOutput):
    x0: torch.Tensor
    noise: torch.Tensor
    latents: torch.Tensor
    timesteps: torch.Tensor
    next_timesteps: torch.Tensor

class V2FPipeline(DiffusionPipeline):
    def __init__(
        self,
        sm_dit: torch.nn.Module,
        image_encoder: torch.nn.Module,
        scheduler: Union[FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler],
        latent_std: torch.Tensor,
        latent_mean: torch.Tensor,
        # prediction: str,
    ):
        super().__init__()

        self.register_modules(
            sm_dit=sm_dit,
            image_encoder=image_encoder,
            scheduler=scheduler,
            latent_std=latent_std,
            latent_mean=latent_mean,
            # prediction=prediction,
        )
        # self.prediction = prediction
        
    def encode_image(
        self, 
        image: Union[List[Image.Image], torch.Tensor, Image.Image], 
        do_classifier_free_guidance: bool = False,
        device = 'cuda',
        view_indices: Optional[List[int]] = None,
    ):
        if isinstance(image, Image.Image):
            image = [image]
        
        if isinstance(image, list):
            if isinstance(image[0], Image.Image):
                image = [np.array(i).astype(np.float32) / 255.0 for i in image]
                image = np.stack(image, axis=0)
                image = torch.from_numpy(image).permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Invalid image type in list: {type(image[0])}")
        
        assert image.shape[1] == 3
        image = image.to(device)
        image_embeds = self.image_encoder(self.image_encoder.preprocess(image))

        negative_image_embeds = None

        if do_classifier_free_guidance:
            negative_image = torch.full_like(image, 0.5)
            # negative_image = torch.zeros_like(image)
            negative_image_embeds = self.image_encoder(self.image_encoder.preprocess(negative_image))
        
        if view_indices is None:
            return image_embeds, negative_image_embeds, None, None

        n_views = len(view_indices)
        view_indices_tensor = torch.tensor(view_indices, device=device, dtype=torch.int64)
        mv_cu_seqlens = torch.tensor([0, n_views], device=device, dtype=torch.int32)
        return image_embeds, negative_image_embeds, view_indices_tensor, mv_cu_seqlens

    def prepare_latents(
        self,
        num_vertices,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        shape = (num_vertices, self.sm_dit.config.in_channels)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents        

    def _model_output_to_velocity(
        self,
        model_output: torch.Tensor,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        scheduler,
        prediction: str,
    ) -> torch.Tensor:
        if prediction == "x":
            sigma = timestep / scheduler.config.num_train_timesteps
            return (latents - model_output) / torch.clamp(sigma, min=1e-6)
        if prediction == "v":
            return model_output
        raise NotImplementedError(f"Unsupported prediction type: {prediction}")

    def _repeat_timestep_by_sequence(
        self,
        timestep: torch.Tensor,
        seq_lens: torch.Tensor,
        device,
    ) -> torch.Tensor:
        timestep_per_sample = timestep.reshape(1).to(device=device).expand(seq_lens.shape[0])
        return torch.repeat_interleave(timestep_per_sample, seq_lens)

    @torch.no_grad()
    def rollout(
        self,
        vertices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        prediction: str,
        image: Optional[Union[List[Image.Image], np.ndarray, torch.Tensor, Image.Image]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 50,
        device="cuda",
        dtype: torch.dtype = torch.float32,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        scheduler: Optional[Any] = None,
        noise: Optional[torch.Tensor] = None,
        noise_std: float = 1.0,
        return_latents_every_step: bool = True,
    ) -> V2FRolloutPipelineOutput:
        scheduler = scheduler if scheduler is not None else self.scheduler
        assert prediction in ["x", "v"]
        if conditions is None and image is None:
            raise ValueError("Either precomputed conditions or image must be provided")

        vertices = vertices.to(device=device)
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]

        do_classifier_free_guidance = guidance_scale > 1.0
        if conditions is None:
            image_embeds, negative_image_embeds, view_indices_tensor, mv_cu_seqlens = self.encode_image(
                image,
                do_classifier_free_guidance,
                device=device,
                view_indices=None,
            )
        else:
            image_embeds = conditions["encoder_hidden_states"].to(device=device)
            negative_image_embeds = conditions.get("negative_encoder_hidden_states")
            if negative_image_embeds is not None:
                negative_image_embeds = negative_image_embeds.to(device=device)
            view_indices_tensor = conditions.get("view_indices")
            if view_indices_tensor is not None:
                view_indices_tensor = view_indices_tensor.to(device=device)
            mv_cu_seqlens = conditions.get("mv_cu_seqlens")
            if mv_cu_seqlens is not None:
                mv_cu_seqlens = mv_cu_seqlens.to(device=device, dtype=torch.int32)
            if do_classifier_free_guidance and negative_image_embeds is None:
                raise ValueError("Classifier-free guidance with precomputed conditions requires negative_encoder_hidden_states")

        scheduler.set_timesteps(num_inference_steps, device=device)
        scheduler_timesteps = scheduler.timesteps

        if noise is None:
            noise = self.prepare_latents(
                num_vertices=vertices.shape[0],
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        else:
            noise = noise.to(device=device)
        latents = noise * noise_std

        image_embeds = image_embeds.to(dtype)
        if negative_image_embeds is not None:
            negative_image_embeds = negative_image_embeds.to(dtype)

        latents_every_step = []
        timesteps_every_step = []
        next_timesteps_every_step = []

        for step_idx, timestep in enumerate(scheduler_timesteps):
            latent_model_input = latents.to(dtype)
            timestep_tokens = self._repeat_timestep_by_sequence(
                timestep,
                seq_lens,
                device=device,
            )
            pred_cond = self.sm_dit(
                hidden_states=latent_model_input,
                timesteps=timestep_tokens,
                encoder_hidden_states=image_embeds,
                hidden_states_position=vertices,
                cu_seqlens=cu_seqlens,
                view_indices=view_indices_tensor,
                mv_cu_seqlens=mv_cu_seqlens,
            )
            if do_classifier_free_guidance:
                pred_uncond = self.sm_dit(
                    hidden_states=latent_model_input,
                    timesteps=timestep_tokens,
                    encoder_hidden_states=negative_image_embeds,
                    hidden_states_position=vertices,
                    cu_seqlens=cu_seqlens,
                    view_indices=view_indices_tensor,
                    mv_cu_seqlens=mv_cu_seqlens,
                )
                pred_cond = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            pred_cond = pred_cond.to(latents.dtype)
            velocity = self._model_output_to_velocity(
                pred_cond,
                latents,
                timestep,
                scheduler,
                prediction,
            )
            latents = scheduler.step(
                velocity,
                timestep,
                latents,
                return_dict=False,
                generator=generator,
            )[0]

            if return_latents_every_step:
                latents_every_step.append(latents.detach().clone())
                timesteps_every_step.append(timestep_tokens.detach().clone())
                if step_idx + 1 < len(scheduler_timesteps):
                    next_timestep = scheduler_timesteps[step_idx + 1]
                else:
                    next_timestep = torch.zeros_like(timestep)
                next_timesteps_every_step.append(
                    self._repeat_timestep_by_sequence(
                        next_timestep,
                        seq_lens,
                        device=device,
                    ).detach().clone()
                )

        if return_latents_every_step:
            latents_out = torch.stack(latents_every_step, dim=0)
            timesteps_out = torch.stack(timesteps_every_step, dim=0)
            next_timesteps_out = torch.stack(next_timesteps_every_step, dim=0)
        else:
            latents_out = latents.detach().unsqueeze(0)
            timesteps_out = torch.empty(0, device=device, dtype=latents.dtype)
            next_timesteps_out = torch.empty(0, device=device, dtype=latents.dtype)

        return V2FRolloutPipelineOutput(
            x0=latents.detach(),
            noise=noise.detach(),
            latents=latents_out.detach(),
            timesteps=timesteps_out.detach(),
            next_timesteps=next_timesteps_out.detach(),
        )

    @torch.inference_mode()
    def __call__(
        self,
        vertices: Union[np.ndarray, torch.Tensor],
        image: Union[List[Image.Image], np.ndarray, torch.Tensor, Image.Image],
        prediction: str,
        guidance_scale: float = 3,
        num_inference_steps: int = 50,
        device = 'cuda',
        dtype: torch.dtype = torch.float32,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_latents_every_step: bool = False,
        scheduler: Optional[Union[FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler]] = None,
        view_indices: Optional[List[int]] = None,
        quad_ratio: Optional[float] = None,
        symmetries: Optional[Union[List[int], torch.Tensor]] = None,
    ):
        scheduler = scheduler if scheduler is not None else self.scheduler
        assert prediction in ["x", "v"]

        do_classifier_free_guidance = guidance_scale > 1.0
        image_embeds, negative_image_embeds, view_indices_tensor, mv_cu_seqlens = self.encode_image(image, do_classifier_free_guidance, device=device, view_indices=view_indices)

        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).to(device=device)
        
        cu_seqlens = torch.tensor([0, vertices.shape[0]], device=device, dtype=torch.int32)
        seq_len = cu_seqlens[1:] - cu_seqlens[:-1]
        
        assert vertices.dim() == 2 # (num_vertices, 3)
        
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        latents = self.prepare_latents(
            num_vertices=vertices.shape[0],
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        self._num_timesteps = len(timesteps)        

        image_embeds = image_embeds.to(dtype)
        if do_classifier_free_guidance:
            negative_image_embeds = negative_image_embeds.to(dtype)
        
        quad_ratios_tensor = None
        if quad_ratio is not None:
            quad_ratios_tensor = torch.tensor([quad_ratio], device=device, dtype=torch.float32)
        # (1, 4) int64 symmetry tokens [x=0, y=0, z=0, any-of-xyz], 1=symmetric 0=uncertain.
        # When None the model defaults to all-uncertain.
        symmetries_tensor = None
        if symmetries is not None:
            if isinstance(symmetries, torch.Tensor):
                symmetries_tensor = symmetries.to(device=device, dtype=torch.long).view(1, 4)
            else:
                symmetries_tensor = torch.tensor([symmetries], device=device, dtype=torch.long)

        latents_every_step = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # print(t)
                latent_model_input = latents.to(dtype)
                timesteps = torch.repeat_interleave(t, seq_len)
                pred_cond = self.sm_dit(
                    hidden_states=latent_model_input,
                    timesteps=timesteps,
                    encoder_hidden_states=image_embeds,
                    hidden_states_position=vertices,
                    cu_seqlens=cu_seqlens,
                    view_indices=view_indices_tensor,
                    mv_cu_seqlens=mv_cu_seqlens,
                    quad_ratios=quad_ratios_tensor,
                    symmetries=symmetries_tensor,
                )
                if do_classifier_free_guidance:
                    pred_uncond = self.sm_dit(
                        hidden_states=latent_model_input,
                        timesteps=timesteps,
                        encoder_hidden_states=negative_image_embeds,
                        hidden_states_position=vertices,
                        cu_seqlens=cu_seqlens,
                        view_indices=view_indices_tensor,
                        mv_cu_seqlens=mv_cu_seqlens,
                        quad_ratios=quad_ratios_tensor,
                        symmetries=symmetries_tensor,
                    )
                    pred_cond = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                # .to(latents.dtype)
                pred_cond = pred_cond.to(latents.dtype)
                if prediction == "x":
                    sigma = t / scheduler.num_train_timesteps
                    v_pred = (latents - pred_cond) / sigma
                elif prediction == "v":
                    v_pred = pred_cond

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(v_pred, t, latents, return_dict=False, generator=generator)[0]
                if return_latents_every_step:
                    latents_every_step.append(latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()
        
        latents = (latents.float() * self.latent_std + self.latent_mean)
        if return_latents_every_step:
            latents_every_step = torch.stack(latents_every_step, dim=0)
            latents_every_step = (latents_every_step.float() * self.latent_std.unsqueeze(0) + self.latent_mean.unsqueeze(0))
            return V2FPipelineOutput(
                vertices_embed=latents,
                vertices_embed_every_step=latents_every_step,
            )
        else:
            return V2FPipelineOutput(
                vertices_embed=latents,
            )
    
