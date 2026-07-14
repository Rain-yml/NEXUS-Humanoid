from typing import Optional, Union, List, Any, Dict, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import trimesh
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torchtitan.experiments.lumon.mesh_tokenizers import TokenizerSpec
from torchtitan.experiments.lumon.models.generation_utils import SamplingParams


@dataclass
class LumonI2MPipelineOutput(BaseOutput):
    mesh: List[trimesh.Trimesh]
    latents: torch.Tensor
    errs: List[str]

class LumonI2MPipeline(DiffusionPipeline):
    def __init__(
        self,
        image_encoder: torch.nn.Module,
        dit: torch.nn.Module,
        ar_decoder: Optional[torch.nn.Module],
        scheduler: FlowMatchEulerDiscreteScheduler,
        tokenizer: TokenizerSpec,
        latent_std: torch.Tensor,
        latent_mean: torch.Tensor,
    ):
        super().__init__()

        self.register_modules(
            image_encoder=image_encoder,
            dit=dit,
            ar_decoder=ar_decoder,
            scheduler=scheduler,
            tokenizer=tokenizer,
            latent_std=latent_std,
            latent_mean=latent_mean,
        )
        # assert self.dit.config.in_channels == self.ar_decoder.config.hidden_size_bottleneck
    
    def encode_image(
        self, 
        image: Union[List[Image.Image], torch.Tensor, Image.Image], 
        do_classifier_free_guidance: bool = False
    ):
        if isinstance(image, Image.Image):
            image = [image]
        
        if isinstance(image, list):
            if isinstance(image[0], Image.Image):
                image = [np.array(i).astype(np.float32) / 255.0 for i in image]
                image = np.stack(image, axis=0)
                image = torch.from_numpy(image).permute(2, 0, 1)
            else:
                raise ValueError(f"Invalid image type in list: {type(image[0])}")
        
        assert image.shape[1] == 3
        image = image.to(self.device)
        image_embeds = self.image_encoder(self.image_encoder.preprocess(image))

        negative_image_embeds = None

        if do_classifier_free_guidance:
            # negative_image = torch.full_like(image, 0.5)
            negative_image = torch.zeros_like(image)
            negative_image_embeds = self.image_encoder(self.image_encoder.preprocess(negative_image))
        
        return image_embeds, negative_image_embeds
    
    def prepare_latents(
        self,
        batch_size,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        if num_tokens is None:
            num_tokens = self.ar_decoder.config.num_tokens
            if isinstance(num_tokens, list):
                num_tokens = max(num_tokens)
        shape = (batch_size, num_tokens, self.dit.config.in_channels)
        print("latents shape:", shape)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents        

    @torch.no_grad()
    def __call__(
        self,
        image: Union[List[Image.Image], np.ndarray, torch.Tensor, Image.Image],
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        top_k: int = -1,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_tokens: int = 10000,
        latents_only: bool = False,
        num_faces: Optional[Union[int, List[int], torch.Tensor]] = None,
        num_tokens: Optional[int] = None,
        enable_progress: bool = True,
        dtype=torch.float32,
    ):
        device = self.device
        do_classifier_free_guidance = guidance_scale > 1.0
        image_embeds, negative_image_embeds = self.encode_image(image, do_classifier_free_guidance)
        n_inputs = image_embeds.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            batch_size=batch_size * n_inputs,
            dtype=torch.float32,
            device=device,
            generator=generator,
            num_tokens=num_tokens,
        )

        if num_faces is None:
            cond_face_num = None
        elif isinstance(num_faces, torch.Tensor):
            cond_face_num = num_faces.repeat_interleave(batch_size, dim=0)
        else:
            if isinstance(num_faces, int):
                cond_face_num = torch.tensor([num_faces] * n_inputs, dtype=torch.long, device=device)
            else:
                assert len(num_faces) == n_inputs
                cond_face_num = torch.tensor(num_faces, dtype=torch.long, device=device)
            
            cond_face_num = cond_face_num.repeat_interleave(batch_size, dim=0)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)        

        image_embeds = image_embeds.repeat_interleave(batch_size, dim=0)
        image_embeds = image_embeds.to(dtype)
        if do_classifier_free_guidance:
            negative_image_embeds = negative_image_embeds.repeat_interleave(batch_size, dim=0)
            negative_image_embeds = negative_image_embeds.to(dtype)
        
        batch_size = batch_size * n_inputs
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents.to(dtype)
                timesteps = t.expand(batch_size)
                noise_pred = self.dit(
                    hidden_states=latent_model_input,
                    timesteps=timesteps,
                    encoder_hidden_states=image_embeds,
                    cond_face_num=cond_face_num,
                )

                if do_classifier_free_guidance:
                    noise_uncond = self.dit(
                        hidden_states=latent_model_input,
                        timesteps=timesteps,
                        encoder_hidden_states=negative_image_embeds,
                        cond_face_num=cond_face_num,
                    )
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        latents = (latents.float() * self.latent_std + self.latent_mean).to(dtype=dtype)
        if latents_only:
            return LumonI2MPipelineOutput(
                mesh=[None] * batch_size,
                latents=latents,
                errs=[''] * batch_size,
            )
        sampling_params = SamplingParams(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            max_positions=max_tokens,
        )

        result = self.ar_decoder.decode(
            latents,
            sampling_params=sampling_params,
        )
        meshes = []
        errs = []
        for i in range(batch_size):
            seq_length = result['sequence_length'][i]
            tokens = result['sequences'][i, :seq_length].detach().cpu().numpy()
            v, f, err_msg = self.tokenizer.detokenize(tokens)
            if f.shape[0] > 0:
                mesh = trimesh.Trimesh(vertices=v, faces=f)
                meshes.append(mesh)
            else:
                meshes.append(None)
            errs.append(err_msg)

        # Offload all models
        # self.maybe_free_model_hooks()
        return LumonI2MPipelineOutput(
            mesh=meshes,
            latents=latents,
            errs=errs,
        )
    

# class TrellisSLATPipeline(DiffusionPipeline):
#     def __init__(
#         self,
#         image_encoder: torch.nn.Module,
#         transformer: torch.nn.Module,
#         vae: torch.nn.Module,
#         scheduler: FlowMatchEulerDiscreteScheduler,
#         image_encoder_2: Optional[torch.nn.Module] = None,
#     ):
#         super().__init__()
        
#         self.register_modules(
#             image_encoder=image_encoder,
#             image_encoder_2=image_encoder_2,
#             transformer=transformer,
#             vae=vae,
#             scheduler=scheduler,
#         )

#     def prepare_latents(
#         self,
#         batch_size: int,
#         num_channels_latents: int = 8,
#         num_voxels: int = 0,
#         dtype: Optional[torch.dtype] = None,
#         device: Optional[torch.device] = None,
#         generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
#         latents: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         if latents is not None:
#             return latents.to(device=device, dtype=dtype)
        
#         shape = (
#             batch_size,
#             num_voxels,
#             num_channels_latents,
#         )
#         if isinstance(generator, list) and len(generator) != batch_size:
#             raise ValueError(
#                 f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
#                 f" size of {batch_size}. Make sure the batch size matches the length of the generators."
#             )

#         latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
#         return latents        
    
#     @property
#     def guidance_scale(self):
#         return self._guidance_scale

#     @property
#     def do_classifier_free_guidance(self):
#         return self._guidance_scale > 1.0

#     @property
#     def num_timesteps(self):
#         return self._num_timesteps

#     @property
#     def current_timestep(self):
#         return self._current_timestep
    
#     @property
#     def attention_kwargs(self):
#         return self._attention_kwargs

#     def _encode_image(self, image_encoder, image: Image.Image, do_classifier_free_guidance: bool, num_shapes_per_prompt: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
#         image_np = np.array(image).astype(np.float32) / 255.0
#         image_np = image_np[...,0:3] * image_np[...,3:4] + 0 * (1 - image_np[...,3:4])
#         image_pt = torch.from_numpy(image_np).permute(2, 0, 1)[None].to(device)
#         image_embeds = image_encoder(image_pt)
#         image_embeds = image_embeds.repeat_interleave(num_shapes_per_prompt, dim=0)
#         negative_image_embeds = None
#         if do_classifier_free_guidance:
#             negative_image_embeds = image_encoder(torch.zeros_like(image_pt))
#             negative_image_embeds = negative_image_embeds.repeat_interleave(num_shapes_per_prompt, dim=0)
#         return image_embeds, negative_image_embeds

#     def encode_image(self, image: Image.Image, image_2: Optional[Image.Image] = None, do_classifier_free_guidance: bool = False, num_shapes_per_prompt: int = 1, concat_hidden_states_along_channel: bool = False, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
#         # TODO: hard-coded
#         image_embeds, negative_image_embeds = self._encode_image(self.image_encoder, image.convert('RGBA'), do_classifier_free_guidance, num_shapes_per_prompt, device)
#         image_embeds_2, negative_image_embeds_2 = None, None
#         if self.image_encoder_2 is not None:
#             if image_2 is None:
#                 image_2 = image
#             image_embeds_2, negative_image_embeds_2 = self._encode_image(self.image_encoder_2, image_2.convert('RGBA'), do_classifier_free_guidance, num_shapes_per_prompt, device)
        
#         if concat_hidden_states_along_channel:
#             image_embeds = torch.cat([image_embeds, image_embeds_2], dim=-1)
#             negative_image_embeds = torch.cat([negative_image_embeds, negative_image_embeds_2], dim=-1)
#             image_embeds_2 = None
#             negative_image_embeds_2 = None

#         return image_embeds, negative_image_embeds, image_embeds_2, negative_image_embeds_2

