from typing import Optional, Union, List, Any, Dict, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from tqdm import tqdm


@dataclass
class I2VPipelineOutput(BaseOutput):
    vertices: torch.Tensor
    vertices_layers: List[torch.Tensor]

class I2VPipeline(DiffusionPipeline):
    def __init__(
        self,
        image_encoder: torch.nn.Module,
        octree_dit: torch.nn.Module,
        scheduler,
    ):
        super().__init__()

        self.register_modules(
            image_encoder=image_encoder,
            octree_dit=octree_dit,
            scheduler=scheduler,
        )
        # assert self.dit.config.in_channels == self.ar_decoder.config.hidden_size_bottleneck
    
    def encode_image(
        self, 
        image: Union[List[Image.Image], torch.Tensor, Image.Image], 
        device,
        do_classifier_free_guidance: bool = False,
        view_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode one or more view images for a single mesh.

        Single-view (view_indices=None):
            Returns (embeds, neg_embeds, None, None).
            embeds shape: (1, num_tokens, hidden_dim)

        Multiview (view_indices provided):
            image must be a list of PIL images, one per view.
            Returns (embeds, neg_embeds, view_indices_tensor, mv_cu_seqlens).
            embeds shape: (N_views, num_tokens, hidden_dim)
            view_indices_tensor: (N_views,) int64
            mv_cu_seqlens: tensor([0, N_views]) int32
        """
        if isinstance(image, Image.Image):
            image = [image]
        
        if isinstance(image, list):
            if isinstance(image[0], Image.Image):
                image = [np.array(i).astype(np.float32) / 255.0 for i in image]
                image = np.stack(image, axis=0) # (B, H, W, C)
                image = torch.from_numpy(image).permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Invalid image type in list: {type(image[0])}")
        
        assert image.shape[1] == 3
        image = image.to(device)
        image_embeds = self.image_encoder(self.image_encoder.preprocess(image))

        negative_image_embeds = None

        if do_classifier_free_guidance:
            negative_image = torch.full_like(image, 0.5)
            negative_image_embeds = self.image_encoder(self.image_encoder.preprocess(negative_image))

        if view_indices is None:
            return image_embeds, negative_image_embeds, None, None

        n_views = len(view_indices)
        view_indices_tensor = torch.tensor(view_indices, device=device, dtype=torch.int64)
        mv_cu_seqlens = torch.tensor([0, n_views], device=device, dtype=torch.int32)
        return image_embeds, negative_image_embeds, view_indices_tensor, mv_cu_seqlens
    
    def prepare_latents(
        self,
        shape,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents        
    
    def sample_layer(
        self,
        scheduler,
        centers,
        depth: int,
        device,
        dtype,
        generator,
        sub_voxel_size,
        num_inference_steps: int,
        image_embeds: Optional[torch.Tensor] = None,
        negative_image_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.0,
        num_vertices: Optional[int] = None,
        quad_ratio: Optional[float] = None,
        symmetries: Optional[Union[List[int], torch.Tensor]] = None,
        prediction: str = 'x',
        view_indices: Optional[torch.Tensor] = None,
        mv_cu_seqlens: Optional[torch.Tensor] = None,
    ):
        n_tokens = centers.shape[0]
        x_t = self.prepare_latents(
            shape=(n_tokens, 8),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        self._num_timesteps = len(timesteps)        
        cu_seqlens = torch.tensor([0, n_tokens], device=device, dtype=torch.int32)
        depths = torch.tensor([depth] * n_tokens, device=device, dtype=torch.long)

        do_classifier_free_guidance = guidance_scale > 1.0
        num_vertices_tensor = torch.tensor([num_vertices], device=device, dtype=torch.int32)
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

        for i, t in enumerate(timesteps):
            timesteps = torch.repeat_interleave(t, n_tokens)
            noise_pred = self.octree_dit(
                x_t=x_t.to(dtype),
                t=timesteps,
                centers=centers,
                depths=depths,
                cu_seqlens_q=cu_seqlens,
                num_layers_per_mesh=[1],
                encoder_hidden_states=image_embeds,
                num_vertices=num_vertices_tensor,
                quad_ratios=quad_ratios_tensor,
                symmetries=symmetries_tensor,
                view_indices=view_indices,
                mv_cu_seqlens=mv_cu_seqlens,
            )

            if do_classifier_free_guidance:
                noise_uncond = self.octree_dit(
                    x_t=x_t.to(dtype),
                    t=timesteps,
                    centers=centers,
                    depths=depths,
                    cu_seqlens_q=cu_seqlens,
                    num_layers_per_mesh=[1],
                    encoder_hidden_states=negative_image_embeds,
                    num_vertices=num_vertices_tensor,
                    quad_ratios=quad_ratios_tensor,
                    symmetries=symmetries_tensor,
                    view_indices=view_indices,
                    mv_cu_seqlens=mv_cu_seqlens,
                )
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            
            noise_pred = noise_pred.to(x_t.dtype)

            if prediction == "x":
                sigma = t / scheduler.num_train_timesteps
                v_pred = (x_t - noise_pred) / sigma
            elif prediction == "v":
                v_pred = noise_pred

            # compute the previous noisy sample x_t -> x_t-1
            # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            x_t = scheduler.step(v_pred, t, x_t, return_dict=False, generator=generator)[0]
        
        child_offsets = torch.tensor([[
            [-1, -1, -1],
            [-1, -1,  1],
            [-1,  1, -1],
            [-1,  1,  1],
            [ 1, -1, -1],
            [ 1, -1,  1],
            [ 1,  1, -1],
            [ 1,  1,  1],
        ]], dtype=centers.dtype, device=centers.device)
        sub_voxel_centers = centers.view(n_tokens, 1, 3) + child_offsets * sub_voxel_size // 2
        sub_centers = sub_voxel_centers[x_t.float() > 0]
        return sub_centers

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[List[Image.Image], np.ndarray, torch.Tensor, Image.Image],
        scheduler,
        device,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_vertices: Optional[int] = None,
        quad_ratio: Optional[float] = None,
        symmetries: Optional[Union[List[int], torch.Tensor]] = None,
        enable_progress: bool = True,
        grid_size: int = 512,
        max_depth: int = 9,
        dtype=torch.float32,
        prediction: str = 'x',
        view_indices: Optional[List[int]] = None,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0
        image_embeds, negative_image_embeds, view_indices_tensor, mv_cu_seqlens = self.encode_image(
            image, device, do_classifier_free_guidance, view_indices=view_indices
        )
        centers = torch.full((1, 3), grid_size // 2, dtype=torch.long, device=device)

        image_embeds = image_embeds.to(dtype)
        if do_classifier_free_guidance:
            negative_image_embeds = negative_image_embeds.to(dtype)
        
        sub_voxel_size = grid_size // 2

        vertices_layers = [centers]
        
        for depth in tqdm(range(max_depth), disable=not enable_progress):
            sub_centers = self.sample_layer(
                scheduler=scheduler,
                centers=centers,
                depth=depth,
                device=device,
                dtype=dtype,
                generator=generator,
                sub_voxel_size=sub_voxel_size,
                num_inference_steps=num_inference_steps,
                image_embeds=image_embeds,
                negative_image_embeds=negative_image_embeds,
                guidance_scale=guidance_scale,
                num_vertices=num_vertices,
                quad_ratio=quad_ratio,
                symmetries=symmetries,
                prediction=prediction,
                view_indices=view_indices_tensor,
                mv_cu_seqlens=mv_cu_seqlens,
            )
            print('centers', centers.shape, 'sub_centers', sub_centers.shape)

            sub_voxel_size = sub_voxel_size // 2
            centers = sub_centers
            if centers.shape[0] == 0:
                break
            vertices_layers.append(centers)

        return I2VPipelineOutput(
            vertices=centers,
            vertices_layers=vertices_layers,
        )
