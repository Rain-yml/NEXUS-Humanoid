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
from torchtitan.experiments.vem.mesh_tokenizers import TokenizerSpec
from torchtitan.experiments.vem.models.generation_utils import SamplingParams
import math


@dataclass
class V2EPipelineOutput(BaseOutput):
    vertices: Optional[np.array] = None
    vertices_embed: Optional[np.array] = None
    edges: Optional[np.array] = None

class V2EPipeline(DiffusionPipeline):
    def __init__(
        self,
        sm_dit: torch.nn.Module,
        scheduler: FlowMatchEulerDiscreteScheduler,
        latent_std: torch.Tensor,
        latent_mean: torch.Tensor,
    ):
        super().__init__()

        self.register_modules(
            sm_dit=sm_dit,
            scheduler=scheduler,
            latent_std=latent_std,
            latent_mean=latent_mean,
        )
    
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

    @torch.no_grad()
    def __call__(
        self,
        vertices: Union[np.ndarray, torch.Tensor],
        num_inference_steps: int = 50,
        dtype: torch.dtype = torch.float32,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        device = self.device

        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).to(device=device, dtype=torch.float32)
        
        cu_seqlens = torch.tensor([0, vertices.shape[0]], device=device, dtype=torch.int32)
        seq_len = cu_seqlens[1:] - cu_seqlens[:-1]
        
        assert vertices.dim() == 2 # (num_vertices, 3)
        
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            num_vertices=vertices.shape[0],
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)        

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents.to(dtype)
                timesteps = torch.repeat_interleave(t, seq_len)
                x_pred = self.sm_dit(
                    hidden_states=latent_model_input,
                    timesteps=timesteps,
                    encoder_hidden_states=vertices,
                    cu_seqlens=cu_seqlens,
                ).to(latents.dtype)
                sigma = t / self.scheduler.num_train_timesteps
                v_pred = (latents - x_pred) / sigma

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(v_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        latents = (latents.float() * self.latent_std + self.latent_mean)
        dim = latents.shape[-1]

        node_t = latents[:, :dim//2]
        node_s = latents[:, dim//2:]

        dt = (node_t[:, None, :] - node_t[None, :, :]).pow(2).sum(dim=-1) / (2 * math.sqrt(dim))
        ds = (node_s[:, None, :] - node_s[None, :, :]).pow(2).sum(dim=-1) / (2 * math.sqrt(dim))
        e1, e2 = torch.where((dt - ds) > 0)
        edge_pred = torch.stack([e1, e2], dim=-1)
        edge_pred = torch.sort(edge_pred, dim=-1)[0].unique(dim=0)

        return V2EPipelineOutput(
            vertices=vertices.detach().cpu().numpy(),
            vertices_embed=latents.detach().cpu().numpy(),
            edges=edge_pred.detach().cpu().numpy(),
        )
    
