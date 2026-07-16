"""Single-stream joint rollout conditioned on clean ground-truth mesh layers."""

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm import tqdm


@dataclass
class SingleStreamTeacherForcedMeshLayer:
    centers: torch.Tensor
    occupancy: torch.Tensor
    depth: int


@dataclass
class ImageMeshToSingleStreamJointOctreePipelineOutput(BaseOutput):
    joints: torch.Tensor
    joint_layers: List[torch.Tensor]


class ImageMeshToSingleStreamJointOctreePipeline(DiffusionPipeline):
    """Denoise joints alongside clean mesh tokens in one 3D-RoPE stream."""

    def __init__(self, image_encoder: torch.nn.Module, octree_dit: torch.nn.Module, scheduler):
        super().__init__()
        self.register_modules(
            image_encoder=image_encoder,
            octree_dit=octree_dit,
            scheduler=scheduler,
        )

    def encode_image(
        self,
        image: Union[List[Image.Image], torch.Tensor, Image.Image],
        device,
        do_classifier_free_guidance: bool = False,
        view_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list):
            if not isinstance(image[0], Image.Image):
                raise ValueError(f"Invalid image type in list: {type(image[0])}")
            image = torch.from_numpy(
                np.stack([np.asarray(item).astype(np.float32) / 255.0 for item in image])
            ).permute(0, 3, 1, 2)

        if image.shape[1] != 3:
            raise ValueError(f"Expected RGB image tensor, got shape {tuple(image.shape)}")
        image = image.to(device)
        image_embeds = self.image_encoder(self.image_encoder.preprocess(image))

        negative_image_embeds = None
        if do_classifier_free_guidance:
            negative_image = torch.full_like(image, 0.5)
            negative_image_embeds = self.image_encoder(
                self.image_encoder.preprocess(negative_image)
            )

        if view_indices is None:
            return image_embeds, negative_image_embeds, None, None

        n_views = len(view_indices)
        return (
            image_embeds,
            negative_image_embeds,
            torch.tensor(view_indices, device=device, dtype=torch.int64),
            torch.tensor([0, n_views], device=device, dtype=torch.int32),
        )

    def prepare_latents(self, shape, dtype=None, device=None, generator=None) -> torch.Tensor:
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def sample_layer(
        self,
        *,
        scheduler,
        mesh_layer: SingleStreamTeacherForcedMeshLayer,
        joint_centers: torch.Tensor,
        joint_ids: torch.Tensor,
        device,
        dtype,
        generator,
        sub_voxel_size: int,
        num_inference_steps: int,
        image_embeds: torch.Tensor,
        negative_image_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        num_vertices: Optional[int] = None,
        prediction: str = "x",
        view_indices: Optional[torch.Tensor] = None,
        mv_cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mesh_centers = mesh_layer.centers.to(device=device, dtype=torch.long)
        mesh_x_0 = mesh_layer.occupancy.to(device=device, dtype=torch.float32)
        n_mesh_tokens = mesh_centers.shape[0]
        n_joint_tokens = joint_centers.shape[0]

        joint_x_t = self.prepare_latents(
            (n_joint_tokens, 8), dtype=torch.float32, device=device, generator=generator
        )

        joint_scheduler = copy.deepcopy(scheduler)
        joint_scheduler.set_timesteps(num_inference_steps, device=device)
        self._num_timesteps = len(joint_scheduler.timesteps)

        all_centers = torch.cat([mesh_centers, joint_centers], dim=0)
        n_tokens = n_mesh_tokens + n_joint_tokens
        cu_seqlens = torch.tensor([0, n_tokens], device=device, dtype=torch.int32)
        depths = torch.full(
            (n_tokens,), mesh_layer.depth, device=device, dtype=torch.long
        )
        token_joint_ids = torch.cat(
            [
                torch.full((n_mesh_tokens,), -1, device=device, dtype=torch.long),
                joint_ids,
            ]
        )
        joint_mask = token_joint_ids >= 0
        num_vertices_tensor = torch.tensor([num_vertices], device=device, dtype=torch.int32)
        do_classifier_free_guidance = guidance_scale > 1.0

        def predict(condition: torch.Tensor, timestep: torch.Tensor):
            x_t = torch.cat([mesh_x_0, joint_x_t], dim=0)
            token_timesteps = torch.cat(
                [
                    torch.zeros(
                        n_mesh_tokens, device=device, dtype=timestep.dtype
                    ),
                    torch.full(
                        (n_joint_tokens,),
                        timestep,
                        device=device,
                        dtype=timestep.dtype,
                    ),
                ]
            )
            prediction = self.octree_dit(
                x_t=x_t.to(dtype),
                t=token_timesteps,
                centers=all_centers,
                depths=depths,
                cu_seqlens_q=cu_seqlens,
                num_layers_per_mesh=[1],
                encoder_hidden_states=condition,
                num_vertices=num_vertices_tensor,
                view_indices=view_indices,
                mv_cu_seqlens=mv_cu_seqlens,
                joint_ids=token_joint_ids,
                joint_mask=joint_mask,
            )
            return prediction[joint_mask]

        for timestep in joint_scheduler.timesteps:
            sigma = timestep / joint_scheduler.config.num_train_timesteps
            joint_prediction = predict(image_embeds, timestep)

            if do_classifier_free_guidance:
                joint_uncond = predict(negative_image_embeds, timestep)
                joint_prediction = joint_uncond + guidance_scale * (
                    joint_prediction - joint_uncond
                )

            joint_prediction = joint_prediction.to(joint_x_t.dtype)
            if prediction == "x":
                joint_v = (joint_x_t - joint_prediction) / sigma.clamp_min(0.05)
            elif prediction == "v":
                joint_v = joint_prediction
            else:
                raise ValueError(f"Unsupported prediction type: {prediction}")
            joint_x_t = joint_scheduler.step(
                joint_v, timestep, joint_x_t, return_dict=False, generator=generator
            )[0]

        child_offsets = torch.tensor(
            [[
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ]],
            dtype=joint_centers.dtype,
            device=device,
        )
        joint_children_xyz = (
            joint_centers.view(n_joint_tokens, 1, 3)
            + child_offsets * sub_voxel_size // 2
        )
        joint_children = joint_x_t.argmax(dim=-1)
        return joint_children_xyz[
            torch.arange(n_joint_tokens, device=device), joint_children
        ]

    @torch.inference_mode()
    def __call__(
        self,
        *,
        image: Union[List[Image.Image], np.ndarray, torch.Tensor, Image.Image],
        mesh_layers: List[SingleStreamTeacherForcedMeshLayer],
        scheduler,
        device,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator=None,
        num_vertices: Optional[int] = None,
        enable_progress: bool = True,
        grid_size: int = 512,
        dtype=torch.float32,
        prediction: str = "x",
        view_indices: Optional[List[int]] = None,
        num_joint_tokens: int = 28,
    ) -> ImageMeshToSingleStreamJointOctreePipelineOutput:
        if not mesh_layers:
            raise ValueError("mesh_layers must contain at least one GT octree layer")
        expected_depths = list(range(len(mesh_layers)))
        actual_depths = [layer.depth for layer in mesh_layers]
        if actual_depths != expected_depths:
            raise ValueError(
                f"Expected contiguous GT mesh depths {expected_depths}, got {actual_depths}"
            )

        do_cfg = guidance_scale > 1.0
        image_embeds, negative_image_embeds, view_indices_tensor, mv_cu_seqlens = (
            self.encode_image(image, device, do_cfg, view_indices=view_indices)
        )
        image_embeds = image_embeds.to(dtype)
        if do_cfg:
            negative_image_embeds = negative_image_embeds.to(dtype)

        joint_centers = torch.full(
            (num_joint_tokens, 3), grid_size // 2, dtype=torch.long, device=device
        )
        joint_ids = torch.arange(num_joint_tokens, dtype=torch.long, device=device)
        joint_layers = [joint_centers]
        sub_voxel_size = grid_size // 2

        for mesh_layer in tqdm(mesh_layers, disable=not enable_progress):
            joint_centers = self.sample_layer(
                scheduler=scheduler,
                mesh_layer=mesh_layer,
                joint_centers=joint_centers,
                joint_ids=joint_ids,
                device=device,
                dtype=dtype,
                generator=generator,
                sub_voxel_size=sub_voxel_size,
                num_inference_steps=num_inference_steps,
                image_embeds=image_embeds,
                negative_image_embeds=negative_image_embeds,
                guidance_scale=guidance_scale,
                num_vertices=num_vertices,
                prediction=prediction,
                view_indices=view_indices_tensor,
                mv_cu_seqlens=mv_cu_seqlens,
            )
            joint_layers.append(joint_centers)
            sub_voxel_size //= 2

        return ImageMeshToSingleStreamJointOctreePipelineOutput(
            joints=joint_centers,
            joint_layers=joint_layers,
        )
