from __future__ import annotations

import gc
import os
import random
import traceback
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.experiments.vem.datasets.json_utils import load_json
from torchtitan.experiments.vem.datasets.mesh_gen import shard_interleave
from torchtitan.experiments.vem.datasets.mesh_gen_quad import SpaceTimeRGBGenQuadPackDataset
from torchtitan.tools.logging import logger


class SpaceTimeRGBGenRLPackDataset(SpaceTimeRGBGenQuadPackDataset):
    """Packed RL rollout dataset using the quad VAE data contract.

    RL rollouts still only diffuse vertex latents and score generated meshes,
    but the VAE encoder needs the same graph fields prepared by the quad-pack
    dataset to produce those latents.
    """

    def _collate_fn(self, batch):
        batch_collated = super()._collate_fn(batch)
        if "encoder_position" in batch_collated:
            batch_collated["decoder_position"] = batch_collated["encoder_position"][
                batch_collated["node_type"] == 0
            ]
        return batch_collated


class ImageVerticesRLDataset(Stateful, IterableDataset):
    worker_shard_data = ["samples"]

    def __init__(
        self,
        data_json: str,
        repeats: int = 1,
        shuffle_seed: int = 0,
        image_resolution: int = 512,
        infinite: bool = True,
        force_divisible_by: int = 1,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        self.data_json = data_json
        self.repeats = repeats
        self.shuffle_seed = shuffle_seed
        self.image_resolution = image_resolution
        self.infinite = infinite
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.sample_idx = 0

        raw_samples = load_json(data_json)
        samples = self._normalize_samples(raw_samples, data_json) * repeats
        if force_divisible_by > 1 and len(samples) >= force_divisible_by:
            samples = samples[: len(samples) // force_divisible_by * force_divisible_by]
            logger.info(f"Image-vertices RL sample number after dropping: {len(samples)}")
        elif force_divisible_by > 1:
            logger.info(
                "Image-vertices RL sample number is smaller than force_divisible_by; "
                f"keeping all {len(samples)} samples"
            )
        if not samples:
            raise ValueError(f"No image/vertices samples found in {data_json}")

        samples = shard_interleave(samples, dp_world_size, dp_rank)
        rng = random.Random(shuffle_seed)
        rng.shuffle(samples)
        self.samples = samples

    @staticmethod
    def _normalize_samples(raw_samples: list[Any], data_json: str) -> list[tuple[str, str]]:
        samples = []
        data_root = os.path.dirname(data_json)
        for idx, sample in enumerate(raw_samples):
            if isinstance(sample, dict):
                image_path = sample.get("image") or sample.get("image_path")
                vertices_path = (
                    sample.get("vertices")
                    or sample.get("vertices_path")
                    or sample.get("npz")
                    or sample.get("npz_path")
                )
            elif isinstance(sample, (list, tuple)) and len(sample) == 2:
                image_path, vertices_path = sample
            else:
                raise ValueError(
                    f"Sample {idx} in {data_json} must be [image_path, npz_path] or a dict"
                )

            if not isinstance(image_path, str) or not isinstance(vertices_path, str):
                raise ValueError(f"Sample {idx} in {data_json} has invalid paths: {sample}")
            if not os.path.isabs(image_path):
                image_path = os.path.join(data_root, image_path)
            if not os.path.isabs(vertices_path):
                vertices_path = os.path.join(data_root, vertices_path)
            samples.append((image_path, vertices_path))
        return samples

    def _load_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGBA")
        image = image.resize(
            (self.image_resolution, self.image_resolution),
            resample=Image.Resampling.BICUBIC,
        )
        image_np = np.asarray(image).astype(np.float32) / 255.0
        image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + 0.5 * (1 - image_np[:, :, 3:4])
        return torch.from_numpy(image_np).permute(2, 0, 1).clamp(0, 1)

    @staticmethod
    def _load_vertices(vertices_path: str) -> torch.Tensor:
        with np.load(vertices_path) as data:
            if "vertices" not in data:
                raise KeyError(f"{vertices_path} does not contain a 'vertices' array")
            vertices = np.asarray(data["vertices"])
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"{vertices_path} vertices must have shape (N, 3), got {vertices.shape}")
        return torch.from_numpy(vertices.astype(np.int64, copy=False)).long()

    def get_data(self, sample: tuple[str, str]):
        image_path, vertices_path = sample
        vertices = self._load_vertices(vertices_path)
        return {
            "image": self._load_image(image_path),
            "vertices": vertices,
            "decoder_position": vertices * 3,
            "instance_id": os.path.splitext(os.path.basename(vertices_path))[0],
        }

    def __iter__(self):
        while True:
            sample_idx = self.sample_idx
            while True:
                try:
                    yield [self.get_data(self.samples[sample_idx])]
                    break
                except GeneratorExit:
                    raise
                except Exception:
                    logger.warning(
                        f"Failed to load image-vertices RL sample {self.samples[sample_idx]}: "
                        f"{traceback.format_exc()}"
                    )
                    sample_idx = random.randint(0, len(self.samples) - 1)

            self.sample_idx += 1
            if self.sample_idx >= len(self.samples):
                if not self.infinite:
                    logger.warning("Dataset has run out of data.")
                    break
                self.sample_idx = 0
                logger.warning("Dataset is being re-looped.")

    def load_state_dict(self, state_dict):
        self.sample_idx = state_dict["sample_idx"]
        self.samples = state_dict["samples"]

    def state_dict(self):
        return {
            "samples": self.samples,
            "sample_idx": self.sample_idx,
        }

    def _collate_fn(self, batch):
        batch = [item for sublist in batch for item in sublist]
        vertex_offsets = [0]
        vertices = []
        decoder_positions = []

        for sample in batch:
            vertices.append(sample["vertices"])
            decoder_positions.append(sample["decoder_position"])
            vertex_offsets.append(vertex_offsets[-1] + sample["vertices"].shape[0])

        batch_collated = {
            "vertices": torch.cat(vertices, dim=0),
            "decoder_position": torch.cat(decoder_positions, dim=0),
            "cu_seqlens": torch.tensor(vertex_offsets, dtype=torch.int32),
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "instance_ids": [sample["instance_id"] for sample in batch],
        }
        del batch
        gc.collect()
        return batch_collated

    def collate_fn(self, batch):
        return self._collate_fn(batch)
