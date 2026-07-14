"""Rigged humanoid multiview dataset derived from NEXUS vertex octree data."""

from __future__ import annotations

import gc
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.experiments.humanoid.data.joint_schema import JointSchema
from torchtitan.experiments.humanoid.data.joint_octree import build_joint_layers
from torchtitan.experiments.humanoid.data.manifest import parse_bos_uri, read_manifest
from torchtitan.experiments.vem.datasets.octree_utils import (
    OctreeBatch,
    OctreeData,
    build_octree_by_layer,
    discretize,
)
from torchtitan.tools.logging import logger


@dataclass
class JointOctreeBatch(OctreeBatch):
    joint_ids_flat: torch.Tensor | None = None
    joint_mask_flat: torch.Tensor | None = None

    def to(self, device: torch.device) -> "JointOctreeBatch":
        values = vars(super().to(device))
        values.update(
            joint_ids_flat=self.joint_ids_flat.to(device),
            joint_mask_flat=self.joint_mask_flat.to(device),
        )
        return JointOctreeBatch(**values)


def _normalize_like_nexus(vertices: np.ndarray, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    extent = float((maximum - minimum).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError(f"Invalid mesh extent: {extent}")
    scale = 2.0 / extent
    return (vertices - center) * scale, (joints - center) * scale


class RiggedHumanoidJointOctreeDataset(IterableDataset, Stateful):
    """Load mesh, canonical joints, and four views from an SSOT Parquet manifest."""

    worker_shard_data = ["rows"]

    def __init__(
        self,
        manifest_path: str,
        joint_schema_path: str,
        split: str = "train",
        repeats: int = 1,
        shuffle_seed: int = 42,
        grid_size: int = 512,
        max_depth: int = 9,
        image_resolution: int = 512,
        drop_image_rate: float = 0.1,
        infinite: bool = True,
        max_sample_retries: int = 3,
        force_divisible_by: int = 1,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        if grid_size != 2**max_depth:
            raise ValueError(f"grid_size={grid_size} must equal 2 ** max_depth={max_depth}")
        self.manifest_path = str(manifest_path)
        self.schema = JointSchema.load(joint_schema_path)
        self.grid_size = grid_size
        self.max_depth = max_depth
        self.image_resolution = image_resolution
        self.drop_image_rate = drop_image_rate
        self.infinite = infinite
        self.max_sample_retries = max_sample_retries
        self.sample_idx = 0
        self._bos_client = None

        frame = read_manifest(manifest_path, split=split)
        schema_names = set(frame["joint_schema"].dropna().astype(str))
        if schema_names != {self.schema.name}:
            raise ValueError(
                f"Manifest joint_schema values {sorted(schema_names)} do not match {self.schema.name!r}"
            )
        rows = frame.to_dict("records") * repeats
        random.Random(shuffle_seed).shuffle(rows)
        if force_divisible_by > 1:
            rows = rows[: len(rows) // force_divisible_by * force_divisible_by]
        self.rows = rows[dp_rank::dp_world_size]
        if not self.rows:
            raise ValueError(f"No rows assigned to rank {dp_rank} from {manifest_path}")
        logger.info(
            f"RiggedHumanoidJointOctreeDataset: manifest={manifest_path}, split={split}, "
            f"rows={len(self.rows)}, joints={len(self.schema.joints)}"
        )

    @property
    def bos_client(self):
        if self._bos_client is None:
            from torchtitan.experiments.humanoid.data.bos import BOSClient

            self._bos_client = BOSClient()
        return self._bos_client

    def _read_uri(self, uri: str) -> io.BytesIO:
        if uri.startswith("bos://"):
            bucket, key = parse_bos_uri(uri)
            return self.bos_client.get_file(bucket, key)
        if uri.startswith("file://"):
            uri = uri.removeprefix("file://")
        return io.BytesIO(Path(uri).read_bytes())

    def _load_rig(self, row: dict[str, Any]) -> tuple[OctreeData, OctreeData]:
        with np.load(self._read_uri(row["rig_npz_uri"]), allow_pickle=True) as rig:
            vertices = np.asarray(rig["vertices"], dtype=np.float32)
            joints = self.schema.select(
                rig["joint_semantics"].tolist(),
                np.asarray(rig["joint_positions"], dtype=np.float32),
                np.asarray(rig["parents"], dtype=np.int64),
            )
        vertices, joints = _normalize_like_nexus(vertices, joints)
        mesh_points = np.unique(discretize(vertices, self.grid_size), axis=0)
        joint_points = discretize(joints, self.grid_size)
        mesh_octree = build_octree_by_layer(
            torch.from_numpy(mesh_points).long(), self.grid_size, self.max_depth
        )
        joint_octree = build_joint_layers(
            torch.from_numpy(joint_points).long(), self.grid_size, self.max_depth
        )
        return mesh_octree, joint_octree

    def _load_image(self, uri: str) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self._read_uri(uri)).convert("RGBA")
        rgba = np.asarray(image, dtype=np.float32) / 255.0
        alpha = rgba[..., 3]
        foreground = np.where(alpha > 0)
        if foreground[0].size:
            y0, y1 = foreground[0].min(), foreground[0].max() + 1
            x0, x1 = foreground[1].min(), foreground[1].max() + 1
            size = max(y1 - y0, x1 - x0)
            margin = max(1, int(round(size * 0.05)))
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            half = size // 2 + margin
            y0, y1 = max(0, cy - half), min(rgba.shape[0], cy + half)
            x0, x1 = max(0, cx - half), min(rgba.shape[1], cx + half)
            rgba = rgba[y0:y1, x0:x1]
        rgb = rgba[..., :3] * rgba[..., 3:4] + 0.5 * (1.0 - rgba[..., 3:4])
        rgb_image = Image.fromarray(np.round(rgb * 255).astype(np.uint8))
        rgb_image = rgb_image.resize((self.image_resolution, self.image_resolution), Image.Resampling.LANCZOS)
        image_tensor = torch.from_numpy(np.asarray(rgb_image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mask_image = Image.fromarray(np.round(rgba[..., 3] * 255).astype(np.uint8))
        mask_image = mask_image.resize((self.image_resolution, self.image_resolution), Image.Resampling.NEAREST)
        mask_tensor = torch.from_numpy(np.asarray(mask_image) > 0)
        return image_tensor.contiguous(), mask_tensor

    def get_data(self, row: dict[str, Any]) -> dict[str, Any]:
        mesh_octree, joint_octree = self._load_rig(row)
        images, masks = zip(
            *(self._load_image(row[f"color_view_{index}_uri"]) for index in range(4))
        )
        image_tensor = torch.stack(images)
        if self.drop_image_rate > 0 and random.random() < self.drop_image_rate:
            image_tensor = torch.full_like(image_tensor, 0.5)
        return {
            "instance_id": str(row["uuid"]),
            "mesh_octree": mesh_octree,
            "joint_octree": joint_octree,
            "images": image_tensor,
            "image_masks": torch.stack(masks),
            "view_indices": torch.arange(4, dtype=torch.long),
        }

    def __iter__(self):
        while True:
            row = self.rows[self.sample_idx % len(self.rows)]
            last_error = None
            for _ in range(self.max_sample_retries):
                try:
                    yield self.get_data(row)
                    last_error = None
                    break
                except GeneratorExit:
                    raise
                except Exception as error:
                    last_error = error
                    logger.warning(f"Failed humanoid sample {row['uuid']}: {error}")
            if last_error is not None:
                raise RuntimeError(f"Failed humanoid sample {row['uuid']}") from last_error
            self.sample_idx += 1
            if self.sample_idx >= len(self.rows):
                if not self.infinite:
                    return
                self.sample_idx = 0

    def state_dict(self):
        return {"sample_idx": self.sample_idx, "rows": self.rows}

    def load_state_dict(self, state_dict):
        self.sample_idx = state_dict["sample_idx"]
        self.rows = state_dict["rows"]

    def collate_fn(self, batch: list[dict[str, Any]]) -> JointOctreeBatch:
        occupancies, centers, depths = [], [], []
        joint_ids, joint_masks, sequence_lengths, instance_ids = [], [], [], []
        layers_per_mesh = []
        for sample in batch:
            mesh = sample["mesh_octree"]
            joints = sample["joint_octree"]
            layers_per_mesh.append(mesh.num_layers)
            for layer_index in range(mesh.num_layers):
                mesh_occupancy = mesh.layer_occupancy[layer_index] * 2 - 1
                joint_occupancy = joints.layer_occupancy[layer_index] * 2 - 1
                occupancies.append(torch.cat([mesh_occupancy, joint_occupancy]).float())
                centers.append(
                    torch.cat(
                        [mesh.layer_parent_centers[layer_index], joints.layer_parent_centers[layer_index]]
                    )
                )
                count_mesh = mesh_occupancy.shape[0]
                count_joints = joint_occupancy.shape[0]
                depths.append(torch.full((count_mesh + count_joints,), layer_index, dtype=torch.long))
                joint_ids.append(
                    torch.cat(
                        [torch.full((count_mesh,), -1), torch.arange(count_joints, dtype=torch.long)]
                    )
                )
                joint_masks.append(
                    torch.cat(
                        [torch.zeros(count_mesh, dtype=torch.bool), torch.ones(count_joints, dtype=torch.bool)]
                    )
                )
                sequence_lengths.append(count_mesh + count_joints)
                instance_ids.append(sample["instance_id"])

        cu_seqlens = torch.zeros(len(sequence_lengths) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(torch.tensor(sequence_lengths, dtype=torch.int32), dim=0)
        views_per_mesh = [sample["images"].shape[0] for sample in batch]
        mv_cu_seqlens = torch.zeros(len(batch) + 1, dtype=torch.int32)
        mv_cu_seqlens[1:] = torch.cumsum(torch.tensor(views_per_mesh, dtype=torch.int32), dim=0)
        images = torch.cat([sample["images"] for sample in batch])
        result = JointOctreeBatch(
            layer_occupancy_flat=torch.cat(occupancies),
            layer_parent_centers_flat=torch.cat(centers),
            layer_depths_flat=torch.cat(depths),
            cu_seqlens=cu_seqlens,
            max_seqlen=max(sequence_lengths),
            layer_idx=-1,
            batch_size=len(sequence_lengths),
            instance_ids=instance_ids,
            num_layers_per_mesh=layers_per_mesh,
            images=images,
            image_masks=torch.cat([sample["image_masks"] for sample in batch]),
            uncond_images=torch.full_like(images, 0.5),
            num_vertices=torch.tensor(
                [sample["mesh_octree"].num_vertices for sample in batch], dtype=torch.int32
            ),
            view_indices=torch.cat([sample["view_indices"] for sample in batch]),
            mv_cu_seqlens=mv_cu_seqlens,
            joint_ids_flat=torch.cat(joint_ids),
            joint_mask_flat=torch.cat(joint_masks),
        )
        del batch
        gc.collect()
        return result
