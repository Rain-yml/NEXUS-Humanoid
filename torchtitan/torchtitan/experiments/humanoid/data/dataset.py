"""Rigged humanoid multiview dataset derived from NEXUS vertex octree data."""

from __future__ import annotations

import gc
import gzip
import io
import json
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchvision.transforms import v2 as transforms_v2

from torchtitan.experiments.humanoid.data.joint_schema import JointSchema
from torchtitan.experiments.humanoid.data.joint_octree import (
    JointOctreeData,
    build_joint_specific_layer,
)
from torchtitan.experiments.humanoid.data.manifest import parse_bos_uri, read_manifest
from torchtitan.experiments.vem.datasets.mesh_utils import rand_with_pt
from torchtitan.experiments.vem.datasets.octree_utils import (
    OctreeBatch,
    OctreeData,
    build_octree_specific_layer,
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


@dataclass
class NormalizedHumanoidRig:
    vertices: np.ndarray
    joints: np.ndarray
    mesh_points: torch.Tensor
    joint_points: torch.Tensor


def _normalize_like_nexus(vertices: np.ndarray, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    extent = float((maximum - minimum).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError(f"Invalid mesh extent: {extent}")
    scale = 2.0 / extent
    return (vertices - center) * scale, (joints - center) * scale


def build_humanoid_image_transform():
    identity = transforms_v2.Lambda(lambda image: image)
    return transforms_v2.Compose(
        [
            transforms_v2.RandomApply(
                [
                    transforms_v2.ColorJitter(hue=0.3),
                    transforms_v2.RandomChoice(
                        [transforms_v2.GaussianBlur(kernel_size=(3, 7)), identity]
                    ),
                    transforms_v2.RandomChoice(
                        [transforms_v2.JPEG(quality=(20, 80)), identity]
                    ),
                ],
                p=0.5,
            )
        ]
    )


def preprocess_humanoid_image(
    image: Image.Image,
    image_resolution: int,
    transform,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the image preprocessing shared by training and validation."""
    image_np = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    mask = image_np[..., 3]
    pixels_y, pixels_x = np.where(mask > 0)

    if len(pixels_y) > 0:
        height, width = mask.shape
        h0, h1 = max(pixels_y.min() - 1, 0), min(pixels_y.max() + 1, height)
        w0, w1 = max(pixels_x.min() - 1, 0), min(pixels_x.max() + 1, width)
        crop_h0, crop_h1, crop_w0, crop_w1 = (
            np.random.random(4) < 0.05
        ).tolist()
        height_fg, width_fg = h1 - h0, w1 - w0
        crop_h0_pixels, crop_h1_pixels = (
            np.random.random(2) * height_fg * 0.02
        ).astype(np.int32).tolist()
        crop_w0_pixels, crop_w1_pixels = (
            np.random.random(2) * width_fg * 0.02
        ).astype(np.int32).tolist()
        if crop_h0:
            h0 += crop_h0_pixels
        if crop_h1:
            h1 -= crop_h1_pixels
        if crop_w0:
            w0 += crop_w0_pixels
        if crop_w1:
            w1 -= crop_w1_pixels

        height_fg, width_fg = h1 - h0, w1 - w0
        pad_ratio = random.uniform(0.05, 0.2)
        size_padded = int(max(height_fg, width_fg) / (1 - pad_ratio))
        padded = np.zeros((size_padded, size_padded, 4), dtype=np.float32)
        start_h = (size_padded - height_fg) // 2
        start_w = (size_padded - width_fg) // 2
        padded[start_h : start_h + height_fg, start_w : start_w + width_fg] = image_np[
            h0:h1, w0:w1
        ]
        image_np = padded
        mask = image_np[..., 3] > 0

        foreground_gray = image_np[..., :3][image_np[..., 3] > 0].mean()
        background = np.random.rand(1)
        while foreground_gray - 0.2 < background < foreground_gray + 0.2:
            background = np.random.rand(1)
        background = np.repeat(background, 3)
        image_np = (
            image_np[..., :3] * image_np[..., 3:4]
            + background[None, None] * (1 - image_np[..., 3:4])
        )
    else:
        image_np = np.zeros_like(image_np[..., :3])
        mask = np.zeros(mask.shape, dtype=bool)

    rgb_image = Image.fromarray((image_np * 255).clip(0, 255).astype(np.uint8))
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    rgb_image = rgb_image.resize(
        (image_resolution, image_resolution), Image.Resampling.LANCZOS
    )
    mask_image = mask_image.resize(
        (image_resolution, image_resolution), Image.Resampling.NEAREST
    )
    rgb_image = transform(rgb_image)
    image_tensor = torch.from_numpy(
        np.asarray(rgb_image, dtype=np.float32) / 255.0
    ).permute(2, 0, 1)
    mask_tensor = torch.from_numpy(np.asarray(mask_image) > 0)
    return image_tensor.contiguous(), mask_tensor


class RiggedHumanoidJointOctreeDataset(IterableDataset, Stateful):
    """Load mesh, canonical joints, and four views from an SSOT Parquet manifest."""

    worker_shard_data = ["batches"]

    def __init__(
        self,
        manifest_path: str,
        batches_packed: str,
        joint_schema_path: str,
        split: str = "train",
        repeats: int = 1,
        shuffle_seed: int = 42,
        grid_size: int = 512,
        max_depth: int = 9,
        image_resolution: int = 512,
        view_indices: list[int] | None = None,
        drop_image_rate: float = 0.1,
        infinite: bool = True,
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
        self.view_indices = tuple(view_indices if view_indices is not None else range(4))
        if not self.view_indices or any(index not in range(4) for index in self.view_indices):
            raise ValueError(f"view_indices must be a non-empty subset of [0, 1, 2, 3]: {view_indices}")
        if len(set(self.view_indices)) != len(self.view_indices):
            raise ValueError(f"view_indices must not contain duplicates: {view_indices}")
        self.drop_image_rate = drop_image_rate
        self.infinite = infinite
        self.sample_idx = 0
        self._bos_client = None

        self.transform = build_humanoid_image_transform()

        frame = read_manifest(manifest_path, split=split)
        schema_names = set(frame["joint_schema"].dropna().astype(str))
        if schema_names != {self.schema.name}:
            raise ValueError(
                f"Manifest joint_schema values {sorted(schema_names)} do not match {self.schema.name!r}"
            )
        self.records = frame.set_index("uuid").to_dict("index")
        packed_path = Path(batches_packed)
        opener = gzip.open if packed_path.suffix == ".gz" else open
        with opener(packed_path, "rt", encoding="utf-8") as file:
            batches = json.load(file)["batches"] * repeats
        if force_divisible_by > 1:
            batches = batches[: len(batches) // force_divisible_by * force_divisible_by]
        if not batches:
            raise ValueError(
                f"No packed batches remain after enforcing divisibility by {force_divisible_by}"
            )
        missing = sorted(
            {item[0] for packed_batch in batches for item in packed_batch} - self.records.keys()
        )
        if missing:
            raise ValueError(f"Packed batches contain UUIDs absent from the manifest: {missing[:5]}")
        self.batches = batches[dp_rank::dp_world_size]
        random.Random(shuffle_seed).shuffle(self.batches)
        if not self.batches:
            raise ValueError(f"No packed batches assigned to rank {dp_rank}")
        logger.info(
            f"RiggedHumanoidJointOctreeDataset: manifest={manifest_path}, split={split}, "
            f"rank_batches={len(self.batches)}, "
            f"views={self.view_indices}, joints={len(self.schema.joints)}"
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

    def _load_normalized_rig(self, row: dict[str, Any]) -> NormalizedHumanoidRig:
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
        return NormalizedHumanoidRig(
            vertices=vertices,
            joints=joints,
            mesh_points=torch.from_numpy(mesh_points).long(),
            joint_points=torch.from_numpy(joint_points).long(),
        )

    def _build_rig_layer(
        self, rig: NormalizedHumanoidRig, layer_id: int
    ) -> tuple[OctreeData, JointOctreeData]:
        mesh_octree = build_octree_specific_layer(
            rig.mesh_points, layer_id, self.grid_size, self.max_depth
        )
        joint_octree = build_joint_specific_layer(
            rig.joint_points, self.grid_size, layer_id
        )
        return mesh_octree, joint_octree

    def _load_rig(
        self, row: dict[str, Any], layer_id: int
    ) -> tuple[OctreeData, JointOctreeData]:
        return self._build_rig_layer(self._load_normalized_rig(row), layer_id)

    def load_rig_layers_from_row(
        self, row: dict[str, Any]
    ) -> tuple[NormalizedHumanoidRig, list[OctreeData]]:
        rig = self._load_normalized_rig(row)
        mesh_layers = [
            self._build_rig_layer(rig, depth)[0] for depth in range(self.max_depth)
        ]
        return rig, mesh_layers

    def load_rig_layers(
        self, instance_id: str
    ) -> tuple[NormalizedHumanoidRig, list[OctreeData]]:
        return self.load_rig_layers_from_row(self.records[instance_id])

    def _load_image(self, uri: str) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self._read_uri(uri))
        return preprocess_humanoid_image(image, self.image_resolution, self.transform)

    def get_data(self, instance_id: str, layer_id: int) -> dict[str, Any]:
        row = self.records[instance_id]
        mesh_octree, joint_octree = self._load_rig(row, layer_id)
        images, masks = zip(
            *(self._load_image(row[f"color_view_{index}_uri"]) for index in self.view_indices)
        )
        image_tensor = torch.stack(images)
        if self.drop_image_rate > 0 and rand_with_pt([0, 1]) < self.drop_image_rate:
            image_tensor = torch.full_like(image_tensor, 0.5)
        return {
            "instance_id": instance_id,
            "mesh_octree": mesh_octree,
            "joint_octree": joint_octree,
            "images": image_tensor,
            "image_masks": torch.stack(masks),
            "view_indices": torch.tensor(self.view_indices, dtype=torch.long),
            "num_vertices": int(row["num_vertices"]),
        }

    def __iter__(self):
        while True:
            sample_idx = self.sample_idx
            while True:
                try:
                    yield [
                        self.get_data(instance_id, layer_id)
                        for instance_id, _sample_id, layer_id in self.batches[sample_idx]
                    ]
                    break
                except GeneratorExit:
                    raise
                except Exception:
                    logger.warning(
                        "Failed humanoid packed batch %s: %s",
                        self.batches[sample_idx],
                        traceback.format_exc(),
                    )
                    sample_idx = random.randrange(len(self.batches))
            self.sample_idx += 1
            if self.sample_idx >= len(self.batches):
                if not self.infinite:
                    return
                self.sample_idx = 0

    def state_dict(self):
        return {"sample_idx": self.sample_idx, "batches": self.batches}

    def load_state_dict(self, state_dict):
        self.sample_idx = state_dict["sample_idx"]
        self.batches = state_dict["batches"]

    def collate_fn(self, batch: list[dict[str, Any]]) -> JointOctreeBatch:
        batch = [item for packed_batch in batch for item in packed_batch]
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
                depth = mesh.layer_depths[layer_index]
                depths.append(torch.full((count_mesh + count_joints,), depth, dtype=torch.long))
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
                [sample["num_vertices"] for sample in batch], dtype=torch.int32
            ),
            view_indices=torch.cat([sample["view_indices"] for sample in batch]),
            mv_cu_seqlens=mv_cu_seqlens,
            joint_ids_flat=torch.cat(joint_ids),
            joint_mask_flat=torch.cat(joint_masks),
        )
        del batch
        gc.collect()
        return result
