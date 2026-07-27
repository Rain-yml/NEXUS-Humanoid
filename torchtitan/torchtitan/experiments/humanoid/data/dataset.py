"""Rigged humanoid multiview dataset derived from NEXUS vertex octree data."""

from __future__ import annotations

import gc
import io
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
from torchtitan.experiments.humanoid.data.rig_records import (
    HUMANOID_RESAVE_FORMAT,
    load_rig_record,
)
from torchtitan.experiments.humanoid.data.skeleton_augmentation import (
    augment_reference_skeleton,
    skeleton_edges,
)
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
    reference_joint_positions_flat: torch.Tensor | None = None
    reference_joint_rope_positions_flat: torch.Tensor | None = None
    reference_edge_index_flat: torch.Tensor | None = None
    reference_cu_seqlens: torch.Tensor | None = None

    def to(self, device: torch.device) -> "JointOctreeBatch":
        values = vars(super().to(device))
        values.update(
            joint_ids_flat=(
                self.joint_ids_flat.to(device) if self.joint_ids_flat is not None else None
            ),
            joint_mask_flat=(
                self.joint_mask_flat.to(device) if self.joint_mask_flat is not None else None
            ),
            reference_joint_positions_flat=(
                self.reference_joint_positions_flat.to(device)
                if self.reference_joint_positions_flat is not None
                else None
            ),
            reference_joint_rope_positions_flat=(
                self.reference_joint_rope_positions_flat.to(device)
                if self.reference_joint_rope_positions_flat is not None
                else None
            ),
            reference_edge_index_flat=(
                self.reference_edge_index_flat.to(device)
                if self.reference_edge_index_flat is not None
                else None
            ),
            reference_cu_seqlens=(
                self.reference_cu_seqlens.to(device)
                if self.reference_cu_seqlens is not None
                else None
            ),
        )
        return JointOctreeBatch(**values)


@dataclass
class NormalizedHumanoidRig:
    vertices: np.ndarray
    joints: np.ndarray
    parents: torch.Tensor
    joint_ids: torch.Tensor
    mesh_points: torch.Tensor
    joint_points: torch.Tensor


class OversizedHumanoidRigError(ValueError):
    pass


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

    worker_shard_data = ["items"]

    def __init__(
        self,
        manifest_path: str,
        joint_schema_path: str | None = None,
        split: str = "train",
        repeats: int = 1,
        shuffle_seed: int = 42,
        grid_size: int = 512,
        max_depth: int = 9,
        image_resolution: int = 512,
        view_indices: list[int] | None = None,
        drop_image_rate: float = 0.1,
        infinite: bool = True,
        max_merged_vertices: int = 11_000,
        joint_selection: str = "strict",
        reference_skeleton_augmentation: bool = False,
        reference_global_scale_log_std: float = 0.08,
        reference_local_scale_log_std: float = 0.10,
        reference_warp_num_segments: int = 4,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        if grid_size != 2**max_depth:
            raise ValueError(f"grid_size={grid_size} must equal 2 ** max_depth={max_depth}")
        self.manifest_path = str(manifest_path)
        self.schema = JointSchema.load(joint_schema_path) if joint_schema_path else None
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
        if joint_selection not in {"strict", "available"}:
            raise ValueError(
                f"joint_selection must be 'strict' or 'available', got {joint_selection!r}"
            )
        self.joint_selection = joint_selection
        self.reference_skeleton_augmentation = reference_skeleton_augmentation
        self.reference_global_scale_log_std = reference_global_scale_log_std
        self.reference_local_scale_log_std = reference_local_scale_log_std
        self.reference_warp_num_segments = reference_warp_num_segments
        if max_merged_vertices < 1:
            raise ValueError("max_merged_vertices must be positive")
        self.max_merged_vertices = max_merged_vertices
        self.sample_idx = 0
        self._bos_client = None

        self.transform = build_humanoid_image_transform()

        frame = read_manifest(
            manifest_path, split=split, view_indices=self.view_indices
        )
        if self.schema is not None:
            if "joint_schema" not in frame:
                raise ValueError("A canonical joint schema requires a joint_schema column")
            schema_names = set(frame["joint_schema"].dropna().astype(str))
            if schema_names != {self.schema.name}:
                raise ValueError(
                    f"Manifest joint_schema values {sorted(schema_names)} do not match {self.schema.name!r}"
                )
        self.records = frame.set_index("uuid").to_dict("index")
        items = [
            (uuid, layer_id)
            for uuid in self.records
            for layer_id in range(self.max_depth)
        ] * repeats
        self.items = items[dp_rank::dp_world_size]
        random.Random(shuffle_seed).shuffle(self.items)
        if not self.items:
            raise ValueError(f"No samples assigned to rank {dp_rank}")
        logger.info(
            f"RiggedHumanoidJointOctreeDataset: manifest={manifest_path}, split={split}, "
            f"rank_items={len(self.items)}, views={self.view_indices}, "
            f"joint_schema={self.schema.name if self.schema is not None else 'per-asset'}, "
            f"joint_selection={self.joint_selection}, "
            f"reference_skeleton_augmentation={self.reference_skeleton_augmentation}, "
            f"max_merged_vertices={self.max_merged_vertices}"
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
            rig_format = row.get("rig_format", HUMANOID_RESAVE_FORMAT)
            if not isinstance(rig_format, str) or not rig_format:
                rig_format = HUMANOID_RESAVE_FORMAT
            record = load_rig_record(
                rig,
                rig_format=rig_format,
                schema=self.schema,
                joint_selection=self.joint_selection,
            )
        vertices, joints = _normalize_like_nexus(record.vertices, record.joints)
        mesh_points = np.unique(discretize(vertices, self.grid_size), axis=0)
        if len(mesh_points) > self.max_merged_vertices:
            raise OversizedHumanoidRigError(
                f"Merged vertex count {len(mesh_points)} exceeds "
                f"max_merged_vertices={self.max_merged_vertices}"
            )
        joint_points = discretize(joints, self.grid_size)
        return NormalizedHumanoidRig(
            vertices=vertices,
            joints=joints,
            parents=torch.from_numpy(record.parents).long(),
            joint_ids=torch.from_numpy(record.joint_ids).long(),
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
    ) -> tuple[OctreeData, JointOctreeData, torch.Tensor]:
        rig = self._load_normalized_rig(row)
        mesh_octree, joint_octree = self._build_rig_layer(rig, layer_id)
        return mesh_octree, joint_octree, rig.joint_ids

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
        rig = self._load_normalized_rig(row)
        mesh_octree, joint_octree = self._build_rig_layer(rig, layer_id)
        images, masks = zip(
            *(self._load_image(row[f"color_view_{index}_uri"]) for index in self.view_indices)
        )
        image_tensor = torch.stack(images)
        if self.drop_image_rate > 0 and rand_with_pt([0, 1]) < self.drop_image_rate:
            image_tensor = torch.full_like(image_tensor, 0.5)
        result = {
            "instance_id": instance_id,
            "mesh_octree": mesh_octree,
            "joint_octree": joint_octree,
            "joint_ids": rig.joint_ids,
            "images": image_tensor,
            "image_masks": torch.stack(masks),
            "view_indices": torch.tensor(self.view_indices, dtype=torch.long),
            "num_vertices": int(mesh_octree.num_vertices),
        }
        if self.reference_skeleton_augmentation:
            reference_joints = augment_reference_skeleton(
                torch.from_numpy(rig.joints),
                rig.parents,
                global_scale_log_std=self.reference_global_scale_log_std,
                local_scale_log_std=self.reference_local_scale_log_std,
                num_segments=self.reference_warp_num_segments,
            ).clamp(-1.0, 1.0)
            result.update(
                reference_joint_positions=reference_joints,
                reference_joint_rope_positions=torch.from_numpy(
                    discretize(reference_joints.numpy(), self.grid_size)
                ).long(),
                joint_parents=rig.parents,
            )
        return result

    def __iter__(self):
        while True:
            if self.sample_idx >= len(self.items):
                if self.infinite:
                    self.sample_idx = 0
                else:
                    return
            item = self.items[self.sample_idx]
            self.sample_idx += 1
            try:
                yield [self.get_data(*item)]
            except GeneratorExit:
                raise
            except Exception:
                logger.warning(
                    "Failed humanoid sample %s: %s", item, traceback.format_exc()
                )

            if not self.infinite and self.sample_idx >= len(self.items):
                return

    def state_dict(self):
        return {
            "sample_idx": self.sample_idx,
            "items": self.items,
        }

    def load_state_dict(self, state_dict):
        self.sample_idx = state_dict["sample_idx"]
        self.items = state_dict["items"]

    def collate_fn(self, batch: list[dict[str, Any]]) -> JointOctreeBatch:
        batch = [item for packed_batch in batch for item in packed_batch]
        occupancies, centers, depths = [], [], []
        joint_ids, joint_masks, sequence_lengths, instance_ids = [], [], [], []
        reference_positions, reference_rope_positions = [], []
        reference_edges, reference_sequence_lengths = [], []
        reference_offset = 0
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
                sample_joint_ids = sample["joint_ids"]
                if sample_joint_ids.shape != (count_joints,):
                    raise ValueError(
                        f"Joint ID count {sample_joint_ids.shape} does not match "
                        f"joint token count {count_joints}"
                    )
                depth = mesh.layer_depths[layer_index]
                depths.append(torch.full((count_mesh + count_joints,), depth, dtype=torch.long))
                joint_ids.append(
                    torch.cat(
                        [torch.full((count_mesh,), -1), sample_joint_ids]
                    )
                )
                joint_masks.append(
                    torch.cat(
                        [torch.zeros(count_mesh, dtype=torch.bool), torch.ones(count_joints, dtype=torch.bool)]
                    )
                )
                sequence_lengths.append(count_mesh + count_joints)
                instance_ids.append(sample["instance_id"])
                if "reference_joint_positions" in sample:
                    if sample["reference_joint_positions"].shape != (count_joints, 3):
                        raise ValueError("Reference joint count does not match target joints")
                    reference_positions.append(sample["reference_joint_positions"])
                    reference_rope_positions.append(
                        sample["reference_joint_rope_positions"]
                    )
                    reference_edges.append(
                        skeleton_edges(sample["joint_parents"]) + reference_offset
                    )
                    reference_sequence_lengths.append(count_joints)
                    reference_offset += count_joints

        has_references = bool(reference_positions)
        if has_references and len(reference_sequence_lengths) != len(sequence_lengths):
            raise ValueError("Reference skeletons must be present for every packed sequence")
        reference_cu_seqlens = None
        if has_references:
            reference_cu_seqlens = torch.zeros(
                len(reference_sequence_lengths) + 1, dtype=torch.int32
            )
            reference_cu_seqlens[1:] = torch.cumsum(
                torch.tensor(reference_sequence_lengths, dtype=torch.int32), dim=0
            )

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
            reference_joint_positions_flat=(
                torch.cat(reference_positions) if has_references else None
            ),
            reference_joint_rope_positions_flat=(
                torch.cat(reference_rope_positions) if has_references else None
            ),
            reference_edge_index_flat=(
                torch.cat(reference_edges, dim=1) if has_references else None
            ),
            reference_cu_seqlens=reference_cu_seqlens,
        )
        del batch
        gc.collect()
        return result
