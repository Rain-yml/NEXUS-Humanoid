"""Views of a packed humanoid batch split into mesh and semantic-joint streams."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.experiments.humanoid.data.dataset import JointOctreeBatch


@dataclass(frozen=True)
class OctreeStream:
    occupancy: torch.Tensor
    centers: torch.Tensor
    depths: torch.Tensor
    cu_seqlens: torch.Tensor
    joint_ids: torch.Tensor | None = None


def _filtered_cu_seqlens(mask: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    lengths = torch.diff(cu_seqlens).to(dtype=torch.long)
    sequence_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=mask.device), lengths
    )
    counts = torch.zeros(lengths.numel(), dtype=torch.int32, device=mask.device)
    counts.scatter_add_(0, sequence_ids, mask.to(dtype=torch.int32))
    result = torch.zeros(counts.numel() + 1, dtype=torch.int32, device=mask.device)
    result[1:] = torch.cumsum(counts, dim=0)
    return result


def split_octree_streams(batch: JointOctreeBatch) -> tuple[OctreeStream, OctreeStream]:
    """Split existing packed tensors without changing dataset semantics or ordering."""
    joint_mask = batch.joint_mask_flat
    joint_ids = batch.joint_ids_flat
    if joint_mask is None or joint_ids is None:
        raise ValueError("Dual-branch training requires joint masks and semantic IDs")
    if not joint_mask.any() or joint_mask.all():
        raise ValueError("A dual-branch batch must contain both mesh and joint tokens")

    mesh_mask = ~joint_mask
    mesh = OctreeStream(
        occupancy=batch.layer_occupancy_flat[mesh_mask],
        centers=batch.layer_parent_centers_flat[mesh_mask],
        depths=batch.layer_depths_flat[mesh_mask],
        cu_seqlens=_filtered_cu_seqlens(mesh_mask, batch.cu_seqlens),
    )
    joints = OctreeStream(
        occupancy=batch.layer_occupancy_flat[joint_mask],
        centers=batch.layer_parent_centers_flat[joint_mask],
        depths=batch.layer_depths_flat[joint_mask],
        cu_seqlens=_filtered_cu_seqlens(joint_mask, batch.cu_seqlens),
        joint_ids=joint_ids[joint_mask],
    )
    if mesh.cu_seqlens.numel() != joints.cu_seqlens.numel():
        raise ValueError("Mesh and joint streams must contain the same packed sequences")
    return mesh, joints


__all__ = ["OctreeStream", "split_octree_streams"]
