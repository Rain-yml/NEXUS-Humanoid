"""Octree labels for persistent semantic joint tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class JointOctreeData:
    layer_occupancy: list[torch.Tensor]
    layer_parent_centers: list[torch.Tensor]
    layer_depths: list[int]
    num_vertices: int


def build_joint_layers(points: torch.Tensor, grid_size: int, max_depth: int) -> JointOctreeData:
    occupancies = []
    centers = []
    for depth in range(max_depth):
        voxel_size = grid_size // (2**depth)
        half = voxel_size // 2
        voxel_coords = points // voxel_size
        local_points = points - voxel_coords * voxel_size
        child_bits = (local_points >= half).long()
        child_ids = child_bits[:, 0] * 4 + child_bits[:, 1] * 2 + child_bits[:, 2]
        occupancy = torch.zeros((points.shape[0], 8), dtype=torch.long)
        occupancy.scatter_(1, child_ids[:, None], 1)
        occupancies.append(occupancy)
        centers.append(voxel_coords * voxel_size + half)
    return JointOctreeData(
        layer_occupancy=occupancies,
        layer_parent_centers=centers,
        layer_depths=list(range(max_depth)),
        num_vertices=points.shape[0],
    )
