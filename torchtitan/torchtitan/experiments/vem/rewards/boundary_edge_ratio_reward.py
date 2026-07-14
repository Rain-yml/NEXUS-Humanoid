from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch

from torchtitan.experiments.vem.rewards.base import MeshReward, MeshRewardOutput


def _face_edges(face: np.ndarray) -> list[tuple[int, int]]:
    return [
        tuple(sorted((int(face[i]), int(face[(i + 1) % len(face)]))))
        for i in range(len(face))
    ]


def compute_boundary_edge_ratio(
    triangles: np.ndarray,
    quads: np.ndarray,
) -> tuple[float, float, float]:
    """Return reward, boundary edge count, and total unique edge count."""
    edge_counts: Counter[tuple[int, int]] = Counter()

    triangles = np.asarray(triangles, dtype=np.int64)
    quads = np.asarray(quads, dtype=np.int64)
    if triangles.size:
        for tri in triangles.reshape(-1, 3):
            edge_counts.update(_face_edges(tri))
    if quads.size:
        for quad in quads.reshape(-1, 4):
            edge_counts.update(_face_edges(quad))

    num_total_edges = float(len(edge_counts))
    if num_total_edges == 0:
        return 0.0, 0.0, 0.0

    num_manifold_edges = float(sum(1 for count in edge_counts.values() if count == 2))
    ratio = num_manifold_edges / num_total_edges
    return float(ratio), num_manifold_edges, num_total_edges


class BoundaryEdgeRatioReward(MeshReward):
    """Reward meshes with fewer boundary edges.

    Defined as 1 - num_boundary_edge / num_total_edges over all unique edges from
    recovered triangles and quads.
    """

    def __init__(self, **kwargs) -> None:
        del kwargs

    def __call__(
        self,
        meshes: list[tuple[Any, Any, Any]],
        metadata: list[dict[str, Any]] | None = None,
        device: torch.device | str | None = None,
    ) -> MeshRewardOutput:
        del metadata

        ratios: list[float] = []
        num_manifold_edges: list[float] = []
        total_edges: list[float] = []
        for _verts, triangles, quads in meshes:
            ratio, num_manifold_edge, num_total_edges = compute_boundary_edge_ratio(
                triangles,
                quads,
            )
            ratios.append(ratio)
            num_manifold_edges.append(num_manifold_edge)
            total_edges.append(num_total_edges)

        reward_tensor = torch.tensor(ratios, dtype=torch.float32, device=device)
        manifold_tensor = torch.tensor(num_manifold_edges, dtype=torch.float32, device=device)
        total_tensor = torch.tensor(total_edges, dtype=torch.float32, device=device)

        metrics = {
            "reward/boundary_edge_ratio_mean": reward_tensor.mean().item() if reward_tensor.numel() else 0.0,
            "reward/n_manifold_edge_mean": manifold_tensor.mean().item() if manifold_tensor.numel() else 0.0,
            "reward/n_total_edge_mean": total_tensor.mean().item() if total_tensor.numel() else 0.0,
        }
        return MeshRewardOutput(
            rewards=reward_tensor,
            metrics=metrics,
            per_sample={
                "boundary_edge_ratio": reward_tensor,
                "n_manifold_edge": manifold_tensor,
                "n_total_edge": total_tensor,
            },
        )
