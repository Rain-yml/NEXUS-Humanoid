from __future__ import annotations

from typing import Any

import torch

from torchtitan.experiments.vem.rewards.base import MeshReward, MeshRewardOutput


class QuadRatioReward(MeshReward):
    """Reward meshes with more quads than triangles.

    Defined as 2 * n_quad / (2 * n_quad + n_tri).
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

        quad_ratios: list[float] = []
        n_triangles: list[float] = []
        n_quads: list[float] = []

        for _verts, triangles, quads in meshes:
            n_tri = float(len(triangles))
            n_quad = float(len(quads))
            denom = 2.0 * n_quad + n_tri
            ratio = (2.0 * n_quad / denom) if denom > 0 else 0.0
            # ratio = (n_tri / denom) if denom > 0 else 0.0
            quad_ratios.append(ratio)
            n_triangles.append(n_tri)
            n_quads.append(n_quad)

        reward_tensor = torch.tensor(quad_ratios, dtype=torch.float32, device=device)
        tri_tensor = torch.tensor(n_triangles, dtype=torch.float32, device=device)
        quad_tensor = torch.tensor(n_quads, dtype=torch.float32, device=device)

        metrics = {
            "reward/quad_ratio_mean": reward_tensor.mean().item() if reward_tensor.numel() else 0.0,
            "reward/n_tri_mean": tri_tensor.mean().item() if tri_tensor.numel() else 0.0,
            "reward/n_quad_mean": quad_tensor.mean().item() if quad_tensor.numel() else 0.0,
        }
        return MeshRewardOutput(
            rewards=reward_tensor,
            metrics=metrics,
            per_sample={
                "quad_ratio": reward_tensor,
                "n_tri": tri_tensor,
                "n_quad": quad_tensor,
            },
        )
