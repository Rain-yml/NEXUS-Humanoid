from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from torchtitan.experiments.vem.rewards.base import MeshReward, MeshRewardOutput
from torchtitan.experiments.vem.rewards.boundary_edge_ratio_reward import (
    BoundaryEdgeRatioReward,
)
from torchtitan.experiments.vem.rewards.loop_simplicity_reward import (
    LoopSimplicityReward,
)
from torchtitan.experiments.vem.rewards.quad_ratio_reward import QuadRatioReward


REWARD_REGISTRY = {
    "loop_simplicity": LoopSimplicityReward,
    "quad_ratio": QuadRatioReward,
    "boundary_edge_ratio": BoundaryEdgeRatioReward,
}


@dataclass(frozen=True)
class RewardSpec:
    name: str
    weight: float = 1.0


class WeightedReward(MeshReward):
    def __init__(
        self,
        reward_names: Sequence[str],
        reward_weights: Sequence[float] | None = None,
        reward_weighting_mode: str = "raw",
        **reward_kwargs,
    ) -> None:
        if not reward_names:
            raise ValueError("WeightedReward requires at least one reward name")
        if reward_weights is None:
            reward_weights = [1.0] * len(reward_names)
        if len(reward_names) != len(reward_weights):
            raise ValueError(
                "reward_names and reward_weights must have the same length"
            )
        if reward_weighting_mode not in {"raw", "batch_standardize"}:
            raise ValueError(
                "reward_weighting_mode must be one of {'raw', 'batch_standardize'}"
            )

        self.reward_weighting_mode = reward_weighting_mode
        self.reward_specs = [
            RewardSpec(name=name, weight=float(weight))
            for name, weight in zip(reward_names, reward_weights)
        ]
        self.rewards = [
            build_reward(name, **reward_kwargs)
            for name in reward_names
        ]

    @staticmethod
    def _standardize(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        mean = x.mean()
        std = x.std(unbiased=False)
        return (x - mean) / (std + 1e-4)

    def __call__(
        self,
        meshes: list[tuple[Any, Any, Any]],
        metadata: list[dict[str, Any]] | None = None,
        device: torch.device | str | None = None,
    ) -> MeshRewardOutput:
        component_outputs = [
            reward_fn(meshes, metadata=metadata, device=device)
            for reward_fn in self.rewards
        ]
        if not component_outputs:
            empty = torch.zeros(0, dtype=torch.float32, device=device)
            return MeshRewardOutput(rewards=empty)

        weighted_components: list[torch.Tensor] = []
        metrics: dict[str, float] = {}
        per_sample: dict[str, torch.Tensor] = {}

        for spec, output in zip(self.reward_specs, component_outputs):
            raw_reward = output.rewards.float()
            component = (
                self._standardize(raw_reward)
                if self.reward_weighting_mode == "batch_standardize"
                else raw_reward
            )
            weighted_components.append(component * spec.weight)
            metrics.update(output.metrics)
            for key, value in output.per_sample.items():
                per_sample[f"{spec.name}/{key}"] = value
            per_sample[f"{spec.name}/reward"] = raw_reward
            metrics[f"reward/{spec.name}_weight"] = float(spec.weight)

        reward_tensor = torch.stack(weighted_components, dim=0).sum(dim=0)
        metrics["reward/weighted_mean"] = reward_tensor.mean().item() if reward_tensor.numel() else 0.0
        metrics["reward/num_components"] = float(len(component_outputs))
        metrics["reward/weighting_mode"] = 0.0
        return MeshRewardOutput(
            rewards=reward_tensor,
            metrics=metrics,
            per_sample=per_sample,
        )


def build_reward(name: str, **kwargs) -> MeshReward:
    if name == "weighted":
        reward_names = kwargs.pop("reward_names", None)
        reward_weights = kwargs.pop("reward_weights", None)
        reward_weighting_mode = kwargs.pop("reward_weighting_mode", "raw")
        if reward_names is None:
            raise ValueError("weighted reward requires reward_names")
        return WeightedReward(
            reward_names=reward_names,
            reward_weights=reward_weights,
            reward_weighting_mode=reward_weighting_mode,
            **kwargs,
        )
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward {name!r}. Available rewards: {sorted(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name](**kwargs)


__all__ = [
    "MeshReward",
    "MeshRewardOutput",
    "LoopSimplicityReward",
    "QuadRatioReward",
    "BoundaryEdgeRatioReward",
    "REWARD_REGISTRY",
    "RewardSpec",
    "WeightedReward",
    "build_reward",
]
