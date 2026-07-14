from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class MeshRewardOutput:
    rewards: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)
    per_sample: dict[str, torch.Tensor] = field(default_factory=dict)


class MeshReward(ABC):
    @abstractmethod
    def __call__(
        self,
        meshes: list[tuple[Any, Any, Any]],
        metadata: list[dict[str, Any]] | None = None,
        device: torch.device | str | None = None,
    ) -> MeshRewardOutput:
        ...
