"""Canonical semantic joint schemas used by humanoid octree experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class JointSchema:
    name: str
    joints: tuple[str, ...]
    parents: tuple[int, ...]

    @classmethod
    def load(cls, path: str | Path) -> "JointSchema":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        joints = tuple(data["joints"])
        parents = tuple(int(parent) for parent in data["parents"])
        if len(joints) != len(parents) or len(set(joints)) != len(joints):
            raise ValueError(f"Invalid joint schema: {path}")
        return cls(name=data["name"], joints=joints, parents=parents)

    def select(
        self,
        semantics: Sequence[str],
        positions: np.ndarray,
        source_parents: np.ndarray,
    ) -> np.ndarray:
        semantics = [str(value) for value in semantics]
        selected: list[np.ndarray] = []
        for semantic in self.joints:
            if semantic in {"spine_0", "spine_1"}:
                continue
            matches = [index for index, value in enumerate(semantics) if value == semantic]
            if len(matches) != 1:
                raise ValueError(f"Expected one {semantic!r} joint, found {len(matches)}")
            selected.append(positions[matches[0]])

        spine_indices = [index for index, value in enumerate(semantics) if value == "spine"]
        if len(spine_indices) != 2:
            raise ValueError(f"Expected two spine joints, found {len(spine_indices)}")
        spine_set = set(spine_indices)
        spine_indices.sort(key=lambda index: self._ancestor_depth(index, source_parents, spine_set))
        by_name = {
            semantic: position
            for semantic, position in zip(
                (joint for joint in self.joints if joint not in {"spine_0", "spine_1"}),
                selected,
            )
        }
        by_name["spine_0"] = positions[spine_indices[0]]
        by_name["spine_1"] = positions[spine_indices[1]]
        return np.stack([by_name[name] for name in self.joints]).astype(np.float32)

    def select_available(
        self,
        semantics: Sequence[str],
        positions: np.ndarray,
        source_parents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select unambiguous joints and return positions with stable schema IDs."""
        semantics = [str(value) for value in semantics]
        selected: dict[int, np.ndarray] = {}
        for joint_id, semantic in enumerate(self.joints):
            if semantic in {"spine_0", "spine_1"}:
                continue
            matches = [index for index, value in enumerate(semantics) if value == semantic]
            if len(matches) == 1:
                selected[joint_id] = positions[matches[0]]

        spine_indices = [index for index, value in enumerate(semantics) if value == "spine"]
        if len(spine_indices) == 2:
            spine_set = set(spine_indices)
            spine_indices.sort(
                key=lambda index: self._ancestor_depth(index, source_parents, spine_set)
            )
            for semantic, source_index in zip(
                ("spine_0", "spine_1"), spine_indices, strict=True
            ):
                if semantic in self.joints:
                    selected[self.joints.index(semantic)] = positions[source_index]

        if not selected:
            raise ValueError("No unambiguous joints match the canonical schema")
        joint_ids = np.asarray(sorted(selected), dtype=np.int64)
        joint_positions = np.stack([selected[int(joint_id)] for joint_id in joint_ids])
        return joint_positions.astype(np.float32), joint_ids

    def parents_for_ids(self, joint_ids: Sequence[int]) -> np.ndarray:
        """Map schema parents into a subset, skipping absent ancestors."""
        joint_ids = [int(joint_id) for joint_id in joint_ids]
        local_by_global = {
            global_id: local_id for local_id, global_id in enumerate(joint_ids)
        }
        local_parents = []
        for global_id in joint_ids:
            parent = self.parents[global_id]
            while parent >= 0 and parent not in local_by_global:
                parent = self.parents[parent]
            local_parents.append(local_by_global.get(parent, -1))
        return np.asarray(local_parents, dtype=np.int64)

    @staticmethod
    def _ancestor_depth(index: int, parents: np.ndarray, spine_set: set[int]) -> int:
        depth = 0
        cursor = int(parents[index])
        visited = {index}
        while cursor >= 0 and cursor not in visited:
            visited.add(cursor)
            if cursor in spine_set:
                depth += 1
            cursor = int(parents[cursor])
        return depth
