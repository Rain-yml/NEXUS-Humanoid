"""Normalize source-specific rig files into one training data contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from torchtitan.experiments.humanoid.data.joint_schema import JointSchema


HUMANOID_RESAVE_FORMAT = "humanoid_resave_v1"
ANIGEN_VOXELIZED_FORMAT = "anigen_voxelized_v1"


@dataclass(frozen=True)
class RigRecord:
    vertices: np.ndarray
    joints: np.ndarray
    parents: np.ndarray
    joint_ids: np.ndarray


def _validate_rig(record: RigRecord) -> RigRecord:
    vertices = np.asarray(record.vertices, dtype=np.float32)
    joints = np.asarray(record.joints, dtype=np.float32)
    parents = np.asarray(record.parents, dtype=np.int64)
    joint_ids = np.asarray(record.joint_ids, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Expected non-empty vertices shaped (N, 3), got {vertices.shape}")
    if joints.ndim != 2 or joints.shape[1] != 3 or len(joints) == 0:
        raise ValueError(f"Expected non-empty joints shaped (J, 3), got {joints.shape}")
    if parents.shape != (len(joints),):
        raise ValueError(f"Expected {len(joints)} parents, got {parents.shape}")
    if joint_ids.shape != (len(joints),):
        raise ValueError(f"Expected {len(joints)} joint IDs, got {joint_ids.shape}")
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise ValueError("Rig contains non-finite coordinates")
    if ((parents < -1) | (parents >= len(joints))).any():
        raise ValueError("Rig contains an out-of-range parent index")
    if np.any(parents == np.arange(len(joints))):
        raise ValueError("Rig contains a self-parented joint")

    for joint_index in range(len(joints)):
        cursor = joint_index
        visited: set[int] = set()
        while cursor >= 0:
            if cursor in visited:
                raise ValueError("Rig contains a cycle in its skeleton graph")
            visited.add(cursor)
            cursor = int(parents[cursor])

    return RigRecord(
        vertices=np.ascontiguousarray(vertices),
        joints=np.ascontiguousarray(joints),
        parents=np.ascontiguousarray(parents),
        joint_ids=np.ascontiguousarray(joint_ids),
    )


def _object_parents(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [-1 if value is None else int(value) for value in values.tolist()],
        dtype=np.int64,
    )


def load_rig_record(
    arrays: Mapping[str, Any],
    *,
    rig_format: str,
    schema: JointSchema | None,
    joint_selection: str,
) -> RigRecord:
    """Load either project resaves or AniGenP voxelized rigs without path logic."""
    vertices = np.asarray(arrays["vertices"], dtype=np.float32)

    if rig_format == ANIGEN_VOXELIZED_FORMAT:
        joints = np.asarray(arrays["joints"], dtype=np.float32)
        parents = _object_parents(np.asarray(arrays["parents"], dtype=object))
        return _validate_rig(
            RigRecord(
                vertices=vertices,
                joints=joints,
                parents=parents,
                joint_ids=np.arange(len(joints), dtype=np.int64),
            )
        )

    if rig_format != HUMANOID_RESAVE_FORMAT:
        raise ValueError(f"Unsupported rig_format={rig_format!r}")
    if schema is None:
        raise ValueError(f"{HUMANOID_RESAVE_FORMAT} requires a joint schema")

    semantics = arrays["joint_semantics"].tolist()
    source_joints = np.asarray(arrays["joint_positions"], dtype=np.float32)
    source_parents = np.asarray(arrays["parents"], dtype=np.int64)
    if joint_selection == "available":
        joints, joint_ids = schema.select_available(
            semantics, source_joints, source_parents
        )
    elif joint_selection == "strict":
        joints = schema.select(semantics, source_joints, source_parents)
        joint_ids = np.arange(len(schema.joints), dtype=np.int64)
    else:
        raise ValueError(f"Unsupported joint_selection={joint_selection!r}")

    return _validate_rig(
        RigRecord(
            vertices=vertices,
            joints=joints,
            parents=schema.parents_for_ids(joint_ids),
            joint_ids=joint_ids,
        )
    )


__all__ = [
    "ANIGEN_VOXELIZED_FORMAT",
    "HUMANOID_RESAVE_FORMAT",
    "RigRecord",
    "load_rig_record",
]
