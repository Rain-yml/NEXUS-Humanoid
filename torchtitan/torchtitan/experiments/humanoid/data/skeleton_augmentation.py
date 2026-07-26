"""Topology-preserving reference-skeleton augmentation."""

from __future__ import annotations

import math

import torch


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm().clamp_min(1e-8)
    x, y, z = axis.unbind()
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)
    skew = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return identity + angle.sin() * skew + (1.0 - angle.cos()) * (skew @ skew)


def _topological_order(parents: torch.Tensor) -> list[int]:
    pending = set(range(len(parents)))
    order: list[int] = []
    while pending:
        ready = [
            index
            for index in pending
            if int(parents[index]) < 0 or int(parents[index]) in order
        ]
        if not ready:
            raise ValueError("Skeleton parents do not form an acyclic forest")
        ready.sort()
        order.extend(ready)
        pending.difference_update(ready)
    return order


def augment_reference_skeleton(
    joints: torch.Tensor,
    parents: torch.Tensor,
    *,
    max_local_rotation_degrees: float,
    bone_length_log_std: float,
    root_translation_std: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Distort a skeleton through forward kinematics, never pointwise noise.

    Parent-child topology is unchanged. Local rotations accumulate down each
    tree, and each bone receives a bounded multiplicative length change.
    """
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError(f"Expected joints shaped (J, 3), got {tuple(joints.shape)}")
    if parents.shape != (len(joints),):
        raise ValueError(f"Expected parents shaped ({len(joints)},), got {tuple(parents.shape)}")
    if max_local_rotation_degrees < 0:
        raise ValueError("max_local_rotation_degrees must be non-negative")
    if bone_length_log_std < 0 or root_translation_std < 0:
        raise ValueError("Augmentation standard deviations must be non-negative")

    joints = joints.to(dtype=torch.float32)
    parents = parents.to(dtype=torch.long)
    order = _topological_order(parents)
    result = torch.empty_like(joints)
    rotations = torch.empty(
        len(joints), 3, 3, dtype=joints.dtype, device=joints.device
    )
    max_angle = math.radians(max_local_rotation_degrees)

    for joint_index in order:
        axis = torch.randn(
            3, dtype=joints.dtype, device=joints.device, generator=generator
        )
        angle = (
            torch.rand(
                (), dtype=joints.dtype, device=joints.device, generator=generator
            )
            * 2.0
            - 1.0
        ) * max_angle
        local_rotation = _axis_angle_matrix(axis, angle)
        parent = int(parents[joint_index])
        if parent < 0:
            rotations[joint_index] = local_rotation
            translation = torch.randn(
                3, dtype=joints.dtype, device=joints.device, generator=generator
            ) * root_translation_std
            result[joint_index] = joints[joint_index] + translation
            continue

        rotations[joint_index] = rotations[parent] @ local_rotation
        bone = joints[joint_index] - joints[parent]
        log_scale = torch.randn(
            (), dtype=joints.dtype, device=joints.device, generator=generator
        ) * bone_length_log_std
        scale = log_scale.clamp(-0.25, 0.25).exp()
        result[joint_index] = result[parent] + rotations[parent] @ (bone * scale)

    return result


def skeleton_edges(parents: torch.Tensor, *, bidirectional: bool = True) -> torch.Tensor:
    """Return packed graph edges in source-target format."""
    children = torch.arange(len(parents), dtype=torch.long)
    valid = parents >= 0
    edges = torch.stack([children[valid], parents[valid].long()])
    if bidirectional and edges.shape[1] > 0:
        edges = torch.cat([edges, edges.flip(0)], dim=1)
    return edges


__all__ = ["augment_reference_skeleton", "skeleton_edges"]
