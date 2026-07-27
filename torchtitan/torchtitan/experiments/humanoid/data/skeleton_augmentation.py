"""Shared spatial augmentation for reference skeletons."""

from __future__ import annotations

import torch


def _validate_skeleton(joints: torch.Tensor, parents: torch.Tensor) -> None:
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError(f"Expected joints shaped (J, 3), got {tuple(joints.shape)}")
    if parents.shape != (len(joints),):
        raise ValueError(
            f"Expected parents shaped ({len(joints)},), got {tuple(parents.shape)}"
        )

    parents = parents.to(dtype=torch.long)
    pending = set(range(len(parents)))
    visited: set[int] = set()
    while pending:
        ready = [
            index
            for index in pending
            if int(parents[index]) < 0 or int(parents[index]) in visited
        ]
        if not ready:
            raise ValueError("Skeleton parents do not form an acyclic forest")
        visited.update(ready)
        pending.difference_update(ready)


def _stretch_axis(
    values: torch.Tensor,
    *,
    global_scale_log_std: float,
    local_scale_log_std: float,
    num_segments: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    lower = values.min()
    upper = values.max()
    extent = upper - lower
    if extent <= 1e-8:
        return values.clone()

    global_log_scale = torch.randn(
        (), dtype=values.dtype, device=values.device, generator=generator
    ) * global_scale_log_std
    global_scale = global_log_scale.clamp(-0.25, 0.25).exp()

    segment_log_scales = torch.randn(
        num_segments,
        dtype=values.dtype,
        device=values.device,
        generator=generator,
    ) * local_scale_log_std
    segment_scales = segment_log_scales.clamp(-0.25, 0.25).exp()
    segment_widths = segment_scales / segment_scales.sum()
    boundaries = torch.cat(
        [
            torch.zeros(1, dtype=values.dtype, device=values.device),
            segment_widths.cumsum(dim=0),
        ]
    )

    normalized = ((values - lower) / extent).clamp(0.0, 1.0)
    segment_position = normalized * num_segments
    segment_index = segment_position.floor().long().clamp(max=num_segments - 1)
    fraction = segment_position - segment_index.to(values.dtype)
    warped = boundaries[segment_index] + (
        boundaries[segment_index + 1] - boundaries[segment_index]
    ) * fraction

    midpoint = (lower + upper) * 0.5
    warped_values = lower + warped * extent
    return midpoint + (warped_values - midpoint) * global_scale


def augment_reference_skeleton(
    joints: torch.Tensor,
    parents: torch.Tensor,
    *,
    global_scale_log_std: float,
    local_scale_log_std: float,
    num_segments: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply one monotonic, axis-aligned spatial warp to every joint.

    The deformation changes proportions without articulating individual bones:
    joints that share a coordinate receive the same transformed coordinate, and
    coordinate ordering is preserved along every axis.
    """
    _validate_skeleton(joints, parents)
    if global_scale_log_std < 0 or local_scale_log_std < 0:
        raise ValueError("Augmentation standard deviations must be non-negative")
    if num_segments < 1:
        raise ValueError("num_segments must be positive")

    joints = joints.to(dtype=torch.float32)
    if global_scale_log_std == 0 and local_scale_log_std == 0:
        return joints.clone()

    return torch.stack(
        [
            _stretch_axis(
                joints[:, axis],
                global_scale_log_std=global_scale_log_std,
                local_scale_log_std=local_scale_log_std,
                num_segments=num_segments,
                generator=generator,
            )
            for axis in range(3)
        ],
        dim=-1,
    )


def skeleton_edges(
    parents: torch.Tensor, *, bidirectional: bool = True
) -> torch.Tensor:
    """Return packed graph edges in source-target format."""
    children = torch.arange(len(parents), dtype=torch.long)
    valid = parents >= 0
    edges = torch.stack([children[valid], parents[valid].long()])
    if bidirectional and edges.shape[1] > 0:
        edges = torch.cat([edges, edges.flip(0)], dim=1)
    return edges


__all__ = ["augment_reference_skeleton", "skeleton_edges"]
