from __future__ import annotations

import os
import re
from typing import Iterable

import numpy as np
import meshio
import torch
from PIL import Image


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)

    scale = (vmax - vmin).max()
    center = (vmax + vmin) / 2.0

    return (vertices - center) / scale

_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z._-]+")


def resolve_rollout_log_dir(dump_folder: str) -> str:
    return os.path.join(os.path.expanduser(dump_folder), "rollout_meshes")


def sanitize_path_component(value: str) -> str:
    value = value.replace(os.sep, "_")
    if os.altsep:
        value = value.replace(os.altsep, "_")
    value = value.replace("#", "_")
    value = _SAFE_NAME_RE.sub("_", value)
    value = value.strip("._")
    return value or "mesh"


def _validate_faces(faces: np.ndarray, expected_width: int, face_kind: str) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return faces.reshape(0, expected_width)
    if faces.ndim != 2 or faces.shape[1] != expected_width:
        raise ValueError(f"{face_kind} faces must have shape (N, {expected_width}), got {faces.shape}")
    return faces


def write_mixed_obj(
    obj_path: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    quads: np.ndarray,
) -> None:
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError(f"Vertices must have shape (N, >=3), got {vertices.shape}")

    os.makedirs(os.path.dirname(obj_path), exist_ok=True)
    cells = []
    if triangles.size > 0:
        cells.append(("triangle", triangles))
    if quads.size > 0:
        cells.append(("quad", quads))
    meshio.write_points_cells(obj_path, vertices, cells)


def write_image(image_path: str, image: torch.Tensor | np.ndarray) -> None:
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu().numpy()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    image = (image.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    Image.fromarray(image).save(image_path)


def save_rollout_meshes(
    rollout_log_dir: str,
    step: int,
    rank: int,
    batch_idx: int,
    meshes: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    instance_ids: Iterable[str],
    input_images: Iterable[torch.Tensor | np.ndarray] | None = None,
    input_instance_ids: Iterable[str] | None = None,
) -> None:
    meshes = list(meshes)
    instance_ids = list(instance_ids)
    if len(meshes) != len(instance_ids):
        raise ValueError(
            "Mesh and instance_id counts must match: "
            f"{len(meshes)} != {len(instance_ids)}"
        )

    batch_dir = os.path.join(
        rollout_log_dir,
        f"step-{step:07d}",
        f"rank-{rank:02d}",
        f"batch-{batch_idx:03d}",
    )
    os.makedirs(batch_dir, exist_ok=True)

    if input_images is not None:
        input_images = list(input_images)
        if input_instance_ids is None:
            input_instance_ids = [str(i) for i in range(len(input_images))]
        input_instance_ids = list(input_instance_ids)
        if len(input_images) != len(input_instance_ids):
            raise ValueError(
                "Input image and input instance_id counts must match: "
                f"{len(input_images)} != {len(input_instance_ids)}"
            )
        for image_idx, (image, instance_id) in enumerate(zip(input_images, input_instance_ids)):
            file_name = (
                f"input-{image_idx:03d}-"
                f"{sanitize_path_component(str(instance_id))}.png"
            )
            write_image(os.path.join(batch_dir, file_name), image)

    for mesh_idx, ((vertices, triangles, quads), instance_id) in enumerate(
        zip(meshes, instance_ids)
    ):
        file_name = (
            f"sample-{mesh_idx:03d}-"
            f"{sanitize_path_component(str(instance_id))}.obj"
        )
        vertices = normalize_vertices(vertices)
        # rotate along x clockwise 90 degrees
        vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1]
        write_mixed_obj(
            os.path.join(batch_dir, file_name),
            vertices,
            triangles,
            quads,
        )
