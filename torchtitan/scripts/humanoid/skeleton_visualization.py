"""Prediction-only mesh and skeleton visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


_VIEW_SPECS = {
    "front": ((0, 1.0), (2, -1.0), (1, 1.0)),
    "right": ((1, -1.0), (2, -1.0), (0, 1.0)),
    "back": ((0, -1.0), (2, -1.0), (1, -1.0)),
    "left": ((1, 1.0), (2, -1.0), (0, -1.0)),
}


def _mesh_frame(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    extent = float((maximum - minimum).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError(f"Invalid mesh extent: {extent}")
    return center, extent


def mesh_space_from_nexus(
    points: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """Invert NEXUS bbox normalization for points belonging to ``vertices``."""
    center, extent = _mesh_frame(vertices)
    return points.astype(np.float32, copy=False) * (0.5 * extent) + center


def render_mesh_skeleton_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    parents: list[int],
    path: Path,
    *,
    view: str,
    size: int = 700,
) -> None:
    """Render one orthographic mesh view with the predicted skeleton overlaid."""
    if view not in _VIEW_SPECS:
        raise ValueError(f"Unknown view {view!r}; expected one of {tuple(_VIEW_SPECS)}")

    image = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    center, extent = _mesh_frame(vertices)
    verts = (vertices.astype(np.float32, copy=False) - center) / extent
    normalized_joints = (joints.astype(np.float32, copy=False) - center) / extent

    (u_axis, u_sign), (v_axis, v_sign), (d_axis, d_sign) = _VIEW_SPECS[view]

    def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.stack(
            [points[:, u_axis] * u_sign, points[:, v_axis] * v_sign], axis=1
        )
        xy = xy * (size * 0.82) + size * 0.5
        return xy, points[:, d_axis] * d_sign

    vertex_xy, vertex_depth = project(verts)
    valid_faces = faces.astype(np.int64, copy=False)
    valid = (valid_faces >= 0).all(axis=1) & (valid_faces < len(vertices)).all(axis=1)
    valid_faces = valid_faces[valid]
    order = np.argsort(vertex_depth[valid_faces].mean(axis=1))
    for face in valid_faces[order]:
        depth = float(vertex_depth[face].mean())
        shade = int(np.clip(180 + depth * 70, 125, 215))
        color = (shade - 45, shade - 20, shade)
        points = [tuple(float(value) for value in point) for point in vertex_xy[face]]
        draw.polygon(points, fill=color)

    joint_xy, _ = project(normalized_joints)
    for joint_id, parent_id in enumerate(parents):
        if parent_id < 0:
            continue
        start = tuple(float(value) for value in joint_xy[parent_id])
        end = tuple(float(value) for value in joint_xy[joint_id])
        draw.line([start, end], fill=(245, 183, 26), width=5)
    for joint_id, point in enumerate(joint_xy):
        x, y = (float(value) for value in point)
        radius = 5 if joint_id else 7
        fill = (220, 45, 45) if joint_id == 0 else (35, 205, 105)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline=(25, 25, 25),
            width=2,
        )
    image.save(path)


def render_prediction_multiview(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    parents: list[int],
    output_dir: Path,
) -> tuple[Path, dict[str, Path]]:
    """Render front/right/back/left views into one per-sample sheet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for view in _VIEW_SPECS:
        path = output_dir / f"prediction_{view}.png"
        render_mesh_skeleton_view(
            vertices, faces, joints, parents, path, view=view
        )
        paths[view] = path

    tile_size = 700
    label_height = 34
    sheet = Image.new(
        "RGB", (tile_size * 2, (tile_size + label_height) * 2), (255, 255, 255)
    )
    draw = ImageDraw.Draw(sheet)
    for index, view in enumerate(_VIEW_SPECS):
        col, row = index % 2, index // 2
        x = col * tile_size
        y = row * (tile_size + label_height)
        draw.text((x + 14, y + 9), view, fill=(0, 0, 0))
        sheet.paste(Image.open(paths[view]).convert("RGB"), (x, y + label_height))
    sheet_path = output_dir / "prediction_multiview.png"
    sheet.save(sheet_path)
    return sheet_path, paths


def export_mesh_skeleton_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    parents: list[int],
    path: Path,
) -> None:
    """Export a native-coordinate mesh and predicted skeleton for 3D inspection."""
    _, extent = _mesh_frame(vertices)
    scene = trimesh.Scene()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = np.array([120, 150, 205, 150], dtype=np.uint8)
    scene.add_geometry(mesh, node_name="mesh")

    joint_radius = extent * 0.006
    bone_radius = extent * 0.0025
    for joint_id, position in enumerate(joints):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=joint_radius)
        sphere.apply_translation(position)
        color = [220, 45, 45, 255] if joint_id == 0 else [35, 205, 105, 255]
        sphere.visual.vertex_colors = np.asarray(color, dtype=np.uint8)
        scene.add_geometry(sphere, node_name=f"joint_{joint_id:02d}")

    for joint_id, parent_id in enumerate(parents):
        if parent_id < 0:
            continue
        start = joints[parent_id]
        end = joints[joint_id]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            continue
        transform = trimesh.geometry.align_vectors(
            np.array([0.0, 0.0, 1.0]), direction / length
        )
        transform[:3, 3] = 0.5 * (start + end)
        bone = trimesh.creation.cylinder(
            radius=bone_radius, height=length, sections=12, transform=transform
        )
        bone.visual.vertex_colors = np.asarray([245, 183, 26, 255], dtype=np.uint8)
        scene.add_geometry(
            bone, node_name=f"bone_{parent_id:02d}_{joint_id:02d}"
        )
    scene.export(path)
