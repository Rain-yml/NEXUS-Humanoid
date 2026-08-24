from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
from vrenderer.blender_utils import bake_scene, clear_scene

from contract import Rejection
from producer_normalize import normalize_with_joint_markers
from rig import RiggedMesh


@dataclass
class NormalizedRig:
    rig: RiggedMesh
    source_to_normalized: np.ndarray
    producer_mode: str


def _world_points(obj) -> np.ndarray:
    coordinates = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", coordinates)
    coordinates = coordinates.reshape(-1, 3)
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    return coordinates @ matrix[:3, :3].T + matrix[:3, 3]


def _mesh_objects():
    return sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "MESH"),
        key=lambda obj: obj.name,
    )


def _capture_skeleton():
    armatures = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"),
        key=lambda obj: obj.name,
    )
    joint_keys = []
    names = []
    positions = []
    parent_keys = []
    for armature in armatures:
        matrix = np.asarray(armature.matrix_world, dtype=np.float64)
        armature.data.pose_position = "REST"
        for bone in armature.data.bones:
            key = (armature.name, bone.name)
            joint_keys.append(key)
            names.append(bone.name)
            head = np.asarray(bone.head_local, dtype=np.float64)
            positions.append(head @ matrix[:3, :3].T + matrix[:3, 3])
            parent_keys.append(
                (armature.name, bone.parent.name) if bone.parent is not None else None
            )
    if not joint_keys:
        raise Rejection("no_rig", "Blender import contains no armature bones")
    index = {key: slot for slot, key in enumerate(joint_keys)}
    parents = np.asarray([index.get(key, -1) for key in parent_keys], dtype=np.int32)
    return index, np.asarray(names, dtype=np.str_), np.asarray(positions), parents


def _mesh_group_maps(joint_index):
    result = {}
    for obj in _mesh_objects():
        armatures = {
            modifier.object.name
            for modifier in obj.modifiers
            if modifier.type == "ARMATURE" and modifier.object is not None
        }
        if not armatures:
            continue
        if len(armatures) != 1:
            modifiers = [
                f"{modifier.name}:{modifier.type}:{getattr(modifier.object, 'name', None)}"
                for modifier in obj.modifiers
            ]
            raise Rejection(
                "ambiguous_mesh_armature",
                f"mesh {obj.name!r} vertices={len(obj.data.vertices)} "
                f"faces={len(obj.data.polygons)} references {len(armatures)} "
                f"armatures; modifiers={modifiers}",
            )
        armature = next(iter(armatures))
        mapping = {}
        for group in obj.vertex_groups:
            joint = joint_index.get((armature, group.name))
            if joint is not None:
                mapping[group.index] = joint
        if not mapping:
            raise Rejection("unskinned_geometry", f"mesh {obj.name!r} has no bone groups")
        result[obj.name] = mapping
    return result


def _weights(obj, group_map):
    rows = []
    width = 0
    for vertex in obj.data.vertices:
        row = [
            (group_map[membership.group], float(membership.weight))
            for membership in vertex.groups
            if membership.group in group_map and membership.weight > 0
        ]
        row.sort(key=lambda item: (-item[1], item[0]))
        if not row:
            raise Rejection("unweighted_vertex", f"mesh {obj.name!r} has an unweighted vertex")
        total = sum(value for _, value in row)
        rows.append([(joint, value / total) for joint, value in row])
        width = max(width, len(row))
    indices = np.full((len(rows), width), -1, dtype=np.int32)
    values = np.zeros((len(rows), width), dtype=np.float64)
    for vertex, row in enumerate(rows):
        for slot, (joint, value) in enumerate(row):
            indices[vertex, slot] = joint
            values[vertex, slot] = value
    return indices, values


def _geometry(group_maps):
    vertices = []
    triangles = []
    joint_indices = []
    weight_values = []
    offset = 0
    objects = _mesh_objects()
    missing = sorted(set(group_maps) - {obj.name for obj in objects})
    if missing:
        raise Rejection("topology_changed", f"bake removed weighted meshes: {missing[:4]}")
    for obj in objects:
        if obj.name not in group_maps:
            raise Rejection("topology_changed", f"unexpected baked mesh {obj.name!r}")
        points = _world_points(obj)
        indices, values = _weights(obj, group_maps[obj.name])
        object_triangles = []
        for polygon in obj.data.polygons:
            face = list(polygon.vertices)
            if len(face) != 3:
                raise Rejection("non_triangle_source", f"mesh {obj.name!r} contains non-triangles after bake")
            object_triangles.append(face)
        if not object_triangles:
            raise Rejection("empty_source_mesh", f"mesh {obj.name!r} has no faces")
        vertices.append(points)
        triangles.append(np.asarray(object_triangles, dtype=np.int64) + offset)
        joint_indices.append(indices)
        weight_values.append(values)
        offset += len(points)
    max_width = max(chunk.shape[1] for chunk in joint_indices)
    padded_indices = []
    padded_values = []
    for indices, values in zip(joint_indices, weight_values):
        pad = max_width - indices.shape[1]
        padded_indices.append(np.pad(indices, ((0, 0), (0, pad)), constant_values=-1))
        padded_values.append(np.pad(values, ((0, 0), (0, pad))))
    return (
        np.concatenate(vertices),
        np.concatenate(triangles),
        np.concatenate(padded_indices),
        np.concatenate(padded_values),
    )


def normalize_source(payload: bytes, *, plus_y_front: bool) -> NormalizedRig:
    mode = "plus_y_front" if plus_y_front else "native_front"
    cleanup_scene()
    try:
        with tempfile.TemporaryDirectory(prefix="tri2quad-rig-source-") as directory:
            path = Path(directory) / "source.glb"
            path.write_bytes(payload)
            bpy.ops.import_scene.gltf(filepath=str(path), merge_vertices=True)
        joint_index, names, source_joints, parents = _capture_skeleton()
        bpy.context.view_layer.update()
        group_maps = _mesh_group_maps(joint_index)

        bake_scene()
        normalized_joints, source_to_normalized, _, _ = normalize_with_joint_markers(
            source_joints,
            (0, 0, 180) if plus_y_front else None,
        )
        post_vertices, triangles, weight_nodes, weight_values = _geometry(group_maps)
        joint_nodes = np.arange(len(names), dtype=np.int32)
        return NormalizedRig(
            rig=RiggedMesh(
                vertices=post_vertices,
                triangles=triangles,
                weight_joint_nodes=weight_nodes,
                weight_values=weight_values,
                joint_nodes=joint_nodes,
                joint_names=names,
                joint_positions=normalized_joints,
                joint_parents=parents,
            ),
            source_to_normalized=source_to_normalized,
            producer_mode=mode,
        )
    finally:
        cleanup_scene()
def cleanup_scene():
    try:
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass
    clear_scene()
    try:
        bpy.ops.outliner.orphans_purge(do_recursive=True)
        bpy.ops.wm.read_factory_settings(use_empty=True)
    except RuntimeError:
        pass
