from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Vector
from vrenderer.blender_utils import (
    compute_objects_bbox_np,
    object_context,
    scene_mesh_objects,
    scene_root_objects,
)

from contract import Rejection


def _marker(name: str, location: np.ndarray):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "PLAIN_AXES"
    obj.location = Vector(np.asarray(location, dtype=np.float64))
    bpy.context.scene.collection.objects.link(obj)
    return obj


def normalize_with_joint_markers(
    joint_positions: np.ndarray,
    rotation_euler: tuple[float, float, float] | None,
):
    """Run the VRenderer 2.5.1 normalization with joints on the same root.

    This is a local copy of the producer's `normalize` path. The only added
    behavior is creating non-geometry joint/probe empties and reading their
    world positions before the producer removes its normalization root.
    """
    joint_markers = [
        _marker(f"__rig_joint_{index:06d}", point)
        for index, point in enumerate(joint_positions)
    ]
    probe_markers = [
        _marker(f"__rig_probe_{index}", point)
        for index, point in enumerate(np.vstack([np.zeros(3), np.eye(3)]))
    ]

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_set(False)
            obj.hide_viewport = False
    bpy.context.view_layer.update()

    mesh_objs = list(scene_mesh_objects())
    if not mesh_objs:
        raise Rejection("empty_source_mesh", "normalization found no visible meshes")
    bbox_min, bbox_max = compute_objects_bbox_np("bound_box", mesh_objs)
    span = float((bbox_max - bbox_min).max())
    if span <= 0:
        raise Rejection("degenerate_geometry", "normalization bbox has zero extent")
    center = (bbox_min + bbox_max) / 2.0
    scaling_factor = 1.0 / span

    bpy.ops.object.empty_add(type="PLAIN_AXES")
    root_object = bpy.context.object
    root_object.name = "__rig_normalization_root"
    for obj in list(scene_root_objects()):
        if obj != root_object:
            matrix_world = obj.matrix_world.copy()
            obj.parent = root_object
            obj.matrix_world = matrix_world

    root_object.location = Vector(-center)
    with object_context(root_object):
        bpy.ops.object.transform_apply(location=True)

    if rotation_euler is not None:
        root_object.rotation_mode = "XYZ"
        root_object.rotation_euler = [math.radians(value) for value in rotation_euler]
    root_object.scale = (scaling_factor,) * 3
    bpy.context.view_layer.update()

    bbox_min, bbox_max = compute_objects_bbox_np("bound_box", scene_mesh_objects())
    post_center = (bbox_min + bbox_max) / 2.0
    if np.linalg.norm(post_center) > 1e-4:
        root_object.location = -Vector(post_center)
        bpy.context.view_layer.update()
        bbox_min, bbox_max = compute_objects_bbox_np("bound_box", scene_mesh_objects())

    transformed_joints = np.asarray(
        [np.asarray(marker.matrix_world.translation) for marker in joint_markers],
        dtype=np.float64,
    )
    probes = np.asarray(
        [np.asarray(marker.matrix_world.translation) for marker in probe_markers],
        dtype=np.float64,
    )
    source_to_normalized = np.eye(4, dtype=np.float64)
    source_to_normalized[:3, 3] = probes[0]
    source_to_normalized[:3, :3] = (probes[1:] - probes[0]).T

    for obj in list(bpy.context.scene.objects):
        if obj.type == "MESH" and obj.parent is not None:
            matrix_world = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = matrix_world
    bpy.data.objects.remove(root_object, do_unlink=True)
    bpy.context.view_layer.update()
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            with object_context(obj):
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.context.view_layer.update()

    return transformed_joints, source_to_normalized, bbox_min, bbox_max
