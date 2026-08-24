from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from contract import MAX_COORDINATE_ERROR, Rejection, WEIGHT_ATOL, structured_keys
from rig import RiggedMesh


@dataclass
class Transfer:
    source_to_final: np.ndarray
    final_vertices: np.ndarray
    final_triangles: np.ndarray
    final_quads: np.ndarray
    joint_positions: np.ndarray
    joint_names: np.ndarray
    joint_parents: np.ndarray
    weight_indices: np.ndarray
    weight_values: np.ndarray
    max_coordinate_error: float


def _canonical_source(
    rig: RiggedMesh,
    merge_center: np.ndarray,
    merge_scale: float,
):
    # Tri2Quad merges after normalizing to [-1, 1]. These coordinates are
    # used only as merge keys. Geometry remains in the Blender-produced
    # space shared by the mesh and joint markers.
    merge_coordinates = (rig.vertices - merge_center) * merge_scale
    keys = structured_keys(merge_coordinates)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap_groups = np.empty(len(order), dtype=np.int64)
    remap_groups[order] = np.arange(len(order))
    inverse = remap_groups[inverse]
    first = np.sort(first)
    merged_source_vertices = rig.vertices[first]
    merged_faces = inverse[rig.triangles]

    # Build skinning by canonical position, rejecting seam duplicates that disagree.
    node_to_joint = {int(node): index for index, node in enumerate(rig.joint_nodes)}
    rows = []
    for source_index in range(len(rig.vertices)):
        row = {
            node_to_joint[int(node)]: float(value)
            for node, value in zip(rig.weight_joint_nodes[source_index], rig.weight_values[source_index])
            if value > 0
        }
        rows.append(row)
    grouped = [[] for _ in range(len(merged_source_vertices))]
    for source_index, canonical_index in enumerate(inverse):
        grouped[int(canonical_index)].append(rows[source_index])
    canonical_weights = []
    for duplicates in grouped:
        reference = duplicates[0]
        for candidate in duplicates[1:]:
            joints = set(reference) | set(candidate)
            if any(abs(reference.get(j, 0.0) - candidate.get(j, 0.0)) > WEIGHT_ATOL for j in joints):
                raise Rejection("conflicting_duplicate_weights", "merged source vertices have different skin weights")
        canonical_weights.append(reference)

    merged_keys = merge_coordinates[first]
    a = merged_keys[merged_faces[:, 0]]
    b = merged_keys[merged_faces[:, 1]]
    c = merged_keys[merged_faces[:, 2]]
    edge = np.maximum.reduce(
        [np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1)]
    )
    double_area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    height = np.divide(double_area, edge, out=np.zeros_like(double_area), where=edge > 0)
    nondegenerate = height > 1e-8
    sorted_faces = np.sort(merged_faces, axis=1)
    _, unique_face_indices = np.unique(sorted_faces, axis=0, return_index=True)
    unique_faces = np.zeros(len(merged_faces), dtype=bool)
    unique_faces[unique_face_indices] = True
    faces = merged_faces[nondegenerate & unique_faces]
    referenced = np.unique(faces.reshape(-1))
    old_to_new = np.full(len(merged_source_vertices), -1, dtype=np.int64)
    old_to_new[referenced] = np.arange(len(referenced))
    faces = old_to_new[faces]
    return (
        merged_source_vertices[referenced],
        faces,
        [canonical_weights[index] for index in referenced],
    )


def transfer_rig(
    rig: RiggedMesh,
    final_vertices: np.ndarray,
    final_triangles: np.ndarray,
    final_quads: np.ndarray,
    source_to_normalized: np.ndarray,
) -> Transfer:
    final_vertices = np.asarray(final_vertices, dtype=np.float64)
    source_center = 0.5 * (rig.vertices.min(axis=0) + rig.vertices.max(axis=0))
    source_extent = float(np.ptp(rig.vertices, axis=0).max())
    if source_extent <= 0:
        raise Rejection("degenerate_geometry", "normalized source has zero extent")
    source_vertices, _, weights = _canonical_source(
        rig, source_center, 2.0 / source_extent
    )
    if len(source_vertices) != len(final_vertices):
        raise Rejection(
            "vertex_count_mismatch",
            f"canonical source has {len(source_vertices)} vertices; final has {len(final_vertices)}",
        )
    target_tree = cKDTree(final_vertices)
    _, target_indices = target_tree.query(source_vertices)
    if len(np.unique(target_indices)) != len(final_vertices):
        raise Rejection("non_bijective_vertices", "normalized source does not biject to final vertices")
    error = np.abs(source_vertices - final_vertices[target_indices])
    maximum = float(error.max(initial=0.0))
    if maximum > MAX_COORDINATE_ERROR:
        raise Rejection("vertex_mismatch", f"normalized-to-final residual {maximum:.9g}")
    source_for_target = np.empty(len(target_indices), dtype=np.int64)
    source_for_target[target_indices] = np.arange(len(target_indices))
    ordered_weights = [weights[index] for index in source_for_target]
    width = max((len(row) for row in ordered_weights), default=0)
    weight_indices = np.full((len(final_vertices), width), -1, dtype=np.int32)
    weight_values = np.zeros((len(final_vertices), width), dtype=np.float32)
    for vertex, row in enumerate(ordered_weights):
        for slot, (joint, value) in enumerate(
            sorted(row.items(), key=lambda item: (-item[1], item[0]))
        ):
            weight_indices[vertex, slot] = joint
            weight_values[vertex, slot] = value
    return Transfer(
        source_to_final=np.asarray(source_to_normalized, dtype=np.float64),
        final_vertices=np.asarray(final_vertices, dtype=np.float32),
        final_triangles=np.asarray(final_triangles, dtype=np.int32),
        final_quads=np.asarray(final_quads, dtype=np.int32),
        joint_positions=np.asarray(rig.joint_positions, dtype=np.float32),
        joint_names=rig.joint_names,
        joint_parents=rig.joint_parents,
        weight_indices=weight_indices,
        weight_values=weight_values,
        max_coordinate_error=maximum,
    )
