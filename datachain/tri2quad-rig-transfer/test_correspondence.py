import numpy as np
import pytest

from contract import Rejection
from correspondence import transfer_rig
from rig import RiggedMesh


def rig(weights=None):
    vertices = np.asarray(
        [[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3]], dtype=np.float64
    )
    values = np.ones((4, 1), dtype=np.float64) if weights is None else weights
    return RiggedMesh(
        vertices=vertices,
        triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        weight_joint_nodes=np.zeros((4, 1), dtype=np.int32),
        weight_values=values,
        joint_nodes=np.asarray([0], dtype=np.int32),
        joint_names=np.asarray(["root"]),
        joint_positions=np.asarray([[0, 0, 0]], dtype=np.float64),
        joint_parents=np.asarray([-1], dtype=np.int32),
    )


def test_different_vertex_count_is_rejected():
    source = rig()
    with pytest.raises(Rejection, match="canonical source has 4 vertices; final has 3") as caught:
        transfer_rig(
            source,
            source.vertices[:3],
            np.asarray([[0, 1, 2]]),
            np.empty((0, 4), dtype=np.int32),
            np.eye(4),
        )
    assert caught.value.code == "vertex_count_mismatch"


def test_normalized_transfer_only_reorders_shared_scene_coordinates():
    source = rig()
    order = np.asarray([2, 0, 3, 1])
    scene_transform = np.asarray(
        [
            [0.5, 0.0, 0.0, -0.25],
            [0.0, 0.5, 0.0, 0.125],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transformed_vertices = (
        source.vertices @ scene_transform[:3, :3].T + scene_transform[:3, 3]
    )
    transformed_joint = (
        source.joint_positions @ scene_transform[:3, :3].T
        + scene_transform[:3, 3]
    )
    source.vertices = transformed_vertices
    source.joint_positions = transformed_joint

    result = transfer_rig(
        source,
        transformed_vertices[order],
        np.asarray([[0, 1, 2], [0, 2, 3]]),
        np.empty((0, 4), dtype=np.int32),
        scene_transform,
    )

    np.testing.assert_allclose(result.final_vertices, transformed_vertices[order])
    np.testing.assert_allclose(result.joint_positions, transformed_joint)
    np.testing.assert_allclose(result.source_to_final, scene_transform)
    assert result.max_coordinate_error == pytest.approx(0.0)
