import json

import numpy as np

from torchtitan.experiments.humanoid.data.joint_schema import JointSchema


def test_schema_selects_spine_in_chain_order(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps({"name": "test", "joints": ["root", "spine_0", "spine_1"], "parents": [-1, 0, 1]})
    )
    schema = JointSchema.load(path)
    positions = np.asarray([[0, 2, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32)
    selected = schema.select(["spine", "root", "spine"], positions, np.asarray([2, -1, 1]))
    np.testing.assert_array_equal(selected[:, 1], [0, 1, 2])


def test_schema_selects_available_joints_with_stable_ids(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps(
            {
                "name": "test",
                "joints": ["root", "spine_0", "spine_1", "head", "left_eye"],
                "parents": [-1, 0, 1, 2, 3],
            }
        )
    )
    schema = JointSchema.load(path)
    positions = np.asarray(
        [[0, 3, 0], [0, 1, 0], [0, 0, 0], [0, 2, 0]], dtype=np.float32
    )

    selected, joint_ids = schema.select_available(
        ["head", "spine", "root", "spine"],
        positions,
        np.asarray([3, 2, -1, 1]),
    )

    np.testing.assert_array_equal(joint_ids, [0, 1, 2, 3])
    np.testing.assert_array_equal(selected[:, 1], [0, 1, 2, 3])


def test_schema_omits_ambiguous_optional_joint(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps(
            {
                "name": "test",
                "joints": ["root", "head"],
                "parents": [-1, 0],
            }
        )
    )
    schema = JointSchema.load(path)
    positions = np.asarray([[0, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=np.float32)

    selected, joint_ids = schema.select_available(
        ["root", "head", "head"], positions, np.asarray([-1, 0, 0])
    )

    np.testing.assert_array_equal(joint_ids, [0])
    np.testing.assert_array_equal(selected, positions[:1])


def test_schema_remaps_parents_across_missing_ancestors(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps(
            {
                "name": "test",
                "joints": ["root", "spine", "chest", "hand"],
                "parents": [-1, 0, 1, 2],
            }
        )
    )
    schema = JointSchema.load(path)

    np.testing.assert_array_equal(schema.parents_for_ids([0, 2, 3]), [-1, 0, 1])
