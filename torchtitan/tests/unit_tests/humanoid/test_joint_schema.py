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
