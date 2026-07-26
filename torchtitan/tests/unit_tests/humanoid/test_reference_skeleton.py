import numpy as np
import torch

from torchtitan.experiments.humanoid.data.rig_records import (
    ANIGEN_VOXELIZED_FORMAT,
    load_rig_record,
)
from torchtitan.experiments.humanoid.data.skeleton_augmentation import (
    augment_reference_skeleton,
    skeleton_edges,
)


def test_anigen_rig_record_preserves_arbitrary_joint_order_and_parents():
    record = load_rig_record(
        {
            "vertices": np.asarray(
                [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32
            ),
            "joints": np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.2, 0.8, 0.0]],
                dtype=np.float32,
            ),
            "parents": np.asarray([None, 0, 1], dtype=object),
        },
        rig_format=ANIGEN_VOXELIZED_FORMAT,
        schema=None,
        joint_selection="strict",
    )

    np.testing.assert_array_equal(record.parents, [-1, 0, 1])
    np.testing.assert_array_equal(record.joint_ids, [0, 1, 2])


def test_reference_augmentation_is_fk_not_point_noise():
    joints = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.4, 0.5, 0.0],
        ]
    )
    parents = torch.tensor([-1, 0, 1, 1])
    augmented = augment_reference_skeleton(
        joints,
        parents,
        max_local_rotation_degrees=20.0,
        bone_length_log_std=0.08,
        root_translation_std=0.03,
        generator=torch.Generator().manual_seed(7),
    )

    assert not torch.equal(augmented, joints)
    original_lengths = (joints[1:] - joints[parents[1:]]).norm(dim=-1)
    augmented_lengths = (augmented[1:] - augmented[parents[1:]]).norm(dim=-1)
    ratios = augmented_lengths / original_lengths
    assert torch.all(ratios >= torch.exp(torch.tensor(-0.25)))
    assert torch.all(ratios <= torch.exp(torch.tensor(0.25)))


def test_zero_reference_augmentation_is_identity():
    joints = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.2, 0.8, 0.0]]
    )
    parents = torch.tensor([-1, 0, 1])
    augmented = augment_reference_skeleton(
        joints,
        parents,
        max_local_rotation_degrees=0.0,
        bone_length_log_std=0.0,
        root_translation_std=0.0,
        generator=torch.Generator().manual_seed(2),
    )
    torch.testing.assert_close(augmented, joints)


def test_skeleton_edges_are_bidirectional():
    edges = skeleton_edges(torch.tensor([-1, 0, 1, 1]))
    expected = {
        (1, 0),
        (2, 1),
        (3, 1),
        (0, 1),
        (1, 2),
        (1, 3),
    }
    assert set(map(tuple, edges.t().tolist())) == expected
