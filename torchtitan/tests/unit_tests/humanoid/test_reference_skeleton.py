import numpy as np
import pytest
import torch

from torchtitan.experiments.humanoid.data.dataset import (
    DegenerateHumanoidRigError,
    RiggedHumanoidJointOctreeDataset,
)
from torchtitan.experiments.humanoid.data.rig_records import (
    ANIGEN_VOXELIZED_FORMAT,
    load_rig_record,
)
from torchtitan.experiments.humanoid.data.skeleton_augmentation import (
    augment_reference_skeleton,
    skeleton_edges,
)
from torchtitan.experiments.humanoid.models.reference_skeleton_encoder import (
    ReferenceSkeletonEncoder,
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


def test_reference_augmentation_is_shared_monotonic_spatial_stretch():
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
        global_scale_log_std=0.08,
        local_scale_log_std=0.10,
        num_segments=4,
        generator=torch.Generator().manual_seed(7),
    )

    assert not torch.equal(augmented, joints)
    original_differences = joints[:, None] - joints[None, :]
    augmented_differences = augmented[:, None] - augmented[None, :]
    assert torch.all(original_differences * augmented_differences >= -1e-7)
    torch.testing.assert_close(augmented[0, 0], augmented[1, 0])
    torch.testing.assert_close(augmented[0, 2], augmented[1, 2])


def test_zero_reference_augmentation_is_identity():
    joints = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.2, 0.8, 0.0]]
    )
    parents = torch.tensor([-1, 0, 1])
    augmented = augment_reference_skeleton(
        joints,
        parents,
        global_scale_log_std=0.0,
        local_scale_log_std=0.0,
        num_segments=4,
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


def test_dataset_rejects_skeleton_without_bones(tmp_path):
    rig_path = tmp_path / "one_joint.npz"
    np.savez(
        rig_path,
        vertices=np.asarray(
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32
        ),
        joints=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        parents=np.asarray([None], dtype=object),
    )
    dataset = object.__new__(RiggedHumanoidJointOctreeDataset)
    dataset._bos_client = None
    dataset.schema = None
    dataset.joint_selection = "strict"
    dataset.grid_size = 512
    dataset.max_merged_vertices = 11_000

    with pytest.raises(DegenerateHumanoidRigError, match="has no bones"):
        dataset._load_normalized_rig(
            {
                "rig_npz_uri": str(rig_path),
                "rig_format": ANIGEN_VOXELIZED_FORMAT,
            }
        )


def test_reference_encoder_rejects_skeleton_without_bones():
    encoder = ReferenceSkeletonEncoder(
        hidden_dim=16,
        edge_hidden_dim=8,
        num_blocks=1,
        num_attention_heads=4,
        intermediate_size=32,
        grid_size=16,
        use_flash_attn_3=False,
    )

    with pytest.raises(ValueError, match="at least one bone"):
        encoder(
            positions=torch.zeros(1, 3),
            rope_positions=torch.zeros(1, 3, dtype=torch.long),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        )


def test_reference_encoder_does_not_optimize_discarded_final_edge_update():
    encoder = ReferenceSkeletonEncoder(
        hidden_dim=16,
        edge_hidden_dim=8,
        num_blocks=3,
        num_attention_heads=4,
        intermediate_size=32,
        grid_size=16,
        use_flash_attn_3=False,
    )
    final_graph_layer = encoder.graph_layers[-1]

    assert all(
        not parameter.requires_grad
        for module in (
            final_graph_layer.attn.edge_mlp,
            final_graph_layer.norm2_edge,
            final_graph_layer.ffn_edge,
        )
        for parameter in module.parameters()
    )
    assert all(
        parameter.requires_grad
        for module in (
            final_graph_layer.attn.query_proj,
            final_graph_layer.attn.key_proj,
            final_graph_layer.attn.value_proj,
            final_graph_layer.ffn_node,
        )
        for parameter in module.parameters()
    )
