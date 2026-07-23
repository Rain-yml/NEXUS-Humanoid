import torch

from torchtitan.experiments.humanoid.data.dataset import (
    RiggedHumanoidJointOctreeDataset,
)
from torchtitan.experiments.humanoid.data.joint_octree import JointOctreeData
from torchtitan.experiments.vem.datasets.octree_utils import OctreeData


def make_dataset(items, get_data):
    dataset = object.__new__(RiggedHumanoidJointOctreeDataset)
    dataset.items = items
    dataset.sample_idx = 0
    dataset.infinite = False
    dataset.get_data = get_data
    return dataset


def test_runtime_dataset_emits_one_sequence_per_item():
    dataset = make_dataset(
        [("asset-a", 2), ("asset-b", 7)],
        lambda uuid, layer: {"uuid": uuid, "layer": layer},
    )

    assert list(dataset) == [
        [{"uuid": "asset-a", "layer": 2}],
        [{"uuid": "asset-b", "layer": 7}],
    ]


def test_runtime_dataset_skips_failed_items():
    def get_data(uuid, layer):
        if uuid == "broken":
            raise RuntimeError("broken asset")
        return {"uuid": uuid, "layer": layer}

    dataset = make_dataset([("broken", 0), ("valid", 1)], get_data)

    assert list(dataset) == [[{"uuid": "valid", "layer": 1}]]


def test_collate_preserves_global_joint_ids():
    dataset = object.__new__(RiggedHumanoidJointOctreeDataset)
    mesh = OctreeData(
        layer_occupancy=[torch.zeros((2, 8), dtype=torch.long)],
        layer_parent_centers=[torch.zeros((2, 3), dtype=torch.long)],
        layer_depths=[4],
        num_vertices=2,
    )
    joints = JointOctreeData(
        layer_occupancy=[torch.zeros((2, 8), dtype=torch.long)],
        layer_parent_centers=[torch.zeros((2, 3), dtype=torch.long)],
        layer_depths=[4],
        num_vertices=2,
    )
    sample = {
        "instance_id": "asset",
        "mesh_octree": mesh,
        "joint_octree": joints,
        "joint_ids": torch.tensor([0, 5]),
        "images": torch.zeros((1, 3, 4, 4)),
        "image_masks": torch.zeros((1, 4, 4), dtype=torch.bool),
        "view_indices": torch.tensor([0]),
        "num_vertices": 2,
    }

    batch = dataset.collate_fn([[sample]])

    torch.testing.assert_close(batch.joint_ids_flat, torch.tensor([-1, -1, 0, 5]))
