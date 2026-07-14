import torch

from torchtitan.experiments.humanoid.data.joint_octree import build_joint_layers


def test_each_joint_has_exactly_one_child_at_every_depth():
    points = torch.tensor([[0, 0, 0], [7, 7, 7], [3, 5, 2]])
    octree = build_joint_layers(points, grid_size=8, max_depth=3)
    assert len(octree.layer_occupancy) == 3
    for occupancy in octree.layer_occupancy:
        assert occupancy.shape == (3, 8)
        torch.testing.assert_close(occupancy.sum(dim=1), torch.ones(3, dtype=torch.long))
