from torchtitan.experiments.humanoid.data.dataset import (
    RiggedHumanoidJointOctreeDataset,
)


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
