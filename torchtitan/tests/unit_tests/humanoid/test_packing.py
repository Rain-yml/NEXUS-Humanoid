from scripts.humanoid.pack_manifest import first_fit_decreasing


def test_first_fit_decreasing_caps_sequences_per_batch():
    items = [(1, [f"asset-{index}", index, 0]) for index in range(10)]

    batches = first_fit_decreasing(items, budget=100, max_sequences_per_batch=4)

    assert [len(batch) for batch in batches] == [4, 4, 2]
    assert sorted(item[0] for batch in batches for item in batch) == [
        f"asset-{index}" for index in range(10)
    ]
