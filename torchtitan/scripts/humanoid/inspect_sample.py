"""Load one manifest row through the production dataset and report tensor shapes."""

from __future__ import annotations

import argparse
from pathlib import Path

from torchtitan.experiments.humanoid.data.dataset import RiggedHumanoidJointOctreeDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("torchtitan/experiments/humanoid/data/humanoid_28_v1.json"),
    )
    parser.add_argument(
        "--joint-selection",
        choices=("strict", "available"),
        default="strict",
    )
    args = parser.parse_args()
    dataset = RiggedHumanoidJointOctreeDataset(
        manifest_path=str(args.manifest),
        joint_schema_path=str(args.schema),
        split=args.split,
        infinite=False,
        drop_image_rate=0.0,
        joint_selection=args.joint_selection,
    )
    sample = next(iter(dataset))[0]
    print(f"uuid={sample['instance_id']}")
    print(f"mesh_nodes={[x.shape[0] for x in sample['mesh_octree'].layer_occupancy]}")
    print(f"joint_nodes={[x.shape[0] for x in sample['joint_octree'].layer_occupancy]}")
    print(f"joint_ids={sample['joint_ids'].tolist()}")
    print(f"images={tuple(sample['images'].shape)}")


if __name__ == "__main__":
    main()
