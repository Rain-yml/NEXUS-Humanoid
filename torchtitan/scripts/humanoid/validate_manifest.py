"""Validate a humanoid experiment manifest without loading model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from torchtitan.experiments.humanoid.data.manifest import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"))
    args = parser.parse_args()
    frame = read_manifest(args.manifest, args.split)
    print(f"rows={len(frame):,}")
    print(frame["split"].value_counts().sort_index().to_string())
    print(frame["joint_schema"].value_counts().to_string())


if __name__ == "__main__":
    main()
