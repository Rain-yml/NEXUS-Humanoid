"""Create a deterministic experiment subset from a canonical SSOT manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from torchtitan.experiments.humanoid.data.manifest import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--size", required=True, type=int)
    args = parser.parse_args()

    frame = read_manifest(args.input, args.split).copy()
    frame["_subset_order"] = frame["uuid"].astype(str).map(
        lambda value: hashlib.sha256(f"humanoid-subset-v1:{value}".encode()).hexdigest()
    )
    frame = frame.sort_values("_subset_order").head(args.size).drop(columns="_subset_order")
    if len(frame) != args.size:
        raise ValueError(f"Requested {args.size} rows, found {len(frame)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"Wrote {len(frame)} {args.split} rows to {args.output}")


if __name__ == "__main__":
    main()
