"""Convert AniGenP's processed manifest into project SSOT Parquets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from torchtitan.experiments.humanoid.data.rig_records import (
    ANIGEN_VOXELIZED_FORMAT,
)


def stable_key(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def split_record(instance_id: str, is_test: bool, val_fraction: float) -> str:
    if is_test:
        return "test"
    value = int(hashlib.sha256(instance_id.encode()).hexdigest()[:16], 16) / 2**64
    return "val" if value < val_fraction else "train"


def convert_record(record: dict, val_fraction: float) -> dict:
    instance_id = str(record["instance_id"])
    image_path = str(record["image_path"])
    image_dir = str(record.get("image_dir") or Path(image_path).parent)
    return {
        "uuid": instance_id,
        "split": split_record(
            instance_id, bool(record.get("is_test", False)), val_fraction
        ),
        "source": "anigenp",
        "rig_format": ANIGEN_VOXELIZED_FORMAT,
        "joint_schema": None,
        "rig_npz_uri": str(record["skeleton_path"]),
        "mesh_glb_uri": None,
        "render_meta_uri": str(Path(image_dir) / "meta.json"),
        "color_view_0_uri": image_path,
        "num_vertices": int(record["num_vertices"]),
        "num_faces": int(record["num_faces"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--full-output", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--selection-seed", default="anigenp-reference-skeleton-v1")
    parser.add_argument("--val-fraction", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_limit < 1:
        raise ValueError("--train-limit must be positive")
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")

    with args.input.open("r", encoding="utf-8") as handle:
        source_records = json.load(handle)
    rows = [convert_record(record, args.val_fraction) for record in source_records]
    frame = pd.DataFrame(rows)
    if frame["uuid"].duplicated().any():
        duplicates = frame.loc[frame["uuid"].duplicated(), "uuid"].head().tolist()
        raise ValueError(f"AniGenP manifest has duplicate instance IDs: {duplicates}")

    args.full_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.full_output, index=False)

    train = frame.loc[frame["split"] == "train"].copy()
    train["_selection_key"] = train["uuid"].map(
        lambda value: stable_key(value, args.selection_seed)
    )
    train = (
        train.sort_values("_selection_key")
        .head(min(args.train_limit, len(train)))
        .drop(columns="_selection_key")
    )
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.train_output, index=False)

    print(f"full={len(frame):,} -> {args.full_output}")
    print(frame["split"].value_counts().sort_index().to_string())
    print(f"train_subset={len(train):,} -> {args.train_output}")


if __name__ == "__main__":
    main()
