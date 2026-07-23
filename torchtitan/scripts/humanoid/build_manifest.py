"""Build a lightweight humanoid experiment manifest from an accepted asset list."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


RIG_BUCKET = "biped-data-resave-glb-npz"
RENDER_BUCKET = "biped-data-render-rgb-normal-4v"


def split_for_uuid(uuid: str, train_fraction: float, val_fraction: float) -> str:
    value = int(hashlib.sha256(uuid.encode("utf-8")).hexdigest()[:16], 16) / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def selection_key(uuid: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{uuid}".encode()).hexdigest()


def artifact_row(
    row: dict,
    *,
    dataset_prefix: str,
    render_prefix: str,
    rig_subdir: str,
    mesh_subdir: str,
    joint_schema: str,
    train_fraction: float,
    val_fraction: float,
    split: str | None,
) -> dict:
    uuid = str(row["uuid"])
    sharded_uuid = f"{uuid[:2]}/{uuid}"
    result = dict(row)
    result.update(
        split=split or split_for_uuid(uuid, train_fraction, val_fraction),
        joint_schema=joint_schema,
        rig_npz_uri=(
            f"bos://{RIG_BUCKET}/{dataset_prefix}/{rig_subdir}/{sharded_uuid}.npz"
        ),
        mesh_glb_uri=(
            f"bos://{RIG_BUCKET}/{dataset_prefix}/{mesh_subdir}/{sharded_uuid}.glb"
        ),
        render_meta_uri=(
            f"bos://{RENDER_BUCKET}/{render_prefix}/{sharded_uuid}/meta.json"
        ),
    )
    for index in range(4):
        result[f"color_view_{index}_uri"] = (
            f"bos://{RENDER_BUCKET}/{render_prefix}/{sharded_uuid}/"
            f"color_{index:04d}.webp"
        )
        result[f"normal_view_{index}_uri"] = (
            f"bos://{RENDER_BUCKET}/{render_prefix}/{sharded_uuid}/"
            f"normal_{index:04d}.webp"
        )
    return result


def parse_split_counts(value: str | None) -> list[int] | None:
    if value is None:
        return None
    counts = [int(part) for part in value.split(",")]
    if len(counts) != 3 or any(count < 0 for count in counts):
        raise ValueError("--split-counts must contain three non-negative integers")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-prefix", required=True)
    parser.add_argument(
        "--render-prefix",
        help="Defaults to --dataset-prefix.",
    )
    parser.add_argument("--rig-subdir", default="rig_npz")
    parser.add_argument("--mesh-subdir", default="glb")
    parser.add_argument("--joint-schema", default="humanoid-28-v1")
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument(
        "--include-untextured",
        action="store_true",
        help="Keep assets without color renders; intended for normal-conditioned configs.",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--selection-seed", default="humanoid-manifest-v1")
    parser.add_argument(
        "--split-counts",
        help="Exact train,val,test counts for deterministic smoke manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_fraction < 0 or args.val_fraction < 0:
        raise ValueError("Split fractions must be non-negative")
    if args.train_fraction + args.val_fraction >= 1.0:
        raise ValueError("Train and validation fractions must sum to less than one")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive")

    split_counts = parse_split_counts(args.split_counts)
    if split_counts is not None:
        requested_rows = sum(split_counts)
        if args.max_rows is not None and args.max_rows != requested_rows:
            raise ValueError("--max-rows must equal the sum of --split-counts")
        args.max_rows = requested_rows

    accepted = pd.read_parquet(args.accepted)
    if "uuid" not in accepted:
        raise ValueError(f"{args.accepted} has no uuid column")
    if accepted["uuid"].duplicated().any():
        raise ValueError(f"Duplicate UUIDs in {args.accepted}")
    if not args.include_untextured:
        if "has_texture" not in accepted:
            raise ValueError(
                f"{args.accepted} has no has_texture column; "
                "pass --include-untextured only for a non-color experiment"
            )
        accepted = accepted.loc[accepted["has_texture"]]

    accepted = accepted.copy()
    accepted["_selection_order"] = accepted["uuid"].astype(str).map(
        lambda uuid: selection_key(uuid, args.selection_seed)
    )
    accepted = accepted.sort_values("_selection_order").drop(columns="_selection_order")
    if args.max_rows is not None:
        accepted = accepted.head(args.max_rows)
        if len(accepted) != args.max_rows:
            raise ValueError(f"Requested {args.max_rows} assets, found {len(accepted)}")

    split_labels = None
    if split_counts is not None:
        split_labels = (
            ["train"] * split_counts[0]
            + ["val"] * split_counts[1]
            + ["test"] * split_counts[2]
        )

    render_prefix = args.render_prefix or args.dataset_prefix
    rows = [
        artifact_row(
            row,
            dataset_prefix=args.dataset_prefix.strip("/"),
            render_prefix=render_prefix.strip("/"),
            rig_subdir=args.rig_subdir.strip("/"),
            mesh_subdir=args.mesh_subdir.strip("/"),
            joint_schema=args.joint_schema,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            split=split_labels[index] if split_labels is not None else None,
        )
        for index, row in enumerate(accepted.to_dict("records"))
    ]
    if not rows:
        raise ValueError("No assets selected")

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(f"Wrote {len(output):,} rows to {args.output}")
    print(output["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
