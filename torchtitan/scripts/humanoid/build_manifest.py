"""Build the versioned SSOT Parquet consumed by humanoid experiments."""

from __future__ import annotations

import argparse
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


RIG_BUCKET = "biped-data-resave-glb-npz"
RENDER_BUCKET = "biped-data-render-rgb-normal-4v"
_thread_state = threading.local()


def split_for_uuid(uuid: str, train_fraction: float, val_fraction: float) -> str:
    value = int(hashlib.sha256(uuid.encode("utf-8")).hexdigest()[:16], 16) / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def artifact_row(row, train_fraction: float, val_fraction: float) -> dict:
    uuid = str(row.uuid)
    prefix = f"{uuid[:2]}/{uuid}"
    result = row._asdict()
    result.update(
        split=split_for_uuid(uuid, train_fraction, val_fraction),
        joint_schema="humanoid-28-v1",
        rig_npz_uri=f"bos://{RIG_BUCKET}/rig_npz/{uuid[:2]}/{uuid}.npz",
        mesh_glb_uri=f"bos://{RIG_BUCKET}/glb/{uuid[:2]}/{uuid}.glb",
        render_meta_uri=f"bos://{RENDER_BUCKET}/{prefix}/meta.json",
    )
    for index in range(4):
        result[f"color_view_{index}_uri"] = (
            f"bos://{RENDER_BUCKET}/{prefix}/color_{index:04d}.webp"
        )
        result[f"normal_view_{index}_uri"] = (
            f"bos://{RENDER_BUCKET}/{prefix}/normal_{index:04d}.webp"
        )
    return result


def render_is_complete(uuid: str) -> bool:
    from torchtitan.experiments.humanoid.data.bos import BOSClient

    if not hasattr(_thread_state, "bos_client"):
        _thread_state.bos_client = BOSClient().client
    client = _thread_state.bos_client
    key = f"{uuid[:2]}/{uuid}/meta.json"
    return bool(client.does_object_exist(RENDER_BUCKET, key))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--skip-render-check", action="store_true")
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_fraction + args.val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")
    accepted = pd.read_parquet(args.accepted)
    if accepted["uuid"].duplicated().any():
        raise ValueError(f"Duplicate UUIDs in {args.accepted}")
    if "has_texture" in accepted:
        accepted = accepted.loc[accepted["has_texture"]]
    accepted = accepted.sort_values("uuid").reset_index(drop=True)

    if not args.skip_render_check:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            complete = list(executor.map(render_is_complete, accepted["uuid"].astype(str)))
        accepted = accepted.loc[complete].reset_index(drop=True)
    if args.max_rows is not None:
        accepted = accepted.iloc[: args.max_rows]
    if accepted.empty:
        raise ValueError("No completed textured assets were found")

    rows = [
        artifact_row(row, args.train_fraction, args.val_fraction)
        for row in accepted.itertuples(index=False)
    ]
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(f"Wrote {len(output):,} rows to {args.output}")
    print(output["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
