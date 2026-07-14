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


def artifact_row(row, train_fraction: float, val_fraction: float, split: str | None = None) -> dict:
    uuid = str(row.uuid)
    prefix = f"{uuid[:2]}/{uuid}"
    result = row._asdict()
    result.update(
        split=split or split_for_uuid(uuid, train_fraction, val_fraction),
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
    try:
        client.get_object_meta_data(RENDER_BUCKET, key)
    except Exception:
        return False
    return True


def select_completed(frame: pd.DataFrame, workers: int, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            complete = list(executor.map(render_is_complete, frame["uuid"].astype(str)))
        return frame.loc[complete].reset_index(drop=True)

    candidates = frame.copy()
    candidates["_selection_order"] = candidates["uuid"].astype(str).map(
        lambda value: hashlib.sha256(f"humanoid-smoke-v1:{value}".encode()).hexdigest()
    )
    candidates = candidates.sort_values("_selection_order").drop(columns="_selection_order")
    selected = []
    chunk_size = max(64, workers * 4)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates.iloc[start : start + chunk_size]
            complete = list(executor.map(render_is_complete, chunk["uuid"].astype(str)))
            selected.extend(chunk.loc[complete].to_dict("records"))
            if len(selected) >= max_rows:
                break
    return pd.DataFrame(selected[:max_rows], columns=frame.columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--skip-render-check", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--split-counts",
        help="Exact train,val,test counts, for example 50,10,10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_fraction + args.val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")
    split_labels = None
    if args.split_counts:
        counts = [int(value) for value in args.split_counts.split(",")]
        if len(counts) != 3 or any(value < 0 for value in counts):
            raise ValueError("--split-counts must contain three non-negative integers")
        requested_rows = sum(counts)
        if args.max_rows is not None and args.max_rows != requested_rows:
            raise ValueError("--max-rows must equal the sum of --split-counts")
        args.max_rows = requested_rows
        split_labels = ["train"] * counts[0] + ["val"] * counts[1] + ["test"] * counts[2]

    accepted = pd.read_parquet(args.accepted)
    if accepted["uuid"].duplicated().any():
        raise ValueError(f"Duplicate UUIDs in {args.accepted}")
    if "has_texture" in accepted:
        accepted = accepted.loc[accepted["has_texture"]]
    accepted = accepted.sort_values("uuid").reset_index(drop=True)

    if not args.skip_render_check:
        accepted = select_completed(accepted, args.workers, args.max_rows)
    elif args.max_rows is not None:
        accepted = accepted.iloc[: args.max_rows]
    if accepted.empty:
        raise ValueError("No completed textured assets were found")

    rows = []
    for index, row in enumerate(accepted.itertuples(index=False)):
        split = split_labels[index] if split_labels is not None else None
        rows.append(artifact_row(row, args.train_fraction, args.val_fraction, split=split))
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(f"Wrote {len(output):,} rows to {args.output}")
    print(output["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
