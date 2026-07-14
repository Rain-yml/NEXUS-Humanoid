"""Build a versioned humanoid SSOT Parquet from BOS assets."""

from __future__ import annotations

import argparse
import hashlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchtitan.experiments.humanoid.data.dataset import _normalize_like_nexus
from torchtitan.experiments.humanoid.data.joint_schema import JointSchema
from torchtitan.experiments.vem.datasets.octree_utils import (
    build_octree_specific_layer,
    discretize,
)


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


def bos_client():
    from torchtitan.experiments.humanoid.data.bos import BOSClient

    if not hasattr(_thread_state, "bos_client"):
        _thread_state.bos_client = BOSClient().client
    return _thread_state.bos_client


def required_render_names() -> set[str]:
    names = {"meta.json"}
    for index in range(4):
        names.add(f"color_{index:04d}.webp")
        names.add(f"normal_{index:04d}.webp")
    return names


def artifacts_are_complete(uuid: str, rig_prefix: str, mesh_prefix: str) -> bool:
    client = bos_client()
    asset_prefix = f"{uuid[:2]}/{uuid}"
    try:
        client.get_object_meta_data(RIG_BUCKET, f"{rig_prefix}/{asset_prefix}.npz")
        client.get_object_meta_data(RIG_BUCKET, f"{mesh_prefix}/{asset_prefix}.glb")
        response = client.list_objects(
            RENDER_BUCKET,
            prefix=f"{asset_prefix}/",
            max_keys=100,
        )
    except Exception:
        return False
    names = {item.key.rsplit("/", 1)[-1] for item in response.contents or []}
    return required_render_names() <= names


def inspect_rig(
    uuid: str,
    rig_prefix: str,
    schema: JointSchema,
    grid_size: int,
    max_depth: int,
) -> dict:
    key = f"{rig_prefix}/{uuid[:2]}/{uuid}.npz"
    payload = bos_client().get_object_as_string(RIG_BUCKET, key)
    with np.load(io.BytesIO(payload), allow_pickle=True) as rig:
        vertices = np.asarray(rig["vertices"], dtype=np.float32)
        faces = np.asarray(rig["faces"])
        joints = schema.select(
            rig["joint_semantics"].tolist(),
            np.asarray(rig["joint_positions"], dtype=np.float32),
            np.asarray(rig["parents"], dtype=np.int64),
        )

    vertices, _ = _normalize_like_nexus(vertices, joints)
    merged = np.unique(discretize(vertices, grid_size), axis=0)
    points = torch.from_numpy(merged).long()
    layer_tokens = [
        build_octree_specific_layer(points, depth, grid_size, max_depth).get_layer_num_nodes(0)
        + len(schema.joints)
        for depth in range(max_depth)
    ]
    return {
        "num_vertices": int(len(vertices)),
        "num_faces": int(len(faces)),
        "num_merged_vertices": int(len(merged)),
        "octree_layer_tokens": layer_tokens,
        "max_layer_tokens": int(max(layer_tokens)),
    }


def artifact_row(
    row: dict,
    metadata: dict,
    *,
    rig_prefix: str,
    mesh_prefix: str,
    train_fraction: float,
    val_fraction: float,
    split: str | None,
) -> dict:
    uuid = str(row["uuid"])
    asset_prefix = f"{uuid[:2]}/{uuid}"
    result = dict(row)
    result.update(metadata)
    result.update(
        split=split or split_for_uuid(uuid, train_fraction, val_fraction),
        joint_schema="humanoid-28-v1",
        rig_npz_uri=f"bos://{RIG_BUCKET}/{rig_prefix}/{asset_prefix}.npz",
        mesh_glb_uri=f"bos://{RIG_BUCKET}/{mesh_prefix}/{asset_prefix}.glb",
        render_meta_uri=f"bos://{RENDER_BUCKET}/{asset_prefix}/meta.json",
    )
    for index in range(4):
        result[f"color_view_{index}_uri"] = (
            f"bos://{RENDER_BUCKET}/{asset_prefix}/color_{index:04d}.webp"
        )
        result[f"normal_view_{index}_uri"] = (
            f"bos://{RENDER_BUCKET}/{asset_prefix}/normal_{index:04d}.webp"
        )
    return result


def inspect_candidate(row: dict, args, schema: JointSchema) -> tuple[dict, dict] | None:
    uuid = str(row["uuid"])
    if not args.skip_artifact_check and not artifacts_are_complete(
        uuid, args.rig_prefix, args.mesh_prefix
    ):
        return None
    try:
        metadata = inspect_rig(
            uuid,
            args.rig_prefix,
            schema,
            args.grid_size,
            args.max_depth,
        )
    except Exception:
        return None
    if metadata["num_faces"] > args.max_faces:
        return None
    if args.max_layer_tokens and metadata["max_layer_tokens"] > args.max_layer_tokens:
        return None
    return row, metadata


def select_assets(frame: pd.DataFrame, args, schema: JointSchema) -> list[tuple[dict, dict]]:
    candidates = frame.copy()
    candidates["_selection_order"] = candidates["uuid"].astype(str).map(
        lambda value: hashlib.sha256(
            f"humanoid-{args.selection_seed}:{value}".encode()
        ).hexdigest()
    )
    candidates = candidates.sort_values("_selection_order").drop(columns="_selection_order")
    rows = candidates.to_dict("records")
    selected: list[tuple[dict, dict]] = []
    chunk_size = max(32, args.workers * 2)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            inspected = executor.map(
                lambda row: inspect_candidate(row, args, schema),
                chunk,
            )
            selected.extend(item for item in inspected if item is not None)
            if args.max_rows is not None and len(selected) >= args.max_rows:
                return selected[: args.max_rows]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--joint-schema", required=True, type=Path)
    parser.add_argument("--rig-prefix", default="rig_npz")
    parser.add_argument("--mesh-prefix", default="glb")
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-artifact-check", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-faces", type=int, default=20_000)
    parser.add_argument("--max-layer-tokens", type=int)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--selection-seed", default="qem20k-smoke-v1")
    parser.add_argument(
        "--split-counts",
        help="Exact train,val,test counts, for example 100,10,10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid_size != 2**args.max_depth:
        raise ValueError("--grid-size must equal 2 ** --max-depth")
    if args.train_fraction + args.val_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum to less than one")

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

    schema = JointSchema.load(args.joint_schema)
    selected = select_assets(accepted, args, schema)
    if not selected:
        raise ValueError("No assets passed artifact and geometry validation")
    if args.max_rows is not None and len(selected) != args.max_rows:
        raise ValueError(f"Requested {args.max_rows} assets, found {len(selected)}")

    rows = []
    for index, (row, metadata) in enumerate(selected):
        split = split_labels[index] if split_labels is not None else None
        rows.append(
            artifact_row(
                row,
                metadata,
                rig_prefix=args.rig_prefix,
                mesh_prefix=args.mesh_prefix,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                split=split,
            )
        )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(f"Wrote {len(output):,} rows to {args.output}")
    print(output["split"].value_counts().sort_index().to_string())
    print(output[["num_vertices", "num_faces", "num_merged_vertices", "max_layer_tokens"]].describe().to_string())


if __name__ == "__main__":
    main()
