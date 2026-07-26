"""Convert AniGenP's processed manifest into project SSOT Parquets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

from torchtitan.experiments.humanoid.data.rig_records import (
    ANIGEN_VOXELIZED_FORMAT,
)

POSE_SUFFIX = re.compile(r"_pose_(\d+)$")
FRONT_DIRECTION = (0.0, -1.0, 0.0)


def stable_key(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def unit_interval(value: str, seed: str) -> float:
    digest = stable_key(value, seed)
    return int(digest[:16], 16) / 2**64


def parse_instance_id(instance_id: str) -> tuple[str, int | None]:
    match = POSE_SUFFIX.search(instance_id)
    if match is None:
        return instance_id, None
    return instance_id[: match.start()], int(match.group(1))


def split_asset(
    asset_id: str,
    *,
    seed: str,
    val_fraction: float,
    test_fraction: float,
) -> str:
    value = unit_interval(asset_id, seed)
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def select_front_frame(image_dir: Path) -> dict:
    transforms_path = image_dir / "transforms.json"
    with transforms_path.open("r", encoding="utf-8") as handle:
        transforms = json.load(handle)

    candidates = []
    for frame_index, frame in enumerate(transforms.get("frames", [])):
        matrix = frame.get("transform_matrix")
        if not isinstance(matrix, list) or len(matrix) < 3:
            continue
        camera = tuple(float(matrix[axis][3]) for axis in range(3))
        distance = math.sqrt(sum(component * component for component in camera))
        if distance <= 1e-8:
            continue
        direction = tuple(component / distance for component in camera)
        front_similarity = sum(
            component * target
            for component, target in zip(direction, FRONT_DIRECTION, strict=True)
        )
        azimuth = math.degrees(math.atan2(camera[1], camera[0]))
        elevation = math.degrees(
            math.atan2(camera[2], math.hypot(camera[0], camera[1]))
        )
        frame_path = image_dir / str(frame["file_path"])
        if frame_path.is_file():
            candidates.append(
                (
                    -front_similarity,
                    abs(elevation),
                    frame_index,
                    frame_path,
                    azimuth,
                    elevation,
                )
            )

    if not candidates:
        raise ValueError(f"No valid rendered camera frames in {transforms_path}")

    _, _, frame_index, frame_path, azimuth, elevation = min(candidates)
    angular_error = math.degrees(
        math.acos(
            max(
                -1.0,
                min(
                    1.0,
                    math.cos(math.radians(azimuth + 90.0))
                    * math.cos(math.radians(elevation)),
                ),
            )
        )
    )
    return {
        "path": str(frame_path),
        "frame_index": frame_index,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "angular_error_deg": angular_error,
        "transforms_path": str(transforms_path),
    }


def convert_record(
    record: dict,
    *,
    split_seed: str,
    val_fraction: float,
    test_fraction: float,
) -> dict:
    instance_id = str(record["instance_id"])
    image_path = str(record["image_path"])
    image_dir = str(record.get("image_dir") or Path(image_path).parent)
    asset_id, pose_index = parse_instance_id(instance_id)
    front = select_front_frame(Path(image_dir))
    return {
        "uuid": instance_id,
        "base_asset_id": asset_id,
        "pose_index": pose_index,
        "split": split_asset(
            asset_id,
            seed=split_seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        ),
        "source": "anigenp",
        "rig_format": ANIGEN_VOXELIZED_FORMAT,
        "joint_schema": None,
        "rig_npz_uri": str(record["skeleton_path"]),
        "mesh_glb_uri": None,
        "render_meta_uri": front["transforms_path"],
        "color_view_0_uri": front["path"],
        "condition_view": "front",
        "condition_frame_index": front["frame_index"],
        "condition_camera_azimuth_deg": front["azimuth_deg"],
        "condition_camera_elevation_deg": front["elevation_deg"],
        "condition_camera_angle_deg": front["angular_error_deg"],
        "num_vertices": int(record["num_vertices"]),
        "num_faces": int(record["num_faces"]),
    }


def select_asset_groups(
    frame: pd.DataFrame,
    *,
    row_limit: int,
    seed: str,
) -> pd.DataFrame:
    train = frame.loc[frame["split"] == "train"].copy()
    asset_ids = sorted(
        train["base_asset_id"].unique(),
        key=lambda asset_id: stable_key(asset_id, seed),
    )

    selected_assets = []
    selected_rows = 0
    group_sizes = train.groupby("base_asset_id").size().to_dict()
    for asset_id in asset_ids:
        group_size = int(group_sizes[asset_id])
        if selected_rows + group_size > row_limit:
            continue
        selected_assets.append(asset_id)
        selected_rows += group_size
        if selected_rows == row_limit:
            break

    return (
        train.loc[train["base_asset_id"].isin(selected_assets)]
        .sort_values(["base_asset_id", "pose_index", "uuid"])
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--full-output", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--selection-seed", default="anigenp-asset-subset-v2")
    parser.add_argument("--split-seed", default="anigenp-asset-split-v2")
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--max-front-angle-degrees", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_limit < 1:
        raise ValueError("--train-limit must be positive")
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")
    if not 0.0 <= args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be in [0, 1)")
    if args.val_fraction + args.test_fraction >= 1.0:
        raise ValueError("--val-fraction + --test-fraction must be less than 1")
    if not 0.0 < args.max_front_angle_degrees <= 90.0:
        raise ValueError("--max-front-angle-degrees must be in (0, 90]")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    with args.input.open("r", encoding="utf-8") as handle:
        source_records = json.load(handle)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(
            executor.map(
                lambda record: convert_record(
                    record,
                    split_seed=args.split_seed,
                    val_fraction=args.val_fraction,
                    test_fraction=args.test_fraction,
                ),
                source_records,
            )
        )
    source_row_count = len(rows)
    frame = pd.DataFrame(rows)
    frame = frame.loc[
        frame["condition_camera_angle_deg"] <= args.max_front_angle_degrees
    ].reset_index(drop=True)
    if frame["uuid"].duplicated().any():
        duplicates = frame.loc[frame["uuid"].duplicated(), "uuid"].head().tolist()
        raise ValueError(f"AniGenP manifest has duplicate instance IDs: {duplicates}")
    split_counts = frame.groupby("base_asset_id")["split"].nunique()
    if int(split_counts.max()) != 1:
        raise ValueError("Base assets cross dataset splits")

    args.full_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.full_output, index=False)

    train = select_asset_groups(
        frame,
        row_limit=args.train_limit,
        seed=args.selection_seed,
    )
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.train_output, index=False)

    print(
        f"front_accepted={len(frame):,} / {source_row_count:,}; "
        f"max_angle={args.max_front_angle_degrees:.1f}deg -> {args.full_output}"
    )
    print(frame["split"].value_counts().sort_index().to_string())
    print(f"assets={frame['base_asset_id'].nunique():,}")
    print(
        f"train_subset={len(train):,} rows / "
        f"{train['base_asset_id'].nunique():,} assets -> {args.train_output}"
    )
    print(
        "front camera error: "
        f"median={frame['condition_camera_angle_deg'].median():.2f}deg, "
        f"max={frame['condition_camera_angle_deg'].max():.2f}deg"
    )


if __name__ == "__main__":
    main()
