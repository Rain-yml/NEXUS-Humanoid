#!/usr/bin/env python3
"""Batch single-stream joint inference on locally prepared evaluation assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from dual_branch_i2v_v2f_validate import (
    latest_dcp_checkpoint,
    load_condition_image,
    load_model,
    make_scheduler,
    read_toml,
    require_paths,
    save_tensor_image,
)
from skeleton_visualization import (
    export_mesh_skeleton_glb,
    mesh_space_from_nexus,
    render_prediction_multiview,
)
from torchtitan.experiments.humanoid.data.bos import BOSClient
from torchtitan.experiments.humanoid.data.dataset import _normalize_like_nexus
from torchtitan.experiments.humanoid.pipelines.image_mesh_to_single_stream_joint_octree import (
    ImageMeshToSingleStreamJointOctreePipeline,
    SingleStreamTeacherForcedMeshLayer,
)
from torchtitan.experiments.vem.datasets.octree_utils import (
    build_octree_specific_layer,
    discretize,
    undiscretize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--stage1-output-root",
        default=(
            "/mnt/pfs/users/liyumeng/code/NEXUS-Humanoid-single-stream/"
            "torchtitan/outputs/humanoid_single_stream_front_qem20k_vroid_train10k"
        ),
    )
    parser.add_argument("--stage1-ckpt", default="")
    parser.add_argument(
        "--stage1-config",
        default=(
            "torchtitan/experiments/humanoid/configs/single_stream/"
            "front_qem20k_vroid_train10k.toml"
        ),
    )
    parser.add_argument(
        "--stage1-ema", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["euler", "heun", "dpm"], default="euler")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--max-wait-seconds", type=float, default=43_200.0)
    return parser.parse_args()


def load_prepared_mesh(
    path: Path,
    *,
    grid_size: int,
    max_depth: int,
    max_merged_vertices: int,
) -> tuple[np.ndarray, np.ndarray, list[SingleStreamTeacherForcedMeshLayer], int]:
    with np.load(path, allow_pickle=False) as mesh:
        vertices = np.asarray(mesh["vertices"], dtype=np.float32)
        faces = np.asarray(mesh["faces"], dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Prepared mesh sidecar has no triangle mesh: {path}")

    normalized_vertices, _ = _normalize_like_nexus(
        vertices, np.zeros((1, 3), dtype=np.float32)
    )
    merged = np.unique(discretize(normalized_vertices, grid_size), axis=0)
    if len(merged) > max_merged_vertices:
        raise ValueError(
            f"Merged point count {len(merged)} exceeds {max_merged_vertices}: {path}"
        )
    mesh_points = torch.from_numpy(merged).long()
    layers = []
    for depth in range(max_depth):
        layer = build_octree_specific_layer(
            mesh_points, depth, grid_size, max_depth
        )
        layers.append(
            SingleStreamTeacherForcedMeshLayer(
                centers=layer.layer_parent_centers[0],
                occupancy=(layer.layer_occupancy[0] * 2 - 1).float(),
                depth=depth,
            )
        )
    return vertices, faces, layers, len(merged)


def condition_path(prepared: dict) -> Path:
    files = [Path(path) for path in prepared["render_files"]]
    preferred = "color_0000.webp" if prepared["has_texture"] else "normal_0000.webp"
    selected = next((path for path in files if path.name == preferred), None)
    if selected is None:
        selected = next((path for path in files if path.name == "normal_0000.webp"), None)
    if selected is None:
        raise FileNotFoundError(f"No front condition render in {files}")
    return selected


def stable_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return (base_seed ^ int.from_bytes(digest[:8], "big")) & ((1 << 63) - 1)


def write_error(path: Path, source: Path, exc: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "error",
                "source": str(source),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")

    sources = sorted(args.input_root.glob("*.glb"))
    assigned = sources[args.shard_index :: args.num_shards]
    if not assigned:
        raise ValueError(f"No inputs assigned from {args.input_root}")

    stage1_root = Path(args.stage1_output_root)
    stage1_ckpt = (
        Path(args.stage1_ckpt)
        if args.stage1_ckpt
        else latest_dcp_checkpoint(stage1_root / "ckpts")
    )
    stage1_config = Path(args.stage1_config)
    config_dict = read_toml(stage1_config)
    dataset_kwargs = config_dict["training"]["dataset_kwargs"]
    joint_schema_path = Path(dataset_kwargs["joint_schema_path"])
    require_paths(
        {
            "stage1_ckpt": stage1_ckpt,
            "stage1_config": stage1_config,
            "joint_schema": joint_schema_path,
        }
    )

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    schema = json.loads(joint_schema_path.read_text(encoding="utf-8"))
    grid_size = int(dataset_kwargs["grid_size"])
    max_depth = int(dataset_kwargs["max_depth"])
    image_size = int(dataset_kwargs["image_resolution"])
    max_merged_vertices = int(dataset_kwargs.get("max_merged_vertices", 11_000))
    view_index = int(dataset_kwargs.get("view_indices", [0])[0])

    print(f"Loading stage1 model once: {stage1_ckpt}", flush=True)
    stage1_cfg, _, model, image_encoder, _ = load_model(
        stage1_ckpt,
        stage1_config,
        device,
        dtype=dtype,
        ema=args.stage1_ema,
        init_weights=True,
    )
    prediction = "v" if stage1_cfg.training.loss_type == "vpred-vloss" else "x"
    pipeline = ImageMeshToSingleStreamJointOctreePipeline(
        image_encoder=image_encoder,
        octree_dit=model,
        scheduler=None,
    ).to(device)
    bos_client = BOSClient()

    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = {source.stem: source for source in assigned}
    complete = 0
    failures = 0
    wait_started = time.monotonic()
    while pending:
        made_progress = False
        for stem, source in list(pending.items()):
            output_dir = args.output_root / stem
            output_summary = output_dir / "summary.json"
            if output_summary.is_file():
                complete += 1
                pending.pop(stem)
                made_progress = True
                continue

            prepared_dir = args.prepared_root / stem
            prepared_summary = prepared_dir / "summary.json"
            prepared_error = prepared_dir / "error.json"
            if prepared_error.is_file() and not prepared_summary.is_file():
                error = RuntimeError(f"Preprocessing failed; see {prepared_error}")
                write_error(output_dir / "error.json", source, error)
                failures += 1
                pending.pop(stem)
                made_progress = True
                continue
            if not prepared_summary.is_file():
                continue

            try:
                prepared = json.loads(prepared_summary.read_text(encoding="utf-8"))
                mesh_path = Path(prepared["qem_npz"])
                image_path = condition_path(prepared)
                vertices, faces, mesh_layers, num_vertices = load_prepared_mesh(
                    mesh_path,
                    grid_size=grid_size,
                    max_depth=max_depth,
                    max_merged_vertices=max_merged_vertices,
                )

                sample_seed = stable_seed(args.seed, stem)
                random.seed(sample_seed)
                np.random.seed(sample_seed % (2**32))
                torch.manual_seed(sample_seed)
                condition = load_condition_image(
                    str(image_path), image_size, bos_client
                )
                generator = torch.Generator(device=device).manual_seed(sample_seed)
                result = pipeline(
                    image=condition,
                    mesh_layers=mesh_layers,
                    scheduler=make_scheduler(args.scheduler),
                    device=device,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    generator=generator,
                    num_vertices=num_vertices,
                    enable_progress=False,
                    grid_size=grid_size,
                    dtype=dtype,
                    prediction=prediction,
                    view_indices=[view_index],
                    num_joint_tokens=len(schema["joints"]),
                )

                output_dir.mkdir(parents=True, exist_ok=True)
                saved_condition = output_dir / "condition.png"
                save_tensor_image(condition, saved_condition)
                predicted_nexus = undiscretize(result.joints.cpu().numpy(), grid_size)
                predicted_mesh = mesh_space_from_nexus(predicted_nexus, vertices)
                np.save(output_dir / "predicted_joints_nexus_normalized.npy", predicted_nexus)
                skeleton_path = output_dir / "predicted_skeleton_mesh_space.npz"
                np.savez_compressed(
                    skeleton_path,
                    positions=predicted_mesh,
                    joint_names=np.asarray(schema["joints"]),
                    parents=np.asarray(schema["parents"], dtype=np.int64),
                    coordinate_space=np.asarray("prepared_qem_npz"),
                    prepared_qem_npz=np.asarray(str(mesh_path)),
                    source_glb=np.asarray(str(source)),
                )
                multiview_path, view_paths = render_prediction_multiview(
                    vertices,
                    faces,
                    predicted_mesh,
                    schema["parents"],
                    output_dir,
                )
                glb_path = output_dir / "mesh_with_predicted_skeleton.glb"
                export_mesh_skeleton_glb(
                    vertices, faces, predicted_mesh, schema["parents"], glb_path
                )
                summary = {
                    "status": "complete",
                    "source": str(source),
                    "prepared_summary": str(prepared_summary),
                    "prepared_qem_npz": str(mesh_path),
                    "condition_source": str(image_path),
                    "condition_image": str(saved_condition),
                    "stage1_ckpt": str(stage1_ckpt),
                    "stage1_config": str(stage1_config),
                    "stage1_ema": args.stage1_ema,
                    "teacher_forced_mesh": True,
                    "mesh_prediction_used": False,
                    "merged_point_count": num_vertices,
                    "joint_count": int(predicted_mesh.shape[0]),
                    "predicted_skeleton": str(skeleton_path),
                    "mesh_with_predicted_skeleton": str(glb_path),
                    "multiview": str(multiview_path),
                    "views": {name: str(path) for name, path in view_paths.items()},
                    "seed": sample_seed,
                }
                temporary = output_summary.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                )
                temporary.replace(output_summary)
                (output_dir / "error.json").unlink(missing_ok=True)
                complete += 1
                print(
                    f"[{complete + failures:03d}/{len(assigned):03d}] complete {stem} "
                    f"merged={num_vertices}",
                    flush=True,
                )
            except Exception as exc:
                write_error(output_dir / "error.json", source, exc)
                failures += 1
                print(f"error {stem}: {type(exc).__name__}: {exc}", flush=True)
            pending.pop(stem)
            made_progress = True

        if pending and not made_progress:
            waited = time.monotonic() - wait_started
            if waited >= args.max_wait_seconds:
                raise TimeoutError(
                    f"Timed out waiting for {len(pending)} prepared assets after {waited:.0f}s"
                )
            print(
                f"waiting for preprocessing: pending={len(pending)} "
                f"complete={complete} failures={failures}",
                flush=True,
            )
            time.sleep(args.poll_seconds)

    shard_summary = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned": len(assigned),
        "complete": complete,
        "failures": failures,
        "stage1_ckpt": str(stage1_ckpt),
    }
    (args.output_root / f"shard-{args.shard_index:02d}.json").write_text(
        json.dumps(shard_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(shard_summary, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
