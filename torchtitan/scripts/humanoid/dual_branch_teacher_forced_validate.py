#!/usr/bin/env python3
"""Validate semantic joints with GT mesh-octree teacher forcing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dual_branch_i2v_v2f_validate import (
    latest_dcp_checkpoint,
    load_condition_image,
    load_model,
    load_parquet_sample,
    make_contact_sheet,
    make_scheduler,
    read_toml,
    render_mesh_preview,
    require_paths,
    save_tensor_image,
)
from torchtitan.experiments.humanoid.data.bos import BOSClient
from torchtitan.experiments.humanoid.data.dataset import (
    RiggedHumanoidJointOctreeDataset,
)
from torchtitan.experiments.humanoid.pipelines.image_mesh_to_joint_octree import (
    ImageMeshToJointOctreePipeline,
    TeacherForcedMeshLayer,
)
from torchtitan.experiments.vem.datasets.octree_utils import undiscretize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict joints while teacher-forcing GT mesh octree layers."
    )
    parser.add_argument(
        "--stage1-output-root",
        default="./outputs/humanoid_dual_branch_front_qem20k_smoke100",
    )
    parser.add_argument("--stage1-ckpt", default="")
    parser.add_argument(
        "--stage1-config",
        default=(
            "torchtitan/experiments/humanoid/configs/dual_branch/"
            "front_qem20k_smoke100.toml"
        ),
    )
    parser.add_argument("--stage1-ema", action="store_true", default=False)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=(
            "./outputs/humanoid_dual_branch_front_qem20k_smoke100/"
            "validation_teacher_forced"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["euler", "heun", "dpm"], default="euler")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataset_reader(
    config: dict, manifest: Path
) -> RiggedHumanoidJointOctreeDataset:
    kwargs = dict(config["training"]["dataset_kwargs"])
    return RiggedHumanoidJointOctreeDataset(
        manifest_path=str(manifest),
        joint_schema_path=kwargs["joint_schema_path"],
        split="train",
        repeats=1,
        shuffle_seed=int(kwargs.get("shuffle_seed", 42)),
        grid_size=int(kwargs["grid_size"]),
        max_depth=int(kwargs["max_depth"]),
        image_resolution=int(kwargs["image_resolution"]),
        view_indices=list(kwargs.get("view_indices", [0])),
        drop_image_rate=0.0,
        infinite=False,
        max_merged_vertices=int(kwargs.get("max_merged_vertices", 11_000)),
    )


def main() -> int:
    args = parse_args()
    stage1_root = Path(args.stage1_output_root)
    stage1_ckpt = (
        Path(args.stage1_ckpt)
        if args.stage1_ckpt
        else latest_dcp_checkpoint(stage1_root / "ckpts")
    )
    stage1_config = Path(args.stage1_config)
    config_dict = read_toml(stage1_config)
    dataset_kwargs = config_dict["training"]["dataset_kwargs"]
    manifest = Path(args.manifest or dataset_kwargs["manifest_path"])
    joint_schema_path = Path(dataset_kwargs["joint_schema_path"])
    require_paths(
        {
            "stage1_ckpt": stage1_ckpt,
            "stage1_config": stage1_config,
            "manifest": manifest,
            "joint_schema": joint_schema_path,
        }
    )

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    row = load_parquet_sample(manifest, args.split, args.sample_index)
    sample_uuid = str(row["uuid"])
    view_index = int(dataset_kwargs.get("view_indices", [0])[0])
    grid_size = int(dataset_kwargs["grid_size"])
    max_depth = int(dataset_kwargs["max_depth"])
    image_size = int(dataset_kwargs["image_resolution"])
    schema = json.loads(joint_schema_path.read_text(encoding="utf-8"))

    dataset_reader = build_dataset_reader(config_dict, manifest)
    rig, octree_layers = dataset_reader.load_rig_layers_from_row(row)
    mesh_layers = [
        TeacherForcedMeshLayer(
            centers=layer.layer_parent_centers[0],
            occupancy=(layer.layer_occupancy[0] * 2 - 1).float(),
            depth=depth,
        )
        for depth, layer in enumerate(octree_layers)
    ]

    out_dir = Path(args.output_dir) / (
        f"{stage1_ckpt.name}_{args.split}_{args.sample_index:04d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    condition_image = load_condition_image(
        str(row[f"color_view_{view_index}_uri"]), image_size, BOSClient()
    )
    condition_path = out_dir / "condition.png"
    save_tensor_image(condition_image, condition_path)

    print(f"Loading stage1 model: {stage1_ckpt}")
    stage1_cfg, _, model, image_encoder, _ = load_model(
        stage1_ckpt,
        stage1_config,
        device,
        dtype=dtype,
        ema=args.stage1_ema,
        init_weights=True,
    )
    prediction = "v" if stage1_cfg.training.loss_type == "vpred-vloss" else "x"
    pipeline = ImageMeshToJointOctreePipeline(
        image_encoder=image_encoder,
        octree_dit=model,
        scheduler=None,
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipeline(
        image=condition_image,
        mesh_layers=mesh_layers,
        scheduler=make_scheduler(args.scheduler),
        device=device,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator,
        num_vertices=int(rig.mesh_points.shape[0]),
        enable_progress=True,
        grid_size=grid_size,
        dtype=dtype,
        prediction=prediction,
        view_indices=[view_index],
        num_joint_tokens=len(schema["joints"]),
    )

    predicted_joints = undiscretize(result.joints.cpu().numpy(), grid_size)
    gt_joints = rig.joints
    joint_errors = np.linalg.norm(predicted_joints - gt_joints, axis=1)
    np.save(out_dir / "predicted_joints.npy", predicted_joints)
    np.save(out_dir / "gt_joints.npy", gt_joints)

    empty_faces = np.empty((0, 3), dtype=np.int64)
    gt_preview = out_dir / "gt_mesh_gt_joints.png"
    predicted_preview = out_dir / "gt_mesh_predicted_joints.png"
    render_mesh_preview(
        rig.vertices,
        empty_faces,
        gt_preview,
        joints=gt_joints,
        joint_parents=schema["parents"],
    )
    render_mesh_preview(
        rig.vertices,
        empty_faces,
        predicted_preview,
        joints=predicted_joints,
        joint_parents=schema["parents"],
    )
    contact_sheet = out_dir / "contact_sheet.png"
    make_contact_sheet(
        [
            (f"{args.split} condition (front)", condition_path),
            ("GT mesh + GT joints", gt_preview),
            ("GT mesh + predicted joints", predicted_preview),
        ],
        contact_sheet,
    )

    summary = {
        "sample_uuid": sample_uuid,
        "split": args.split,
        "stage1_ckpt": str(stage1_ckpt),
        "stage1_config": str(stage1_config),
        "teacher_forced_mesh": True,
        "mesh_layers": len(mesh_layers),
        "mesh_prediction_used": False,
        "joint_count": int(predicted_joints.shape[0]),
        "mean_joint_error": float(joint_errors.mean()),
        "max_joint_error": float(joint_errors.max()),
        "per_joint_error": joint_errors.tolist(),
        "condition_image": str(condition_path),
        "gt_preview": str(gt_preview),
        "predicted_preview": str(predicted_preview),
        "contact_sheet": str(contact_sheet),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
