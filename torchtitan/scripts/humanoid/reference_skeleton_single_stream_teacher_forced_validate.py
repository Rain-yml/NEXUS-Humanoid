#!/usr/bin/env python3
"""Validate reference-conditioned joints with clean GT mesh tokens."""

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
    make_contact_sheet,
    make_scheduler,
    read_toml,
    require_paths,
    save_tensor_image,
)
from torchtitan.experiments.humanoid.data.bos import BOSClient
from torchtitan.experiments.humanoid.data.dataset import (
    OversizedHumanoidRigError,
    RiggedHumanoidJointOctreeDataset,
)
from torchtitan.experiments.humanoid.data.skeleton_augmentation import (
    augment_reference_skeleton,
    skeleton_edges,
)
from torchtitan.experiments.humanoid.pipelines.image_mesh_reference_skeleton_to_single_stream_joint_octree import (
    ImageMeshReferenceSkeletonToSingleStreamJointOctreePipeline,
    SingleStreamTeacherForcedMeshLayer,
)
from torchtitan.experiments.vem.datasets.octree_utils import discretize, undiscretize
from skeleton_visualization import (
    export_mesh_skeleton_glb,
    mesh_space_from_nexus,
    render_prediction_multiview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict joints from a reference skeleton and clean GT mesh layers."
    )
    parser.add_argument(
        "--stage1-output-root",
        default="./outputs/humanoid_reference_skeleton_front_anigen100k_v2",
    )
    parser.add_argument("--stage1-ckpt", default="")
    parser.add_argument(
        "--stage1-config",
        default=(
            "torchtitan/experiments/humanoid/configs/single_stream/"
            "front_anigen100k_reference_skeleton.toml"
        ),
    )
    parser.add_argument("--stage1-ema", action="store_true", default=False)
    parser.add_argument(
        "--manifest",
        default=(
            "/mnt/pfs/users/liyumeng/data/rigged_humanoid/datasets/"
            "anigenp_asset_front_full_v2.parquet"
        ),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=(
            "./outputs/humanoid_reference_skeleton_front_anigen100k_v2/"
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
    parser.add_argument(
        "--prediction-only-multiview",
        action="store_true",
        help="Save native-space predictions and four-view renders without GT panels.",
    )
    return parser.parse_args()


def build_dataset_reader(
    config: dict, manifest: Path, split: str
) -> RiggedHumanoidJointOctreeDataset:
    kwargs = dict(config["training"]["dataset_kwargs"])
    return RiggedHumanoidJointOctreeDataset(
        manifest_path=str(manifest),
        joint_schema_path=kwargs.get("joint_schema_path"),
        split=split,
        repeats=1,
        shuffle_seed=int(kwargs.get("shuffle_seed", 42)),
        grid_size=int(kwargs["grid_size"]),
        max_depth=int(kwargs["max_depth"]),
        image_resolution=int(kwargs["image_resolution"]),
        view_indices=list(kwargs.get("view_indices", [0])),
        drop_image_rate=0.0,
        infinite=False,
        max_merged_vertices=int(kwargs.get("max_merged_vertices", 11_000)),
        joint_selection=str(kwargs.get("joint_selection", "strict")),
        reference_skeleton_augmentation=False,
    )


def load_training_eligible_sample(
    dataset: RiggedHumanoidJointOctreeDataset, sample_index: int
):
    """Resolve a base-asset index after applying training's vertex limit."""
    if sample_index < 0:
        raise ValueError(f"sample_index must be non-negative, got {sample_index}")

    asset_rows: dict[str, list[tuple[int, dict]]] = {}
    for manifest_index, (sample_uuid, record) in enumerate(dataset.records.items()):
        row = {"uuid": sample_uuid, **record}
        asset_id = str(row.get("base_asset_id") or sample_uuid)
        asset_rows.setdefault(asset_id, []).append((manifest_index, row))

    eligible_index = 0
    oversized_count = 0
    for rows in asset_rows.values():
        eligible_pose = None
        for manifest_index, row in rows:
            try:
                rig, octree_layers = dataset.load_rig_layers_from_row(row)
            except OversizedHumanoidRigError:
                oversized_count += 1
                continue
            eligible_pose = (row, rig, octree_layers, manifest_index)
            break
        if eligible_pose is None:
            continue
        if eligible_index == sample_index:
            return (*eligible_pose, oversized_count)
        eligible_index += 1

    raise IndexError(
        f"Eligible asset index {sample_index} is outside {dataset.manifest_path}; "
        f"found {eligible_index} assets after skipping {oversized_count} oversized poses"
    )


def joint_name(value: object, index: int) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    text = str(value)
    return text if text else f"joint_{index:03d}"


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
    require_paths(
        {
            "stage1_ckpt": stage1_ckpt,
            "stage1_config": stage1_config,
            "manifest": manifest,
        }
    )

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    dataset_reader = build_dataset_reader(config_dict, manifest, args.split)
    row, rig, octree_layers, manifest_index, oversized_before_sample = (
        load_training_eligible_sample(dataset_reader, args.sample_index)
    )
    sample_uuid = str(row["uuid"])
    view_index = int(dataset_kwargs.get("view_indices", [0])[0])
    grid_size = int(dataset_kwargs["grid_size"])
    max_depth = int(dataset_kwargs["max_depth"])
    image_size = int(dataset_kwargs["image_resolution"])

    gt_joints = rig.joints
    gt_joint_ids = rig.joint_ids.cpu().numpy()
    gt_parents = rig.parents.cpu().numpy()

    with np.load(
        dataset_reader._read_uri(str(row["rig_npz_uri"])), allow_pickle=True
    ) as raw_rig:
        mesh_vertices = np.asarray(raw_rig["vertices"], dtype=np.float32)
        mesh_faces = np.asarray(raw_rig["faces"], dtype=np.int64)
        raw_names = np.asarray(
            raw_rig["names"]
            if "names" in raw_rig
            else [f"joint_{index:03d}" for index in range(len(gt_joints))]
        )
    if len(raw_names) != len(gt_joints):
        raise ValueError(
            f"Expected {len(gt_joints)} joint names, found {len(raw_names)}"
        )
    joint_names = [
        joint_name(value, index) for index, value in enumerate(raw_names.tolist())
    ]

    reference_generator = torch.Generator().manual_seed(args.seed)
    reference_joints_tensor = augment_reference_skeleton(
        torch.from_numpy(gt_joints),
        rig.parents,
        max_local_rotation_degrees=float(
            dataset_kwargs["reference_max_local_rotation_degrees"]
        ),
        bone_length_log_std=float(dataset_kwargs["reference_bone_length_log_std"]),
        root_translation_std=float(
            dataset_kwargs["reference_root_translation_std"]
        ),
        generator=reference_generator,
    ).clamp(-1.0, 1.0)
    reference_joints = reference_joints_tensor.numpy()
    reference_rope_positions = torch.from_numpy(
        discretize(reference_joints, grid_size)
    ).long()
    reference_edge_index = skeleton_edges(rig.parents)
    mesh_layers = [
        SingleStreamTeacherForcedMeshLayer(
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
    pipeline = ImageMeshReferenceSkeletonToSingleStreamJointOctreePipeline(
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
        joint_ids=rig.joint_ids,
        reference_joint_positions=reference_joints_tensor,
        reference_joint_rope_positions=reference_rope_positions,
        reference_edge_index=reference_edge_index,
    )

    predicted_joints = undiscretize(result.joints.cpu().numpy(), grid_size)
    if not np.array_equal(result.joint_ids.cpu().numpy(), gt_joint_ids):
        raise RuntimeError("Pipeline output joint IDs do not match the requested subset")
    joint_errors = np.linalg.norm(predicted_joints - gt_joints, axis=1)
    np.save(out_dir / "predicted_joints_nexus_normalized.npy", predicted_joints)
    np.save(out_dir / "predicted_joint_ids.npy", gt_joint_ids)

    mesh_space_joints = mesh_space_from_nexus(predicted_joints, mesh_vertices)
    mesh_space_gt_joints = mesh_space_from_nexus(gt_joints, mesh_vertices)
    mesh_space_reference_joints = mesh_space_from_nexus(
        reference_joints, mesh_vertices
    )
    skeleton_path = out_dir / "predicted_skeleton_mesh_space.npz"
    np.savez_compressed(
        skeleton_path,
        positions=mesh_space_joints,
        joint_ids=gt_joint_ids,
        joint_names=np.asarray(joint_names),
        parents=gt_parents,
        uuid=np.asarray(sample_uuid),
        coordinate_space=np.asarray("rig_npz_and_mesh_glb"),
        rig_npz_uri=np.asarray(str(row["rig_npz_uri"])),
        mesh_glb_uri=np.asarray(str(row["mesh_glb_uri"])),
    )

    np.save(out_dir / "predicted_joints.npy", predicted_joints)
    np.save(out_dir / "reference_joints.npy", reference_joints)
    np.save(out_dir / "gt_joints.npy", gt_joints)

    predicted_multiview, predicted_views = render_prediction_multiview(
        mesh_vertices,
        mesh_faces,
        mesh_space_joints,
        gt_parents.tolist(),
        out_dir / "predicted",
    )
    _, reference_views = render_prediction_multiview(
        mesh_vertices,
        mesh_faces,
        mesh_space_reference_joints,
        gt_parents.tolist(),
        out_dir / "reference",
    )
    _, gt_views = render_prediction_multiview(
        mesh_vertices,
        mesh_faces,
        mesh_space_gt_joints,
        gt_parents.tolist(),
        out_dir / "gt",
    )
    glb_path = out_dir / "mesh_with_predicted_skeleton.glb"
    export_mesh_skeleton_glb(
        mesh_vertices,
        mesh_faces,
        mesh_space_joints,
        gt_parents.tolist(),
        glb_path,
    )

    if args.prediction_only_multiview:
        panels = [
            ("condition", condition_path),
            ("predicted left", predicted_views["left"]),
            ("predicted front", predicted_views["front"]),
            ("predicted right", predicted_views["right"]),
        ]
    else:
        panels = [
            (f"{args.split} condition", condition_path),
            ("distorted reference", reference_views["front"]),
            ("predicted target", predicted_views["front"]),
            ("GT target", gt_views["front"]),
        ]
    contact_sheet = out_dir / "contact_sheet.png"
    make_contact_sheet(panels, contact_sheet)

    summary = {
        "sample_uuid": sample_uuid,
        "base_asset_id": str(row.get("base_asset_id") or sample_uuid),
        "split": args.split,
        "eligible_asset_index": args.sample_index,
        "manifest_row_index": manifest_index,
        "oversized_poses_skipped_before_asset": oversized_before_sample,
        "stage1_ckpt": str(stage1_ckpt),
        "stage1_config": str(stage1_config),
        "teacher_forced_mesh": True,
        "mesh_layers": len(mesh_layers),
        "mesh_prediction_used": False,
        "reference_augmentation_seed": args.seed,
        "joint_count": int(predicted_joints.shape[0]),
        "evaluated_joint_ids": gt_joint_ids.tolist(),
        "mean_joint_error": float(joint_errors.mean()),
        "max_joint_error": float(joint_errors.max()),
        "per_joint_error": joint_errors.tolist(),
        "condition_image": str(condition_path),
        "predicted_multiview": str(predicted_multiview),
        "mesh_with_predicted_skeleton": str(glb_path),
        "predicted_skeleton": str(skeleton_path),
        "contact_sheet": str(contact_sheet),
        "source": str(row.get("source", "")),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
