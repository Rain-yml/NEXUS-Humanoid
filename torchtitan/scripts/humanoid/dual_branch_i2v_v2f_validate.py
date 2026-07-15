#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.distributed.checkpoint as dcp
import trimesh
from PIL import Image, ImageDraw

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from diffusers.schedulers import (
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
)

import torchtitan.experiments.humanoid.train_spec  # noqa: F401 - register humanoid train spec
import torchtitan.experiments.vem  # noqa: F401 - register VEM train specs
from torchtitan.experiments.vem.datasets.octree_utils import (
    undiscretize,
)
from torchtitan.experiments.humanoid.data.bos import BOSClient
from torchtitan.experiments.humanoid.data.manifest import parse_bos_uri
from torchtitan.experiments.vem.image_encoder import (
    DINOv2ImageEncoder,
    DINOv2ImageEncoderWithoutPooler,
    DINOv3ImageEncoder,
    SigLIP2ImageEncoder,
)
from torchtitan.experiments.vem.models.transformer import to_dtype_except_keep_precision
from torchtitan.experiments.humanoid.pipelines.image_to_dual_branch_octree import (
    ImageToDualBranchOctreePipeline,
)
from torchtitan.experiments.vem.pipelines.v2f import V2FPipeline
import torchtitan.protocols.train_spec as train_spec_module


@dataclass
class RunPaths:
    stage1_ckpt: Path
    stage1_config: Path
    stage2_ckpt: Path
    stage2_config: Path
    encoder_ckpt: Path
    latent_stats: Path
    manifest: Path
    output_dir: Path


class DictToObj:
    def __init__(self, dict_: dict[str, Any]):
        for key, value in dict_.items():
            if isinstance(value, dict):
                value = DictToObj(value)
            setattr(self, key, value)


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def require_paths(paths: dict[str, Path]) -> None:
    missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def latest_dcp_checkpoint(ckpt_root: Path) -> Path:
    candidates = []
    for path in ckpt_root.glob("step-*"):
        if path.is_dir() and (path / ".metadata").exists():
            try:
                step = int(path.name.split("-", 1)[1])
            except ValueError:
                continue
            candidates.append((step, path))
    if not candidates:
        raise FileNotFoundError(f"No complete DCP step-* checkpoints found in {ckpt_root}")
    return max(candidates, key=lambda item: item[0])[1]


def load_parquet_sample(manifest: Path, split: str, sample_index: int) -> dict[str, Any]:
    frame = pd.read_parquet(manifest)
    frame = frame.loc[frame["split"] == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows found for split={split!r} in {manifest}")
    return frame.iloc[sample_index % len(frame)].to_dict()


def read_uri(uri: str, bos_client: BOSClient) -> io.BytesIO:
    if uri.startswith("bos://"):
        bucket, key = parse_bos_uri(uri)
        return bos_client.get_file(bucket, key)
    if uri.startswith("file://"):
        uri = uri.removeprefix("file://")
    return io.BytesIO(Path(uri).read_bytes())


def load_condition_image(uri: str, image_size: int, bos_client: BOSClient) -> torch.Tensor:
    image = Image.open(read_uri(uri, bos_client)).convert("RGBA")
    image = image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + 0.5 * (1 - image_np[:, :, 3:4])
    return torch.from_numpy(image_np).permute(2, 0, 1).clamp(0, 1).unsqueeze(0)


def save_tensor_image(image: torch.Tensor, path: Path) -> None:
    image_np = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    Image.fromarray((image_np * 255).clip(0, 255).astype(np.uint8)).save(path)


def prepare_image_encoder(
    image_encoder_type: str,
    image_encoder_model: str,
    image_encoder_return_the_nth_hidden_states: int,
    device,
    dtype,
):
    if image_encoder_type == "dinov2":
        image_encoder = DINOv2ImageEncoder(
            model_name=image_encoder_model,
            return_the_nth_hidden_states=image_encoder_return_the_nth_hidden_states,
        ).to(device=device, dtype=dtype)
    elif image_encoder_type == "dinov3":
        image_encoder = DINOv3ImageEncoder(
            model_name=image_encoder_model,
            return_the_nth_hidden_states=image_encoder_return_the_nth_hidden_states,
        ).to(device=device, dtype=dtype)
    elif image_encoder_type == "dinov2_without_pooler":
        image_encoder = DINOv2ImageEncoderWithoutPooler(
            model_name=image_encoder_model,
            return_the_nth_hidden_states=image_encoder_return_the_nth_hidden_states,
        ).to(device=device, dtype=dtype)
    elif image_encoder_type == "siglip2":
        image_encoder = SigLIP2ImageEncoder(
            model_name=image_encoder_model,
            return_the_nth_hidden_states=image_encoder_return_the_nth_hidden_states,
        ).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Invalid image encoder type: {image_encoder_type}")
    image_encoder.eval()
    return image_encoder


def load_dcp_state(state: dict[str, Any], checkpoint_path: Path) -> None:
    try:
        dcp.load(state, checkpoint_id=str(checkpoint_path), no_dist=True)
    except TypeError:
        dcp.load(state, checkpoint_id=str(checkpoint_path))


def load_model(
    model_ckpt_path: Path,
    config_path: Path,
    device,
    *,
    dtype: torch.dtype,
    ema: bool = False,
    init_weights: bool = True,
    skip_pretrained_init: bool = False,
    load_checkpoint: bool = True,
    pretrained_init_path: Optional[Path] = None,
):
    config_dict = read_toml(config_path)
    config = DictToObj(config_dict)
    train_spec = train_spec_module.get_train_spec(config.model.name)

    model_args = train_spec.config[config.model.flavor]
    model = train_spec.cls.from_model_args(model_args)
    if pretrained_init_path is not None:
        model.config.pretrained_path = str(pretrained_init_path)
    if skip_pretrained_init and hasattr(model, "config") and hasattr(model.config, "pretrained_path"):
        model.config.pretrained_path = None
    if init_weights and hasattr(model, "init_weights"):
        model.init_weights(buffer_device=device)

    trainable_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    to_bf16_function = partial(to_dtype_except_keep_precision, dtype)
    model.apply(to_bf16_function)
    model.to(device)
    model.requires_grad_(False)
    model.eval()

    if load_checkpoint:
        if model_ckpt_path.is_dir():
            if ema:
                state = {"model": model.state_dict()}
                load_dcp_state(state, model_ckpt_path)
                model.load_state_dict(state["model"])
                model_state = model.state_dict()
                ema_state = {
                    "ema": {
                        "model": {name: model_state[name] for name in trainable_names}
                    }
                }
                load_dcp_state(ema_state, model_ckpt_path)
                model.load_state_dict(ema_state["ema"]["model"], strict=False)
            else:
                state = {"model": model.state_dict()}
                load_dcp_state(state, model_ckpt_path)
                model.load_state_dict(state["model"])
        else:
            ckpt = torch.load(model_ckpt_path, map_location="cpu")
            if ema:
                if "model" in ckpt:
                    model.load_state_dict(ckpt["model"])
                model.load_state_dict(ckpt["ema"]["model"], strict=False)
            elif "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)

    image_encoder = prepare_image_encoder(
        config.training.image_encoder_type,
        config.training.image_encoder_model,
        config.training.image_encoder_return_the_nth_hidden_states,
        device=device,
        dtype=torch.float32,
    )
    return config, config_dict, model, image_encoder, train_spec


def make_scheduler(name: str, shift: float = 1.0):
    if name == "heun":
        return FlowMatchHeunDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    if name == "euler":
        return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    if name == "dpm":
        return DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=shift,
        )
    raise ValueError(f"Unsupported scheduler {name!r}")


def nx_all_triangles(graph, nbunch=None):
    if nbunch is None:
        nbunch = relevant_nodes = graph
    else:
        nbunch = dict.fromkeys(graph.nbunch_iter(nbunch))
        relevant_nodes = chain(
            nbunch,
            (nbr for node in nbunch for nbr in graph.neighbors(node) if nbr not in nbunch),
        )

    node_to_id = {node: i for i, node in enumerate(relevant_nodes)}
    triangles = []
    for u in nbunch:
        u_id = node_to_id[u]
        u_nbrs = graph._adj[u].keys()
        for v in u_nbrs:
            v_id = node_to_id.get(v, -1)
            if v_id <= u_id:
                continue
            v_nbrs = graph._adj[v].keys()
            for w in v_nbrs & u_nbrs:
                if node_to_id.get(w, -1) > v_id:
                    triangles.append((u, v, w))
    return np.array(triangles)


def faces_from_vertices_embed(node_embed, st_decoder, node_position=None):
    node_embed = node_embed.to(dtype=st_decoder.dec_proj_in.weight.dtype)
    cu_seqlens = torch.tensor([0, node_embed.shape[0]], device=node_embed.device, dtype=torch.int32)
    st_feat, hidden = st_decoder.forward_decoder(node_embed, cu_seqlens, position=node_position)
    orient_embed = st_decoder.dec_orient(hidden)

    if st_decoder.face_dim_split:
        edge_feat = st_feat[:, :-st_decoder.face_dim]
        face_feat = st_feat[:, -st_decoder.face_dim:]
    else:
        edge_feat = st_feat
        face_feat = st_feat

    num_vertices = node_embed.shape[0]
    e1, e2 = torch.triu_indices(num_vertices, num_vertices, offset=1, device=st_feat.device)
    candidate_edges = torch.stack([e1, e2], dim=-1)
    d_edge = st_decoder.edge_loss.spacetime_distance(edge_feat, candidate_edges)
    edges = candidate_edges[d_edge > 0].cpu().numpy()

    graph = nx.Graph()
    graph.add_edges_from(edges)
    candidate_triangles = nx_all_triangles(graph)
    if candidate_triangles.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    candidate_triangles = np.sort(candidate_triangles, axis=1)
    candidate_triangles = torch.from_numpy(candidate_triangles).to(dtype=torch.long, device=st_feat.device)

    d_face = st_decoder.face_loss.spacetime_area(face_feat, candidate_triangles)
    faces = candidate_triangles[d_face > 0]
    if faces.numel() == 0:
        return np.empty((0, 3), dtype=np.int64)

    face_orient = st_decoder.orient_loss.orient(orient_embed, faces)
    correct_mask = face_orient > 0
    faces[~correct_mask] = torch.flip(faces[~correct_mask], dims=[1])
    return faces.cpu().numpy()


def render_mesh_preview(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: Path,
    *,
    joints: Optional[np.ndarray] = None,
    joint_parents: Optional[list[int]] = None,
    size: int = 768,
) -> None:
    image = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    if vertices.size == 0:
        image.save(path)
        return

    vertices = vertices.astype(np.float32, copy=False)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) * 0.5
    scale = float((maxs - mins).max())
    if scale <= 1e-6:
        scale = 1.0
    verts = (vertices - center) / scale
    xy = verts[:, [0, 2]]
    xy[:, 1] *= -1
    z = verts[:, 1]
    xy = xy * (size * 0.72) + size * 0.5

    if faces.size:
        faces = faces.astype(np.int64, copy=False)
        valid = (faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1)
        faces = faces[valid]
        order = np.argsort(z[faces].mean(axis=1))
        light = np.array([0.2, -0.4, 0.9], dtype=np.float32)
        light /= np.linalg.norm(light)
        for face in faces[order]:
            pts3 = verts[face]
            normal = np.cross(pts3[1] - pts3[0], pts3[2] - pts3[0])
            norm = np.linalg.norm(normal)
            shade = 0.55
            if norm > 1e-6:
                normal = normal / norm
                shade = 0.45 + 0.4 * max(float(np.dot(normal, light)), 0.0)
            color = tuple(int(c * shade) for c in (118, 150, 214))
            pts2 = [tuple(p) for p in xy[face]]
            draw.polygon(pts2, fill=color)
            draw.line(pts2 + [pts2[0]], fill=(72, 92, 130), width=1)
    else:
        for p in xy:
            x, y = float(p[0]), float(p[1])
            draw.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=(78, 110, 180))

    if joints is not None and joints.shape[0] > 0 and joint_parents is not None:
        joints_normalized = (joints.astype(np.float32, copy=False) - center) / scale
        joints_xy = joints_normalized[:, [0, 2]]
        joints_xy[:, 1] *= -1
        joints_xy = joints_xy * (size * 0.72) + size * 0.5
        for joint_id, parent_id in enumerate(joint_parents):
            if parent_id < 0:
                continue
            start = tuple(float(value) for value in joints_xy[parent_id])
            end = tuple(float(value) for value in joints_xy[joint_id])
            draw.line([start, end], fill=(245, 190, 35), width=5)
        for joint_id, point in enumerate(joints_xy):
            x, y = float(point[0]), float(point[1])
            radius = 5 if joint_id else 7
            fill = (220, 45, 45) if joint_id == 0 else (40, 205, 105)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(25, 25, 25), width=2)
    image.save(path)


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)


def make_contact_sheet(paths: list[tuple[str, Path]], out_path: Path) -> None:
    tile_w, tile_h = 512, 560
    sheet = Image.new("RGB", (tile_w * len(paths), tile_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(paths):
        img = Image.open(path).convert("RGB")
        img.thumbnail((tile_w, tile_w))
        x = i * tile_w + (tile_w - img.width) // 2
        y = 36 + (tile_w - img.height) // 2
        sheet.paste(img, (x, y))
        draw.text((i * tile_w + 16, 12), label, fill=(0, 0, 0))
    sheet.save(out_path)


def resolve_paths(args: argparse.Namespace) -> RunPaths:
    stage1_root = Path(args.stage1_output_root)
    stage1_ckpt = Path(args.stage1_ckpt) if args.stage1_ckpt else latest_dcp_checkpoint(stage1_root / "ckpts")
    stage1_config = Path(args.stage1_config)
    stage1_config_dict = read_toml(stage1_config)
    manifest = Path(args.manifest or stage1_config_dict["training"]["dataset_kwargs"]["manifest_path"])

    return RunPaths(
        stage1_ckpt=stage1_ckpt,
        stage1_config=stage1_config,
        stage2_ckpt=Path(args.stage2_ckpt),
        stage2_config=Path(args.stage2_config),
        encoder_ckpt=Path(args.encoder_ckpt),
        latent_stats=Path(args.latent_stats),
        manifest=manifest,
        output_dir=Path(args.output_dir),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dual-branch humanoid stage 1 and the unchanged NEXUS stage 2."
    )
    parser.add_argument(
        "--stage1-output-root", default="./outputs/humanoid_dual_branch_front_overfit"
    )
    parser.add_argument("--stage1-ckpt", default="")
    parser.add_argument(
        "--stage1-config",
        default="torchtitan/experiments/humanoid/configs/dual_branch/front_overfit.toml",
    )
    parser.add_argument("--stage1-ema", action="store_true", default=False)
    parser.add_argument("--stage2-ckpt", default="/mnt/pfs/users/liyumeng/checkpoints/nexus/2B_16kl_rope_reg4_pack64_tokavg_mv/ckpts/step-200000")
    parser.add_argument("--stage2-config", default="/mnt/pfs/users/liyumeng/checkpoints/nexus/2B_16kl_rope_reg4_pack64_tokavg_mv/config.toml")
    parser.add_argument("--no-stage2-ema", dest="stage2_ema", action="store_false")
    parser.add_argument("--encoder-ckpt", default="/mnt/pfs/users/liyumeng/checkpoints/nexus/muon_16kl_zero_nobias_rope_pack/ckpts/step-40000.pth")
    parser.add_argument("--latent-stats", default="/mnt/pfs/users/liyumeng/checkpoints/nexus/muon_16kl_zero_nobias_rope_pack/ckpts/latent_stat_nf20k_noaug_step-40000.pth")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--joint-schema", default="torchtitan/experiments/humanoid/data/humanoid_28_v1.json")
    parser.add_argument(
        "--output-dir",
        default="./outputs/humanoid_dual_branch_front_overfit/validation_i2v_v2f",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--stage1-steps", type=int, default=20)
    parser.add_argument("--stage2-steps", type=int, default=20)
    parser.add_argument("--stage1-cfg", type=float, default=1.0)
    parser.add_argument("--stage2-cfg", type=float, default=2.5)
    parser.add_argument("--stage1-scheduler", choices=["euler", "heun", "dpm"], default="euler")
    parser.add_argument("--stage2-scheduler", choices=["euler", "heun", "dpm"], default="euler")
    parser.add_argument("--stage2-shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=0)
    parser.add_argument("--skip-stage2", action="store_true")
    parser.set_defaults(stage2_ema=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    require_paths(
        {
            "stage1_ckpt": paths.stage1_ckpt,
            "stage1_config": paths.stage1_config,
            "stage2_ckpt": paths.stage2_ckpt,
            "stage2_config": paths.stage2_config,
            "encoder_ckpt": paths.encoder_ckpt,
            "latent_stats": paths.latent_stats,
            "manifest": paths.manifest,
            "joint_schema": Path(args.joint_schema),
        }
    )

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    stage1_cfg = read_toml(paths.stage1_config)
    dataset_kwargs = stage1_cfg["training"]["dataset_kwargs"]
    max_depth = args.max_depth or int(dataset_kwargs["max_depth"])
    grid_size = args.grid_size or int(dataset_kwargs["grid_size"])
    row = load_parquet_sample(paths.manifest, args.split, args.sample_index)
    view_index = int(dataset_kwargs.get("view_indices", [0])[0])
    bos_client = BOSClient()
    schema = json.loads(Path(args.joint_schema).read_text(encoding="utf-8"))

    out_dir = paths.output_dir / f"{paths.stage1_ckpt.name}_{args.split}_{args.sample_index:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    condition_image = load_condition_image(
        str(row[f"color_view_{view_index}_uri"]), args.image_size, bos_client
    )
    condition_path = out_dir / "condition.png"
    save_tensor_image(condition_image, condition_path)

    num_vertices = int(row["num_vertices"])

    print(f"Loading stage1 model: {paths.stage1_ckpt}")
    stage1_config, _, stage1_model, stage1_image_encoder, _ = load_model(
        paths.stage1_ckpt,
        paths.stage1_config,
        device,
        dtype=dtype,
        ema=args.stage1_ema,
        init_weights=True,
    )
    stage1_prediction = "v" if stage1_config.training.loss_type == "vpred-vloss" else "x"
    stage1_pipe = ImageToDualBranchOctreePipeline(
        image_encoder=stage1_image_encoder,
        octree_dit=stage1_model,
        scheduler=None,
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        stage1_kwargs = dict(
            image=condition_image,
            scheduler=make_scheduler(args.stage1_scheduler),
            device=device,
            num_inference_steps=args.stage1_steps,
            guidance_scale=args.stage1_cfg,
            generator=generator,
            num_vertices=num_vertices,
            enable_progress=True,
            dtype=dtype,
            grid_size=grid_size,
            max_depth=max_depth,
            prediction=stage1_prediction,
            view_indices=[view_index],
        )
        stage1_kwargs["num_joint_tokens"] = len(schema["joints"])
        stage1 = stage1_pipe(**stage1_kwargs)

    pred_vertices_discrete = stage1.vertices.detach()
    pred_vertices = undiscretize(pred_vertices_discrete.cpu().numpy(), grid_size)
    pred_joints_discrete = stage1.joints.detach()
    pred_joints = undiscretize(pred_joints_discrete.cpu().numpy(), grid_size)
    joints_path = out_dir / "stage1_joints.npy"
    np.save(joints_path, pred_joints)
    vertex_cloud_path = out_dir / "stage1_vertices.ply"
    vertex_preview_path = out_dir / "stage1_vertices_preview.png"
    if pred_vertices.shape[0] > 0:
        trimesh.PointCloud(pred_vertices).export(vertex_cloud_path)
    else:
        vertex_cloud_path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n",
            encoding="utf-8",
        )
    render_mesh_preview(
        pred_vertices,
        np.empty((0, 3), dtype=np.int64),
        vertex_preview_path,
        joints=pred_joints,
        joint_parents=schema["parents"],
    )

    summary = {
        "sample_uuid": str(row["uuid"]),
        "split": args.split,
        "stage1_ckpt": str(paths.stage1_ckpt),
        "stage1_config": str(paths.stage1_config),
        "stage2_ckpt": str(paths.stage2_ckpt),
        "stage2_config": str(paths.stage2_config),
        "encoder_ckpt": str(paths.encoder_ckpt),
        "latent_stats": str(paths.latent_stats),
        "condition_image": str(condition_path),
        "stage1_vertices": str(vertex_cloud_path),
        "stage1_joints": str(joints_path),
        "stage1_vertex_count": int(pred_vertices_discrete.shape[0]),
        "conditioned_raw_vertex_count": num_vertices,
        "stage1_joint_count": int(pred_joints_discrete.shape[0]),
        "stage1_steps": args.stage1_steps,
        "stage2_steps": args.stage2_steps,
        "stage1_cfg": args.stage1_cfg,
        "stage2_cfg": args.stage2_cfg,
    }

    generated_preview_path = None
    generated_mesh_path = None
    if not args.skip_stage2 and pred_vertices_discrete.shape[0] > 2:
        print(f"Loading stage2 model: {paths.stage2_ckpt}")
        stage2_config, _, stage2_model, stage2_image_encoder, stage2_train_spec = load_model(
            paths.stage2_ckpt,
            paths.stage2_config,
            device,
            dtype=dtype,
            ema=args.stage2_ema,
            init_weights=True,
            skip_pretrained_init=True,
        )
        encoder_args = stage2_train_spec.encoder_config[stage2_config.training.encoder_flavor]
        encoder = stage2_train_spec.encoder_cls.from_model_args(encoder_args).to(device=device, dtype=dtype).eval()
        if hasattr(encoder, "init_weights"):
            encoder.init_weights(buffer_device=device)
        encoder_state = torch.load(paths.encoder_ckpt, map_location="cpu")
        encoder.load_state_dict(encoder_state["model"], strict=True)
        encoder.requires_grad_(False)
        encoder.eval()

        stats = torch.load(paths.latent_stats, map_location="cpu")
        latent_mean = stats["mean_ch"].view(1, -1).to(device=device)
        latent_std = stats["std_ch"].view(1, -1).to(device=device)
        stage2_prediction = "v" if stage2_config.training.loss_type == "vpred-vloss" else "x"
        stage2_pipe = V2FPipeline(
            sm_dit=stage2_model,
            image_encoder=stage2_image_encoder,
            latent_mean=latent_mean,
            latent_std=latent_std,
            scheduler=None,
        ).to(device)

        stage2_scheduler = make_scheduler(args.stage2_scheduler, shift=args.stage2_shift)
        if getattr(stage2_config.scheduler, "dynamic_shift", False):
            dynamic_shift = math.sqrt(pred_vertices_discrete.shape[0] / stage2_config.scheduler.shift_base)
            stage2_scheduler = make_scheduler(args.stage2_scheduler, shift=dynamic_shift)
            summary["stage2_dynamic_shift"] = dynamic_shift

        with torch.inference_mode():
            stage2 = stage2_pipe(
                vertices=pred_vertices_discrete,
                image=condition_image,
                guidance_scale=args.stage2_cfg,
                num_inference_steps=args.stage2_steps,
                prediction=stage2_prediction,
                device=device,
                dtype=dtype,
                generator=generator,
                scheduler=stage2_scheduler,
                view_indices=[view_index],
            )

        node_position = pred_vertices_discrete * 3 if getattr(encoder, "has_rope", False) else None
        faces_pred = faces_from_vertices_embed(stage2.vertices_embed, encoder, node_position=node_position)
        generated_mesh_path = out_dir / "generated.obj"
        generated_preview_path = out_dir / "generated_mesh_joints_overlay.png"
        write_mesh(generated_mesh_path, pred_vertices, faces_pred)
        render_mesh_preview(
            pred_vertices,
            faces_pred,
            generated_preview_path,
            joints=pred_joints,
            joint_parents=schema["parents"],
        )
        summary.update(
            {
                "generated_mesh": str(generated_mesh_path),
                "generated_preview": str(generated_preview_path),
                "generated_faces": int(faces_pred.shape[0]),
            }
        )

    sheet_items = [
        ("test condition (front)", condition_path),
        ("stage1 points + joints", vertex_preview_path),
    ]
    if generated_preview_path is not None:
        sheet_items.append(("generated mesh + joints", generated_preview_path))
    contact_sheet_path = out_dir / "contact_sheet.png"
    make_contact_sheet(sheet_items, contact_sheet_path)
    summary["contact_sheet"] = str(contact_sheet_path)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
