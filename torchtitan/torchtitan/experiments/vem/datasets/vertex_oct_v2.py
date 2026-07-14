from __future__ import annotations

import gc
import random
import traceback
from typing import List, Optional

import numpy as np
import torch
import trimesh
from PIL import Image
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchvision.transforms import v2 as transforms_v2

from torchtitan.experiments.vem.datasets.mesh_utils import MeshProcessor, rand_int_with_pt, rand_with_pt
from torchtitan.experiments.vem.datasets.octree_utils import (
    OctreeBatch,
    OctreeData,
    build_octree_by_layer,
    build_octree_specific_layer,
    discretize,
)
from torchtitan.experiments.vem.datasets.path_io import DatasetPathIO
from torchtitan.tools.logging import logger


def shard_interleave(l, num_shards: int, index: int):
    if not 0 <= index < num_shards:
        raise ValueError("index should be in [0, num_shards-1]")
    return [l[i] for i in range(len(l)) if i % num_shards == index]


class _VertexOctreeV2Base(IterableDataset, Stateful):
    _metadata_columns = {
        "uuid",
        "mesh_path",
        "image_dir",
        "num_vertices",
        "num_faces",
    }

    def _init_common(
        self,
        metadata_parquet: str,
        repeats: int,
        shuffle_seed: int,
        grid_size: int,
        max_depth: int,
        yup_to_zup: bool,
        cache_instances: bool,
        image_resolution: int,
        drop_image_rate: float,
        alignment: str,
        infinite: bool,
        dp_rank: int,
        dp_world_size: int,
        mv_uuids_json: Optional[str],
        mv_root: str,
        mv_prob: float,
        mv_num_views_probs: Optional[List[float]],
        symmetry_condition: bool = False,
        symmetry_drop_rate: float = 0.1,
        quad_ratio_uncond_rate: float = 0.1,
    ) -> None:
        self.path_io = DatasetPathIO()
        self.metadata_parquet = metadata_parquet
        self.repeats = repeats
        self.shuffle_seed = shuffle_seed
        self.infinite = infinite
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.sample_idx = 0

        self.alignment = alignment
        assert self.alignment in ["none", "near_align"]
        self.grid_size = grid_size
        self.max_depth = max_depth
        assert grid_size == 2**max_depth
        self.yup_to_zup = yup_to_zup
        self.image_resolution = image_resolution
        self.drop_image_rate = drop_image_rate

        self.symmetry_condition = symmetry_condition
        self.symmetry_drop_rate = symmetry_drop_rate
        self.quad_ratio_uncond_rate = quad_ratio_uncond_rate
        # Symmetry labels live in extra parquet columns; only require them when used.
        if symmetry_condition:
            self._metadata_columns = self._metadata_columns | {
                "symmetry_x",
                "symmetry_y",
                "symmetry_z",
            }

        self.mv_prob = mv_prob
        self.mv_root = mv_root
        self.mv_num_views_probs = mv_num_views_probs if mv_num_views_probs is not None else [0.2, 0.4, 0.4]
        assert len(self.mv_num_views_probs) == 3, "mv_num_views_probs must have 3 elements: P(2), P(3), P(4)"
        if mv_uuids_json is not None:
            mv_uuids_data = self.path_io.read_json(mv_uuids_json)
            self.mv_uuids = set(mv_uuids_data if isinstance(mv_uuids_data, list) else mv_uuids_data["uuids"])
            logger.info(f"Loaded {len(self.mv_uuids)} UUIDs with mv renders from {mv_uuids_json}")
        else:
            self.mv_uuids = None

        self.records = self._load_metadata(metadata_parquet)
        self.mp = MeshProcessor()

        identity = transforms_v2.Lambda(lambda x: x)
        self.transform = transforms_v2.Compose(
            [
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(hue=0.3),
                        transforms_v2.RandomChoice(
                            [
                                transforms_v2.GaussianBlur(kernel_size=(3, 7)),
                                identity,
                            ]
                        ),
                        transforms_v2.RandomChoice(
                            [
                                transforms_v2.JPEG(quality=(20, 80)),
                                identity,
                            ]
                        ),
                    ],
                    p=0.5,
                )
            ]
        )

        print("VertexOctreeV2Dataset initialized")
        print("grid_size", grid_size, "max_depth", max_depth)
        print("image_resolution:", image_resolution, "drop_image_rate:", drop_image_rate)
        print("dp_rank", dp_rank, "dp_world_size", dp_world_size)

    def _load_metadata(self, metadata_parquet: str) -> dict[str, dict]:
        df = self.path_io.read_parquet(metadata_parquet)
        missing_columns = self._metadata_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in {metadata_parquet}: {sorted(missing_columns)}")
        if df["uuid"].duplicated().any():
            duplicated = df.loc[df["uuid"].duplicated(), "uuid"].head(5).tolist()
            raise ValueError(f"Duplicate uuid rows in {metadata_parquet}: {duplicated}")

        return df.set_index("uuid").to_dict(orient="index")

    def _filter_uuids(
        self,
        num_face_range: Optional[List[int]],
        num_vertex_range: Optional[List[int]],
    ) -> list[str]:
        items = []
        for uuid, rec in self.records.items():
            if num_face_range is not None and not (num_face_range[0] <= rec["num_faces"] <= num_face_range[1]):
                continue
            if num_vertex_range is not None and not (num_vertex_range[0] <= rec["num_vertices"] <= num_vertex_range[1]):
                continue
            items.append(uuid)
        return items

    def process_image(self, image, image_size):
        image_np = np.array(image).astype(np.float32) / 255.0
        mask = image_np[..., 3]
        height, width = mask.shape

        pixels_y, pixels_x = np.where(mask > 0)

        if len(pixels_y) > 0:
            h0, h1 = max(pixels_y.min() - 1, 0), min(pixels_y.max() + 1, height)
            w0, w1 = max(pixels_x.min() - 1, 0), min(pixels_x.max() + 1, width)

            crop_edge_prob = 0.05
            crop_h0, crop_h1, crop_w0, crop_w1 = (np.random.random(4) < crop_edge_prob).tolist()
            height_fg, width_fg = h1 - h0, w1 - w0
            crop_edge_max_ratio = 0.02
            crop_h0_pixels, crop_h1_pixels = (
                (np.random.random(2) * height_fg * crop_edge_max_ratio).astype(np.int32).tolist()
            )
            crop_w0_pixels, crop_w1_pixels = (
                (np.random.random(2) * width_fg * crop_edge_max_ratio).astype(np.int32).tolist()
            )
            if crop_h0 > 0:
                h0 = h0 + crop_h0_pixels
            if crop_h1 > 0:
                h1 = h1 - crop_h1_pixels
            if crop_w0 > 0:
                w0 = w0 + crop_w0_pixels
            if crop_w1 > 0:
                w1 = w1 - crop_w1_pixels

            height_fg, width_fg = h1 - h0, w1 - w0
            pad_ratio = random.uniform(0.05, 0.2)
            size_padded = int((height_fg if height_fg > width_fg else width_fg) / (1 - pad_ratio))

            image_np_padded = np.zeros((size_padded, size_padded, 4), dtype=np.float32)
            start_h = (size_padded - height_fg) // 2
            start_w = (size_padded - width_fg) // 2
            image_np_padded[start_h : start_h + height_fg, start_w : start_w + width_fg] = image_np[h0:h1, w0:w1]
            image_np = image_np_padded
            mask = image_np[..., 3] > 0

            fg_grayscale = image_np[..., :3][image_np[..., 3] > 0].mean()
            bg_color = np.random.rand(1)
            while bg_color > fg_grayscale - 0.2 and bg_color < fg_grayscale + 0.2:
                bg_color = np.random.rand(1)

            bg_color = np.concatenate([bg_color, bg_color, bg_color])
            image_np = image_np[..., :3] * image_np[..., 3:4] + bg_color[None, None] * (1 - image_np[..., 3:4])
        else:
            image_np = np.zeros_like(image_np[..., :3])
            mask = np.zeros(mask.shape, dtype=bool)

        image = Image.fromarray((image_np * 255.0).clip(0, 255).astype(np.uint8))
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255))
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        mask_image = mask_image.resize((image_size, image_size), Image.Resampling.NEAREST)
        image = self.transform(image)

        image_pt = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_pt = torch.from_numpy(np.array(mask_image) > 0)
        return image, image_pt, mask_pt

    def _get_image(self, rec: dict):
        image_dir = rec["image_dir"]
        meta = self.path_io.read_json(self.path_io.join(image_dir, "meta.json"))

        valid_image_indices = list(range(len(meta["locations"])))
        image_idx = np.random.choice(valid_image_indices)
        transform_matrix = np.array(meta["locations"][image_idx]["transform_matrix"])
        camera_loc = transform_matrix[:3, 3]
        azimuth_deg = np.rad2deg(np.arctan2(camera_loc[1], camera_loc[0]))
        rotation_idx = int(((azimuth_deg + 90 + 45) % 360) // 90)

        image_fn = meta["locations"][image_idx]["frames"][0]["name"]
        image = self.path_io.read_image(self.path_io.join(image_dir, image_fn), mode="RGBA")
        image, image_pt, image_mask = self.process_image(image, self.image_resolution)

        return {
            "image_pil": image,
            "image_pt": image_pt,
            "image_mask": image_mask,
            "rotation_idx": rotation_idx,
        }

    def _get_mv_images(self, uuid: str):
        group = random.randint(0, 1)
        n_views = np.random.choice([2, 3, 4], p=self.mv_num_views_probs)
        view_indices = sorted(random.sample([0, 1, 2, 3], n_views))

        images = []
        masks = []
        for vi in view_indices:
            filename = f"color_{group * 4 + vi:04d}.webp"
            image_path = self.path_io.join(self.mv_root, uuid[:2], uuid, filename)
            img = self.path_io.read_image(image_path, mode="RGBA")
            _, img_pt, mask_pt = self.process_image(img, self.image_resolution)
            images.append(img_pt)
            masks.append(mask_pt)

        return {
            "images": torch.stack(images, dim=0),
            "image_masks": torch.stack(masks, dim=0),
            "view_indices": torch.tensor(view_indices, dtype=torch.long),
        }

    @staticmethod
    def _compute_quad_ratio(mesh_mixed) -> float:
        num_quads = int(np.count_nonzero(mesh_mixed.is_quad))
        num_triangles = int(mesh_mixed.faces.shape[0] - num_quads)
        denom = float(num_triangles + 2 * num_quads)
        return 2.0 * num_quads / denom if denom > 0 else 0.0

    def _build_octree(self, mesh_mixed, layer_id: Optional[int], rotation_transform):
        mesh_tri, _, _ = mesh_mixed.to_triangle_mesh()
        vertices, faces = self.mp.process_vfinput(mesh_tri.vertices, mesh_tri.faces[:, :3], z_up=self.yup_to_zup)
        mesh_process = trimesh.Trimesh(vertices, faces)
        mesh_process = self.mp.clear_mesh(mesh_process, digits_vertex=6)
        if rotation_transform is not None:
            mesh_process.apply_transform(rotation_transform)
        vertices = np.copy(mesh_process.vertices)

        discrete_points = discretize(vertices, self.grid_size)
        discrete_points = np.unique(discrete_points, axis=0)
        discrete_points = torch.from_numpy(discrete_points).long()

        if layer_id is None:
            octree_data = build_octree_by_layer(
                discrete_points=discrete_points,
                grid_size=self.grid_size,
                max_depth=self.max_depth,
            )
        else:
            octree_data = build_octree_specific_layer(
                discrete_points=discrete_points,
                depth=layer_id,
                grid_size=self.grid_size,
                max_depth=self.max_depth,
            )

        return octree_data

    def get_data(self, uuid: str, layer_id: Optional[int] = None):
        rec = self.records[uuid]
        mesh_mixed = self.path_io.read_mixed_mesh(rec["mesh_path"])
        ret = {"instance_id": uuid}
        ret['num_vertices'] = int(rec["num_vertices"])
        ret['num_faces'] = int(rec["num_faces"])
        ret['quad_ratio'] = self._compute_quad_ratio(mesh_mixed)
        # With some probability, mark quad_ratio as "unknown" (negative); the model
        # then uses its unconditional quad-ratio embedding instead.
        if self.quad_ratio_uncond_rate > 0 and rand_with_pt([0, 1]) < self.quad_ratio_uncond_rate:
            ret['quad_ratio'] = -1.0

        rotation_transform = None
        # Number of 90-degree turns about Z applied by near_align (parity is what
        # matters for remapping symmetry planes: odd turns swap the x=0 / y=0 planes).
        near_align_quarter_turns = 0
        use_mv = self.mv_uuids is not None and uuid in self.mv_uuids and random.random() < self.mv_prob

        if use_mv:
            mv = self._get_mv_images(uuid)
            if self.drop_image_rate > 0 and rand_with_pt([0, 1]) < self.drop_image_rate:
                gray = torch.full(
                    (3, self.image_resolution, self.image_resolution),
                    fill_value=0.5,
                    dtype=torch.float32,
                )
                ret["mv_images"] = gray.unsqueeze(0).expand(mv["images"].shape[0], -1, -1, -1).clone()
            else:
                ret["mv_images"] = mv["images"]
            ret["mv_image_masks"] = mv["image_masks"]
            ret["mv_view_indices"] = mv["view_indices"]
            if self.alignment == "near_align":
                rotate_idx = rand_int_with_pt([0, 4])
                ret["mv_view_indices"] = (ret["mv_view_indices"] + rotate_idx) % 4
                rotation_transform = trimesh.transformations.rotation_matrix(np.pi / 2 * rotate_idx, [0, 0, 1])
                near_align_quarter_turns = rotate_idx
        else:
            image_dict = self._get_image(rec)
            if self.drop_image_rate > 0 and rand_with_pt([0, 1]) < self.drop_image_rate:
                ret["image"] = torch.full(
                    (3, self.image_resolution, self.image_resolution),
                    fill_value=0.5,
                    dtype=torch.float32,
                )
            else:
                ret["image"] = image_dict["image_pt"]
            ret["image_mask"] = image_dict["image_mask"]
            if self.alignment == "near_align" and image_dict["rotation_idx"] != 0:
                rotation_transform = trimesh.transformations.rotation_matrix(
                    -np.pi / 2 * image_dict["rotation_idx"],
                    [0, 0, 1],
                )
                near_align_quarter_turns = image_dict["rotation_idx"]

        if self.symmetry_condition:
            ret["symmetries"] = self._build_symmetries(rec, near_align_quarter_turns)

        ret["octree"] = self._build_octree(mesh_mixed, layer_id, rotation_transform)
        return ret

    def _build_symmetries(self, rec: dict, near_align_quarter_turns: int) -> List[int]:
        """Build the per-mesh symmetry token states.

        Reads sym_x/y/z from metadata, remaps the x=0 / y=0 planes if near_align
        applied an odd number of 90-degree Z rotations, derives a 4th "any plane"
        direction (true if any of x/y/z is symmetric), then randomly drops each
        direction to "uncertain".

        Returns a length-4 list of states for directions [x=0, y=0, z=0, any-of-xyz]
        with values 1=symmetric, 0=uncertain.
        """
        sym_x = bool(rec["symmetry_x"])
        sym_y = bool(rec["symmetry_y"])
        sym_z = bool(rec["symmetry_z"])

        # A 90/270-degree rotation about Z swaps the x=0 and y=0 planes; z=0 is
        # invariant. A 180-degree rotation leaves all three planes invariant.
        if near_align_quarter_turns % 2 == 1:
            sym_x, sym_y = sym_y, sym_x

        # 4th direction: symmetric about "some" axis-aligned plane.
        sym_any = sym_x or sym_y or sym_z

        flags = [sym_x, sym_y, sym_z, sym_any]
        states = []
        for is_sym in flags:
            # Only positive symmetry assertions can be dropped to "uncertain";
            # a non-symmetric direction is always reported as uncertain.
            if is_sym and rand_with_pt([0, 1]) < self.symmetry_drop_rate:
                is_sym = False
            states.append(1 if is_sym else 0)
        return states


    def collate_fn(self, batch, train_all_layers=True):
        def flatten_list(l):
            return [item for sublist in l for item in sublist]

        batch = flatten_list(batch)
        octrees: List[OctreeData] = [b["octree"] for b in batch]
        instance_ids = [b["instance_id"] for b in batch]
        min_layers = min(o.num_layers for o in octrees)

        occupancy_list = []
        centers_list = []
        depths_list = []
        seqlens = []
        batch_instance_ids = []
        num_layers_per_mesh = []

        if train_all_layers:
            for sample_idx, octree in enumerate(octrees):
                num_layers = octree.num_layers
                num_layers_per_mesh.append(num_layers)

                for layer_idx in range(num_layers):
                    occ = octree.layer_occupancy[layer_idx] * 2 - 1
                    centers = octree.layer_parent_centers[layer_idx]
                    depth = octree.layer_depths[layer_idx]
                    num_nodes = occ.shape[0]

                    occupancy_list.append(occ)
                    centers_list.append(centers)
                    depths_list.append(torch.full((num_nodes,), depth, dtype=torch.long))
                    seqlens.append(num_nodes)
                    batch_instance_ids.append(instance_ids[sample_idx])
        else:
            layer_idx = random.randint(0, min_layers - 1)
            for sample_idx, octree in enumerate(octrees):
                occ = octree.layer_occupancy[layer_idx] * 2 - 1
                centers = octree.layer_parent_centers[layer_idx]
                depth = octree.layer_depths[layer_idx]
                num_nodes = occ.shape[0]

                occupancy_list.append(occ)
                centers_list.append(centers)
                depths_list.append(torch.full((num_nodes,), depth, dtype=torch.long))
                seqlens.append(num_nodes)
                batch_instance_ids.append(instance_ids[sample_idx])

        occupancy_flat = torch.cat(occupancy_list, dim=0).float()
        centers_flat = torch.cat(centers_list, dim=0)
        depths_flat = torch.cat(depths_list, dim=0)
        total_seqs = len(seqlens)
        cu_seqlens = torch.zeros(total_seqs + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), dim=0)

        images = None
        image_masks = None
        view_indices = None
        mv_cu_seqlens = None
        if "image" in batch[0] or "mv_images" in batch[0]:
            images_flat_list = []
            image_masks_flat_list = []
            view_idx_list = []
            for b in batch:
                if "mv_images" in b:
                    images_flat_list.append(b["mv_images"])
                    image_masks_flat_list.append(b["mv_image_masks"])
                    view_idx_list.append(b["mv_view_indices"])
                else:
                    images_flat_list.append(b["image"].unsqueeze(0))
                    image_masks_flat_list.append(b["image_mask"].unsqueeze(0))
                    view_idx_list.append(torch.zeros(1, dtype=torch.long))
            num_views_per_mesh = [x.shape[0] for x in images_flat_list]
            images = torch.cat(images_flat_list, dim=0)
            image_masks = torch.cat(image_masks_flat_list, dim=0)
            view_indices = torch.cat(view_idx_list, dim=0)
            mv_cu_seqlens = torch.zeros(len(num_views_per_mesh) + 1, dtype=torch.int32)
            mv_cu_seqlens[1:] = torch.cumsum(torch.tensor(num_views_per_mesh, dtype=torch.int32), dim=0)

        uncond_images = torch.full_like(images, 0.5) if images is not None else None
        num_vertices = torch.tensor([b["num_vertices"] for b in batch], dtype=torch.int32)
        num_faces = torch.tensor([b["num_faces"] for b in batch], dtype=torch.int32)
        quad_ratios = torch.tensor([b["quad_ratio"] for b in batch], dtype=torch.float32)
        symmetries = (
            torch.tensor([b["symmetries"] for b in batch], dtype=torch.long)
            if "symmetries" in batch[0]
            else None
        )

        result = OctreeBatch(
            layer_occupancy_flat=occupancy_flat,
            layer_parent_centers_flat=centers_flat,
            layer_depths_flat=depths_flat,
            cu_seqlens=cu_seqlens,
            max_seqlen=max(seqlens),
            layer_idx=-1 if train_all_layers else layer_idx,
            batch_size=total_seqs,
            instance_ids=batch_instance_ids,
            num_layers_per_mesh=num_layers_per_mesh if train_all_layers else None,
            images=images,
            image_masks=image_masks,
            uncond_images=uncond_images,
            num_vertices=num_vertices,
            num_faces=num_faces,
            quad_ratios=quad_ratios,
            symmetries=symmetries,
            view_indices=view_indices,
            mv_cu_seqlens=mv_cu_seqlens,
        )

        del batch
        gc.collect()
        return result


class VertexOctreeRGBGenV2(_VertexOctreeV2Base):
    worker_shard_data = ["instance_ids"]

    def __init__(
        self,
        metadata_parquet: str,
        repeats: int = 1,
        shuffle_seed: int = 0,
        grid_size: int = 128,
        max_depth: int = 7,
        yup_to_zup: bool = False,
        cache_instances: bool = False,
        num_face_range: Optional[List[int]] = [0, 20000],
        num_vertex_range: Optional[List[int]] = [0, 100000],
        image_resolution: int = 518,
        drop_image_rate: float = 0.0,
        alignment: str = "none",
        mv_uuids_json: Optional[str] = None,
        mv_root: str = "bos://mesh-data-shapediff-render-mv",
        mv_prob: float = 0.5,
        mv_num_views_probs: Optional[List[float]] = None,
        infinite: bool = True,
        force_divisible_by: int = 1,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        symmetry_condition: bool = False,
        symmetry_drop_rate: float = 0.1,
        quad_ratio_uncond_rate: float = 0.1,
    ) -> None:
        self._init_common(
            metadata_parquet=metadata_parquet,
            repeats=repeats,
            shuffle_seed=shuffle_seed,
            grid_size=grid_size,
            max_depth=max_depth,
            yup_to_zup=yup_to_zup,
            cache_instances=cache_instances,
            image_resolution=image_resolution,
            drop_image_rate=drop_image_rate,
            alignment=alignment,
            infinite=infinite,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            mv_uuids_json=mv_uuids_json,
            mv_root=mv_root,
            mv_prob=mv_prob,
            mv_num_views_probs=mv_num_views_probs,
            symmetry_condition=symmetry_condition,
            symmetry_drop_rate=symmetry_drop_rate,
            quad_ratio_uncond_rate=quad_ratio_uncond_rate,
        )

        instance_ids = self._filter_uuids(num_face_range, num_vertex_range) * repeats
        if force_divisible_by > 1:
            instance_ids = instance_ids[len(instance_ids) % force_divisible_by :]
            logger.info(f"Instance number after dropping: {len(instance_ids)}")

        instance_ids = shard_interleave(instance_ids, dp_world_size, dp_rank)
        rng = random.Random(shuffle_seed)
        rng.shuffle(instance_ids)
        print("rank", dp_rank, instance_ids[:10])
        self.instance_ids = instance_ids

    def __iter__(self):
        while True:
            sample_idx = self.sample_idx
            while True:
                try:
                    yield [self.get_data(self.instance_ids[sample_idx])]
                    break
                except GeneratorExit:
                    raise
                except Exception:
                    logger.warning(
                        f"Failed to load data for instance {self.instance_ids[sample_idx]}: {traceback.format_exc()}"
                    )
                    sample_idx = random.randint(0, len(self.instance_ids) - 1)

            self.sample_idx += 1
            if self.sample_idx >= len(self.instance_ids):
                if not self.infinite:
                    logger.warning("Dataset has run out of data.")
                    break
                self.sample_idx = 0
                logger.warning("Dataset is being re-looped.")

    def load_state_dict(self, state_dict):
        self.sample_idx = state_dict["sample_idx"]
        self.instance_ids = state_dict["instance_ids"]

    def state_dict(self):
        return {
            "instance_ids": self.instance_ids,
            "sample_idx": self.sample_idx,
        }


class VertexOctreePackRGBGenV2(_VertexOctreeV2Base):
    worker_shard_data = ["batches"]

    def __init__(
        self,
        metadata_parquet: str,
        batches_packed: str,
        repeats: int = 1,
        shuffle_seed: int = 0,
        grid_size: int = 128,
        max_depth: int = 7,
        yup_to_zup: bool = False,
        cache_instances: bool = False,
        image_resolution: int = 518,
        drop_image_rate: float = 0.0,
        alignment: str = "none",
        mv_uuids_json: Optional[str] = None,
        mv_root: str = "bos://mesh-data-shapediff-render-mv",
        mv_prob: float = 0.5,
        mv_num_views_probs: Optional[List[float]] = None,
        infinite: bool = True,
        force_divisible_by: int = 1,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        symmetry_condition: bool = False,
        symmetry_drop_rate: float = 0.1,
        quad_ratio_uncond_rate: float = 0.1,
    ) -> None:
        self._init_common(
            metadata_parquet=metadata_parquet,
            repeats=repeats,
            shuffle_seed=shuffle_seed,
            grid_size=grid_size,
            max_depth=max_depth,
            yup_to_zup=yup_to_zup,
            cache_instances=cache_instances,
            image_resolution=image_resolution,
            drop_image_rate=drop_image_rate,
            alignment=alignment,
            infinite=infinite,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            mv_uuids_json=mv_uuids_json,
            mv_root=mv_root,
            mv_prob=mv_prob,
            mv_num_views_probs=mv_num_views_probs,
            symmetry_condition=symmetry_condition,
            symmetry_drop_rate=symmetry_drop_rate,
            quad_ratio_uncond_rate=quad_ratio_uncond_rate,
        )

        batches_packed_json = self.path_io.read_json(batches_packed)
        batches = batches_packed_json["batches"] * repeats
        if force_divisible_by > 1:
            batches = batches[: len(batches) // force_divisible_by * force_divisible_by]
            logger.info(f"Batch number after dropping: {len(batches)}")

        missing_uuids = sorted({item[0] for batch in batches for item in batch} - set(self.records))
        if missing_uuids:
            raise ValueError(f"Packed batches contain uuid missing from metadata parquet: {missing_uuids[:5]}")

        batches = shard_interleave(batches, dp_world_size, dp_rank)
        rng = random.Random(shuffle_seed)
        rng.shuffle(batches)
        print("rank", dp_rank, batches[:10])
        self.batches = batches

    def __iter__(self):
        while True:
            sample_idx = self.sample_idx
            while True:
                try:
                    yield [self.get_data(uid, layer_id) for (uid, _sid, layer_id) in self.batches[sample_idx]]
                    break
                except GeneratorExit:
                    raise
                except Exception:
                    logger.warning(f"Failed to load data for instance {self.batches[sample_idx]}: {traceback.format_exc()}")
                    sample_idx = random.randint(0, len(self.batches) - 1)

            self.sample_idx += 1
            if self.sample_idx >= len(self.batches):
                if not self.infinite:
                    logger.warning("Dataset has run out of data.")
                    break
                self.sample_idx = 0
                logger.warning("Dataset is being re-looped.")

    def load_state_dict(self, state_dict):
        self.sample_idx = state_dict["sample_idx"]
        self.batches = state_dict["batches"]

    def state_dict(self):
        return {
            "batches": self.batches,
            "sample_idx": self.sample_idx,
        }
