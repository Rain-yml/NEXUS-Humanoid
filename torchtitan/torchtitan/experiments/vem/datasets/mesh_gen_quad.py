from __future__ import annotations

import gc
import random
import traceback
from typing import Any, List, Optional

import numpy as np
import torch
import trimesh
from PIL import Image
from torchvision.transforms import v2 as transforms_v2

from torchtitan.experiments.vem.datasets.mesh_stae_quad import SpaceTimeQuadAEDataset, shard_interleave
from torchtitan.experiments.vem.datasets.mesh_utils import Mesh, MeshProcessor, rand_with_pt
from torchtitan.experiments.vem.datasets.octree_utils import discretize, undiscretize
from torchtitan.experiments.vem.datasets.path_io import DatasetPathIO
from torchtitan.tools.logging import logger


class SpaceTimeRGBGenQuadDataset(SpaceTimeQuadAEDataset):
    worker_shard_data = ["instance_ids"]
    _metadata_columns = {"uuid", "mesh_path", "image_dir", "num_faces", "num_vertices"}

    @staticmethod
    def _compute_quad_ratio(mesh_mixed: Mesh) -> float:
        num_quads = int(np.count_nonzero(mesh_mixed.is_quad))
        num_triangles = int(mesh_mixed.faces.shape[0] - num_quads)
        denom = float(num_triangles + 2 * num_quads)
        return 2.0 * num_quads / denom if denom > 0 else 0.0

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

    def __init__(
        self,
        metadata_parquet: str,
        repeats: int = 1,
        shuffle_seed: int = 0,
        yup_to_zup: bool = False,
        vertex_noise: float = 0.0,
        vertex_resolution: int = -1,
        vertex_position_type: str = "float",
        encoder_vertex_position_type: str = "none",
        extra_feat: str = "none",
        num_face_range: Optional[List[int]] = [0, 20000],
        num_vertex_range: Optional[List[int]] = [0, 100000],
        mode: str = "tri_connect",
        return_mesh_mixed: bool = False,
        image_resolution: int = 518,
        drop_image_rate: float = 0.0,
        alignment: str = "none",
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
            yup_to_zup=yup_to_zup,
            vertex_noise=vertex_noise,
            vertex_resolution=vertex_resolution,
            vertex_position_type=vertex_position_type,
            encoder_vertex_position_type=encoder_vertex_position_type,
            extra_feat=extra_feat,
            mode=mode,
            return_mesh_mixed=return_mesh_mixed,
            image_resolution=image_resolution,
            drop_image_rate=drop_image_rate,
            alignment=alignment,
            infinite=infinite,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
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

    def _init_common(
        self,
        metadata_parquet: str,
        repeats: int,
        shuffle_seed: int,
        yup_to_zup: bool,
        vertex_noise: float,
        vertex_resolution: int,
        vertex_position_type: str,
        encoder_vertex_position_type: str,
        extra_feat: str,
        mode: str,
        return_mesh_mixed: bool,
        image_resolution: int,
        drop_image_rate: float,
        alignment: str,
        infinite: bool,
        dp_rank: int,
        dp_world_size: int,
        symmetry_condition: bool = False,
        symmetry_drop_rate: float = 0.1,
        quad_ratio_uncond_rate: float = 0.1,
    ) -> None:
        assert vertex_position_type in ["int", "float"]
        assert encoder_vertex_position_type in ["none", "int"]
        assert mode in ["tri", "tri_connect", "tri_bi_connect", "native_quad", "native_quad_wireframe"]
        assert extra_feat in ["none", "normal", "face_normal"]
        assert alignment in ["none", "near_align"]

        if vertex_position_type == "int" and vertex_resolution <= 0:
            raise ValueError("vertex_position_type='int' requires a positive vertex_resolution")
        if encoder_vertex_position_type == "int" and vertex_resolution <= 0:
            raise ValueError("encoder_vertex_position_type='int' requires a positive vertex_resolution")

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

        self.path_io = DatasetPathIO()
        self.metadata_parquet = metadata_parquet
        self.records = self._load_metadata(metadata_parquet)
        self.repeats = repeats
        self.shuffle_seed = shuffle_seed
        self.infinite = infinite
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.sample_idx = 0
        self.packed = False

        self.mp = MeshProcessor()
        self.yup_to_zup = yup_to_zup
        self.vertex_noise = vertex_noise
        self.vertex_resolution = vertex_resolution
        self.vertex_position_type = vertex_position_type
        self.encoder_vertex_position_type = encoder_vertex_position_type
        self.extra_feat = extra_feat
        self.mode = mode
        self.return_mesh_mixed = return_mesh_mixed
        self.image_resolution = image_resolution
        self.drop_image_rate = drop_image_rate
        self.alignment = alignment

        self.transform = transforms_v2.Lambda(lambda x: x)

        print("dp_rank", dp_rank, "dp_world_size", dp_world_size)

    def _load_metadata(self, metadata_parquet: str) -> dict[str, dict[str, Any]]:
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
            if num_vertex_range is not None and not (
                num_vertex_range[0] <= rec["num_vertices"] <= num_vertex_range[1]
            ):
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

            fg_grayscale = image_np[..., :3][image_np[..., 3] > 0].mean()
            bg_color = np.random.rand(1)
            while bg_color > fg_grayscale - 0.2 and bg_color < fg_grayscale + 0.2:
                bg_color = np.random.rand(1)

            bg_color = np.concatenate([bg_color, bg_color, bg_color])
            image_np = image_np[..., :3] * image_np[..., 3:4] + bg_color[None, None] * (1 - image_np[..., 3:4])
        else:
            image_np = np.zeros_like(image_np[..., :3])

        image = Image.fromarray((image_np * 255.0).clip(0, 255).astype(np.uint8))
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        image = self.transform(image)
        image_pt = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1)
        return image, image_pt

    def _get_image(self, rec: dict[str, Any]):
        image_dir = rec["image_dir"]
        meta = self.path_io.read_json(self.path_io.join(image_dir, "meta.json"))

        image_idx = np.random.choice(list(range(len(meta["locations"]))))
        transform_matrix = np.array(meta["locations"][image_idx]["transform_matrix"])
        camera_loc = transform_matrix[:3, 3]
        azimuth_deg = np.rad2deg(np.arctan2(camera_loc[1], camera_loc[0]))
        rotation_idx = int(((azimuth_deg + 90 + 45) % 360) // 90)

        image_fn = meta["locations"][image_idx]["frames"][0]["name"]
        image = self.path_io.read_image(self.path_io.join(image_dir, image_fn), mode="RGBA")
        image, image_pt = self.process_image(image, self.image_resolution)

        return {
            "image_pil": image,
            "image_pt": image_pt,
            "rotation_idx": rotation_idx,
        }

    def _prepare_mesh_mixed(self, rec: dict[str, Any], rotation_transform=None):
        mesh = self.path_io.read_mixed_mesh(rec["mesh_path"])
        vertices = self.mp.normalize_vertices(mesh.vertices, range=(-1, 1))
        faces = np.copy(mesh.faces)
        if self.yup_to_zup:
            vertices = np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], axis=-1)
        if rotation_transform is not None:
            vertices = trimesh.transformations.transform_points(vertices, rotation_transform)

        if faces.shape[0] == 0:
            raise ValueError(f"Mesh has no triangle or quad faces: {rec.get('uuid', rec['mesh_path'])}")
        faces = faces.astype(np.int32, copy=False)

        if self.vertex_noise > 0:
            if rand_with_pt([0, 1]) < 0.7:
                noise_level = np.random.uniform(0, self.vertex_noise)
                vertices += np.random.randn(*vertices.shape) * noise_level

        if self.vertex_resolution > 0:
            vertices_dis = discretize(vertices, self.vertex_resolution)
            mesh_mixed = Mesh(vertices_dis, faces=faces)
            mesh_mixed.merge_vertices(digits=0)
            mesh_mixed.clean_faces()
            token_pos_np = np.copy(mesh_mixed.vertices)
            vertex_position = np.copy(token_pos_np) * 3
            vertices = undiscretize(token_pos_np, self.vertex_resolution)
            mesh_mixed.vertices = vertices
            if self.vertex_position_type == "int":
                vertices_for_diffusion = torch.from_numpy(token_pos_np.astype(np.int32, copy=False)).long()
            else:
                vertices_for_diffusion = torch.from_numpy(vertices).float()
        else:
            mesh_mixed = Mesh(vertices, faces=faces)
            mesh_mixed.merge_vertices(digits=6)
            mesh_mixed.clean_faces()
            vertices = np.copy(mesh_mixed.vertices)
            vertex_position = None
            vertices_for_diffusion = torch.from_numpy(vertices).float()

        return mesh_mixed, vertex_position, vertices_for_diffusion

    def _get_mesh(self, uuid: str, rec: dict[str, Any], rotation_transform=None):
        mesh_mixed, vertex_position, vertices = self._prepare_mesh_mixed(rec, rotation_transform=rotation_transform)

        if self.mode == "tri":
            ret = self._get_data_tri(
                uuid,
                mesh_mixed,
                vertex_position,
                vertex_position_type=self.encoder_vertex_position_type,
                return_supervision=False,
            )
        elif self.mode == "tri_connect":
            ret = self._get_data_tri_connect(
                uuid,
                mesh_mixed,
                vertex_position,
                vertex_position_type=self.encoder_vertex_position_type,
                return_supervision=False,
            )
        elif self.mode == "tri_bi_connect":
            ret = self._get_data_tri_bi_connect(
                uuid,
                mesh_mixed,
                vertex_position,
                vertex_position_type=self.encoder_vertex_position_type,
                return_supervision=False,
            )
        elif self.mode == "native_quad":
            ret = self._get_data_native_quad(
                uuid,
                mesh_mixed,
                vertex_position,
                vertex_position_type=self.encoder_vertex_position_type,
                return_supervision=False,
            )
        elif self.mode == "native_quad_wireframe":
            ret = self._get_data_native_quad(
                uuid,
                mesh_mixed,
                vertex_position,
                vertex_position_type=self.encoder_vertex_position_type,
                include_wireframe_edges=True,
                return_supervision=False,
            )
        else:
            raise NotImplementedError(self.mode)

        ret["vertices"] = vertices
        if self.encoder_vertex_position_type == "int" and "position" in ret:
            ret["encoder_position"] = ret["position"]
            del ret["position"]
        if self.return_mesh_mixed:
            ret["mesh_mixed"] = mesh_mixed

        if ret["edges"].numel() == 0:
            raise ValueError(f"Mesh graph has no edges: {uuid}")
        ret["quad_ratio"] = self._compute_quad_ratio(mesh_mixed)
        return ret

    def get_data(self, uuid: str):
        rec = self.records[uuid]
        rec["uuid"] = uuid
        ret = {"instance_id": uuid}
        rotation_transform = None
        # Number of 90-degree turns about Z applied by near_align (parity is what
        # matters for remapping symmetry planes: odd turns swap the x=0 / y=0 planes).
        near_align_quarter_turns = 0

        if self.drop_image_rate > 0 and rand_with_pt([0, 1]) < self.drop_image_rate:
            ret["image"] = torch.full(
                (3, self.image_resolution, self.image_resolution),
                fill_value=0.5,
                dtype=torch.float32,
            )
        else:
            image_dict = self._get_image(rec)
            ret["image"] = image_dict["image_pt"]
            if self.alignment == "near_align" and image_dict["rotation_idx"] != 0:
                rotation_transform = trimesh.transformations.rotation_matrix(
                    -np.pi / 2 * image_dict["rotation_idx"],
                    [0, 0, 1],
                )
                near_align_quarter_turns = image_dict["rotation_idx"]

        ret.update(self._get_mesh(uuid, rec, rotation_transform=rotation_transform))

        # With some probability, mark quad_ratio as "unknown" (negative); the model
        # then uses its unconditional quad-ratio embedding instead.
        if self.quad_ratio_uncond_rate > 0 and rand_with_pt([0, 1]) < self.quad_ratio_uncond_rate:
            ret["quad_ratio"] = -1.0

        if self.symmetry_condition:
            ret["symmetries"] = self._build_symmetries(rec, near_align_quarter_turns)

        return ret

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

    def _collate_fn(self, batch):
        def flatten_list(l):
            return [item for sublist in l for item in sublist]

        batch = flatten_list(batch)

        batch_collated = {}
        offset = [0]
        vertex_offset = [0]
        nodes = []
        edges = []
        vertices = []

        for b in batch:
            nodes.append(b["nodes"])
            edges.append(b["edges"] + offset[-1])
            vertices.append(b["vertices"])
            offset.append(offset[-1] + b["nodes"].shape[0])
            vertex_offset.append(vertex_offset[-1] + b["vertices"].shape[0])

        batch_collated["nodes"] = torch.cat(nodes, dim=0)
        batch_collated["edges"] = torch.cat(edges, dim=0)
        batch_collated["offsets"] = torch.tensor(offset, dtype=torch.int32)
        batch_collated["encoder_cu_seqlens"] = torch.tensor(offset, dtype=torch.int32)
        batch_collated["vertices"] = torch.cat(vertices, dim=0)
        batch_collated["cu_seqlens"] = torch.tensor(vertex_offset, dtype=torch.int32)
        batch_collated["instance_ids"] = [b["instance_id"] for b in batch]
        batch_collated["quad_ratios"] = torch.tensor([b["quad_ratio"] for b in batch], dtype=torch.float32)
        if "symmetries" in batch[0]:
            batch_collated["symmetries"] = torch.tensor([b["symmetries"] for b in batch], dtype=torch.long)
        batch_collated["image"] = torch.stack([b["image"] for b in batch], dim=0)
        batch_collated["node_type"] = torch.cat([b["node_type"] for b in batch], dim=0)

        if self.encoder_vertex_position_type == "int":
            batch_collated["encoder_position"] = torch.cat([b["encoder_position"] for b in batch], dim=0)
        if self.return_mesh_mixed:
            batch_collated["mesh_mixed"] = [b["mesh_mixed"] for b in batch]

        del batch
        gc.collect()
        return batch_collated

    def collate_fn(self, batch):
        return self._collate_fn(batch)


class SpaceTimeRGBGenQuadPackDataset(SpaceTimeRGBGenQuadDataset):
    worker_shard_data = ["batches"]

    def __init__(
        self,
        metadata_parquet: str,
        batches_packed: str,
        repeats: int = 1,
        shuffle_seed: int = 0,
        yup_to_zup: bool = False,
        vertex_noise: float = 0.0,
        vertex_resolution: int = -1,
        vertex_position_type: str = "float",
        encoder_vertex_position_type: str = "none",
        extra_feat: str = "none",
        mode: str = "tri_connect",
        return_mesh_mixed: bool = False,
        image_resolution: int = 518,
        drop_image_rate: float = 0.0,
        alignment: str = "none",
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
            yup_to_zup=yup_to_zup,
            vertex_noise=vertex_noise,
            vertex_resolution=vertex_resolution,
            vertex_position_type=vertex_position_type,
            encoder_vertex_position_type=encoder_vertex_position_type,
            extra_feat=extra_feat,
            mode=mode,
            return_mesh_mixed=return_mesh_mixed,
            image_resolution=image_resolution,
            drop_image_rate=drop_image_rate,
            alignment=alignment,
            infinite=infinite,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            symmetry_condition=symmetry_condition,
            symmetry_drop_rate=symmetry_drop_rate,
            quad_ratio_uncond_rate=quad_ratio_uncond_rate,
        )

        batches_packed_json = self.path_io.read_json(batches_packed)
        batches = self._normalize_batches(batches_packed_json["batches"]) * repeats
        if force_divisible_by > 1:
            batches = batches[: len(batches) // force_divisible_by * force_divisible_by]
            logger.info(f"Batch number after dropping: {len(batches)}")

        missing_uuids = sorted({uuid for batch in batches for uuid in batch} - set(self.records))
        if missing_uuids:
            raise ValueError(f"Packed batches contain uuid missing from metadata parquet: {missing_uuids[:5]}")

        batches = shard_interleave(batches, dp_world_size, dp_rank)
        rng = random.Random(shuffle_seed)
        rng.shuffle(batches)
        print("rank", dp_rank, batches[:10])
        self.batches = batches
        self.packed = True

    @staticmethod
    def _normalize_batches(batches):
        normalized = []
        for batch_idx, batch in enumerate(batches):
            if not isinstance(batch, list):
                raise ValueError(f"Packed batch {batch_idx} must be a list")
            normalized_batch = []
            for record_idx, record in enumerate(batch):
                if isinstance(record, str):
                    uuid = record
                elif isinstance(record, (list, tuple)) and len(record) > 0 and isinstance(record[0], str):
                    uuid = record[0]
                else:
                    raise ValueError(
                        f"Packed batch {batch_idx} record {record_idx} must be a uuid string or a non-empty list/tuple"
                    )
                normalized_batch.append(uuid)
            normalized.append(normalized_batch)
        return normalized

    def __iter__(self):
        while True:
            sample_idx = self.sample_idx
            while True:
                try:
                    yield [self.get_data(uuid) for uuid in self.batches[sample_idx]]
                    break
                except GeneratorExit:
                    raise
                except Exception:
                    logger.warning(
                        f"Failed to load data for batch {self.batches[sample_idx]}: {traceback.format_exc()}"
                    )
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
