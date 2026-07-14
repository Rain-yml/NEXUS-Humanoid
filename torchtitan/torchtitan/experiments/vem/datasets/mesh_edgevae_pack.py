"""Packed dataset for the edge VAE.

Encodes a mesh *wireframe* (vertices + edges) into a graph and supervises the
reconstruction of that wireframe (and optionally the quad diagonals) over the
per-vertex latents.  No face / orientation supervision is produced.

Three graph-encoding ``mode`` values (all controlled here, transparent to the model):

  * ``wireframe``      - graph nodes = vertices, graph edges = wireframe edges.
  * ``bipartite``      - graph nodes = vertices + one node per wireframe edge;
                         graph edges = vertex<->edge incidences. The two node kinds
                         are distinguished by ``node_type`` (0 = vertex, 1 = edge).
  * ``bipartite_diag`` - ``bipartite`` plus one node per quad diagonal as a third
                         node type (``node_type`` 2), each connected to its two
                         endpoint vertices.

Supervision (always):
  * ``vertex_pair``       - the wireframe edges (positive pairs for the all-pair edge loss).
Supervision (optional, ``diag_supervision=True``):
  * ``vertex_pair_diag``  - the quad diagonals (positive pairs for the all-pair diag loss).
"""

from typing import List, Tuple
import gc
import random
import traceback

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.tools.logging import logger
from torchtitan.experiments.vem.datasets.json_utils import load_json
from torchtitan.experiments.vem.datasets.mesh_utils import (
    Mesh,
    MeshProcessor,
    rand_int_with_pt,
    rand_with_pt,
)
from torchtitan.experiments.vem.datasets.octree_utils import discretize, undiscretize
from torchtitan.experiments.vem.datasets.path_io import DatasetPathIO
from torchtitan.experiments.vem.datasets.mesh_stae_quad import (
    shard_interleave,
    triangle_normals,
    normalize_rows,
    quad_split_triangles,
    round_half_away_from_zero,
)


class SpaceTimeEdgeAEPackDataset(IterableDataset, Stateful):
    worker_shard_data = ["batches"]

    def __init__(
        self,
        batches_packed: str,
        repeats: int = 1,
        shuffle_seed: int = 0,
        aug_flip: bool = True,
        aug_rotate_all: bool = False,
        aug_rotate_z: bool = True,
        aug_scale: bool = False,
        aug_scale_range: List[float] = [0.8, 1.2],
        yup_to_zup: bool = False,
        vertex_noise: float = 0.0,
        extra_feat: str = "none",
        vertex_resolutions: List[int] = [-1],
        vertex_position_type: str = "none",
        mode: str = "wireframe",
        diag_supervision: bool = False,
        return_mesh_mixed: bool = False,
        # auto assigned args
        infinite: bool = True,
        force_divisible_by: int = 1,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        self.repeats = repeats
        self.shuffle_seed = shuffle_seed
        self.infinite = infinite
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.use_bos = True
        assert vertex_position_type in ["none", "int"]
        assert extra_feat in ["none", "normal"]
        assert mode in ["wireframe", "bipartite", "bipartite_diag"]
        if vertex_position_type == "int" and not any(v > 0 for v in vertex_resolutions):
            raise ValueError("vertex_position_type='int' requires a positive vertex_resolution")
        self.vertex_position_type = vertex_position_type

        self.path_io = DatasetPathIO()
        self.sample_idx = 0

        batches_packed_json = load_json(batches_packed)
        batches = batches_packed_json["batches"]
        self._validate_batches(batches)
        batches = batches * repeats
        if force_divisible_by > 1:
            batches = batches[: len(batches) // force_divisible_by * force_divisible_by]
            logger.info(f"Batch number after dropping: {len(batches)}")

        batches = shard_interleave(batches, dp_world_size, dp_rank)

        rng = random.Random(shuffle_seed)
        rng.shuffle(batches)

        print("rank", dp_rank, batches[:10])

        self.batches = batches
        self.packed = True

        self.mp = MeshProcessor()
        self.aug_flip = aug_flip
        self.aug_rotate_all = aug_rotate_all
        self.aug_rotate_z = aug_rotate_z
        self.aug_scale = aug_scale
        self.aug_scale_range = aug_scale_range
        self.yup_to_zup = yup_to_zup
        self.vertex_noise = vertex_noise
        self.vertex_resolutions = vertex_resolutions
        self.extra_feat = extra_feat
        self.mode = mode
        self.diag_supervision = diag_supervision
        self.return_mesh_mixed = return_mesh_mixed
        print(
            "Augmentations",
            "flip", aug_flip,
            "rotate_all", aug_rotate_all,
            "rotate_z", aug_rotate_z,
            "scale", aug_scale,
            "scale_range", aug_scale_range,
        )
        print("mode", mode, "diag_supervision", diag_supervision, "extra_feat", extra_feat)
        print("dp_rank", dp_rank, "dp_world_size", dp_world_size)

    @staticmethod
    def _validate_batches(batches: List[List[Tuple[str, str, str]]]) -> None:
        for batch_idx, batch in enumerate(batches):
            if not isinstance(batch, list):
                raise ValueError(f"Packed batch {batch_idx} must be a list")
            for record_idx, record in enumerate(batch):
                if not isinstance(record, (list, tuple)) or len(record) != 3:
                    raise ValueError(
                        f"Packed batch {batch_idx} record {record_idx} must be "
                        "a (uuid, bucket, bos_path) triple"
                    )
                if not all(isinstance(value, str) for value in record):
                    raise ValueError(
                        f"Packed batch {batch_idx} record {record_idx} triple values must be strings"
                    )

    # ------------------------------------------------------------------ #
    # geometry helpers
    # ------------------------------------------------------------------ #
    def _vertex_normals(self, vertices, faces, is_quad):
        """Per-vertex normals computed natively over the mixed tri/quad mesh."""
        face_normals = np.zeros((faces.shape[0], 3), dtype=vertices.dtype)
        tri_indices = np.where(~is_quad)[0]
        if len(tri_indices) > 0:
            face_normals[tri_indices] = triangle_normals(vertices, faces[tri_indices, :3])
        quad_indices = np.where(is_quad)[0]
        if len(quad_indices) > 0:
            quad_faces = faces[quad_indices, :4]
            quad_tris = quad_split_triangles(quad_faces)
            quad_tri_normals = triangle_normals(vertices, quad_tris).reshape(4, len(quad_indices), 3)
            face_normals[quad_indices] = normalize_rows(quad_tri_normals.mean(axis=0))

        vertex_normals = np.zeros_like(vertices)
        if len(tri_indices) > 0:
            tri_faces = faces[tri_indices, :3]
            for corner in range(3):
                np.add.at(vertex_normals, tri_faces[:, corner], face_normals[tri_indices])
        if len(quad_indices) > 0:
            quad_faces = faces[quad_indices, :4]
            for corner in range(4):
                np.add.at(vertex_normals, quad_faces[:, corner], face_normals[quad_indices])
        return normalize_rows(vertex_normals)

    def _quad_diagonals(self, mesh_mixed):
        faces = mesh_mixed.faces
        is_quad = mesh_mixed.is_quad
        quad_indices = np.where(is_quad)[0]
        if len(quad_indices) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        quad_faces = faces[quad_indices, :4]
        diag = np.concatenate([quad_faces[:, [0, 2]], quad_faces[:, [1, 3]]], axis=0)
        diag.sort(axis=1)
        diag = np.unique(diag, axis=0).astype(np.int64, copy=False)
        return diag

    def _node_features(self, node_pos, node_normals):
        """Assemble the model's per-node input feature ([pos] or [pos, normal])."""
        if self.extra_feat == "none":
            return node_pos.astype(np.float32, copy=False)
        return np.concatenate([node_pos, node_normals], axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _midpoint_position(int_position, pairs):
        if pairs.shape[0] == 0:
            return np.zeros((0, int_position.shape[1]), dtype=int_position.dtype)
        return round_half_away_from_zero(int_position[pairs].mean(axis=1)).astype(
            int_position.dtype, copy=False
        )

    # ------------------------------------------------------------------ #
    # graph builders, one per mode
    # ------------------------------------------------------------------ #
    def _build_graph(self, uuid, mesh_mixed, vertex_position):
        vertices = np.copy(mesh_mixed.vertices)
        is_quad = np.copy(mesh_mixed.is_quad)
        faces = np.copy(mesh_mixed.faces)
        nv = vertices.shape[0]

        # wireframe edges == positive supervision
        wf_edges = mesh_mixed.edges().astype(np.int64, copy=False)
        wf_edges.sort(axis=1)
        wf_edges = np.unique(wf_edges, axis=0)

        diag_edges = self._quad_diagonals(mesh_mixed)

        if self.extra_feat == "normal":
            vertex_normals = self._vertex_normals(vertices, faces, is_quad)
        else:
            vertex_normals = np.zeros((nv, 3), dtype=vertices.dtype)

        int_pos = vertex_position if self.vertex_position_type == "int" else None

        if self.mode == "wireframe":
            node_pos = vertices
            node_normals = vertex_normals
            node_type = np.zeros(nv, dtype=np.int64)
            edge_index = wf_edges  # vertex<->vertex
            position = int_pos
        elif self.mode == "bipartite":
            E = wf_edges.shape[0]
            edge_mid = vertices[wf_edges].mean(axis=1)
            node_pos = np.concatenate([vertices, edge_mid], axis=0)
            node_normals = np.concatenate([vertex_normals, np.zeros((E, 3), dtype=vertices.dtype)], axis=0)
            node_type = np.concatenate([np.zeros(nv, dtype=np.int64), np.ones(E, dtype=np.int64)])
            edge_node_idx = np.arange(nv, nv + E, dtype=np.int64)
            edge_index = np.stack([np.repeat(edge_node_idx, 2), wf_edges.reshape(-1)], axis=1)
            if int_pos is not None:
                edge_mid_pos = self._midpoint_position(int_pos, wf_edges)
                position = np.concatenate([int_pos, edge_mid_pos], axis=0)
            else:
                position = None
        elif self.mode == "bipartite_diag":
            E = wf_edges.shape[0]
            D = diag_edges.shape[0]
            edge_mid = vertices[wf_edges].mean(axis=1)
            diag_mid = (
                vertices[diag_edges].mean(axis=1)
                if D > 0
                else np.zeros((0, 3), dtype=vertices.dtype)
            )
            node_pos = np.concatenate([vertices, edge_mid, diag_mid], axis=0)
            node_normals = np.concatenate(
                [vertex_normals, np.zeros((E + D, 3), dtype=vertices.dtype)], axis=0
            )
            node_type = np.concatenate(
                [
                    np.zeros(nv, dtype=np.int64),
                    np.ones(E, dtype=np.int64),
                    np.full(D, 2, dtype=np.int64),
                ]
            )
            edge_node_idx = np.arange(nv, nv + E, dtype=np.int64)
            diag_node_idx = np.arange(nv + E, nv + E + D, dtype=np.int64)
            inc_edge = np.stack([np.repeat(edge_node_idx, 2), wf_edges.reshape(-1)], axis=1)
            inc_diag = (
                np.stack([np.repeat(diag_node_idx, 2), diag_edges.reshape(-1)], axis=1)
                if D > 0
                else np.zeros((0, 2), dtype=np.int64)
            )
            edge_index = np.concatenate([inc_edge, inc_diag], axis=0)
            if int_pos is not None:
                edge_mid_pos = self._midpoint_position(int_pos, wf_edges)
                diag_mid_pos = self._midpoint_position(int_pos, diag_edges)
                position = np.concatenate([int_pos, edge_mid_pos, diag_mid_pos], axis=0)
            else:
                position = None
        else:
            raise NotImplementedError(self.mode)

        all_nodes = self._node_features(node_pos, node_normals)

        ret = {
            "nodes": torch.from_numpy(all_nodes).float(),
            "edges": torch.from_numpy(np.ascontiguousarray(edge_index)).long(),
            "instance_id": uuid,
            "node_type": torch.from_numpy(node_type).long(),
            "vertex_pair": torch.from_numpy(wf_edges).long(),
        }
        if self.diag_supervision:
            ret["vertex_pair_diag"] = torch.from_numpy(diag_edges).long()
        if position is not None:
            ret["position"] = torch.from_numpy(position).long()
        return ret

    # ------------------------------------------------------------------ #
    # per-instance pipeline (identical preprocessing to the quad dataset)
    # ------------------------------------------------------------------ #
    def get_data(self, instance_id: Tuple[str, str, str]):
        uuid, bucket, path = instance_id
        if bucket == "":
            mesh = self.path_io.read_mixed_mesh(path)
        else:
            mesh = self.path_io.read_mixed_mesh(f"bos://{bucket}/{path}")

        vertices = self.mp.normalize_vertices(mesh.vertices, range=(-1, 1))
        faces = np.copy(mesh.faces)
        if self.yup_to_zup:
            vertices = np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], axis=-1)

        faces_tri = faces[faces[:, -1] < 0, :3]
        faces_quad = faces[faces[:, -1] >= 0]
        vertices, faces_tri, faces_quad = self.mp.augment_quad(
            vertices=vertices,
            faces=faces_tri,
            faces_quad=faces_quad,
            aug_flip=self.aug_flip,
            aug_rotate_all=self.aug_rotate_all,
            aug_rotate_z=self.aug_rotate_z,
            aug_scale=self.aug_scale,
            aug_scale_range=self.aug_scale_range,
        )
        faces_unified = []
        if faces_tri.shape[0] > 0:
            faces_unified.append(
                np.concatenate(
                    [faces_tri, np.full((faces_tri.shape[0], 1), -1, dtype=faces_tri.dtype)], axis=-1
                )
            )
        if faces_quad.shape[0] > 0:
            faces_unified.append(faces_quad)
        if len(faces_unified) == 0:
            raise ValueError(f"Mesh has no triangle or quad faces: {uuid}")
        faces = np.concatenate(faces_unified, axis=0).astype(np.int32, copy=False)

        if len(self.vertex_resolutions) == 0:
            v_res = -1
        else:
            v_res = self.vertex_resolutions[rand_int_with_pt([0, len(self.vertex_resolutions)])]

        if self.vertex_noise > 0:
            if rand_with_pt([0, 1]) < 0.7:
                noise_level = np.random.uniform(0, self.vertex_noise)
                vertices += np.random.randn(*vertices.shape) * noise_level

        if v_res > 0:
            vertices_dis = discretize(vertices, v_res)
            mesh_mixed = Mesh(vertices_dis, faces=faces)
            mesh_mixed.merge_vertices(digits=0)
            mesh_mixed.clean_faces()
            vertices = np.copy(mesh_mixed.vertices)
            vertex_position = np.copy(vertices) * 3
            vertices = undiscretize(vertices, v_res)
            mesh_mixed.vertices = vertices
        else:
            mesh_mixed = Mesh(vertices, faces=faces)
            mesh_mixed.merge_vertices(digits=6)
            mesh_mixed.clean_faces()
            vertices = np.copy(mesh_mixed.vertices)
            vertex_position = None

        ret = self._build_graph(uuid, mesh_mixed, vertex_position)

        if self.return_mesh_mixed:
            ret["mesh_mixed"] = mesh_mixed
        return ret

    # ------------------------------------------------------------------ #
    # iteration / checkpointing
    # ------------------------------------------------------------------ #
    def __iter__(self):
        while True:
            sample_idx = self.sample_idx
            while True:
                try:
                    yield [self.get_data(record) for record in self.batches[sample_idx]]
                    break
                except GeneratorExit:
                    raise
                except Exception:
                    logger.warning(
                        f"Failed to load data for batch {self.batches[sample_idx]}: "
                        f"{traceback.format_exc()}"
                    )
                    sample_idx = random.randint(0, len(self.batches) - 1)

            self.sample_idx += 1
            if self.sample_idx >= len(self.batches):
                if not self.infinite:
                    logger.warning("Dataset has run out of data.")
                    break
                else:
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

    # ------------------------------------------------------------------ #
    # collation
    # ------------------------------------------------------------------ #
    def _collate_fn(self, batch):
        def flatten_list(l):
            return [item for sublist in l for item in sublist]

        batch = flatten_list(batch)

        batch_collated = {}
        offset = [0]
        nodes = []
        edges = []
        vertex_pair = []
        pair_offsets = [0]
        node_type = []
        has_diag = self.diag_supervision
        if has_diag:
            vertex_pair_diag = []
            diag_pair_offsets = [0]

        for b in batch:
            nodes.append(b["nodes"])
            edges.append(b["edges"] + offset[-1])
            node_type.append(b["node_type"])
            vertex_pair.append(b["vertex_pair"] + offset[-1])
            pair_offsets.append(pair_offsets[-1] + b["vertex_pair"].shape[0])
            if has_diag:
                vertex_pair_diag.append(b["vertex_pair_diag"] + offset[-1])
                diag_pair_offsets.append(diag_pair_offsets[-1] + b["vertex_pair_diag"].shape[0])
            offset.append(offset[-1] + b["nodes"].shape[0])

        batch_collated["nodes"] = torch.cat(nodes, dim=0)
        batch_collated["edges"] = torch.cat(edges, dim=0)
        batch_collated["offsets"] = torch.tensor(offset, dtype=torch.int32)
        batch_collated["node_type"] = torch.cat(node_type, dim=0)
        batch_collated["instance_ids"] = [b["instance_id"] for b in batch]
        batch_collated["vertex_pair"] = torch.cat(vertex_pair, dim=0)
        batch_collated["pair_offsets"] = torch.tensor(pair_offsets, dtype=torch.int32)
        if has_diag:
            batch_collated["vertex_pair_diag"] = torch.cat(vertex_pair_diag, dim=0)
            batch_collated["diag_pair_offsets"] = torch.tensor(diag_pair_offsets, dtype=torch.int32)
        if self.vertex_position_type == "int":
            batch_collated["position"] = torch.cat([b["position"] for b in batch], dim=0)
        if self.return_mesh_mixed:
            batch_collated["mesh_mixed"] = [b["mesh_mixed"] for b in batch]

        del batch
        gc.collect()
        return batch_collated

    def collate_fn(self, batch):
        return self._collate_fn(batch)
