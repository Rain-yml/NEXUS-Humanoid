from dataclasses import asdict
from typing import Optional, List, Union, Tuple
import json
import random
import os
import numpy as np
import trimesh
from PIL import Image
import traceback
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.experiments.vem.dataloader import ParallelAwareDataloader
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger
from torchtitan.experiments.vem.datasets.bos import BOSClient
from torchtitan.experiments.vem.datasets.json_utils import load_json
from scipy.spatial.transform import Rotation
from torchtitan.experiments.vem.datasets.mesh_utils import (
    Mesh,
    MeshProcessor,
    rand_int_with_pt,
    rand_with_pt,
)
import math
from torchtitan.experiments.vem.datasets.renderer.moderngl_rasterizer import FaceNormalRenderer
from torchtitan.experiments.vem.datasets.octree_utils import (
    discretize,
    undiscretize,
)
from itertools import chain
import pandas as pd
import networkx as nx

from torchtitan.experiments.vem.datasets.path_io import DatasetPathIO

def nx_all_triangles(G, nbunch=None):
    if nbunch is None:
        nbunch = relevant_nodes = G
    else:
        nbunch = dict.fromkeys(G.nbunch_iter(nbunch))
        relevant_nodes = chain(
            nbunch,
            (nbr for node in nbunch for nbr in G.neighbors(node) if nbr not in nbunch),
        )

    node_to_id = {node: i for i, node in enumerate(relevant_nodes)}

    triangles = []
    for u in nbunch:
        u_id = node_to_id[u]
        u_nbrs = G._adj[u].keys()
        for v in u_nbrs:
            v_id = node_to_id.get(v, -1)
            if v_id <= u_id:
                continue
            v_nbrs = G._adj[v].keys()
            for w in v_nbrs & u_nbrs:
                if node_to_id.get(w, -1) > v_id:
                    triangles.append((u, v, w))
    return np.array(triangles)


def shard(size: int, num_shards: int, index: int):
    if not 0 <= index < num_shards:
        raise ValueError("index should be in [0, num_shards-1]")
    
    div = size // num_shards
    mod = size % num_shards
    start = div * index + min(index, mod)
    end = start + div + (1 if index < mod else 0)
    
    return start, end

def shard_interleave(l, num_shards: int, index: int):
    if not 0 <= index < num_shards:
        raise ValueError("index should be in [0, num_shards-1]")

    return [l[i] for i in range(len(l)) if i % num_shards == index]

# Largest int64 (exclusive). Row-key encoding is used only when base**ncols stays
# below this; otherwise we fall back to the original structured-dtype path.
_INT63 = 1 << 63


def _encode_rows(rows, base):
    """Pack each row of non-negative int indices into a single int64 key.

    The key ``r0*base**(k-1) + ... + r_{k-1}`` is monotonic with the lexicographic
    row order, so 1-D ``np.unique`` / ``np.setdiff1d`` on the keys reproduce the
    row-wise (axis=0) results exactly while avoiding the slow structured/
    lexicographic row sort that dominated dataloader CPU time.
    """
    rows = np.ascontiguousarray(rows)
    key = rows[:, 0].astype(np.int64)
    for j in range(1, rows.shape[1]):
        key = key * base + rows[:, j].astype(np.int64)
    return key


def _decode_rows(keys, base, k):
    out = np.empty((keys.shape[0], k), dtype=np.int64)
    rem = keys.astype(np.int64, copy=True)
    for j in range(k - 1, -1, -1):
        out[:, j] = rem % base
        rem //= base
    return out


def _row_base(*arrays):
    """Encoding base (= max index + 1) that fits every row of the given arrays."""
    maxv = 0
    for arr in arrays:
        if arr.size:
            maxv = max(maxv, int(arr.max()))
    return maxv + 1


def unique_rows(rows):
    """Equivalent to ``np.unique(rows, axis=0)`` (sorted unique rows), but fast.

    Encodes each row to an int64 key so deduplication is a 1-D unique instead of a
    lexicographic row sort. Falls back to ``np.unique(axis=0)`` on overflow.
    """
    if rows.shape[0] == 0:
        return rows
    k = rows.shape[1]
    base = _row_base(rows)
    if base ** k >= _INT63:
        return np.unique(rows, axis=0)
    uniq = np.unique(_encode_rows(rows, base))
    return _decode_rows(uniq, base, k).astype(rows.dtype)


def rows_in_A_not_in_B(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(f"Expected 2D arrays with matching row widths, got {A.shape} and {B.shape}")

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    k = A.shape[1]
    if A.shape[0] == 0:
        return A.reshape(0, k)

    # Encode each row as a single int64 key and use fast 1-D set ops. The key is
    # monotonic with lexicographic row order, so the returned rows (and their
    # order) match the structured-dtype implementation below bit-for-bit, without
    # the row-wise sort that dominated dataloader CPU time.
    base = _row_base(A, B)
    if base ** k < _INT63:
        key_A = _encode_rows(A, base)
        key_B = _encode_rows(B, base) if B.shape[0] else np.zeros(0, dtype=np.int64)
        remain = np.setdiff1d(key_A, key_B)
        return _decode_rows(remain, base, k)

    # Fallback for the (practically unreachable) overflow case: original
    # structured-dtype set difference.
    A_view = A.view([('', A.dtype)] * k)
    B_view = B.view([('', B.dtype)] * k)
    C_view = np.setdiff1d(A_view, B_view)
    return C_view.view(A.dtype).reshape(-1, k)

def random_non_face_triplet_with_edge(edges, faces, num_vertices, n):
    """
    Sample n triplets (v1, v2, v3) such that:
      - (v1, v2) is an edge in edges
      - (v1, v2, v3) is NOT equal to any face (in any ordering)
      - v1, v2, v3 are all distinct
    """
    edges = np.asarray(edges, dtype=np.int32)
    faces = np.asarray(faces, dtype=np.int32)

    # --- Normalize faces into sorted-row form for easy checking ---
    # Any permutation of a face becomes the same sorted triple
    faces_sorted = np.sort(faces, axis=1)
    face_set = {tuple(row) for row in faces_sorted}

    E = len(edges)

    triplets = []
    tries = 0

    while len(triplets) < n and tries < n:
        tries += 1

        v1, v2 = edges[np.random.randint(E)]

        # Random candidate v3
        v3 = np.random.randint(0, num_vertices)
        if v3 == v1 or v3 == v2:
            continue

        # Check not a face
        tri_sorted = tuple(sorted((v1, v2, v3)))
        if tri_sorted in face_set:
            continue

        triplets.append([v1, v2, v3])

    return np.array(triplets, dtype=np.int64)

def random_non_face_triplet_with_edge_fast(edges, faces, num_vertices, n, oversample=4):
    """
    Vectorized version of `random_non_face_triplet_with_edge` (no Python loop).

    Samples triplets (v1, v2, v3) such that:
      - (v1, v2) is an edge in `edges`
      - {v1,v2,v3} is NOT equal to any face (orientation-insensitive)
      - v1, v2, v3 are all distinct
    """
    edges = np.asarray(edges, dtype=np.int32)
    faces = np.asarray(faces, dtype=np.int32)

    if n <= 0 or edges.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    faces_sorted = np.sort(faces, axis=1)
    E = edges.shape[0]

    m = max(int(np.ceil(float(oversample) * n)), n + 1024)
    edge_idx = np.random.randint(E, size=m)
    candidates = np.stack(
        [
            edges[edge_idx, 0],
            edges[edge_idx, 1],
            np.random.randint(0, num_vertices, size=m, dtype=np.int32),
        ],
        axis=1,
    ).astype(np.int32, copy=False)

    # Filter distinct vertices.
    valid = (candidates[:, 0] != candidates[:, 1]) & \
            (candidates[:, 0] != candidates[:, 2]) & \
            (candidates[:, 1] != candidates[:, 2])
    candidates = candidates[valid]
    if candidates.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    # Order doesn't matter: sort rows for dedup + face filtering.
    candidates.sort(axis=1)
    candidates = unique_rows(candidates)

    candidates = rows_in_A_not_in_B(candidates, faces_sorted)
    return candidates[:n].astype(np.int64, copy=False)

def sample_non_face_triplets(faces, num_vertices, n, max_attempts=100000):
    """
    faces: (F, 3) array of vertex indices
    num_vertices: total number of vertices
    n: number of non-face triplets to return
    """

    # Normalize faces (orientation-insensitive)
    face_set = set(map(tuple, np.sort(faces, axis=1)))

    result = set()
    attempts = 0

    while len(result) < n and attempts < max_attempts:
        attempts += 1

        # Sample 3 distinct vertices
        triplet = tuple(sorted(
            np.random.choice(num_vertices, size=3, replace=False)
        ))

        if triplet in face_set or triplet in result:
            continue

        result.add(triplet)

    if len(result) < n:
        raise ValueError(
            f"Only found {len(result)} non-face triplets; requested {n}"
        )

    return np.array(list(result), dtype=int)

def sample_non_face_triplets_fast(faces, num_vertices, n, oversample=4):
    """
    faces: (F, 3) int array
    num_vertices: total number of vertices
    n: target number of triplets (approximate)
    oversample: how much to oversample to compensate for rejections
    """

    # Number of candidates to generate
    m = int(oversample * n)

    # Sample vertices (m, 3), no replacement per row
    candidates = np.random.choice(num_vertices, (m, 3), replace=True)
    valid = (candidates[:, 0] != candidates[:, 1]) & \
            (candidates[:, 1] != candidates[:, 2])
    
    candidates = candidates[valid]
    # Remove duplicates inside batch
    candidates = unique_rows(candidates)
    candidates.sort(axis=1)

    candidates = rows_in_A_not_in_B(candidates, np.sort(faces, axis=1))
    # print('m', m, 'candidates', candidates.shape[0], 'n', n)

    # Remove degenerate duplicates within batch
    return candidates[:n]

def normalize_rows(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > eps)

def triangle_normals(vertices, faces):
    if faces.shape[0] == 0:
        return np.zeros((0, 3), dtype=vertices.dtype)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return normalize_rows(np.cross(v1 - v0, v2 - v0))

def sort_rows(rows):
    rows = rows.astype(np.int64, copy=True)
    rows.sort(axis=1)
    return rows

def round_half_away_from_zero(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def quad_split_triangles(quad_faces):
    if quad_faces.shape[0] == 0:
        return np.zeros((0, 3), dtype=quad_faces.dtype)
    return np.concatenate(
        [
            quad_faces[:, [0, 1, 2]],
            quad_faces[:, [0, 2, 3]],
            quad_faces[:, [0, 1, 3]],
            quad_faces[:, [1, 2, 3]],
        ],
        axis=0,
    )

def nx_triangles_sorted(G):
    triangles = nx_all_triangles(G)
    if triangles.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.sort(triangles, axis=1)

class SpaceTimeQuadAEDataset(IterableDataset, Stateful):
    worker_shard_data = ['instance_ids']
    
    def __init__(
        self,
        instance_list: Union[str, List[str]],
        repeats: int = 1,
        shuffle_seed: int = 0,
        # specify_instance: Optional[str] = None,
        aug_flip: bool = True,
        aug_rotate_all: bool = False,
        aug_rotate_z: bool = True,
        aug_scale: bool = False,
        aug_scale_range: List[float] = [0.8, 1.2],
        yup_to_zup: bool = False,
        vertex_noise: float = 0.0,
        extra_feat: str = 'none',
        include_face: bool = False,
        include_face_orient: bool = False,
        num_face_range: List[int] = [0, 20000],
        num_vertex_range: List[int] = [0, 100000],
        vertex_resolutions: List[int] = [-1], 
        vertex_position_type: str = 'none',
        face_negative: str = 'random',
        mode: str = 'tri_connect',
        diag_as_edge: bool = True,
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
        assert vertex_position_type in ['none', 'int']
        assert face_negative in ['none', 'random']
        assert mode in ['tri', 'tri_connect', 'tri_bi_connect', 'native_quad', 'native_quad_wireframe']
        if not include_face:
            raise ValueError("SpaceTimeQuadAEDataset requires include_face=True")
        if vertex_position_type == 'int' and not any(v > 0 for v in vertex_resolutions):
            raise ValueError("vertex_position_type='int' requires a positive vertex_resolution")
        self.vertex_position_type = vertex_position_type

        self.path_io = DatasetPathIO()
        self.sample_idx = 0

        if isinstance(instance_list, str):
            instance_list = [instance_list]
        
        instance_ids = []
        required_columns = {'uuid', 'bucket', 'bos_path'}
        for jp in instance_list:
            if not jp.endswith('.parquet'):
                raise ValueError("SpaceTimeQuadAEDataset instance_list only accepts parquet files")
            df = pd.read_parquet(jp)
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in {jp}: {sorted(missing_columns)}")
            # df = df[
            #     (df['num_faces'].between(num_face_range[0], num_face_range[1])) &
            #     (df['num_vertices'].between(num_vertex_range[0], num_vertex_range[1]))
            # ]
            instance_ids.extend(list(df[['uuid', 'bucket', 'bos_path']].itertuples(index=False, name=None)))

        instance_ids = instance_ids * repeats
        if force_divisible_by > 1:
            instance_ids = instance_ids[len(instance_ids) % force_divisible_by:]
            logger.info(f"Instance number after dropping: {len(instance_ids)}")        

        instance_ids = shard_interleave(instance_ids, dp_world_size, dp_rank)

        rng = random.Random(shuffle_seed)
        rng.shuffle(instance_ids)

        print("rank", dp_rank, instance_ids[:10])

        self.instance_ids = instance_ids

        self.packed = False

        self.mp = MeshProcessor()
        self.aug_flip = aug_flip
        self.aug_rotate_all = aug_rotate_all
        self.aug_rotate_z = aug_rotate_z
        self.aug_scale = aug_scale
        self.aug_scale_range = aug_scale_range
        self.yup_to_zup = yup_to_zup
        self.vertex_noise = vertex_noise
        self.include_face = include_face
        self.vertex_resolutions = vertex_resolutions
        self.include_face_orient = include_face_orient
        self.extra_feat = extra_feat
        self.face_negative = face_negative
        self.mode = mode
        self.diag_as_edge = diag_as_edge
        self.return_mesh_mixed = return_mesh_mixed
        assert self.extra_feat in ['none', 'normal', 'face_normal']
        print("Augmentations", "flip", aug_flip, "rotate_all", aug_rotate_all, "rotate_z", aug_rotate_z, "scale", aug_scale, "scale_range", aug_scale_range)
        print("dp_rank", dp_rank, "dp_world_size", dp_world_size)

    def _sample_face_triplets(self, positive_faces, edge_positive_faces, graph_edges, graph, nv, n_ref):
        all_triplets = nx_triangles_sorted(graph)
        all_faces = sort_rows(positive_faces)

        remain = rows_in_A_not_in_B(all_triplets, all_faces)
        # print(all_triplets.shape, all_faces.shape, remain.shape)
        assert np.logical_and(remain >= 0, remain < nv).all()
        if self.face_negative == 'random':
            v_triplet_neg = random_non_face_triplet_with_edge_fast(
                graph_edges,
                edge_positive_faces,
                nv,
                10 * n_ref,
                oversample=1.5,
            )
            v_triplet = np.concatenate([all_faces, remain, v_triplet_neg], axis=0)
            v_triplet_label = np.concatenate(
                [np.ones(all_faces.shape[0]), np.zeros(remain.shape[0] + v_triplet_neg.shape[0])],
                axis=0,
            )

            v_triplet_neg_extra = sample_non_face_triplets_fast(v_triplet, nv, 50 * n_ref, oversample=1.5)
            v_triplet = np.concatenate([v_triplet, v_triplet_neg_extra], axis=0)
            v_triplet_label = np.concatenate([v_triplet_label, np.zeros(v_triplet_neg_extra.shape[0])], axis=0)
        else:
            v_triplet = np.concatenate([all_faces, remain], axis=0)
            v_triplet_label = np.concatenate([np.ones(all_faces.shape[0]), np.zeros(remain.shape[0])], axis=0)

        return v_triplet, v_triplet_label

    def _get_data_tri_connect(self, uuid, mesh_mixed, vertex_position, vertex_position_type, return_supervision=True):
        mesh_tri, face_to_tris, _ = mesh_mixed.to_triangle_mesh(return_mesh=True)
        vertices = np.copy(mesh_tri.vertices)
        faces = np.copy(mesh_tri.faces[:, :3])
        mesh_process = trimesh.Trimesh(vertices, faces, process=False)

        quad_indices = np.where(mesh_mixed.is_quad)[0]
        nv = vertices.shape[0]
        if return_supervision:
            edges = np.copy(mesh_process.edges_unique)
            diag_edges = mesh_mixed.faces[quad_indices][:, [0, 2]] if len(quad_indices) > 0 else np.zeros((0, 2), dtype=np.int32)
            diag_edges.sort(axis=1)
            diag_edges = np.unique(diag_edges, axis=0)

            # Store positives only. The all-pair negatives are generated by the
            # quad VAE loss on device to avoid materializing O(V^2) CPU tensors.
            vertex_pair = edges.astype(np.int64, copy=True)
            vertex_pair.sort(axis=1)
            vertex_pair = np.unique(vertex_pair, axis=0)
            vertex_pair_diag = diag_edges.astype(np.int64, copy=True)

        nf = faces.shape[0]
        face_center = vertices[faces].mean(axis=1)
        all_nodes = np.concatenate([vertices, face_center], axis=0)
        if self.extra_feat == 'normal':
            face_normals = np.copy(mesh_process.face_normals)
            vertex_normals = np.copy(mesh_process.vertex_normals)
            extra_feats = np.concatenate([vertex_normals, face_normals], axis=0)
            all_nodes = np.concatenate([all_nodes, extra_feats], axis=1)
        elif self.extra_feat == 'face_normal':
            face_normals = np.copy(mesh_process.face_normals)
            vertex_normals = np.zeros((nv, 3))
            extra_feats = np.concatenate([vertex_normals, face_normals], axis=0)
            all_nodes = np.concatenate([all_nodes, extra_feats], axis=1)
        if vertex_position_type == 'int':
            face_position = vertex_position[faces].mean(axis=1)
            position = np.concatenate([vertex_position, face_position], axis=0)
        node_type = np.ones(nv + nf, dtype=np.int64)
        node_type[:nv] = 0
        if len(quad_indices) > 0:
            node_type[nv + face_to_tris[quad_indices].reshape(-1)] = 2

        face_indices = np.arange(nv, nv + nf, dtype=np.int64)
        face_repeat = np.repeat(face_indices, 3)
        verts_flat = faces.reshape(-1)
        edges_fv = np.stack([face_repeat, verts_flat], axis=1)

        quad_face_edges = []
        if len(quad_indices) > 0:
            quad_tri_indices = face_to_tris[quad_indices]
            quad_face_edges = nv + quad_tri_indices
        if len(quad_face_edges) > 0:
            edges_fv = np.concatenate([edges_fv, quad_face_edges.astype(np.int64, copy=False)], axis=0)

        ret = {
            'nodes': torch.from_numpy(all_nodes).float(),
            'edges': torch.from_numpy(edges_fv).long(),
            'instance_id': uuid,
            'node_type': torch.from_numpy(node_type).long(),
        }

        if return_supervision:
            g = mesh_process.vertex_adjacency_graph
            v_triplet, v_triplet_label = self._sample_face_triplets(faces, faces, edges, g, nv, nf)
            ret.update({
                'vertex_pair': torch.from_numpy(vertex_pair).long(),
                'vertex_pair_diag': torch.from_numpy(vertex_pair_diag).long(),
                'vertex_triplet': torch.from_numpy(v_triplet).long(),
                'vertex_triplet_label': torch.from_numpy(v_triplet_label).long(),
            })

        if return_supervision and self.include_face_orient:
            face_in_order = torch.from_numpy(faces).long()
            orient_label = torch.ones(faces.shape[0], dtype=torch.long)
            ret.update({
                'face_in_order': face_in_order,
                'orient_label': orient_label,
            })
        
        if vertex_position_type == 'int':
            # ret['vertex_position'] = torch.from_numpy(vertex_position).long()
            ret['position'] = torch.from_numpy(position).long()

        return ret

    def _get_data_tri_bi_connect(self, uuid, mesh_mixed, vertex_position, vertex_position_type, return_supervision=True):
        vertices = np.copy(mesh_mixed.vertices)
        mixed_faces = np.copy(mesh_mixed.faces)
        is_quad = np.copy(mesh_mixed.is_quad)
        nv = vertices.shape[0]

        tri_indices = np.where(~is_quad)[0]
        quad_indices = np.where(is_quad)[0]
        tri_faces = mixed_faces[tri_indices, :3]
        quad_faces = mixed_faces[quad_indices, :4]

        face_parts = []
        if len(tri_indices) > 0:
            face_parts.append(tri_faces)
        if len(quad_indices) > 0:
            face_parts.extend([
                quad_faces[:, [0, 1, 2]],
                quad_faces[:, [0, 2, 3]],
                quad_faces[:, [0, 1, 3]],
                quad_faces[:, [1, 2, 3]],
            ])
        if len(face_parts) == 0:
            faces = np.zeros((0, 3), dtype=np.int32)
        else:
            faces = np.concatenate(face_parts, axis=0).astype(np.int32, copy=False)

        nf = faces.shape[0]
        mesh_process = trimesh.Trimesh(vertices, faces, process=False)

        if return_supervision:
            vertex_pair = mesh_mixed.edges().astype(np.int64, copy=True)
            vertex_pair.sort(axis=1)
            vertex_pair = np.unique(vertex_pair, axis=0)

            if len(quad_indices) > 0:
                diag_edges = np.concatenate([quad_faces[:, [0, 2]], quad_faces[:, [1, 3]]], axis=0)
                diag_edges.sort(axis=1)
                vertex_pair_diag = np.unique(diag_edges, axis=0).astype(np.int64, copy=False)
            else:
                vertex_pair_diag = np.zeros((0, 2), dtype=np.int64)

            if self.diag_as_edge and vertex_pair_diag.shape[0] > 0:
                vertex_pair = np.concatenate([vertex_pair, vertex_pair_diag], axis=0)
                vertex_pair = np.unique(vertex_pair, axis=0)

        face_center = vertices[faces].mean(axis=1)
        all_nodes = np.concatenate([vertices, face_center], axis=0)
        if self.extra_feat == 'normal':
            face_normals = np.copy(mesh_process.face_normals)
            vertex_normals = np.copy(mesh_process.vertex_normals)
            extra_feats = np.concatenate([vertex_normals, face_normals], axis=0)
            all_nodes = np.concatenate([all_nodes, extra_feats], axis=1)
        elif self.extra_feat == 'face_normal':
            face_normals = np.copy(mesh_process.face_normals)
            vertex_normals = np.zeros((nv, 3))
            extra_feats = np.concatenate([vertex_normals, face_normals], axis=0)
            all_nodes = np.concatenate([all_nodes, extra_feats], axis=1)
        if vertex_position_type == 'int':
            face_position = vertex_position[faces].mean(axis=1)
            position = np.concatenate([vertex_position, face_position], axis=0)

        node_type = np.ones(nv + nf, dtype=np.int64)
        node_type[:nv] = 0
        if len(quad_indices) > 0:
            quad_face_start = len(tri_indices)
            node_type[nv + quad_face_start:] = 2

        face_indices = np.arange(nv, nv + nf, dtype=np.int64)
        face_repeat = np.repeat(face_indices, 3)
        verts_flat = faces.reshape(-1)
        edges_fv = np.stack([face_repeat, verts_flat], axis=1)

        if len(quad_indices) > 0:
            quad_face_start = nv + len(tri_indices)
            n_quads = len(quad_indices)
            quad_face_edges = np.concatenate(
                [
                    np.stack(
                        [
                            np.arange(quad_face_start, quad_face_start + n_quads, dtype=np.int64),
                            np.arange(quad_face_start + n_quads, quad_face_start + 2 * n_quads, dtype=np.int64),
                        ],
                        axis=1,
                    ),
                    np.stack(
                        [
                            np.arange(quad_face_start + 2 * n_quads, quad_face_start + 3 * n_quads, dtype=np.int64),
                            np.arange(quad_face_start + 3 * n_quads, quad_face_start + 4 * n_quads, dtype=np.int64),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
            edges_fv = np.concatenate([edges_fv, quad_face_edges], axis=0)

        ret = {
            'nodes': torch.from_numpy(all_nodes).float(),
            'edges': torch.from_numpy(edges_fv).long(),
            'instance_id': uuid,
            'node_type': torch.from_numpy(node_type).long(),
        }

        if return_supervision:
            g = nx.Graph()
            g.add_nodes_from(range(nv))
            graph_edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
            graph_edges.sort(axis=1)
            graph_edges = np.unique(graph_edges, axis=0)
            g.add_edges_from(graph_edges)
            v_triplet, v_triplet_label = self._sample_face_triplets(faces, faces, graph_edges, g, nv, nf)
            ret.update({
                'vertex_pair': torch.from_numpy(vertex_pair).long(),
                'vertex_pair_diag': torch.from_numpy(vertex_pair_diag).long(),
                'vertex_triplet': torch.from_numpy(v_triplet).long(),
                'vertex_triplet_label': torch.from_numpy(v_triplet_label).long(),
            })

        if return_supervision and self.include_face_orient:
            face_in_order = torch.from_numpy(faces).long()
            orient_label = torch.ones(faces.shape[0], dtype=torch.long)
            ret.update({
                'face_in_order': face_in_order,
                'orient_label': orient_label,
            })

        if vertex_position_type == 'int':
            ret['position'] = torch.from_numpy(position).long()

        return ret

    def _get_data_tri(self, uuid, mesh_mixed, vertex_position, vertex_position_type, return_supervision=True):
        mesh_tri, _, _ = mesh_mixed.to_triangle_mesh(return_mesh=True)
        vertices = np.copy(mesh_tri.vertices)
        faces = np.copy(mesh_tri.faces[:, :3])
        mesh_process = trimesh.Trimesh(vertices, faces, process=False)
        nv = vertices.shape[0]
        if return_supervision:
            edges = np.copy(mesh_process.edges_unique)
            vertex_pair = edges.astype(np.int64, copy=True)
            vertex_pair.sort(axis=1)
            vertex_pair = np.unique(vertex_pair, axis=0)

        nf = faces.shape[0]
        face_center = vertices[faces].mean(axis=1)
        all_nodes = np.concatenate([vertices, face_center], axis=0)
        if self.extra_feat == 'normal':
            face_normals = np.copy(mesh_process.face_normals)
            vertex_normals = np.copy(mesh_process.vertex_normals)
            extra_feats = np.concatenate([vertex_normals, face_normals], axis=0)
            all_nodes = np.concatenate([all_nodes, extra_feats], axis=1)
        elif self.extra_feat == 'face_normal':
            face_normals = np.copy(mesh_process.face_normals)
            vertex_normals = np.zeros((nv, 3))
            extra_feats = np.concatenate([vertex_normals, face_normals], axis=0)
            all_nodes = np.concatenate([all_nodes, extra_feats], axis=1)
        if vertex_position_type == 'int':
            face_position = vertex_position[faces].mean(axis=1)
            position = np.concatenate([vertex_position, face_position], axis=0)

        node_type = np.ones(nv + nf, dtype=np.int64)
        node_type[:nv] = 0

        face_indices = np.arange(nv, nv + nf, dtype=np.int64)
        face_repeat = np.repeat(face_indices, 3)
        verts_flat = faces.reshape(-1)
        edges_fv = np.stack([face_repeat, verts_flat], axis=1)

        ret = {
            'nodes': torch.from_numpy(all_nodes).float(),
            'edges': torch.from_numpy(edges_fv).long(),
            'instance_id': uuid,
            'node_type': torch.from_numpy(node_type).long(),
        }

        if return_supervision:
            g = mesh_process.vertex_adjacency_graph
            v_triplet, v_triplet_label = self._sample_face_triplets(faces, faces, edges, g, nv, nf)
            ret.update({
                'vertex_pair': torch.from_numpy(vertex_pair).long(),
                'vertex_triplet': torch.from_numpy(v_triplet).long(),
                'vertex_triplet_label': torch.from_numpy(v_triplet_label).long(),
            })

        if return_supervision and self.include_face_orient:
            face_in_order = torch.from_numpy(faces).long()
            orient_label = torch.ones(faces.shape[0], dtype=torch.long)
            ret.update({
                'face_in_order': face_in_order,
                'orient_label': orient_label,
            })

        if vertex_position_type == 'int':
            ret['position'] = torch.from_numpy(position).long()

        return ret

    def _native_face_normals(self, vertices, faces, is_quad):
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
        return face_normals

    def _native_vertex_normals(self, vertices, faces, is_quad, face_normals):
        vertex_normals = np.zeros_like(vertices)
        tri_faces = faces[~is_quad, :3]
        tri_normals = face_normals[~is_quad]
        for corner in range(3):
            np.add.at(vertex_normals, tri_faces[:, corner], tri_normals)
        quad_faces = faces[is_quad, :4]
        quad_normals = face_normals[is_quad]
        for corner in range(4):
            np.add.at(vertex_normals, quad_faces[:, corner], quad_normals)
        return normalize_rows(vertex_normals)

    def _get_data_native_quad(self, uuid, mesh_mixed, vertex_position, vertex_position_type, include_wireframe_edges=False, return_supervision=True):
        vertices = np.copy(mesh_mixed.vertices)
        faces = np.copy(mesh_mixed.faces)
        is_quad = np.copy(mesh_mixed.is_quad)
        nv = vertices.shape[0]
        nf = faces.shape[0]

        tri_indices = np.where(~is_quad)[0]
        quad_indices = np.where(is_quad)[0]
        tri_faces = faces[tri_indices, :3]
        quad_faces = faces[quad_indices, :4]

        face_center = np.zeros((nf, 3), dtype=vertices.dtype)
        if len(tri_indices) > 0:
            face_center[tri_indices] = vertices[tri_faces].mean(axis=1)
        if len(quad_indices) > 0:
            face_center[quad_indices] = vertices[quad_faces].mean(axis=1)
        all_nodes = np.concatenate([vertices, face_center], axis=0)

        if self.extra_feat in ['normal', 'face_normal']:
            face_normals = self._native_face_normals(vertices, faces, is_quad)
            if self.extra_feat == 'normal':
                vertex_normals = self._native_vertex_normals(vertices, faces, is_quad, face_normals)
            else:
                vertex_normals = np.zeros((nv, 3))
            extra_feats = np.concatenate([vertex_normals, face_normals], axis=0)
            all_nodes = np.concatenate([all_nodes, extra_feats], axis=1)

        if vertex_position_type == 'int':
            face_position = np.zeros((nf, vertex_position.shape[1]), dtype=vertex_position.dtype)
            if len(tri_indices) > 0:
                face_position[tri_indices] = round_half_away_from_zero(
                    vertex_position[tri_faces].mean(axis=1)
                ).astype(vertex_position.dtype, copy=False)
            if len(quad_indices) > 0:
                face_position[quad_indices] = round_half_away_from_zero(
                    vertex_position[quad_faces].mean(axis=1)
                ).astype(vertex_position.dtype, copy=False)
            position = np.concatenate([vertex_position, face_position], axis=0)

        node_type = np.zeros(nv + nf, dtype=np.int64)
        node_type[nv + tri_indices] = 1
        node_type[nv + quad_indices] = 2

        edges_fv_parts = []
        if len(tri_indices) > 0:
            face_repeat = np.repeat(nv + tri_indices.astype(np.int64), 3)
            edges_fv_parts.append(np.stack([face_repeat, tri_faces.reshape(-1)], axis=1))
        if len(quad_indices) > 0:
            face_repeat = np.repeat(nv + quad_indices.astype(np.int64), 4)
            edges_fv_parts.append(np.stack([face_repeat, quad_faces.reshape(-1)], axis=1))
        edges_fv = np.concatenate(edges_fv_parts, axis=0).astype(np.int64, copy=False)

        if include_wireframe_edges or return_supervision:
            vertex_pair = mesh_mixed.edges().astype(np.int64, copy=True)
            vertex_pair.sort(axis=1)
            vertex_pair = np.unique(vertex_pair, axis=0)
        if include_wireframe_edges:
            edges_fv = np.concatenate([edges_fv, vertex_pair], axis=0)

        ret = {
            'nodes': torch.from_numpy(all_nodes).float(),
            'edges': torch.from_numpy(edges_fv).long(),
            'instance_id': uuid,
            'node_type': torch.from_numpy(node_type).long(),
        }

        if return_supervision:
            if len(quad_indices) > 0:
                diag_edges = np.concatenate([quad_faces[:, [0, 2]], quad_faces[:, [1, 3]]], axis=0)
                diag_edges.sort(axis=1)
                vertex_pair_diag = np.unique(diag_edges, axis=0).astype(np.int64, copy=False)
            else:
                vertex_pair_diag = np.zeros((0, 2), dtype=np.int64)

            quad_positive_faces = quad_split_triangles(quad_faces)
            positive_faces = np.concatenate([tri_faces, quad_positive_faces], axis=0).astype(np.int64, copy=False)
            # positive_faces_unique = np.sort(positive_faces, axis=1)
            # positive_faces_unique = np.unique(positive_faces_unique, axis=0)
            # print('positive faces', positive_faces_unique.shape, positive_faces.shape)

            g = nx.Graph()
            g.add_nodes_from(range(nv))
            all_edges = np.concatenate([positive_faces[:, [0, 1]], positive_faces[:, [1, 2]], positive_faces[:, [2, 0]]], axis=0)
            all_edges.sort(axis=1)
            all_edges = np.unique(all_edges, axis=0)
            g.add_edges_from(all_edges)
            v_triplet, v_triplet_label = self._sample_face_triplets(
                positive_faces,
                positive_faces,
                vertex_pair,
                g,
                nv,
                positive_faces.shape[0],
            )
            ret.update({
                'vertex_pair': torch.from_numpy(vertex_pair).long(),
                'vertex_pair_diag': torch.from_numpy(vertex_pair_diag).long(),
                'vertex_triplet': torch.from_numpy(v_triplet).long(),
                'vertex_triplet_label': torch.from_numpy(v_triplet_label).long(),
            })

            if self.include_face_orient:
                face_in_order = torch.from_numpy(positive_faces).long()
                orient_label = torch.ones(positive_faces.shape[0], dtype=torch.long)
                ret.update({
                    'face_in_order': face_in_order,
                    'orient_label': orient_label,
                })

        if vertex_position_type == 'int':
            ret['position'] = torch.from_numpy(position).long()

        return ret

    def get_data(self, instance_id: Tuple[str, str, str]):
        uuid, bucket, path = instance_id
        if bucket == "":
            mesh = self.path_io.read_mixed_mesh(path)
        else:
            bos_url = f"bos://{bucket}/{path}"
            mesh = self.path_io.read_mixed_mesh(bos_url)

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
            faces_unified.append(np.concatenate([faces_tri, np.full((faces_tri.shape[0], 1), -1, dtype=faces_tri.dtype)], axis=-1))
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

        if v_res  > 0:
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

        if self.mode == 'tri':
            ret = self._get_data_tri(uuid, mesh_mixed, vertex_position, vertex_position_type=self.vertex_position_type)
        elif self.mode == 'tri_connect':
            ret = self._get_data_tri_connect(uuid, mesh_mixed, vertex_position, vertex_position_type=self.vertex_position_type)
        elif self.mode == 'tri_bi_connect':
            ret = self._get_data_tri_bi_connect(uuid, mesh_mixed, vertex_position, vertex_position_type=self.vertex_position_type)
        elif self.mode == 'native_quad':
            ret = self._get_data_native_quad(uuid, mesh_mixed, vertex_position, vertex_position_type=self.vertex_position_type)
        elif self.mode == 'native_quad_wireframe':
            ret = self._get_data_native_quad(uuid, mesh_mixed, vertex_position, vertex_position_type=self.vertex_position_type, include_wireframe_edges=True)
        else:
            raise NotImplementedError(self.mode)

        if self.return_mesh_mixed:
            ret['mesh_mixed'] = mesh_mixed

        return ret

    def __iter__(self):
        while True:
            sample_idx = self.sample_idx

            while True:
                try:
                    if not self.packed:
                        yield [self.get_data(self.instance_ids[sample_idx])]
                    else:
                        yield [self.get_data(idx) for idx in self.instance_ids[sample_idx]]
                    break
                except GeneratorExit:
                    raise
                except:
                    logger.warning(f"Failed to load data for instance {self.instance_ids[sample_idx]}: {traceback.format_exc()}")
                    sample_idx = random.randint(0, len(self.instance_ids) - 1)

            self.sample_idx += 1

            if self.sample_idx >= len(self.instance_ids):
                if not self.infinite:
                    logger.warning(f"Dataset has run out of data.")
                    break
                else:
                    self.sample_idx = 0
                    # random.shuffle(self.instance_ids)
                    logger.warning(f"Dataset is being re-looped.")

    def load_state_dict(self, state_dict):
        self.sample_idx = state_dict["sample_idx"]
        self.instance_ids = state_dict["instance_ids"]
    
    def state_dict(self):
        return {
            "instance_ids": self.instance_ids,
            "sample_idx": self.sample_idx,
        }
        
    def _collate_fn(self, batch):
        # flatten batches
        def flatten_list(l):
            return [item for sublist in l for item in sublist]
        
        batch = flatten_list(batch)

        # merge the mesh into a large graph
        batch_collated = {}
        offset = [0]
        nodes = []
        edges = []
        vertex_pair = []
        vertex_pair_diag = []
        vertex_triplet = []
        pair_offsets = [0]
        diag_pair_offsets = [0]
        has_diag = any('vertex_pair_diag' in b for b in batch)
        if self.include_face:
            triplet_offsets = [0]
        if self.include_face_orient:
            # orient_offsets = [0]
            face_in_order = []

        for b in batch:
            nodes.append(b['nodes'])
            edges.append(b['edges'] + offset[-1])
            vertex_pair.append(b['vertex_pair'] + offset[-1])
            pair_offsets.append(pair_offsets[-1] + b['vertex_pair'].shape[0])
            if has_diag:
                if 'vertex_pair_diag' not in b:
                    raise ValueError("Cannot collate a batch that mixes diag and non-diag samples")
                vertex_pair_diag.append(b['vertex_pair_diag'] + offset[-1])
                diag_pair_offsets.append(diag_pair_offsets[-1] + b['vertex_pair_diag'].shape[0])
            if self.include_face:
                vertex_triplet.append(b['vertex_triplet'] + offset[-1])
                triplet_offsets.append(triplet_offsets[-1] + b['vertex_triplet'].shape[0])
            if self.include_face_orient:
                face_in_order.append(b['face_in_order'] + offset[-1])
                # orient_offsets.append(orient_offsets[-1] + b['face_in_order'].shape[0])
            offset.append(offset[-1] + b['nodes'].shape[0])
        
        batch_collated['nodes'] = torch.cat(nodes, dim=0)
        batch_collated['edges'] = torch.cat(edges, dim=0)
        batch_collated['offsets'] = torch.tensor(offset, dtype=torch.int32)
        batch_collated['instance_ids'] = [b['instance_id'] for b in batch]
        batch_collated['vertex_pair'] = torch.cat(vertex_pair, dim=0)
        batch_collated['pair_offsets'] = torch.tensor(pair_offsets, dtype=torch.int32)
        if has_diag:
            batch_collated['vertex_pair_diag'] = torch.cat(vertex_pair_diag, dim=0)
            batch_collated['diag_pair_offsets'] = torch.tensor(diag_pair_offsets, dtype=torch.int32)
        if self.include_face:
            batch_collated['node_type'] = torch.cat([b['node_type'] for b in batch], dim=0)
            batch_collated['vertex_triplet_label'] = torch.cat([b['vertex_triplet_label'] for b in batch], dim=0)
            batch_collated['vertex_triplet'] = torch.cat(vertex_triplet, dim=0)
            batch_collated['triplet_offsets'] = torch.tensor(triplet_offsets, dtype=torch.int32)
        
        if self.include_face_orient:
            batch_collated['face_in_order'] = torch.cat(face_in_order, dim=0)
            batch_collated['orient_label'] = torch.cat([b['orient_label'] for b in batch], dim=0)
        
        if self.vertex_position_type == 'int':
            batch_collated['position'] = torch.cat([b['position'] for b in batch], dim=0)

        if self.return_mesh_mixed:
            batch_collated['mesh_mixed'] = [b['mesh_mixed'] for b in batch]

        del batch
        gc.collect()
        return batch_collated

    def collate_fn(self, batch):
        return self._collate_fn(batch)
