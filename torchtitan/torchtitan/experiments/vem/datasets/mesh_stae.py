from dataclasses import asdict
from typing import Any, Optional, List, Union, Dict
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
    # read_mesh,
    # read_mesh_from_bos,
    # normalize_vertices,
    # process,
    # sample,
    # sample_with_dora,
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

def rows_in_A_not_in_B(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(f"Expected 2D arrays with matching row widths, got {A.shape} and {B.shape}")

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    # convert rows to a single structured dtype so we can use set operations
    A_view = A.view([('', A.dtype)] * A.shape[1])
    B_view = B.view([('', B.dtype)] * B.shape[1])
    C_view = np.setdiff1d(A_view, B_view)
    return C_view.view(A.dtype).reshape(-1, A.shape[1])

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
    candidates = np.unique(candidates, axis=0)

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
    candidates = np.unique(candidates, axis=0)
    candidates.sort(axis=1)

    candidates = rows_in_A_not_in_B(candidates, np.sort(faces, axis=1))
    # print('m', m, 'candidates', candidates.shape[0], 'n', n)

    # Remove degenerate duplicates within batch
    return candidates[:n]

class SpaceTimeAEDataset(IterableDataset, Stateful):
    worker_shard_data = ['instance_ids']
    
    def __init__(
        self,
        instance_list: Union[str, List[str]],
        repeats: int = 1,
        shuffle_seed: int = 0,
        # specify_instance: Optional[str] = None,
        use_bos: bool = False,
        bos_bucket: Optional[str] = None,
        aug_flip: bool = True,
        aug_rotate_all: bool = False,
        aug_rotate_z: bool = True,
        aug_scale: bool = False,
        aug_scale_range: List[float] = [0.8, 1.2],
        yup_to_zup: bool = False,
        vertex_noise: float = 0.0,
        random_negative: bool = False,
        extra_feat: str = 'none',
        include_face: bool = False,
        include_face_orient: bool = False,
        num_face_range: List[int] = [0, 20000],
        num_vertex_range: List[int] = [0, 100000],
        vertex_resolutions: List[int] = [-1], 
        vertex_position_type: str = 'none',
        face_negative: str = 'random',
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
        self.use_bos = use_bos
        assert vertex_position_type in ['none', 'int']
        assert face_negative in ['none', 'random']
        self.vertex_position_type = vertex_position_type

        if self.use_bos:
            self.bos_client = BOSClient()
            self.bos_bucket = bos_bucket
        else:
            self.bos_client = None
            self.bos_bucket = None

        self.sample_idx = 0

        if isinstance(instance_list, str):
            instance_list = [instance_list]
        
        instance_ids = []
        for jp in instance_list:
            if jp.endswith(".json") or jp.endswith(".json.gz"):
                instance_ids.extend(load_json(jp))
            elif jp.endswith('.parquet'):
                df = pd.read_parquet(jp)
                df = df[(df['num_faces'].between(num_face_range[0], num_face_range[1])) & (df['num_vertices'].between(num_vertex_range[0], num_vertex_range[1]))]
                instance_ids.extend(df["instance_id"].tolist())
            elif jp.endswith('.ply') or jp.endswith('.obj'):
                instance_ids.append(jp)

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
        self.random_negative = random_negative
        self.include_face = include_face
        self.vertex_resolutions = vertex_resolutions
        self.include_face_orient = include_face_orient
        self.extra_feat = extra_feat
        self.face_negative = face_negative
        assert self.extra_feat in ['none', 'normal', 'face_normal']
        print("Augmentations", "flip", aug_flip, "rotate_all", aug_rotate_all, "rotate_z", aug_rotate_z, "scale", aug_scale, "scale_range", aug_scale_range)
        print("dp_rank", dp_rank, "dp_world_size", dp_world_size)
    
    def get_data(self, instance_id: str):
        if self.use_bos:
            assert self.bos_client is not None
            mesh, pc = self.mp.read_mesh_from_bos(instance_id, self.bos_client, self.bos_bucket, process=True)
        else:
            mesh, pc = self.mp.read_mesh(instance_id)
        
        vertices, faces = self.mp.process(mesh, z_up=self.yup_to_zup)

        vertices, faces = self.mp.augment(
            vertices=vertices,
            faces=faces,
            aug_flip=self.aug_flip,
            aug_rotate_all=self.aug_rotate_all,
            aug_rotate_z=self.aug_rotate_z,
            aug_scale=self.aug_scale,
            aug_scale_range=self.aug_scale_range,
        )
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
            mesh_process = trimesh.Trimesh(vertices_dis, faces)
            mesh_process = self.mp.clear_mesh(mesh_process, digits_vertex=0)
            vertices, faces = np.copy(mesh_process.vertices), np.copy(mesh_process.faces)
            vertex_position = np.copy(vertices) * 3
            vertices = undiscretize(vertices, v_res)
        else:
            mesh_process = trimesh.Trimesh(vertices, faces)
            mesh_process = self.mp.clear_mesh(mesh_process, digits_vertex=6)
            vertices, faces = np.copy(mesh_process.vertices), np.copy(mesh_process.faces)
            vertex_position = None

        edges = np.copy(mesh_process.edges_unique)
        if self.random_negative:
            nv = vertices.shape[0]
            edge_pairs = set(map(tuple, edges))

            non_edge_sample = set()
            # Generate non-edges until we have as many as edges
            for i in range(int(edges.shape[0] * 1.1)):
                # Randomly sample 2 distinct vertices
                u = random.randint(0, nv - 1)
                v = random.randint(0, nv - 1)
                if u == v:
                    continue
                # Use frozenset to handle unordered pairs (u,v) == (v,u)
                pair = (u, v) if u < v else (v, u)
                if pair not in edge_pairs and pair not in non_edge_sample:
                    # Add as a sorted tuple to maintain consistency (optional)
                    non_edge_sample.add(pair)
            
            non_edge_sample = np.array(list(non_edge_sample))

            vertex_pair = np.concatenate([edges, non_edge_sample], axis=0)
            vertex_pair_label = np.zeros(vertex_pair.shape[0], dtype=np.int32)
            vertex_pair_label[:edges.shape[0]] = 1
        else:
            # all other vertex pairs as negative samples
            nv = vertices.shape[0]
            adj = np.zeros((nv, nv), dtype=np.int32)
            adj[edges[:, 0], edges[:, 1]] = 1
            iu, iv = np.triu_indices(nv, k=1)
            vertex_pair = np.stack([iu, iv], axis=1)
            vertex_pair_label = adj[iu, iv]

        # if faces.shape[0] > self.max_num_faces:
        #     print("Too many faces", instance_id, faces.shape[0])

        if not self.include_face:
            v_tensor = torch.from_numpy(vertices).float()
            e_tensor = torch.from_numpy(edges).long()
            ret = {
                'nodes': v_tensor,
                'edges': e_tensor,
                'vertex_pair': torch.from_numpy(vertex_pair).long(),
                'vertex_pair_label': torch.from_numpy(vertex_pair_label).long(),
                'instance_id': instance_id,
            }
        else:
            nv = vertices.shape[0]
            nf = faces.shape[0]
            # face centers as node features (3D)
            face_center = vertices[faces].mean(axis=1)  # (nf, 3)
            # concatenate vertex positions and face centers -> node features
            all_nodes = np.concatenate([vertices, face_center], axis=0)  # (nv+nf, 3)
            if self.extra_feat == 'normal':
                face_normals = np.copy(mesh_process.face_normals)
                vertex_normals = np.copy(mesh_process.vertex_normals)
                extra_feats = np.concatenate([vertex_normals, face_normals], axis=0)
                all_nodes = np.concatenate([all_nodes, extra_feats], axis=1) # (nv+nf, 6)
            elif self.extra_feat == 'face_normal':
                face_normals = np.copy(mesh_process.face_normals)
                vertex_normals = np.zeros((nv, 3))
                extra_feats = np.concatenate([vertex_normals, face_normals], axis=0)
                all_nodes = np.concatenate([all_nodes, extra_feats], axis=1) # (nv+nf, 6)
            if self.vertex_position_type == 'int':
                face_position = vertex_position[faces].mean(axis=1)
                position = np.concatenate([vertex_position, face_position], axis=0)
            vertex_mask = np.zeros(nv+nf, dtype=bool)
            vertex_mask[:nv] = True

            # face node indices in the concatenated node array
            face_indices = np.arange(nv, nv + nf, dtype=np.int64)  # (nf,)

            # For each face, create edges (face_idx <-> each vertex)
            # faces is (nf,3) containing vertex indices per face
            # Create directed edges both ways to represent undirected face-vertex relationship
            # First: face -> vertex (repeat face index 3 times)
            face_repeat = np.repeat(face_indices, 3)                      # (nf*3,)
            verts_flat = faces.reshape(-1)                                # (nf*3,)
            edges_fv = np.stack([face_repeat, verts_flat], axis=1)       # (nf*3, 2)

            # construct triplets
            g = mesh_process.vertex_adjacency_graph

            all_triplets = np.sort(nx_all_triangles(g), axis=1)
            all_faces = np.sort(faces, axis=1)

            remain = rows_in_A_not_in_B(all_triplets, all_faces)
            assert np.logical_and(remain >= 0, remain < nv).all()
            if self.face_negative == 'random':
                v_triplet_neg = random_non_face_triplet_with_edge_fast(edges, faces, nv, 10 * nf, oversample=1.5)
                v_triplet = np.concatenate([all_faces, remain, v_triplet_neg], axis=0)
                v_triplet_label = np.concatenate([np.ones(all_faces.shape[0]), np.zeros(remain.shape[0] + v_triplet_neg.shape[0])], axis=0)

                # also sample some random negative triplets
                v_triplet_neg_extra = sample_non_face_triplets_fast(v_triplet, nv, 50 * nf, oversample=1.5)
                v_triplet = np.concatenate([v_triplet, v_triplet_neg_extra], axis=0)
                v_triplet_label = np.concatenate([v_triplet_label, np.zeros(v_triplet_neg_extra.shape[0])], axis=0)
            else:
                v_triplet = np.concatenate([all_faces, remain], axis=0)
                v_triplet_label = np.concatenate([np.ones(all_faces.shape[0]), np.zeros(remain.shape[0])], axis=0)

            ret = {
                'nodes': torch.from_numpy(all_nodes).float(),            # (nv+nf, 3)
                'edges': torch.from_numpy(edges_fv).long(),
                # vertex_pair should refer to vertex indices only (0..nv-1) — unchanged
                'vertex_pair': torch.from_numpy(vertex_pair).long(),
                'vertex_pair_label': torch.from_numpy(vertex_pair_label).long(),
                'instance_id': instance_id,
                'vertex_mask': torch.from_numpy(vertex_mask).bool(),
                'vertex_triplet': torch.from_numpy(v_triplet).long(),
                'vertex_triplet_label': torch.from_numpy(v_triplet_label).long(),
            }

        if self.include_face_orient:
            face_in_order = torch.from_numpy(faces).long()
            orient_label = torch.ones(faces.shape[0], dtype=torch.long)
            ret.update({
                'face_in_order': face_in_order,
                'orient_label': orient_label,
            })
        
        if self.vertex_position_type == 'int':
            # ret['vertex_position'] = torch.from_numpy(vertex_position).long()
            ret['position'] = torch.from_numpy(position).long()

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
        vertex_triplet = []
        pair_offsets = [0]
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
        batch_collated['vertex_pair_label'] = torch.cat([b['vertex_pair_label'] for b in batch], dim=0)
        batch_collated['pair_offsets'] = torch.tensor(pair_offsets, dtype=torch.int32)
        if self.include_face:
            batch_collated['vertex_mask'] = torch.cat([b['vertex_mask'] for b in batch], dim=0)
            batch_collated['vertex_triplet_label'] = torch.cat([b['vertex_triplet_label'] for b in batch], dim=0)
            batch_collated['vertex_triplet'] = torch.cat(vertex_triplet, dim=0)
            batch_collated['triplet_offsets'] = torch.tensor(triplet_offsets, dtype=torch.int32)
        
        if self.include_face_orient:
            batch_collated['face_in_order'] = torch.cat(face_in_order, dim=0)
            batch_collated['orient_label'] = torch.cat([b['orient_label'] for b in batch], dim=0)
        
        if self.vertex_position_type == 'int':
            batch_collated['position'] = torch.cat([b['position'] for b in batch], dim=0)

        del batch
        gc.collect()
        return batch_collated

    def collate_fn(self, batch):
        return self._collate_fn(batch)




# class SpaceMeshGenDatasetDeprecated(IterableDataset, Stateful):
#     worker_shard_data = ['instance_ids']
    
#     def __init__(
#         self,
#         instance_list: Union[str, List[str]],
#         repeats: int = 1,
#         shuffle_seed: int = 0,
#         # specify_instance: Optional[str] = None,
#         use_bos: bool = False,
#         bos_bucket: Optional[str] = None,
#         aug_flip: bool = True,
#         aug_rotate_all: bool = False,
#         aug_rotate_z: bool = True,
#         aug_scale: bool = False,
#         aug_scale_range: List[float] = [0.8, 1.2],
#         yup_to_zup: bool = False,
#         vertex_noise: float = 0.0,
#         max_num_face: int = -1,
#         # auto assigned args
#         infinite: bool = True,
#         force_divisible_by: int = 1,
#         dp_rank: int = 0,
#         dp_world_size: int = 1,
#     ) -> None:
#         self.repeats = repeats
#         self.shuffle_seed = shuffle_seed
#         self.infinite = infinite
#         self.dp_rank = dp_rank
#         self.dp_world_size = dp_world_size
#         self.use_bos = use_bos

#         if self.use_bos:
#             self.bos_client = BOSClient()
#             self.bos_bucket = bos_bucket
#         else:
#             self.bos_client = None
#             self.bos_bucket = None

#         self.sample_idx = 0

#         if isinstance(instance_list, str):
#             instance_list = [instance_list]
        
#         instance_ids = []
#         for jp in instance_list:
#             if jp.endswith(".json") or jp.endswith(".json.gz"):
#                 instance_ids.extend(load_json(jp))
#             elif jp.endswith('.parquet'):
#                 df = pd.read_parquet(jp)
#                 if max_num_face > 0:
#                     df = df[df['num_faces'] <= max_num_face]
#                 instance_ids.extend(df["instance_id"].tolist())
#             elif jp.endswith('.ply'):
#                 instance_ids.append(jp)
#             elif jp.endswith('.glb'):
#                 instance_ids.append(jp)

#         instance_ids = instance_ids * repeats
#         if force_divisible_by > 1:
#             instance_ids = instance_ids[len(instance_ids) % force_divisible_by:]
#             logger.info(f"Instance number after dropping: {len(instance_ids)}")        

#         instance_ids = shard_interleave(instance_ids, dp_world_size, dp_rank)

#         rng = random.Random(shuffle_seed)
#         rng.shuffle(instance_ids)

#         print("rank", dp_rank, instance_ids[:10])

#         self.instance_ids = instance_ids

#         self.packed = False

#         self.mp = MeshProcessor()
#         self.aug_flip = aug_flip
#         self.aug_rotate_all = aug_rotate_all
#         self.aug_rotate_z = aug_rotate_z
#         self.aug_scale = aug_scale
#         self.aug_scale_range = aug_scale_range
#         self.yup_to_zup = yup_to_zup
#         self.vertex_noise = vertex_noise
#         print("Augmentations", "flip", aug_flip, "rotate_all", aug_rotate_all, "rotate_z", aug_rotate_z, "scale", aug_scale, "scale_range", aug_scale_range)
#         print("dp_rank", dp_rank, "dp_world_size", dp_world_size)
    
#     def get_data(self, instance_id: str):
#         if self.use_bos:
#             assert self.bos_client is not None
#             mesh, pc = self.mp.read_mesh_from_bos(instance_id, self.bos_client, self.bos_bucket, process=True)
#         else:
#             mesh, pc = self.mp.read_mesh(instance_id)
        
#         vertices, faces = self.mp.process(mesh, z_up=self.yup_to_zup)

#         vertices, faces = self.mp.augment(
#             vertices=vertices,
#             faces=faces,
#             aug_flip=self.aug_flip,
#             aug_rotate_all=self.aug_rotate_all,
#             aug_rotate_z=self.aug_rotate_z,
#             aug_scale=self.aug_scale,
#             aug_scale_range=self.aug_scale_range,
#         )

#         mesh_process = trimesh.Trimesh(vertices, faces)
#         mesh_process = self.mp.clear_mesh(mesh_process, digits_vertex=6)

#         vertices, faces = mesh_process.vertices, mesh_process.faces

#         edges = np.copy(mesh_process.edges_unique)

#         v_tensor = torch.from_numpy(vertices).float()
#         e_tensor = torch.from_numpy(edges).long()
#         if self.vertex_noise > 0:
#             noise_level = rand_with_pt([0, self.vertex_noise])
#             v_tensor += torch.randn_like(v_tensor) * noise_level

#         ret = {
#             'vertices': v_tensor,
#             'edges': e_tensor,
#             'instance_id': instance_id,
#         }

#         return ret

#     def __iter__(self):
#         while True:
#             sample_idx = self.sample_idx

#             while True:
#                 try:
#                     if not self.packed:
#                         yield [self.get_data(self.instance_ids[sample_idx])]
#                     else:
#                         yield [self.get_data(idx) for idx in self.instance_ids[sample_idx]]
#                     break
#                 except GeneratorExit:
#                     raise
#                 except:
#                     logger.warning(f"Failed to load data for instance {self.instance_ids[sample_idx]}: {traceback.format_exc()}")
#                     sample_idx = random.randint(0, len(self.instance_ids) - 1)

#             self.sample_idx += 1

#             if self.sample_idx >= len(self.instance_ids):
#                 if not self.infinite:
#                     logger.warning(f"Dataset has run out of data.")
#                     break
#                 else:
#                     self.sample_idx = 0
#                     random.shuffle(self.instance_ids)
#                     logger.warning(f"Dataset is being re-looped.")

#     def load_state_dict(self, state_dict):
#         self.sample_idx = state_dict["sample_idx"]
#         self.instance_ids = state_dict["instance_ids"]
    
#     def state_dict(self):
#         return {
#             "instance_ids": self.instance_ids,
#             "sample_idx": self.sample_idx,
#         }
        
#     def _collate_fn(self, batch):
#         # flatten batches
#         def flatten_list(l):
#             return [item for sublist in l for item in sublist]
        
#         batch = flatten_list(batch)

#         # merge the mesh into a large graph
#         batch_collated = {}
#         offset = [0]
#         vertices = []
#         edges = []

#         for b in batch:
#             vertices.append(b['vertices'])
#             edges.append(b['edges'] + offset[-1])
#             offset.append(offset[-1] + b['vertices'].shape[0])
        
#         batch_collated['vertices'] = torch.cat(vertices, dim=0)
#         batch_collated['edges'] = torch.cat(edges, dim=0)
#         batch_collated['offsets'] = torch.tensor(offset, dtype=torch.int32)
#         batch_collated['instance_ids'] = [b['instance_id'] for b in batch]

#         del batch
#         gc.collect()
#         return batch_collated

#     def collate_fn(self, batch):
#         return self._collate_fn(batch)
