from typing import Optional, Union, List, Any, Dict, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import trimesh
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torchtitan.experiments.vem.models.mesh_gnn import MeshTransformer

from collections import defaultdict
from typing import Tuple, List

def _merge_faces_to_quad(face1, face2, shared_edge):
    """
    Merge two triangle faces into a quad, preserving local vertex order.
    
    Parameters
    ----------
    face1, face2 : (3,) int
        Triangular face indices.
    shared_edge : (2,) int
        Vertices of the shared diagonal edge (unordered).
    
    Returns
    -------
    quad : list[int]
        Quad vertex indices in consistent order.
    """
    f1 = list(face1)
    f2 = list(face2)
    e0, e1 = shared_edge

    f1e0 = f1.index(e0)
    f1e1 = f1.index(e1)
    f2e0 = f2.index(e0)
    f2e1 = f2.index(e1)

    order1 = (f1e0, f1e1) in [(0,1),(1,2),(2,0)]
    order2 = (f2e0, f2e1) in [(0,1),(1,2),(2,0)]
    if order1 == order2:
        return False, None

    if order1:
        return True, [f1[(f1e1+1)%3], f1[f1e0], f2[(f2e0 + 1) % 3], f2[f2e1]]
    else:
        return True, [f2[(f2e1+1)%3], f2[f2e0], f1[(f1e0+1)%3], f1[f1e1]]


def greedy_merge_tris_to_quads(mesh: trimesh.Trimesh,
                               is_quad: np.ndarray,
                               conf: np.ndarray):
    """
    Greedy merge triangle faces into quads using trimesh adjacency.
    Vertex ordering follows the original face order.
    """
    adj_faces = mesh.face_adjacency          # (k,2)
    adj_edges = mesh.face_adjacency_edges    # (k,2)

    edges_unique = mesh.edges_unique
    edge_map = {tuple(sorted(e)): i for i, e in enumerate(edges_unique)}
    adj_edge_idx = np.array([edge_map[tuple(sorted(e))] for e in adj_edges])

    # candidate adjacency pairs
    candidates = np.where(is_quad[adj_edge_idx])[0]
    if candidates.size == 0:
        return mesh.faces.copy(), np.empty((0, 4), dtype=int)

    # sort by confidence
    confs = conf[adj_edge_idx[candidates]]
    order = candidates[np.argsort(-confs)]

    face_removed = np.zeros(len(mesh.faces), dtype=bool)
    chosen_diagonals = set()
    quads = []

    for idx in order:
        f1, f2 = adj_faces[idx]
        e_idx = adj_edge_idx[idx]
        shared = adj_edges[idx]

        if face_removed[f1] or face_removed[f2]:
            continue
        if e_idx in chosen_diagonals:
            continue

        valid, quad = _merge_faces_to_quad(mesh.faces[f1], mesh.faces[f2], shared)
        if valid:
            quads.append(quad)

            face_removed[f1] = True
            face_removed[f2] = True
            chosen_diagonals.add(e_idx)

    remaining_tris = mesh.faces[~face_removed]
    quads = np.array(quads, dtype=int) if quads else np.empty((0, 4), dtype=int)
    return remaining_tris, quads


@dataclass
class T2QPipelineOutput(BaseOutput):
    v: List[np.ndarray]
    t: List[np.ndarray]
    q: List[np.ndarray]

class T2QPipeline(DiffusionPipeline):
    def __init__(
        self,
        mesh_transformer: MeshTransformer,
    ):
        super().__init__()

        self.register_modules(
            mesh_transformer=mesh_transformer,
        )
    
    def preprocess(
        self,
        mesh: trimesh.Trimesh,
    ):
        mesh.vertices, center, extents = self.normalize_vertices(mesh.vertices)
        mesh.merge_vertices(digits_vertex=6) # merge vertices with 1e-6 precision
        mesh.update_faces(mesh.nondegenerate_faces(height=1.e-8) & mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        return mesh, center, extents

    def normalize_vertices(self, vertices, range=(-1, 1)):
        vert_min = vertices.min(axis=0)
        vert_max = vertices.max(axis=0)
        vert_center = 0.5 * (vert_min + vert_max)
        extents = (vert_max - vert_min).max()
        vertices = vertices - vert_center
        vertices = vertices / extents * (range[1] - range[0]) + (range[0] + range[1]) / 2
        return vertices, vert_center, extents
    
    def denormzlie_vertices(self, vertices, vert_center, extents, range=(-1, 1)):
        vertices = vertices - (range[0] + range[1]) / 2
        vertices = vertices * extents / (range[1] - range[0])
        vertices = vertices + vert_center
        return vertices
    
    @torch.inference_mode()
    def __call__(
        self,
        input_mesh: Union[trimesh.Trimesh, List[trimesh.Trimesh]],
        threshold: float = 0.5,
    ):
        device = self.device
        dtype = self.dtype

        if isinstance(input_mesh, trimesh.Trimesh):
            meshes = [input_mesh]
        else:
            meshes = input_mesh
        
        vertices = []
        tri_faces = []
        quad_faces = []
        for i, mesh in enumerate(meshes):
            mesh, center_i, extents_i = self.preprocess(mesh)

            v = mesh.vertices
            e = mesh.edges_unique

            v_tensor = torch.from_numpy(v).to(device, dtype=dtype)
            e_tensor = torch.from_numpy(e).to(device, dtype=torch.long).permute(1, 0).contiguous()

            cu_seqlens = torch.tensor([0, v_tensor.shape[0]], dtype=torch.int32, device=device)

            e_pred = self.mesh_transformer(v_tensor, e_tensor, cu_seqlens=cu_seqlens)
            is_quad = (torch.sigmoid(e_pred) > threshold).detach().cpu().numpy().flatten()
            remaining_tris, quads = greedy_merge_tris_to_quads(mesh, is_quad, e_pred.detach().cpu().numpy().flatten())

            v_origin = self.denormzlie_vertices(v, center_i, extents_i)

            vertices.append(v_origin)
            tri_faces.append(remaining_tris)
            quad_faces.append(quads)

        return T2QPipelineOutput(
            v = vertices,
            t = tri_faces,
            q = quad_faces
        )