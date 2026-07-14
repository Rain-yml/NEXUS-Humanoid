from dataclasses import asdict
from typing import Any, Optional, List, Union, Dict
import json
import random
import os
import numpy as np
import trimesh
import torch
import itertools
import meshio
import tempfile

from scipy.spatial.transform import Rotation
from torchtitan.experiments.vem.datasets.renderer.moderngl_rasterizer import PositionNormalRenderer


def rand_with_pt(r: List[float]):
    return torch.rand(1).item() * (r[1] - r[0]) + r[0]

def rand_int_with_pt(r: List[int]):
    # [r0, r1)
    return torch.randint(r[0], r[1], (1,)).item()

def read_mesh_mock(mesh_path):
    pc = None
    mesh = trimesh.Trimesh(
        vertices=np.random.randn(1000, 3),
        faces=np.random.randint(0, 1000, size=(2000, 3)),
    )

    return mesh, pc

def read_mesh(mesh_path):
    pc = None
    if mesh_path.endswith(".npz"):
        # with np.load(mesh_path) as data:
        data = np.load(mesh_path)
        vertices = data['v']
        faces = data['f']
        if "p" in data:
            pc = data['p']
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    elif mesh_path.endswith(".glb"):
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
    elif mesh_path.endswith(".obj"):
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
    elif mesh_path.endswith(".ply"):
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
    else:
        raise NotImplementedError
    
    return mesh, pc

def read_mesh_from_bos(mesh_path: str, bos_client, bos_bucket: str, process=False):
    pc = None
    if mesh_path.endswith(".npz"):
        file = bos_client.get_file(bos_bucket, mesh_path)
        with np.load(file) as data:
            vertices = data['v'].astype(np.float32)
            faces = data['f'].astype(np.int64)
            if "p" in data:
                pc = data['p'].astype(np.float32)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=process)
    elif mesh_path.endswith(".glb"):
        file = bos_client.get_file(bos_bucket, mesh_path)
        mesh = trimesh.load_mesh(file, file_type="glb", process=process)
    elif mesh_path.endswith(".ply"):
        file = bos_client.get_file(bos_bucket, mesh_path)
        mesh = trimesh.load_mesh(file, file_type="ply", process=process)
    else:
        raise NotImplementedError
    
    return mesh, pc

def normalize_vertices(vertices, range=(-1, 1)):
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    vert_center = 0.5 * (vert_min + vert_max)
    extents = (vert_max - vert_min).max()
    vertices = vertices - vert_center
    vertices = vertices / extents * (range[1] - range[0]) + (range[0] + range[1]) / 2
    return vertices

def _process(mesh, flip_x=True, rotate_z=True, z_up=True):
    vertices, faces = mesh.vertices, mesh.faces
    vertices = normalize_vertices(vertices, range=(-1, 1))
    if z_up:
        vertices = np.stack([vertices[:,0], -vertices[:,2], vertices[:,1]], axis=-1)
    if flip_x:
        # random apply flip in x dir
        if torch.rand(1).item() < 0.5:
            vertices[:,0] = -vertices[:,0]
            faces = faces[:, [0, 2, 1]]  

    if rotate_z:
        rotate_z = torch.randint(4, (1,)).item() * 90
        axis = [0, 0, 1]
        radian = np.pi / 180 * rotate_z
        rotation = Rotation.from_rotvec(radian * np.array(axis))
        vertices = rotation.apply(vertices)

    vertices = normalize_vertices(vertices, range=(-1, 1))
    
    return vertices, faces

def sample(vertices, faces, num_points=1024):
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    sampled_points, face_index = trimesh.sample.sample_surface(m, num_points)
    sampled_normals = m.face_normals[face_index]
    return sampled_points, sampled_normals, face_index

def sample_with_adaptive(vertices, faces, num_points=1024, angle_threshold=30):
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # crease edges from angle threshold
    fa_angles = np.degrees(m.face_adjacency_angles)
    fa_crease_mask = fa_angles > angle_threshold
    fa_edges_crease = m.face_adjacency_edges[fa_crease_mask]

    if fa_edges_crease.shape[0] == 0:
        # fallback: adaptive threshold
        fa_crease_mask = fa_angles > np.median(fa_angles)
        fa_edges_crease = m.face_adjacency_edges[fa_crease_mask]

    # boundary edges
    boundary_edges = m.edges[trimesh.grouping.group_rows(m.edges_sorted, require_count=1)]
    crease_edges = np.concatenate([fa_edges_crease, boundary_edges], axis=0)
    crease_edges.sort(axis=1)
    crease_edges = np.unique(crease_edges, axis=0)

    # first include the sharp edge vertices
    # edge vertices and lengths
    edge_vertices = m.vertices[crease_edges]   # (E, 2, 3)

    edge_lengths = np.linalg.norm(edge_vertices[:, 1] - edge_vertices[:, 0], axis=1)
    L = edge_lengths.sum()
    S = m.area
    lam = 1 # d_surface / d_edge

    num_salient_points = int((-(lam*L)**2 + np.sqrt((lam*L)**4+4*S*num_points*(lam*L)**2)) /(2*S))
    num_surface_points = num_points - num_salient_points

    sharp_vertices = m.vertices[np.unique(crease_edges.flatten())]
    if sharp_vertices.shape[0] < num_salient_points:
        num_salient_points_edge = num_salient_points - sharp_vertices.shape[0]

        start = edge_vertices[:, 0]                # (E, 3)
        end = edge_vertices[:, 1]                  # (E, 3)

        # handle degenerate
        if edge_lengths.sum() == 0:
            edge_probs = np.ones_like(edge_lengths) / len(edge_lengths)
        else:
            edge_probs = edge_lengths / edge_lengths.sum()

        # allocate points proportional to length
        n_points_per_edge = np.floor(edge_probs * num_salient_points_edge).astype(int)
        allocated = n_points_per_edge.sum()
        remaining = num_salient_points_edge - allocated

        if remaining > 0:
            extra_edges = np.random.choice(len(edge_lengths), size=remaining, replace=False, p=edge_probs)
            np.add.at(n_points_per_edge, extra_edges, 1)

        total_samples = n_points_per_edge.sum()
        assert total_samples == num_salient_points_edge
        # build edge indices
        edge_ids = np.repeat(np.arange(len(n_points_per_edge)), n_points_per_edge)

        # relative positions (0,1) evenly spaced per edge
        # formula: t = (k+1)/(n+1), where 0 <= k < n
        counts = n_points_per_edge[n_points_per_edge > 0]        # (E_nonzero,)
        offsets = np.cumsum(np.concatenate([[0], counts[:-1]]))  # start index per edge
        k = np.arange(total_samples) - np.repeat(offsets, counts)  # position index within edge
        n = np.repeat(counts, counts)                             # total samples in that edge
        t = (k + 1) / (n + 1)

        sharp_points = start[edge_ids] * (1 - t[:, None]) + end[edge_ids] * t[:, None]
        sharp_points = np.concatenate([sharp_points, sharp_vertices], axis=0)
    else:
        ind = np.random.choice(sharp_vertices.shape[0], size=num_salient_points, replace=False)
        sharp_points = sharp_vertices[ind].copy()

    # surface sampling
    surface_points, face_index = trimesh.sample.sample_surface(m, num_surface_points)
    sampled_normals = m.face_normals[face_index]

    # points = np.concatenate([surface_points, sharp_points], axis=0)
    # sampled_normals = np.concatenate([sampled_normals, np.zeros((sharp_points.shape[0], 3))], axis=0)

    return surface_points, sampled_normals, sharp_points

def sample_with_dora(vertices, faces, num_points=1024, num_salient_points=1024, angle_threshold=30):
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # crease edges from angle threshold
    fa_angles = np.degrees(m.face_adjacency_angles)
    fa_crease_mask = fa_angles > angle_threshold
    fa_edges_crease = m.face_adjacency_edges[fa_crease_mask]

    if fa_edges_crease.shape[0] == 0:
        # fallback: adaptive threshold
        fa_crease_mask = fa_angles > np.median(fa_angles)
        fa_edges_crease = m.face_adjacency_edges[fa_crease_mask]

    # boundary edges
    boundary_edges = m.edges[trimesh.grouping.group_rows(m.edges_sorted, require_count=1)]
    crease_edges = np.concatenate([fa_edges_crease, boundary_edges], axis=0)
    crease_edges.sort(axis=1)
    crease_edges = np.unique(crease_edges, axis=0)

    # first include the sharp edge vertices
    # edge vertices and lengths
    edge_vertices = m.vertices[crease_edges]   # (E, 2, 3)

    sharp_vertices = m.vertices[np.unique(crease_edges.flatten())]
    if sharp_vertices.shape[0] < num_salient_points:
        num_salient_points_edge = num_salient_points - sharp_vertices.shape[0]

        start = edge_vertices[:, 0]                # (E, 3)
        end = edge_vertices[:, 1]                  # (E, 3)
        edge_lengths = np.linalg.norm(end - start, axis=1)

        # handle degenerate
        if edge_lengths.sum() == 0:
            edge_probs = np.ones_like(edge_lengths) / len(edge_lengths)
        else:
            edge_probs = edge_lengths / edge_lengths.sum()

        # allocate points proportional to length
        n_points_per_edge = np.floor(edge_probs * num_salient_points_edge).astype(int)
        allocated = n_points_per_edge.sum()
        remaining = num_salient_points_edge - allocated

        if remaining > 0:
            extra_edges = np.random.choice(len(edge_lengths), size=remaining, replace=False, p=edge_probs)
            np.add.at(n_points_per_edge, extra_edges, 1)

        total_samples = n_points_per_edge.sum()
        assert total_samples == num_salient_points_edge
        # build edge indices
        edge_ids = np.repeat(np.arange(len(n_points_per_edge)), n_points_per_edge)

        # relative positions (0,1) evenly spaced per edge
        # formula: t = (k+1)/(n+1), where 0 <= k < n
        counts = n_points_per_edge[n_points_per_edge > 0]        # (E_nonzero,)
        offsets = np.cumsum(np.concatenate([[0], counts[:-1]]))  # start index per edge
        k = np.arange(total_samples) - np.repeat(offsets, counts)  # position index within edge
        n = np.repeat(counts, counts)                             # total samples in that edge
        t = (k + 1) / (n + 1)

        sharp_points = start[edge_ids] * (1 - t[:, None]) + end[edge_ids] * t[:, None]
        sharp_points = np.concatenate([sharp_points, sharp_vertices], axis=0)
    else:
        ind = np.random.choice(sharp_vertices.shape[0], size=num_salient_points, replace=False)
        sharp_points = sharp_vertices[ind].copy()

    # surface sampling
    surface_points, face_index = trimesh.sample.sample_surface(m, num_points)
    sampled_normals = m.face_normals[face_index]

    return surface_points, sampled_normals, sharp_points

def find_absent(index, n):
    mask = np.zeros(n, dtype=bool)
    mask[index] = True
    np.nonzero(~mask)[0]
    return ~mask

def find_intersect(a, b, n):
    mask = np.zeros(n, dtype=bool)
    mask[a] = True
    inter = b[mask[b]]
    return inter

def sample_with_dora_normal(vertices, faces, num_points=1024, num_salient_points=1024, angle_threshold=30, equal_salient_sampling=True):
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # crease edges from angle threshold
    fa_angles = np.degrees(m.face_adjacency_angles)
    fa_crease_mask = fa_angles > angle_threshold
    fa_edges_crease = m.face_adjacency_edges[fa_crease_mask]

    if fa_edges_crease.shape[0] == 0:
        # fallback: adaptive threshold
        fa_crease_mask = fa_angles > np.median(fa_angles)
        fa_edges_crease = m.face_adjacency_edges[fa_crease_mask]

    fa_crease = m.face_adjacency[fa_crease_mask]
    fa_edges_normals = m.face_normals[fa_crease].mean(axis=1) # N_c x 2 x 3 -> N_c x 3
    fa_edges_normals = fa_edges_normals / np.clip(np.linalg.norm(fa_edges_normals, axis=1)[:, None], 1e-8, None)

    # find non watertight edge, treat them as individual crease edges
    num_all_edges = m.edges.shape[0]
    twin_edges = trimesh.grouping.group_rows(m.edges_sorted, require_count=2).flatten()
    edge_no_repeat = trimesh.grouping.group_rows(m.edges, require_count=1).flatten()
    wt_edge_index = find_intersect(twin_edges, edge_no_repeat, num_all_edges)

    nonwt_edge_index = find_absent(wt_edge_index, num_all_edges) # non watertight edges
    nonwt_edge = m.edges_sorted[nonwt_edge_index]
    nonwt_edge_normal = m.face_normals[m.edges_face[nonwt_edge_index]]

    crease_edges = np.concatenate([fa_edges_crease, nonwt_edge], axis=0)
    crease_normal = np.concatenate([fa_edges_normals, nonwt_edge_normal], axis=0)

    edge_vertices = m.vertices[crease_edges]   # (E, 2, 3)

    # first include the sharp edge vertices
    # edge vertices and lengths
    sharp_vertices_index = np.unique(crease_edges.flatten())
    sharp_vertices = m.vertices[sharp_vertices_index]
    sharp_vertices_normal = m.vertex_normals[sharp_vertices_index]

    if sharp_vertices.shape[0] < num_salient_points:
        num_salient_points_edge = num_salient_points - sharp_vertices.shape[0]

        start = edge_vertices[:, 0]                # (E, 3)
        end = edge_vertices[:, 1]                  # (E, 3)
        edge_lengths = np.linalg.norm(end - start, axis=1)

        # handle degenerate
        if edge_lengths.sum() == 0:
            edge_probs = np.ones_like(edge_lengths) / len(edge_lengths)
        else:
            edge_probs = edge_lengths / edge_lengths.sum()

        # allocate points proportional to length
        n_points_per_edge = np.floor(edge_probs * num_salient_points_edge).astype(int)
        allocated = n_points_per_edge.sum()
        remaining = num_salient_points_edge - allocated

        if remaining > 0:
            extra_edges = np.random.choice(len(edge_lengths), size=remaining, replace=False, p=edge_probs)
            np.add.at(n_points_per_edge, extra_edges, 1)

        total_samples = n_points_per_edge.sum()
        assert total_samples == num_salient_points_edge
        # build edge indices
        edge_ids = np.repeat(np.arange(len(n_points_per_edge)), n_points_per_edge)

        # relative positions (0,1) evenly spaced per edge
        # formula: t = (k+1)/(n+1), where 0 <= k < n
        if equal_salient_sampling:
            counts = n_points_per_edge[n_points_per_edge > 0]        # (E_nonzero,)
            offsets = np.cumsum(np.concatenate([[0], counts[:-1]]))  # start index per edge
            k = np.arange(total_samples) - np.repeat(offsets, counts)  # position index within edge
            n = np.repeat(counts, counts)                             # total samples in that edge
            t = (k + 1) / (n + 1)
        else:
            t = np.random.rand(total_samples)

        sharp_points = start[edge_ids] * (1 - t[:, None]) + end[edge_ids] * t[:, None]
        sharp_points = np.concatenate([sharp_points, sharp_vertices], axis=0)
        sharp_normals = np.concatenate([crease_normal[edge_ids], sharp_vertices_normal], axis=0)
    else:
        ind = np.random.choice(sharp_vertices.shape[0], size=num_salient_points, replace=False)
        sharp_points = sharp_vertices[ind].copy()
        sharp_normals = sharp_vertices_normal[ind].copy()

    # surface sampling
    surface_points, face_index = trimesh.sample.sample_surface(m, num_points)
    surface_normals = m.face_normals[face_index]

    return surface_points, surface_normals, sharp_points, sharp_normals

def axis_aligned_rotations_24():
    """Return the 24 rotation matrices (3x3) of the cube rotation group (SO(3) symmetries)."""
    mats = []
    perms = list(itertools.permutations([0,1,2]))
    signs = list(itertools.product([1,-1], repeat=3))
    for p in perms:
        P = np.zeros((3,3), dtype=int)
        for i, j in enumerate(p):
            P[i, j] = 1
        for s in signs:
            S = np.diag(s)
            R = S @ P
            if np.linalg.det(R) == 1:  # keep only rotations, not reflections
                mats.append(R.astype(float))
    # remove duplicates
    uniq = []
    seen = set()
    for M in mats:
        key = tuple(M.flatten())
        if key not in seen:
            seen.add(key)
            uniq.append(M)
    assert len(uniq) == 24
    return uniq

# Basic types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md

# PyTorch Tensor type
from torch import Tensor
import trimesh

def _project_to_2d(verts_3d):
    """Project polygon vertices to 2D by dropping the dominant normal axis."""
    n = len(verts_3d)
    normal = np.zeros(3, dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        normal[0] += (verts_3d[i, 1] - verts_3d[j, 1]) * (verts_3d[i, 2] + verts_3d[j, 2])
        normal[1] += (verts_3d[i, 2] - verts_3d[j, 2]) * (verts_3d[i, 0] + verts_3d[j, 0])
        normal[2] += (verts_3d[i, 0] - verts_3d[j, 0]) * (verts_3d[i, 1] + verts_3d[j, 1])
    dominant = np.argmax(np.abs(normal))
    axes = [a for a in range(3) if a != dominant]
    return verts_3d[:, axes]


def _tri_area_2d(v1, v2, v3):
    return (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])


def _quad_rotate_cost(coords, v1_idx, v2_idx, v3_idx, v4_idx):
    """
    Compute cost of flipping the diagonal of a quad (v1,v2,v3,v4).
    Current diagonal: (v2,v4). Flipped diagonal: (v1,v3).
    Returns negative if flipping improves quality; inf if flip is invalid.
    """
    v1, v2, v3, v4 = coords[v1_idx], coords[v2_idx], coords[v3_idx], coords[v4_idx]

    eps = 1e-12

    area_123 = _tri_area_2d(v1, v2, v3)
    area_134 = _tri_area_2d(v1, v3, v4)
    if (area_123 >= 0.0) != (area_134 >= 0.0):
        return float('inf')
    if abs(area_123) <= eps or abs(area_134) <= eps:
        return float('inf')

    area_234 = _tri_area_2d(v2, v3, v4)
    area_241 = _tri_area_2d(v2, v4, v1)
    if (area_234 >= 0.0) != (area_241 >= 0.0):
        return -1e30
    if abs(area_234) <= eps or abs(area_241) <= eps:
        return -1e30

    len_12 = np.linalg.norm(v1 - v2)
    len_23 = np.linalg.norm(v2 - v3)
    len_34 = np.linalg.norm(v3 - v4)
    len_41 = np.linalg.norm(v4 - v1)
    len_13 = np.linalg.norm(v1 - v3)
    len_24 = np.linalg.norm(v2 - v4)

    fac_24 = abs(area_234) / (len_23 + len_34 + len_24) + abs(area_241) / (len_41 + len_12 + len_24)
    fac_13 = abs(area_123) / (len_12 + len_23 + len_13) + abs(area_134) / (len_34 + len_41 + len_13)

    return fac_24 - fac_13


def _build_internal_edges(tris, n, boundary):
    """
    Find all internal edges from the triangle list.
    Returns list of (tri_a, tri_b, v1, v2, v3, v4) where:
      - v2, v4 are the shared edge vertices
      - v1 is the opposite vertex in tri_a
      - v3 is the opposite vertex in tri_b
    """
    edge_to_tri = {}
    for ti, tri in enumerate(tris):
        for j in range(3):
            a, b = tri[j], tri[(j + 1) % 3]
            key = (min(a, b), max(a, b))
            if key in boundary:
                continue
            if key in edge_to_tri:
                edge_to_tri[key].append((ti, j))
            else:
                edge_to_tri[key] = [(ti, j)]

    internal = []
    for key, entries in edge_to_tri.items():
        if len(entries) == 2:
            (ti_a, j_a), (ti_b, j_b) = entries
            v2 = tris[ti_a][j_a]
            v4 = tris[ti_a][(j_a + 1) % 3]
            v1 = tris[ti_a][(j_a + 2) % 3]
            v3 = tris[ti_b][(j_b + 2) % 3]
            internal.append((ti_a, ti_b, v1, v2, v3, v4))
    return internal


def beauty_triangulate_ngon(verts_3d):
    """
    Triangulate an n-gon using Blender's Beauty algorithm.
    verts_3d: (n, 3) array of vertex positions in polygon winding order.
    Returns: (n-2, 3) array of local vertex indices forming triangles.
    """
    n = len(verts_3d)
    if n < 3:
        return np.empty((0, 3), dtype=np.int64)
    if n == 3:
        return np.array([[0, 1, 2]], dtype=np.int64)

    coords = _project_to_2d(verts_3d)

    num_tris = n - 2
    tris = [[0, i + 1, i + 2] for i in range(num_tris)]

    if n == 4:
        v2, v4 = tris[0][2], tris[0][0]  # shared edge is (0,2)
        v1, v3 = tris[0][1], tris[1][2]  # opposite verts: 1, 3
        cost = _quad_rotate_cost(coords, v1, v2, v3, v4)
        if cost < -1e-6:
            tris[0] = [v1, v2, v3]
            tris[1] = [v1, v3, v4]
        return np.array(tris, dtype=np.int64)

    boundary = set()
    for i in range(n):
        a, b = i, (i + 1) % n
        boundary.add((min(a, b), max(a, b)))

    max_iters = (n - 3) * (n - 3)
    for _ in range(max_iters):
        internal = _build_internal_edges(tris, n, boundary)
        best_cost = -1e-6
        best_edge = None
        for edge_info in internal:
            ti_a, ti_b, v1, v2, v3, v4 = edge_info
            cost = _quad_rotate_cost(coords, v1, v2, v3, v4)
            if cost < best_cost:
                best_cost = cost
                best_edge = edge_info

        if best_edge is None:
            break

        ti_a, ti_b, v1, v2, v3, v4 = best_edge
        tris[ti_a] = [v1, v2, v3]
        tris[ti_b] = [v1, v3, v4]

    return np.array(tris, dtype=np.int64)

def _dedup_tris(tris):
    """Helper: 对 tri 数组去重，保留每组第一个出现的（忽略顺序和朝向）"""
    if len(tris) == 0:
        return tris
    tri_sorted = np.sort(tris, axis=1)
    _, first_idx = np.unique(tri_sorted, axis=0, return_index=True)
    keep_mask = np.zeros(len(tris), dtype=bool)
    keep_mask[first_idx] = True
    return tris[keep_mask]

class Mesh:
    @classmethod
    def load(cls, obj_path: str) -> "Mesh":
        m = meshio.read(obj_path)
        vertices = np.asarray(m.points[:, :3], dtype=np.float64)
        triangle_blocks = []
        quad_blocks = []

        for cell_block in m.cells:
            if cell_block.type == "triangle":
                triangles = np.asarray(cell_block.data, dtype=np.int64).reshape((-1, 3))
                triangle_blocks.append(triangles)
            elif cell_block.type == "quad":
                quads = np.asarray(cell_block.data, dtype=np.int64).reshape((-1, 4))
                quad_blocks.append(quads)
            elif cell_block.type == "polygon":
                polygons = np.asarray(cell_block.data, dtype=np.int64)
                n = polygons.shape[1]
                for row_idx in range(polygons.shape[0]):
                    poly_verts = polygons[row_idx]
                    verts_3d = vertices[poly_verts]
                    local_tris = beauty_triangulate_ngon(verts_3d)
                    global_tris = poly_verts[local_tris]
                    triangle_blocks.append(global_tris)
            else:
                raise RuntimeError(f"Unsupported cell type: {cell_block.type}")
        
        triangle_blocks = np.concatenate(triangle_blocks, axis=0) if len(triangle_blocks) > 0 else np.zeros((0, 3), dtype=np.int64)
        quad_blocks = np.concatenate(quad_blocks, axis=0) if len(quad_blocks) > 0 else np.zeros((0, 4), dtype=np.int64)
        faces = np.full((len(triangle_blocks) + len(quad_blocks), 4), -1, dtype=np.int64)
        faces[:len(triangle_blocks), :3] = triangle_blocks
        faces[len(triangle_blocks):] = quad_blocks
        
        return cls(
            vertices=vertices.astype(np.float32),
            faces=faces.astype(np.int32),
        )
    
    @property
    def face_list(self):
        return [face[:-1] if face[-1] == -1 else face for face in self.faces]
    
    def __init__(self, vertices, faces=None, fnv=None, face_list=None) -> None:
        assert(len(vertices.shape) == 2)
        assert(vertices.shape[1] == 3 or vertices.shape[1] == 6)
        if vertices.shape[1] == 6:
            vertices = vertices[:, :3]

        self.vertices = vertices
        if faces is not None:
            assert(face_list is None)
            assert(len(faces.shape) == 2)
            assert(faces.shape[1] == 4)
            self.faces = faces
        else:
            assert(face_list is not None)
            faces = []
            nV = self.vertices.shape[0]
            for f in face_list:
                f = np.array(f, dtype=int)
                if len(f) == 4:
                    faces.append(f)
                elif len(f) == 3:
                    faces.append(np.concatenate([f, [-1]]))
                else:
                    center = self.vertices[f].mean(axis=0)
                    sub_faces = np.stack([f, np.roll(f, -1), np.full_like(f, nV), np.full_like(f, -1)], axis=-1)
                    faces += sub_faces.tolist()
                    self.vertices = np.concatenate([self.vertices, center.reshape(1, 3)], axis=0)
                    nV += 1
            faces = np.array(faces, dtype=int)
            self.faces = faces

        if fnv is None:
            fnv = (faces >= 0).sum(axis=-1)
        self.face_num_vertices = fnv
        self.is_quad = (self.face_num_vertices == 4)
        self.num_quad_faces = self.is_quad.sum()
        self.num_tri_faces = (~self.is_quad).sum()
    
    def edges(self):
        # Compute edges
        quad_faces = self.faces[self.is_quad]
        tri_faces = self.faces[~self.is_quad]
        edges_from_quad_faces = np.concatenate(
            [
                quad_faces[:, [0, 1]],
                quad_faces[:, [1, 2]],
                quad_faces[:, [2, 3]],
                quad_faces[:, [3, 0]],
            ],
            axis=0,
        )
        edges_from_tri_faces = np.concatenate(
            [
                tri_faces[:, [0, 1]],
                tri_faces[:, [1, 2]],
                tri_faces[:, [2, 0]],
            ],
            axis=0,
        )
        edges = np.concatenate([edges_from_quad_faces, edges_from_tri_faces], axis=0)
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        return edges

    def v_degree(self):
        v_degree = np.zeros(self.vertices.shape[0], dtype=np.int32)
        edges = self.edges()
        unique, counts = np.unique(edges, return_counts=True)
        v_degree[unique] = counts
        return v_degree
    
    def merge_vertices(self, digits=0):
        referenced = np.zeros(len(self.vertices), dtype=bool)
        referenced[self.faces[self.faces != -1]] = True

        stacked = [self.vertices * (10**digits)]

        # stack collected vertex properties and round to integer
        stacked = np.column_stack(stacked).round().astype(np.int32)

        # check unique rows of referenced vertices
        u, i = trimesh.grouping.unique_rows(stacked[referenced], keep_order=True)

        # construct an inverse using the subset
        inverse = np.zeros(len(self.vertices), dtype=np.int32)
        inverse[referenced] = i
        # get the vertex mask
        mask = np.nonzero(referenced)[0][u]
        # run the update including normals and UV coordinates
        self.update_vertices(mask=mask, inverse=inverse)
    
    def update_vertices(
        self,
        mask,
        inverse = None,
    ) -> None:
        # copy from trimesh
        """
        Update vertices with a mask.

        Parameters
        ------------
        vertex_mask : (len(self.vertices)) bool
          Array of which vertices to keep
        inverse : (len(self.vertices)) int
          Array to reconstruct vertex references
          such as output by np.unique
        """
        # make sure mask is a numpy array
        mask = np.asanyarray(mask)

        if (mask.dtype.name == "bool" and mask.all()) or len(mask) == 0:
            # mask doesn't remove any vertices so exit early
            return

        # create the inverse mask if not passed
        if inverse is None:
            inverse = np.zeros(len(self.vertices), dtype=np.int32)
            if mask.dtype.kind == "b":
                inverse[mask] = np.arange(mask.sum())
            elif mask.dtype.kind == "i":
                inverse[mask] = np.arange(len(mask))
            else:
                inverse = None

        # re-index faces from inverse
        if inverse is not None:
            self.faces[self.faces >= 0] = inverse[self.faces[self.faces >= 0]]

        # actually apply the mask
        self.vertices = self.vertices[mask]
    
    def to_triangle_mesh(self, return_mesh=True):
        """
        Convert mixed quad/triangle mesh to pure triangle mesh using numpy operations.
        
        Returns:
            if return_mesh=True: (Mesh, face_to_tris, tris_to_face)
            if return_mesh=False: (new_faces, face_to_tris, tris_to_face)
        """
        n_quads = self.is_quad.sum()
        n_tris = (~self.is_quad).sum()
        n_new_faces = n_tris + 2 * n_quads  # each quad becomes 2 triangles
        
        # Pre-allocate output arrays
        new_faces = np.full((n_new_faces, 4), -1, dtype=np.int32)
        face_to_tris = np.full((len(self.faces), 2), -1, dtype=np.int32)
        tris_to_face = np.zeros(n_new_faces, dtype=np.int32)
        
        # Handle triangle faces first
        tri_mask = ~self.is_quad
        tri_indices = np.where(tri_mask)[0]
        n_tri_faces = len(tri_indices)
        
        if n_tri_faces > 0:
            # Copy triangle faces directly
            new_faces[:n_tri_faces] = self.faces[tri_mask]
            face_to_tris[tri_indices, 0] = np.arange(n_tri_faces)
            tris_to_face[:n_tri_faces] = tri_indices
        
        # Handle quad faces
        quad_mask = self.is_quad
        quad_indices = np.where(quad_mask)[0]
        
        if len(quad_indices) > 0:
            quad_faces = self.faces[quad_mask]  # Shape: (n_quads, 4)
            
            # Extract vertices for each quad
            v1, v2, v3, v4 = quad_faces[:, 0], quad_faces[:, 1], quad_faces[:, 2], quad_faces[:, 3]
            
            # Create two triangles per quad: (v1,v2,v3) and (v3,v4,v1)
            tri1_indices = np.arange(n_tri_faces, n_tri_faces + n_quads)
            tri2_indices = np.arange(n_tri_faces + n_quads, n_tri_faces + 2 * n_quads)
            
            # Fill first triangles (v1, v2, v3, -1)
            new_faces[tri1_indices, 0] = v1
            new_faces[tri1_indices, 1] = v2  
            new_faces[tri1_indices, 2] = v3
            # new_faces[tri1_indices, 3] is already -1
            
            # Fill second triangles (v1, v2, v3, -1)
            new_faces[tri2_indices, 0] = v1
            new_faces[tri2_indices, 1] = v3
            new_faces[tri2_indices, 2] = v4
            # new_faces[tri2_indices, 3] is already -1
            
            # Update mapping arrays
            face_to_tris[quad_indices, 0] = tri1_indices
            face_to_tris[quad_indices, 1] = tri2_indices
            tris_to_face[tri1_indices] = quad_indices
            tris_to_face[tri2_indices] = quad_indices
        
        if return_mesh:
            fnv = np.full(n_new_faces, 3, dtype=np.int32)
            return Mesh(self.vertices.copy(), new_faces, fnv), face_to_tris, tris_to_face
        else:
            return new_faces, face_to_tris, tris_to_face
    
    def _clean_faces_deprecated(self):
        """
        Remove degenerate faces and duplicate faces from the mesh using numpy operations.
        
        Degenerate faces: faces that contain duplicate vertices
        Duplicate faces: faces that contain identical vertices with other faces
        """
        # Convert to pure triangle mesh
        tri_faces, face_to_tris, tris_to_face = self.to_triangle_mesh(return_mesh=False)
        
        # Extract triangle indices (remove -1 padding)
        triangle_indices = tri_faces[:, :3]  # Shape: (n_triangles, 3)
        
        # 1. Find degenerate triangles (containing duplicate vertices)
        # Check if all 3 vertices are unique for each triangle
        # Check pairwise equality within each triangle
        v1, v2, v3 = triangle_indices[:, 0], triangle_indices[:, 1], triangle_indices[:, 2]
        degenerate_mask = (v1 == v2) | (v1 == v3) | (v2 == v3)
        
        # 2. Find duplicate triangles
        # Sort vertices of each triangle to create canonical form
        sorted_triangles = np.sort(triangle_indices, axis=1)
        
        # Find unique triangles
        _, unique_indices = np.unique(sorted_triangles, axis=0, return_index=True)
        
        # Create mask for triangles to keep (unique and non-degenerate)
        keep_tri_mask = np.zeros(len(triangle_indices), dtype=bool)
        keep_tri_mask[unique_indices] = True
        keep_tri_mask &= ~degenerate_mask  # Remove degenerate ones
        
        if np.all(keep_tri_mask):
            # No cleaning needed, all triangles are valid
            return
        
        # 3. Handle face reconstruction based on original face types
        quad_mask = self.is_quad
        tri_mask = ~self.is_quad
        
        # For original triangle faces
        original_tri_indices = np.where(tri_mask)[0]
        tri_face_to_tri_idx = face_to_tris[original_tri_indices, 0]  # Get triangle indices
        keep_original_tris = keep_tri_mask[tri_face_to_tri_idx]  # Which original triangles to keep
        
        # For original quad faces  
        original_quad_indices = np.where(quad_mask)[0]
        if len(original_quad_indices) > 0:
            quad_tri1_indices = face_to_tris[original_quad_indices, 0]
            quad_tri2_indices = face_to_tris[original_quad_indices, 1]
            
            keep_tri1 = keep_tri_mask[quad_tri1_indices]
            keep_tri2 = keep_tri_mask[quad_tri2_indices]
            
            # Quad face decisions:
            # both good -> keep quad, only one good -> convert to triangle, both bad -> remove
            keep_as_quad = keep_tri1 & keep_tri2
            convert_to_tri1 = keep_tri1 & ~keep_tri2
            convert_to_tri2 = ~keep_tri1 & keep_tri2
            # remove_quad = ~keep_tri1 & ~keep_tri2 (implicit)
        
        # 4. Build new face arrays
        new_faces_list = []
        new_fnv_list = []
        
        # Add surviving original triangles
        surviving_tri_indices = original_tri_indices[keep_original_tris]
        if len(surviving_tri_indices) > 0:
            new_faces_list.append(self.faces[surviving_tri_indices])
            new_fnv_list.append(np.full(len(surviving_tri_indices), 3))
        
        # Add surviving/converted quad faces
        if len(original_quad_indices) > 0:
            # Keep quads that have both triangles valid
            keep_quad_indices = original_quad_indices[keep_as_quad]
            if len(keep_quad_indices) > 0:
                new_faces_list.append(self.faces[keep_quad_indices])
                new_fnv_list.append(np.full(len(keep_quad_indices), 4))
            
            # Convert quads to triangles (first triangle valid)
            convert_tri1_indices = original_quad_indices[convert_to_tri1]
            if len(convert_tri1_indices) > 0:
                tri1_global_indices = quad_tri1_indices[convert_to_tri1]
                converted_faces = np.column_stack([
                    triangle_indices[tri1_global_indices],
                    np.full(len(tri1_global_indices), -1)
                ])
                new_faces_list.append(converted_faces)
                new_fnv_list.append(np.full(len(convert_tri1_indices), 3))
            
            # Convert quads to triangles (second triangle valid)
            convert_tri2_indices = original_quad_indices[convert_to_tri2]
            if len(convert_tri2_indices) > 0:
                tri2_global_indices = quad_tri2_indices[convert_to_tri2]
                converted_faces = np.column_stack([
                    triangle_indices[tri2_global_indices], 
                    np.full(len(tri2_global_indices), -1)
                ])
                new_faces_list.append(converted_faces)
                new_fnv_list.append(np.full(len(convert_tri2_indices), 3))
        
        # 5. Update mesh with cleaned faces
        if new_faces_list:
            self.faces = np.vstack(new_faces_list).astype(np.int32)
            self.face_num_vertices = np.concatenate(new_fnv_list).astype(np.int32)
        else:
            # No faces left
            self.faces = np.empty((0, 4), dtype=np.int32)
            self.face_num_vertices = np.empty(0, dtype=np.int32)
        
        # Update derived properties
        self.is_quad = (self.face_num_vertices == 4)
        self.num_quad_faces = self.is_quad.sum()
        self.num_tri_faces = (~self.is_quad).sum()
        self.num_faces = len(self.faces)

    def clean_faces(self):
        """
        更严格的去重去退化：直接在 quad 层面判断，不依赖 to_triangle_mesh。

        Phase 1: 退化处理
          - tri 顶点重复 → 删除
          - quad 对角顶点重合 (v0==v2 or v1==v3) → 删除
          - quad 相邻顶点重合 → 退化成 tri（保留 3 个不重复顶点）

        Phase 2: 去重
          - tri-tri 重复（顶点集合相同）→ 保留一个
          - tri-quad 重复（tri 是 quad 的某 3 顶点子集）→ 删 tri
          - quad-quad 完全重复（4 顶点集合相同）→ 保留一个
          - quad-quad 部分重叠（共享 3 顶点）→ 都拆成 2 个 tri，再去重
        """
        if len(self.faces) == 0:
            return

        tri_mask = ~self.is_quad
        quad_mask = self.is_quad

        tri_faces_in = self.faces[tri_mask, :3].copy()
        quad_faces_in = self.faces[quad_mask, :4].copy()

        # === Phase 1: 退化处理 ===
        if len(tri_faces_in) > 0:
            tri_degen = ((tri_faces_in[:, 0] == tri_faces_in[:, 1]) |
                         (tri_faces_in[:, 0] == tri_faces_in[:, 2]) |
                         (tri_faces_in[:, 1] == tri_faces_in[:, 2]))
            valid_tris = tri_faces_in[~tri_degen]
        else:
            valid_tris = np.empty((0, 3), dtype=np.int32)

        if len(quad_faces_in) > 0:
            diag_degen = ((quad_faces_in[:, 0] == quad_faces_in[:, 2]) |
                          (quad_faces_in[:, 1] == quad_faces_in[:, 3]))
            adj_degen = ((quad_faces_in[:, 0] == quad_faces_in[:, 1]) |
                         (quad_faces_in[:, 1] == quad_faces_in[:, 2]) |
                         (quad_faces_in[:, 2] == quad_faces_in[:, 3]) |
                         (quad_faces_in[:, 3] == quad_faces_in[:, 0]))

            quad_keep_as_quad = ~diag_degen & ~adj_degen
            quad_to_tri = ~diag_degen & adj_degen

            valid_quads = quad_faces_in[quad_keep_as_quad]
            quads_to_convert = quad_faces_in[quad_to_tri]
        else:
            valid_quads = np.empty((0, 4), dtype=np.int32)
            quads_to_convert = np.empty((0, 4), dtype=np.int32)

        # 从相邻顶点重合的 quad 提取 3 个不重复顶点（保持原始顺序）
        converted_tris = []
        for quad in quads_to_convert:
            seen = set()
            unique_verts = []
            for v in quad:
                vi = int(v)
                if vi not in seen:
                    seen.add(vi)
                    unique_verts.append(vi)
            if len(unique_verts) == 3:
                converted_tris.append(unique_verts)
        if converted_tris:
            converted_tris = np.array(converted_tris, dtype=np.int32)
        else:
            converted_tris = np.empty((0, 3), dtype=np.int32)

        all_tris = np.vstack([valid_tris, converted_tris]) if (len(valid_tris) + len(converted_tris)) > 0 else np.empty((0, 3), dtype=np.int32)

        # === Phase 2.1: tri-tri 去重 ===
        all_tris = _dedup_tris(all_tris)

        # === Phase 2.2: tri-quad 去重（tri 是 quad 的子三角形 → 删 tri）===
        if len(valid_quads) > 0 and len(all_tris) > 0:
            quad_subset_keys = set()
            quad_combos = [(0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3)]
            for quad in valid_quads:
                for a, b, c in quad_combos:
                    key = tuple(sorted((int(quad[a]), int(quad[b]), int(quad[c]))))
                    quad_subset_keys.add(key)

            tri_sorted = np.sort(all_tris, axis=1)
            keep = np.array([tuple(int(v) for v in row) not in quad_subset_keys for row in tri_sorted])
            all_tris = all_tris[keep]

        # === Phase 2.3: quad-quad 完全重复 ===
        if len(valid_quads) > 0:
            quad_sorted = np.sort(valid_quads, axis=1)
            _, first_idx = np.unique(quad_sorted, axis=0, return_index=True)
            keep_mask = np.zeros(len(valid_quads), dtype=bool)
            keep_mask[first_idx] = True
            valid_quads = valid_quads[keep_mask]

        # === Phase 2.4: quad-quad 部分重叠（共享 3 顶点）→ 按共享子集对应的对角线拆 ===
        if len(valid_quads) > 0:
            # 记录每个 3-subset 出现在哪些 quad 上（以及对应的 missing vertex 的局部索引）
            subset_to_quads = {}  # key: sorted 3-tuple → [(quad_idx, missing_local_idx)]
            for qi, quad in enumerate(valid_quads):
                for k in range(4):
                    sub_verts = [int(quad[i]) for i in range(4) if i != k]
                    key = tuple(sorted(sub_verts))
                    subset_to_quads.setdefault(key, []).append((qi, k))

            # 为每个发生碰撞的 quad 收集应使用的对角线
            # k 是 missing vertex 的局部索引 → 共享 subset = {v_i for i!=k}
            # k=1 或 3 → 共享 tri 在 diagonal v0↔v2 上 → 拆 (v0,v1,v2)+(v0,v2,v3)
            # k=0 或 2 → 共享 tri 在 diagonal v1↔v3 上 → 拆 (v0,v1,v3)+(v1,v2,v3)
            quad_to_diagonals = {}
            for key, entries in subset_to_quads.items():
                if len(entries) > 1:
                    for qi, k in entries:
                        diag = "02" if (k % 2 == 1) else "13"
                        quad_to_diagonals.setdefault(qi, set()).add(diag)

            if quad_to_diagonals:
                split_tris_list = []
                for qi, diagonals in quad_to_diagonals.items():
                    quad = valid_quads[qi]
                    v0, v1, v2, v3 = int(quad[0]), int(quad[1]), int(quad[2]), int(quad[3])
                    if diagonals == {"02"}:
                        split_tris_list.append([v0, v1, v2])
                        split_tris_list.append([v0, v2, v3])
                    elif diagonals == {"13"}:
                        split_tris_list.append([v0, v1, v3])
                        split_tris_list.append([v1, v2, v3])
                    else:
                        # 同一 quad 的不同 subset 撞上了不同对角线 → 全拆 4 个 sub-tri
                        split_tris_list.append([v0, v1, v2])
                        split_tris_list.append([v0, v2, v3])
                        split_tris_list.append([v0, v1, v3])
                        split_tris_list.append([v1, v2, v3])
                split_tris = np.array(split_tris_list, dtype=np.int32)

                keep_quad_mask = np.ones(len(valid_quads), dtype=bool)
                keep_quad_mask[list(quad_to_diagonals.keys())] = False
                valid_quads = valid_quads[keep_quad_mask]

                # 拆出来的 tri 也可能退化（quad 本身不退化时不会，但保险起见过滤）
                split_degen = ((split_tris[:, 0] == split_tris[:, 1]) |
                               (split_tris[:, 0] == split_tris[:, 2]) |
                               (split_tris[:, 1] == split_tris[:, 2]))
                split_tris = split_tris[~split_degen]

                all_tris = np.vstack([all_tris, split_tris]) if len(all_tris) > 0 else split_tris
                all_tris = _dedup_tris(all_tris)

        # === Phase 3: 重建 faces ===
        new_faces_list = []
        new_fnv_list = []
        if len(all_tris) > 0:
            tri_padded = np.column_stack([
                all_tris,
                np.full(len(all_tris), -1, dtype=np.int32),
            ])
            new_faces_list.append(tri_padded)
            new_fnv_list.append(np.full(len(all_tris), 3, dtype=np.int32))
        if len(valid_quads) > 0:
            new_faces_list.append(valid_quads.astype(np.int32))
            new_fnv_list.append(np.full(len(valid_quads), 4, dtype=np.int32))

        if new_faces_list:
            self.faces = np.vstack(new_faces_list).astype(np.int32)
            self.face_num_vertices = np.concatenate(new_fnv_list).astype(np.int32)
        else:
            self.faces = np.empty((0, 4), dtype=np.int32)
            self.face_num_vertices = np.empty(0, dtype=np.int32)
        
        referenced_vertices = np.unique(self.faces[self.faces != -1])
        referenced = np.zeros(len(self.vertices), dtype=bool)
        referenced[referenced_vertices] = True
        if not referenced.all():
            self.update_vertices(mask=referenced)

        self.is_quad = (self.face_num_vertices == 4)
        self.num_quad_faces = self.is_quad.sum()
        self.num_tri_faces = (~self.is_quad).sum()
        self.num_faces = len(self.faces)

        
    def normalize_vertices(self, range=(0.05, 0.95)):
        vmin = self.vertices.min(axis=0)
        vmax = self.vertices.max(axis=0)

        scale = (vmax - vmin).max()
        center = (vmax + vmin) / 2.0

        self.vertices = (self.vertices - center) / scale * (range[1] - range[0]) + (range[0] + range[1]) / 2
        return scale, center
    
    # def export(self, obj_path: str):
    #     with open(obj_path, "w") as fi:
    #         for v in self.vertices:
    #             fi.write(f"v {v[0]} {v[1]} {v[2]}\n")
    #         for i, f in enumerate(self.faces):
    #             if self.is_quad[i]:
    #                 fi.write(f"f {f[0]+1} {f[1]+1} {f[2]+1} {f[3]+1}\n")
    #             else:
    #                 fi.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    
    def export(self, obj_path: str):
        triangle_faces = self.faces[~self.is_quad]
        quad_faces = self.faces[self.is_quad]
        cells = []
        if triangle_faces.shape[0] > 0:
            cells.append(("triangle", triangle_faces[:, :3]))
        if quad_faces.shape[0] > 0:
            cells.append(("quad", quad_faces[:, :4]))
        meshio.write_points_cells(
            obj_path,
            points=self.vertices,
            cells=cells,
        )
    
    def export_np(self, path):
        np.savez(path, verts=self.vertices, faces=self.faces, fnv=self.face_num_vertices)
    
    @classmethod
    def load_np(cls, path):
        with np.load(path) as data:
            vertices = data["verts"]
            faces = data["faces"]
            fnv = data["fnv"]

        return cls(vertices, faces, fnv)
    
    def to_trimesh(self) -> trimesh.Trimesh:
        """
        Convert the Mesh object to a trimesh.Trimesh object.
        """
        new_faces, _, _ = self.to_triangle_mesh(return_mesh=False)
        return trimesh.Trimesh(vertices=self.vertices, faces=new_faces[:, :3], process=False)

def generate_icosahedron_cameras(
    radius: float = 3.0,
    fov_deg: float = 45.0,
    near: float = 0.1,
    far: float = 100.0,
):
    """
    Generate camera parameters for views at icosahedron face centers.
    
    This creates 20 evenly distributed camera viewpoints around an object,
    positioned at the centers of an icosahedron's faces, all looking toward
    the origin.
    
    Args:
        radius: Distance from origin to each camera
        fov_deg: Field of view in degrees (same for all cameras)
        near: Near clipping plane
        far: Far clipping plane
    
    Returns:
        Dictionary with arrays suitable for renderer.render():
            - 'camera_distance': array of shape (20,)
            - 'camera_fovy_deg': array of shape (20,)
            - 'camera_elevation_deg': array of shape (20,)
            - 'camera_azimuth_deg': array of shape (20,)
            - 'render_near': array of shape (20,)
            - 'render_far': array of shape (20,)
    
    Example:
        >>> params = generate_icosahedron_cameras(radius=3.0, fov_deg=45.0)
        >>> results = renderer.render(mesh, **params, smooth=True)
        >>> # results will be a list of 20 dicts with 'pos' and 'normal' keys
    """
    # Define the 12 vertices of an icosahedron
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    
    # Normalize vertices to lie on a unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Define the 20 faces of the icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Calculate face centers
    face_centers = []
    for face in faces:
        # Average the vertices of each face
        center = np.mean([vertices[i] for i in face], axis=0)
        # Normalize to lie on unit sphere
        center = center / np.linalg.norm(center)
        face_centers.append(center)
    
    face_centers = np.array(face_centers)  # Shape: (20, 3)
    
    # Convert Cartesian coordinates to spherical coordinates
    # for each camera position
    distances = []
    elevations = []
    azimuths = []
    
    for center in face_centers:
        # Scale to desired radius
        pos = center * radius
        
        # Calculate spherical coordinates
        # Distance (should all be equal to radius, but compute for verification)
        dist = np.linalg.norm(pos)
        distances.append(dist)
        
        # Elevation: angle from XY plane (in degrees)
        # elevation = arcsin(z / r)
        elevation_rad = np.arcsin(pos[2] / dist)
        elevation_deg = np.degrees(elevation_rad)
        elevations.append(elevation_deg)
        
        # Azimuth: angle in XY plane from X axis (in degrees)
        # azimuth = atan2(y, x)
        azimuth_rad = np.arctan2(pos[1], pos[0])
        azimuth_deg = np.degrees(azimuth_rad)
        azimuths.append(azimuth_deg)
    
    # Convert to numpy arrays
    distances = np.array(distances)
    elevations = np.array(elevations)
    azimuths = np.array(azimuths)
    
    # Create parameter dictionary
    params = {
        'camera_distance': distances,
        'camera_fovy_deg': np.full(20, fov_deg),
        'camera_elevation_deg': elevations,
        'camera_azimuth_deg': azimuths,
        'render_near': np.full(20, near),
        'render_far': np.full(20, far),
    }
    
    return params


class MeshProcessor:
    def __init__(self):
        self.mats_24 = axis_aligned_rotations_24()
        self.renderer = None
    
    def init_render(self):
        if self.renderer is None:
            self.renderer = PositionNormalRenderer()
            self.camera_params = generate_icosahedron_cameras(
                radius=3.0,
                fov_deg=40.0,
                near=0.01,
                far=100.0,
            )

    def read_mesh_mock(self, mesh_path):
        return read_mesh_mock(mesh_path)
    
    def read_mesh_from_bos(self, mesh_path: str, bos_client, bos_bucket: str, process=False):
        return read_mesh_from_bos(mesh_path, bos_client, bos_bucket, process=process)
    
    def read_quad_mesh(self, mesh_path: str):
        if mesh_path.endswith(".npz"):
            with np.load(mesh_path) as data:
                v = data['verts']
                f = data['faces']
            
            tri_f = f[f[:, -1] < 0, :3]
            quad_f = f[f[:,-1] >= 0]
            return v, tri_f, quad_f
        elif mesh_path.endswith(".ply"):
            mesh = meshio.read(mesh_path)
            cells_t = []
            cells_q = []
            for c in mesh.cells:
                if c.type == 'triangle':
                    cells_t.append(c.data)
                elif c.type == 'quad':
                    cells_q.append(c.data)
                else:
                    raise ValueError
            
            cells_t = np.concatenate(cells_t, axis=0).astype(np.int64) if len(cells_t) > 0 else np.zeros((0, 3), dtype=np.int64)
            cells_q = np.concatenate(cells_q, axis=0).astype(np.int64) if len(cells_q) > 0 else np.zeros((0, 4), dtype=np.int64)
            return mesh.points, cells_t, cells_q
        else:
            raise NotImplementedError

    def read_quad_mesh_from_bos(self, mesh_path: str, bos_client: str, bos_bucket: str):
        raise NotImplementedError

    def read_mixed_mesh(self, mesh_path: str):
        if mesh_path.endswith(".ply"):
            return Mesh.load(mesh_path)
        else:
            raise NotImplementedError

    def read_mixed_mesh_from_bos(self, mesh_path: str, bos_client: str, bos_bucket: str):
        if not mesh_path.endswith(".ply"):
            raise NotImplementedError

        file = bos_client.get_file(bos_bucket, mesh_path)
        file.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".ply") as tmp:
            tmp.write(file.read())
            tmp.flush()
            return Mesh.load(tmp.name)

    def read_mesh(self, mesh_path):
        return read_mesh(mesh_path)
    
    def process(self, mesh, z_up=True, up='z', front='-y'):
        vertices, faces = mesh.vertices, mesh.faces
        vertices = normalize_vertices(vertices, range=(-1, 1))
        # if z_up:
        #     vertices = np.stack([vertices[:,0], -vertices[:,2], vertices[:,1]], axis=-1)
        if z_up:
            print("z_up is deprecated, use up='z' instead")
            up = 'z'
        if up == 'z' and front == '-y':
            pass
        elif up == 'y' and front == 'z':
            vertices = np.stack([vertices[:,0], -vertices[:,2], vertices[:,1]], axis=-1)
        elif up == 'y' and front == '-z':
            vertices = np.stack([vertices[:,0], vertices[:,2], vertices[:,1]], axis=-1)
            faces = faces[:, [0, 2, 1]]
        else:
            raise NotImplementedError

        return vertices, faces

    def process_vfinput(self, vertices, faces, z_up=True, up='z', front='-y'):
        vertices = normalize_vertices(vertices, range=(-1, 1))
        # if z_up:
        #     vertices = np.stack([vertices[:,0], -vertices[:,2], vertices[:,1]], axis=-1)
        if z_up:
            print("z_up is deprecated, use up='z' instead")
            up = 'z'
        if up == 'z' and front == '-y':
            pass
        elif up == 'y' and front == 'z':
            vertices = np.stack([vertices[:,0], -vertices[:,2], vertices[:,1]], axis=-1)
        elif up == 'y' and front == '-z':
            vertices = np.stack([vertices[:,0], vertices[:,2], vertices[:,1]], axis=-1)
            faces = faces[:, [0, 2, 1]]
        else:
            raise NotImplementedError

        return vertices, faces
    
    def process_quad(self, vertices, faces_tri, faces_quad, z_up=True):
        vertices = normalize_vertices(vertices, range=(-1, 1))
        if z_up:
            vertices = np.stack([vertices[:,0], -vertices[:,2], vertices[:,1]], axis=-1)
        
        faces_unified = []
        if faces_tri.shape[0] > 0:
            faces_tri = np.concatenate([faces_tri, np.full((faces_tri.shape[0], 1), -1)], axis=-1)
            faces_unified.append(faces_tri)
        if faces_quad.shape[0] > 0:
            faces_unified.append(faces_quad)
        faces_unified = np.concatenate(faces_unified, axis=0)
        
        quad_mesh = Mesh(vertices, faces=faces_unified)
        quad_mesh.merge_vertices(digits=6) # merge vertices with 1e-6 precision
        quad_mesh.clean_faces()
        vertices = quad_mesh.vertices
        faces_quad = quad_mesh.faces[quad_mesh.is_quad]
        faces_tri = quad_mesh.faces[~quad_mesh.is_quad, :3]
        
        return vertices, faces_tri, faces_quad
    
    def clear_mesh(self, mesh: trimesh.Trimesh, digits_vertex=6):
        mesh.merge_vertices(digits_vertex=digits_vertex)
        mesh.update_faces(mesh.nondegenerate_faces(height=1.e-8) & mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        # mesh.fix_normals()
        assert np.all(mesh.area_faces > 0)
        return mesh
    
    def augment(
        self, 
        vertices, 
        faces,
        aug_flip=True,
        aug_rotate_all=True,
        aug_rotate_z=True,
        aug_scale=True,
        aug_scale_range=(0.8, 1.2),
    ):
        assert not (aug_rotate_all and aug_rotate_z), "Cannot enable both aug_rotate_all and aug_rotate_z"
        if aug_flip:
            if rand_with_pt([0, 1]) < 0.5:
                vertices[:,0] = -vertices[:,0]
                faces = faces[:, [0, 2, 1]]
        
        if aug_rotate_all:
            mat_idx = rand_int_with_pt([0, 24])
            R = self.mats_24[mat_idx]
            vertices = vertices @ R.T
        elif aug_rotate_z:
            rotate_z = rand_int_with_pt([0, 4]) * 90
            axis = [0, 0, 1]
            radian = np.pi / 180 * rotate_z
            rotation = Rotation.from_rotvec(radian * np.array(axis))
            vertices = rotation.apply(vertices)
        
        if aug_scale:
            scale = (torch.rand(3) * (aug_scale_range[1] - aug_scale_range[0]) + aug_scale_range[0]).numpy()
            vertices = vertices * scale[None, :]
        
        vertices = normalize_vertices(vertices, range=(-1, 1))
        return vertices, faces
    
    def sample(self, vertices, faces, num_points=1024):
        return sample(vertices, faces, num_points)
    
    def sample_with_dora(self, *args, **kwargs):
        return sample_with_dora(*args, **kwargs)

    def sample_with_dora_normal(self, *args, **kwargs):
        return sample_with_dora_normal(*args, **kwargs)
    
    def sample_with_adaptive(self, vertices, faces, num_points=1024, angle_threshold=30):
        return sample_with_adaptive(vertices, faces, num_points, angle_threshold)
    
    def sample_with_render(self, vertices, faces, num_points):
        if self.renderer is None:
            self.init_render()
        
        m = trimesh.Trimesh(vertices=vertices, faces=faces)
        ret = self.renderer.render(
            mesh=m,
            smooth=False,
            **self.camera_params,
        )

        pos = np.stack([ret_['pos'] for ret_ in ret], axis=0)
        nrm = np.stack([ret_['normal'] for ret_ in ret], axis=0)
        mask = nrm[..., 3] > 128
        points = pos[..., :3][mask]
        normals = nrm[..., :3][mask]

        indices = np.random.choice(points.shape[0], num_points, replace=points.shape[0] < num_points)
        points = points[indices]
        normals = normals[indices]

        normals = normals.astype(np.float32) / 255. * 2.0 - 1.0
        return points, normals
    
    def normalize_vertices(self, vertices, range=(-1, 1)):
        return normalize_vertices(vertices, range)
    
    def augment_quad(
        self, 
        vertices, 
        faces,
        faces_quad,
        aug_flip=True,
        aug_rotate_all=True,
        aug_rotate_z=True,
        aug_scale=True,
        aug_scale_range=(0.8, 1.2),
    ):
        assert not (aug_rotate_all and aug_rotate_z), "Cannot enable both aug_rotate_all and aug_rotate_z"
        if aug_flip:
            if rand_with_pt([0, 1]) < 0.5:
                vertices[:,0] = -vertices[:,0]
                faces = faces[:, [0, 2, 1]]
                faces_quad = faces_quad[:, [0, 3, 2, 1]]
        
        if aug_rotate_all:
            mat_idx = rand_int_with_pt([0, 24])
            R = self.mats_24[mat_idx]
            vertices = vertices @ R.T
        elif aug_rotate_z:
            rotate_z = rand_int_with_pt([0, 4]) * 90
            axis = [0, 0, 1]
            radian = np.pi / 180 * rotate_z
            rotation = Rotation.from_rotvec(radian * np.array(axis))
            vertices = rotation.apply(vertices)
        
        if aug_scale:
            scale = (torch.rand(3) * (aug_scale_range[1] - aug_scale_range[0]) + aug_scale_range[0]).numpy()
            vertices = vertices * scale[None, :]
        
        vertices = normalize_vertices(vertices, range=(-1, 1))
        return vertices, faces, faces_quad
