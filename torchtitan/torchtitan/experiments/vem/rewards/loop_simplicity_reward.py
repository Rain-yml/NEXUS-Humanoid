from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from torchtitan.experiments.vem.rewards.base import MeshReward, MeshRewardOutput


def load_quad_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load OBJ, extract only quad faces."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                vidxs = [int(p.split("/")[0]) - 1 for p in parts]
                if len(vidxs) == 4:
                    faces.append(vidxs)
    return np.array(vertices, np.float64), np.array(faces, np.int64)


def build_adjacency(faces):
    """Build edge→face, vertex→edge, face→edges mappings."""
    edge_faces = defaultdict(list)
    vert_edges = defaultdict(set)
    face_edge_list = {}

    for fi, f in enumerate(faces):
        edges = []
        for i in range(4):
            e = (min(f[i], f[(i + 1) % 4]), max(f[i], f[(i + 1) % 4]))
            edges.append(e)
            edge_faces[e].append(fi)
            vert_edges[f[i]].add(e)
        face_edge_list[fi] = edges

    return dict(edge_faces), dict(vert_edges), face_edge_list


def _vertex_opposite(edge, vertex, edge_faces, vert_edges):
    """Find vertex-opposite edge at a regular vertex. Returns None at irregular/boundary."""
    incident = vert_edges.get(vertex, set())
    if len(incident) not in (3, 4):
        return None
    cur_faces = set(edge_faces.get(edge, []))
    candidates = [
        c for c in incident if c != edge and not cur_faces.intersection(edge_faces.get(c, []))
    ]
    return candidates[0] if len(candidates) == 1 else None


def enumerate_edge_loops(edge_faces, vert_edges, max_iter_per_loop=10000, time_budget=30.0):
    """Return all unique edge-loops (list of edge-tuples each)."""
    visited = set()
    loops = []
    start_time = time.time()

    for start in edge_faces:
        if start in visited:
            continue
        if time.time() - start_time > time_budget:
            break
        chain = [start]
        cur, nxt_v = start, start[1]
        for _ in range(max_iter_per_loop):
            nxt = _vertex_opposite(cur, nxt_v, edge_faces, vert_edges)
            if nxt is None:
                break
            if nxt == start:
                break
            chain.append(nxt)
            nxt_v = nxt[1] if nxt[0] == nxt_v else nxt[0]
            cur = nxt
        visited.update(chain)
        loops.append(chain)
    return loops


def _face_opposite_edge(edge, face_idx, face_edge_list):
    """Return opposite edge in a quad face."""
    edges = face_edge_list[face_idx]
    try:
        return edges[(edges.index(edge) + 2) % 4]
    except ValueError:
        return None


def enumerate_face_loops(edge_faces, faces, face_edge_list, max_iter_per_loop=10000, time_budget=30.0):
    """
    Return all unique face-loops.
    Each quad face has two pairs of opposite edges → belongs to exactly 2 face-loops.
    We track which (face, direction_pair) has been visited.
    """
    visited = set()
    loops = []
    start_time = time.time()

    def pair_idx_of_edge(edge, fi):
        edges = face_edge_list[fi]
        if edge in (edges[0], edges[2]):
            return 0
        if edge in (edges[1], edges[3]):
            return 1
        return None

    for start_edge, adj in edge_faces.items():
        if time.time() - start_time > time_budget:
            break
        for start_fi in adj:
            pidx = pair_idx_of_edge(start_edge, start_fi)
            if pidx is None or (start_fi, pidx) in visited:
                continue

            chain = [start_fi]
            visited.add((start_fi, pidx))
            cur_fi, cur_edge = start_fi, start_edge
            seen_in_chain = {start_fi}
            has_self_intersection = False

            for _ in range(max_iter_per_loop):
                opp = _face_opposite_edge(cur_edge, cur_fi, face_edge_list)
                if opp is None:
                    break
                adj_faces = edge_faces.get(opp, [])
                nxt_fi = None
                for f in adj_faces:
                    if f != cur_fi:
                        nxt_fi = f
                        break
                if nxt_fi is None:
                    break
                if nxt_fi == chain[0]:
                    break
                if nxt_fi in seen_in_chain:
                    has_self_intersection = True
                    break
                p = pair_idx_of_edge(opp, nxt_fi)
                if p is not None:
                    visited.add((nxt_fi, p))
                chain.append(nxt_fi)
                seen_in_chain.add(nxt_fi)
                cur_fi, cur_edge = nxt_fi, opp

            loops.append((chain, has_self_intersection))

    return loops


def rotation_index(points: np.ndarray) -> float:
    """Rotation index = |total turning| / 2π on best-fit plane."""
    n = len(points)
    if n < 3:
        return 0.0
    c = points - points.mean(0)
    try:
        _, _, vh = np.linalg.svd(c, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    p2 = np.column_stack([c @ vh[0], c @ vh[1]])

    total = 0.0
    for i in range(n):
        d1 = p2[i] - p2[i - 1]
        d2 = p2[(i + 1) % n] - p2[i]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        d1 /= n1
        d2 /= n2
        total += np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], d1 @ d2)
    return abs(total) / (2 * np.pi)


def face_areas(verts, faces):
    v = verts[faces]
    t1 = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    t2 = 0.5 * np.linalg.norm(np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]), axis=1)
    return t1 + t2


def is_loop_simple(si, ind):
    return si == 0 and ind <= 1.0 + 1e-6


def compute_loop_simplicity(mesh_path: str, min_loop_len: int = 2) -> Tuple[float, float, float]:
    """
    Compute S_fl, S_el, S_l for a quad mesh.
    Loops shorter than min_loop_len are excluded (they are trivially "simple").
    """
    verts, faces = load_quad_mesh(mesh_path)
    return compute_loop_simplicity_arrays(verts, faces, min_loop_len=min_loop_len)


def compute_loop_simplicity_arrays(
    verts: np.ndarray,
    faces: np.ndarray,
    min_loop_len: int = 2,
) -> tuple[float, float, float]:
    """Compute S_fl, S_el, S_l for in-memory quad arrays."""

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        return 1.0, 1.0, 1.0
    if faces.ndim != 2 or faces.shape[1] != 4:
        raise ValueError(f"Loop simplicity expects quad faces with shape (F, 4), got {faces.shape}")

    ef, ve, fel = build_adjacency(faces)
    areas = face_areas(verts, faces)

    edge_loops = enumerate_edge_loops(ef, ve)
    tot_ea = simp_ea = 0.0
    for loop in edge_loops:
        if len(loop) < min_loop_len:
            continue
        pts = np.array([0.5 * (verts[e[0]] + verts[e[1]]) for e in loop])
        si = len(loop) - len(set(loop))
        ind = rotation_index(pts) if len(pts) >= 3 else 0.0
        area = sum(areas[fi] for e in loop for fi in ef.get(e, []))
        tot_ea += area
        if is_loop_simple(si, ind):
            simp_ea += area
    s_el = simp_ea / tot_ea if tot_ea > 0 else 1.0

    face_loops = enumerate_face_loops(ef, faces, fel)
    tot_fa = simp_fa = 0.0
    for loop, has_si in face_loops:
        if len(loop) < min_loop_len:
            continue
        centers = np.array([verts[faces[fi]].mean(0) for fi in loop])
        ind = rotation_index(centers) if len(centers) >= 3 else 0.0
        simple = (not has_si) and (ind <= 1.0 + 1e-6)
        area = sum(areas[fi] for fi in loop)
        tot_fa += area
        if simple:
            simp_fa += area
    s_fl = simp_fa / tot_fa if tot_fa > 0 else 1.0
    return float(s_fl), float(s_el), float(min(s_fl, s_el))


class LoopSimplicityReward(MeshReward):
    def __init__(self, min_loop_len: int = 2, workers: int = 0, **kwargs) -> None:
        del kwargs
        self.min_loop_len = min_loop_len
        self.workers = workers

    def __call__(
        self,
        meshes: list[tuple[Any, Any, Any]],
        metadata: list[dict[str, Any]] | None = None,
        device: torch.device | str | None = None,
    ) -> MeshRewardOutput:
        del metadata

        def score_one(mesh):
            verts, _triangles, quads = mesh
            return compute_loop_simplicity_arrays(
                verts,
                quads,
                min_loop_len=self.min_loop_len,
            )

        if self.workers and self.workers > 1 and len(meshes) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                scores = list(executor.map(score_one, meshes))
        else:
            scores = [score_one(mesh) for mesh in meshes]

        s_fl_values: list[float] = []
        s_el_values: list[float] = []
        rewards: list[float] = []
        for s_fl, s_el, s_l in scores:
            s_fl_values.append(s_fl)
            s_el_values.append(s_el)
            rewards.append(s_l)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        s_fl_tensor = torch.tensor(s_fl_values, dtype=torch.float32, device=device)
        s_el_tensor = torch.tensor(s_el_values, dtype=torch.float32, device=device)

        metrics = {
            "reward/loop_simplicity_mean": reward_tensor.mean().item() if reward_tensor.numel() else 0.0,
            "reward/S_fl_mean": s_fl_tensor.mean().item() if s_fl_tensor.numel() else 0.0,
            "reward/S_el_mean": s_el_tensor.mean().item() if s_el_tensor.numel() else 0.0,
        }
        return MeshRewardOutput(
            rewards=reward_tensor,
            metrics=metrics,
            per_sample={
                "S_fl": s_fl_tensor,
                "S_el": s_el_tensor,
                "S_l": reward_tensor,
            },
        )
