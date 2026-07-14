from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch

from torchtitan.experiments.vem.models.mesh_quadvae import MeshQuadVAE

try:
    import networkx as nx
except ImportError:  # pragma: no cover - imported on training images.
    nx = None


def nx_all_triangles(graph, nbunch=None, max_triangles=5_000_000):
    if nx is None:
        raise ImportError("networkx is required for quad decoder mesh recovery")
    if nbunch is None:
        nbunch = relevant_nodes = graph
    else:
        from itertools import chain

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
                    if len(triangles) >= max_triangles:
                        warnings.warn(
                            f"nx_all_triangles hit the cap of {max_triangles} candidate "
                            "triangles; the recovered mesh will be truncated (faces "
                            "involving higher-index vertices are dropped). Raise "
                            "max_triangles to avoid silently omitting faces.",
                            stacklevel=2,
                        )
                        return np.array(triangles)
    return np.array(triangles)


def quad_from_oriented_diag_triangles(tri_a, tri_b, diag):
    diag = tuple(sorted(int(v) for v in diag))
    a = [int(v) for v in tri_a]
    b = [int(v) for v in tri_b]

    if not all(v in a and v in b for v in diag):
        return None

    for tri0, tri1 in ((a, b), (b, a)):
        for i in range(3):
            v0, v1, v2 = tri0[i], tri0[(i + 1) % 3], tri0[(i + 2) % 3]
            if tuple(sorted((v0, v2))) != diag:
                continue
            other = [v for v in tri1 if v not in (v0, v2)]
            if len(other) == 1 and other[0] not in (v0, v1, v2):
                return [v0, v1, v2, other[0]]

    other_a = [v for v in a if v not in diag]
    other_b = [v for v in b if v not in diag]
    if len(other_a) != 1 or len(other_b) != 1 or other_a[0] == other_b[0]:
        return None

    d0, d1 = diag
    return [d0, other_a[0], d1, other_b[0]]


def all_vertex_pairs(nv, device):
    e1, e2 = torch.triu_indices(nv, nv, offset=1, device=device)
    return torch.stack([e1, e2], dim=-1)


def predict_edges(model, embed, vertex_global, candidate_local, edge_chunk_size=20_000_000):
    candidate_global = vertex_global[candidate_local]
    num_pairs = candidate_global.shape[0]
    edge_mask = []
    for start in range(0, num_pairs, edge_chunk_size):
        end = min(start + edge_chunk_size, num_pairs)
        d_edge_chunk = model.edge_loss.spacetime_distance(embed, candidate_global[start:end])
        edge_mask.append(d_edge_chunk > 0)
    
    edge_mask = torch.cat(edge_mask, dim=0)
    return candidate_local[edge_mask], candidate_global[edge_mask]


def predict_diags(model, embed, vertex_global, candidate_local, edge_chunk_size=20_000_000):
    if not model.pred_diag:
        raise ValueError("Quad decoder model has pred_diag=False")
    candidate_global = vertex_global[candidate_local]
    num_pairs = candidate_global.shape[0]
    edge_mask = []
    for start in range(0, num_pairs, edge_chunk_size):
        end = min(start + edge_chunk_size, num_pairs)
        d_edge_chunk = model.diag_loss.spacetime_distance(embed, candidate_global[start:end])
        edge_mask.append(d_edge_chunk > 0)
    
    edge_mask = torch.cat(edge_mask, dim=0)
    # d_diag = model.diag_loss.spacetime_distance(embed, candidate_global)
    return candidate_local[edge_mask], candidate_global[edge_mask]


def triangles_from_edges(edges_local, max_triangles=None):
    if nx is None:
        raise ImportError("networkx is required for quad decoder mesh recovery")
    edges = edges_local.detach().cpu().numpy()
    graph = nx.Graph()
    graph.add_edges_from(edges)
    if max_triangles is None:
        # A surface mesh has ~4*F ~= O(V) candidate triangles (each quad becomes a
        # 4-clique = 4 sub-triangles once both diagonals are predicted). Scale the cap
        # with graph size, with a generous floor, so large meshes are never truncated.
        # The previous flat 100k default silently dropped faces on big meshes
        # (candidate count exceeded 100k for ~>24k-quad meshes).
        max_triangles = max(5_000_000, 32 * graph.number_of_nodes())
    candidate_triangles = nx_all_triangles(graph, max_triangles=max_triangles)
    if candidate_triangles.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.sort(candidate_triangles, axis=1).astype(np.int64, copy=False)


def filter_triangles_by_diag_count(triangles_local, diag_edges_local, max_diags=1):
    if triangles_local.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)

    diag_edges = diag_edges_local.detach().cpu().numpy()
    if diag_edges.shape[0] == 0:
        return triangles_local.astype(np.int64, copy=False)

    diag_edges = np.sort(diag_edges.astype(np.int64, copy=False), axis=1)
    diag_edge_set = {tuple(edge) for edge in diag_edges}

    kept_triangles = []
    for tri in np.sort(triangles_local.astype(np.int64, copy=False), axis=1):
        diag_count = 0
        for edge in ((tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])):
            if tuple(edge) in diag_edge_set:
                diag_count += 1
                if diag_count > max_diags:
                    break
        if diag_count <= max_diags:
            kept_triangles.append(tri)

    if len(kept_triangles) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(kept_triangles, dtype=np.int64)


def filter_triangles(model, pred, vertex_global, triangles_local):
    if triangles_local.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.int64),
            torch.zeros((0, 3), dtype=torch.long, device=vertex_global.device),
        )

    triangles_local_t = torch.from_numpy(triangles_local).to(
        dtype=torch.long,
        device=vertex_global.device,
    )
    triangles_global = vertex_global[triangles_local_t]
    # Chunk the area evaluation: candidate triangle counts now scale with mesh size
    # (no longer capped at 100k), so a single unchunked spacetime_area could OOM.
    face_chunk_size = 5_000_000
    if triangles_global.shape[0] <= face_chunk_size:
        d_face = model.face_loss.spacetime_area(pred["face_embed"], triangles_global)
        keep_mask = d_face > 0
    else:
        keep_parts = []
        for start in range(0, triangles_global.shape[0], face_chunk_size):
            d_chunk = model.face_loss.spacetime_area(
                pred["face_embed"], triangles_global[start:start + face_chunk_size]
            )
            keep_parts.append(d_chunk > 0)
        keep_mask = torch.cat(keep_parts, dim=0)
    faces_global = triangles_global[keep_mask]

    if model.pred_orient and faces_global.shape[0] > 0:
        face_orient = model.orient_loss.orient(pred["orient_embed"], faces_global)
        correct_mask = face_orient > 0
        faces_global[~correct_mask] = torch.flip(faces_global[~correct_mask], dims=[1])

    faces_local = (faces_global - vertex_global[0]).detach().cpu().numpy()
    return faces_local, faces_global


def merge_triangles_by_diag(faces_local, valid_diag_local, dedup_quads=False):
    diag_to_face_indices = {}
    faces_sorted = np.sort(faces_local, axis=1)
    for face_idx, tri in enumerate(faces_sorted):
        for edge in ((tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])):
            key = tuple(sorted(int(v) for v in edge))
            diag_to_face_indices.setdefault(key, []).append(face_idx)

    used_faces = set()
    quad_keys = set()
    recovered_quads = []

    for diag in valid_diag_local.detach().cpu().numpy():
        key = tuple(sorted(int(v) for v in diag))
        face_indices = [idx for idx in diag_to_face_indices.get(key, []) if idx not in used_faces]
        if len(face_indices) != 2:
            continue

        tri_a = faces_local[face_indices[0]]
        tri_b = faces_local[face_indices[1]]
        quad = quad_from_oriented_diag_triangles(tri_a, tri_b, key)
        if quad is None:
            continue

        quad_key = tuple(sorted(int(v) for v in quad))
        if dedup_quads and quad_key in quad_keys:
            used_faces.update(face_indices)
            continue
        quad_keys.add(quad_key)

        recovered_quads.append(quad)
        used_faces.update(face_indices)

    remaining_triangles = [
        faces_local[i]
        for i in range(len(faces_local))
        if i not in used_faces
    ]
    triangles = (
        np.array(remaining_triangles, dtype=np.int64)
        if len(remaining_triangles) > 0
        else np.zeros((0, 3), dtype=np.int64)
    )
    quads = (
        np.array(recovered_quads, dtype=np.int64)
        if len(recovered_quads) > 0
        else np.zeros((0, 4), dtype=np.int64)
    )
    return triangles, quads


def _round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def make_quad_decoder_batch(
    vertex_latents: torch.Tensor,
    vertex_positions: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Create the minimal batch-like structure needed for recovery."""

    device = vertex_latents.device
    offsets = cu_seqlens.to(device=device, dtype=torch.int32)
    node_type = torch.zeros(vertex_latents.shape[0], dtype=torch.long, device=device)
    edges = torch.zeros((0, 2), dtype=torch.long, device=device)
    batch = {
        "nodes": vertex_positions.to(device=device),
        "offsets": offsets,
        "node_type": node_type,
        "edges": edges,
    }
    if vertex_positions.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        batch["position"] = vertex_positions
    return batch


def _decoder_inputs_for_model(
    model: MeshQuadVAE,
    vertex_latents: torch.Tensor,
    vertex_positions: torch.Tensor,
):
    position = None
    if getattr(model, "has_rope", False):
        if vertex_positions.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            position = vertex_positions.to(device=vertex_latents.device, dtype=torch.long) * 3
        else:
            position = _round_half_away_from_zero(
                vertex_positions.detach().cpu().numpy() * 3.0
            ).astype(np.int64, copy=False)
            position = torch.from_numpy(position).to(device=vertex_latents.device, dtype=torch.long)
    return vertex_latents.to(dtype=model.dec_proj_in.weight.dtype), position


def decode_vertex_embeddings(
    model: MeshQuadVAE,
    vertex_latents: torch.Tensor,
    vertex_positions: torch.Tensor,
    cu_seqlens: torch.Tensor,
    decoder_positions: Optional[torch.Tensor] = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    batch = make_quad_decoder_batch(vertex_latents, vertex_positions, cu_seqlens)
    if decoder_positions is not None:
        if decoder_positions.shape[0] != vertex_latents.shape[0]:
            raise ValueError(
                "decoder_positions must have the same token count as vertex_latents: "
                f"{decoder_positions.shape[0]} != {vertex_latents.shape[0]}"
            )
        x0_model = vertex_latents.to(dtype=model.dec_proj_in.weight.dtype)
        position = (
            decoder_positions.to(device=vertex_latents.device, dtype=torch.long)
            if getattr(model, "has_rope", False)
            else None
        )
    else:
        x0_model, position = _decoder_inputs_for_model(model, vertex_latents, vertex_positions)
    pred_vertex, _ = model.forward_decoder(
        x0_model,
        cu_seqlens.to(device=vertex_latents.device, dtype=torch.int32),
        position=position,
    )
    edge_embed, diag_embed, face_embed, orient_embed = torch.split(
        pred_vertex,
        [model.edge_dim, model.diag_dim, model.face_dim, model.orient_dim],
        dim=-1,
    )
    pred = {
        "st_feat": pred_vertex,
        "edge_embed": edge_embed,
        "diag_embed": diag_embed,
        "face_embed": face_embed,
        "orient_embed": orient_embed,
    }
    return pred, batch


def recover_one_mesh_from_embeddings(
    batch,
    pred,
    model,
    mesh_idx,
    mode,
):
    node_start = int(batch["offsets"][mesh_idx].item())
    node_end = int(batch["offsets"][mesh_idx + 1].item())
    vertex_global = torch.arange(node_start, node_end, dtype=torch.long, device=pred["edge_embed"].device)
    nv = vertex_global.shape[0]

    vertices_raw = batch["nodes"][vertex_global, :3]
    vertices_np = vertices_raw.detach().cpu().float().numpy()
    if nv < 3:
        return vertices_np, np.zeros((0, 3), dtype=np.int64), np.zeros((0, 4), dtype=np.int64)

    candidate_pairs = all_vertex_pairs(nv, pred["edge_embed"].device)
    valid_edges_local, _ = predict_edges(
        model,
        pred["edge_embed"],
        vertex_global,
        candidate_pairs,
        edge_chunk_size=5_000_000,
    )

    if mode == "tri":
        candidate_triangles = triangles_from_edges(valid_edges_local)
        triangles, _ = filter_triangles(model, pred, vertex_global, candidate_triangles)
        return vertices_np, triangles, np.zeros((0, 4), dtype=np.int64)

    if mode == "tri_connect":
        candidate_triangles = triangles_from_edges(valid_edges_local)
        triangles, _ = filter_triangles(model, pred, vertex_global, candidate_triangles)
        valid_diag_local, _ = predict_diags(
            model,
            pred["diag_embed"],
            vertex_global,
            valid_edges_local,
        )
        triangles, quads = merge_triangles_by_diag(triangles, valid_diag_local, dedup_quads=False)
        return vertices_np, triangles, quads

    if mode in ["native_quad", "native_quad_wireframe"]:
        valid_diag_local, _ = predict_diags(
            model,
            pred["diag_embed"],
            vertex_global,
            candidate_pairs,
        )
        triangle_graph_edges = torch.cat([valid_edges_local, valid_diag_local], dim=0)
        candidate_triangles = triangles_from_edges(triangle_graph_edges)
        candidate_triangles = filter_triangles_by_diag_count(
            candidate_triangles,
            valid_diag_local,
            max_diags=1,
        )
        triangles, _ = filter_triangles(model, pred, vertex_global, candidate_triangles)
        triangles, quads = merge_triangles_by_diag(triangles, valid_diag_local, dedup_quads=True)
        return vertices_np, triangles, quads

    raise NotImplementedError(mode)


@torch.no_grad()
def recover_meshes_from_embeddings(
    model: MeshQuadVAE,
    vertex_latents: torch.Tensor,
    vertex_positions: torch.Tensor,
    cu_seqlens: torch.Tensor,
    mode: str = "native_quad",
    decoder_positions: Optional[torch.Tensor] = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    was_training = model.training
    model.eval()
    pred, batch = decode_vertex_embeddings(
        model,
        vertex_latents,
        vertex_positions,
        cu_seqlens,
        decoder_positions=decoder_positions,
    )
    outputs = [
        recover_one_mesh_from_embeddings(batch, pred, model, mesh_idx, mode)
        for mesh_idx in range(cu_seqlens.shape[0] - 1)
    ]
    if was_training:
        model.train()
    return outputs
