from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    import triton  # noqa: F401
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


class SpacetimeEdgeLoss(nn.Module):
    """Copied materialized-pair spacetime edge loss baseline."""

    def __init__(
        self,
        use_focal_loss: bool = False,
        use_pred_balanced_loss: bool = False,
        use_dice_loss: bool = False,
        dice_weight: float = 1.0,
        pred_loss_weight: float = 1.0,
        gamma: float = 2.0,
        alpha: float = 0.25,
        chunk_size: int = 200000,
        extra_layer: bool = False,
        scale: str = "no",
    ) -> None:
        super().__init__()
        assert scale in ["no", "std", "learn", "learn_nobias"]
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma
        self.alpha = alpha
        self.chunk_size = chunk_size
        self.use_pred_balanced_loss = use_pred_balanced_loss
        self.pred_loss_weight = pred_loss_weight
        self.use_dice_loss = use_dice_loss
        self.dice_weight = dice_weight
        self.extra_layer = extra_layer
        self.scale = scale
        if self.scale == "learn":
            self.scale_d = nn.Linear(1, 1)
        elif self.scale == "learn_nobias":
            self.scale_d = nn.Linear(1, 1, bias=False)

    def spacetime_distance(self, node_embed: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        dim = node_embed.shape[-1]
        p0 = pair[:, 0]
        p1 = pair[:, 1]
        # Cast to fp32 for numerical consistency with the Triton kernel.
        node_t = node_embed[..., : dim // 2].float()
        node_s = node_embed[..., dim // 2 :].float()
        dt = ((node_t[p0] - node_t[p1]) ** 2).sum(dim=-1)
        ds = ((node_s[p0] - node_s[p1]) ** 2).sum(dim=-1)
        d = dt - ds
        return self._scale_distance(d, dim)

    def _scale_distance(self, d: torch.Tensor, dim: int) -> torch.Tensor:
        if self.scale == "std":
            return d / (2 * math.sqrt(dim))
        if self.scale in ["learn", "learn_nobias"]:
            return self.scale_d(d.unsqueeze(-1).to(self.scale_d.weight.dtype)).squeeze(-1).float()
        return d

    def forward(self, node_embed: torch.Tensor, pair: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total_pos_loss: torch.Tensor | float = 0.0
        total_neg_loss: torch.Tensor | float = 0.0
        total_pos = 0
        total_neg = 0
        total_dice_num: torch.Tensor | float = 0.0
        total_dice_den: torch.Tensor | float = 0.0
        total_pred_pos_loss: torch.Tensor | float = 0.0
        total_pred_neg_loss: torch.Tensor | float = 0.0
        total_pred_pos: torch.Tensor | float = 0
        total_pred_neg: torch.Tensor | float = 0
        total_tp: torch.Tensor | float = 0.0
        total_fp: torch.Tensor | float = 0.0
        total_fn: torch.Tensor | float = 0.0

        d = node_embed.new_zeros(())
        for start in range(0, pair.size(0), self.chunk_size):
            end = min(start + self.chunk_size, pair.size(0))
            d = self.spacetime_distance(node_embed, pair[start:end])
            d_float = d.float()
            t_chunk = target[start:end]
            pos_mask = t_chunk == 1
            neg_mask = t_chunk == 0

            if not self.use_focal_loss:
                if pos_mask.any():
                    pos_loss = F.binary_cross_entropy_with_logits(d_float[pos_mask], t_chunk[pos_mask].float(), reduction="mean")
                    total_pos_loss = total_pos_loss + pos_loss * pos_mask.sum()
                    total_pos += int(pos_mask.sum().item())
                if neg_mask.any():
                    neg_loss = F.binary_cross_entropy_with_logits(d_float[neg_mask], t_chunk[neg_mask].float(), reduction="mean")
                    total_neg_loss = total_neg_loss + neg_loss * neg_mask.sum()
                    total_neg += int(neg_mask.sum().item())
            else:
                p = torch.sigmoid(d_float)
                t = t_chunk.float()
                eps = 1e-8
                loss_pos = -self.alpha * (1 - p).pow(self.gamma) * torch.log(p + eps) * t
                loss_neg = -(1 - self.alpha) * p.pow(self.gamma) * torch.log(1 - p + eps) * (1 - t)
                loss_chunk = loss_pos + loss_neg
                if pos_mask.any():
                    total_pos_loss = total_pos_loss + loss_chunk[pos_mask].mean() * pos_mask.sum()
                    total_pos += int(pos_mask.sum().item())
                if neg_mask.any():
                    total_neg_loss = total_neg_loss + loss_chunk[neg_mask].mean() * neg_mask.sum()
                    total_neg += int(neg_mask.sum().item())

            if not self.use_focal_loss and self.use_pred_balanced_loss:
                probs = torch.sigmoid(d_float)
                pred_pos_mask = probs >= 0.5
                if pred_pos_mask.any():
                    pred_pos_loss = F.binary_cross_entropy(probs[pred_pos_mask], t_chunk[pred_pos_mask].float(), reduction="mean")
                    total_pred_pos_loss = total_pred_pos_loss + pred_pos_loss * pred_pos_mask.sum()
                    total_pred_pos = total_pred_pos + pred_pos_mask.sum()
                if (~pred_pos_mask).any():
                    pred_neg_loss = F.binary_cross_entropy(probs[~pred_pos_mask], t_chunk[~pred_pos_mask].float(), reduction="mean")
                    total_pred_neg_loss = total_pred_neg_loss + pred_neg_loss * (~pred_pos_mask).sum()
                    total_pred_neg = total_pred_neg + (~pred_pos_mask).sum()

            if self.use_dice_loss:
                p_chunk = torch.sigmoid(d_float)
                t_float = t_chunk.float()
                total_dice_num = total_dice_num + 2 * (p_chunk * t_float).sum()
                total_dice_den = total_dice_den + (p_chunk + t_float).sum() + 1e-8

            pred_label = d > 0
            total_tp = total_tp + ((pred_label == 1) & (t_chunk == 1)).sum()
            total_fp = total_fp + ((pred_label == 1) & (t_chunk == 0)).sum()
            total_fn = total_fn + ((pred_label == 0) & (t_chunk == 1)).sum()

        zero = torch.tensor(0.0, device=node_embed.device)
        pos_loss = total_pos_loss / total_pos if total_pos > 0 else zero
        neg_loss = total_neg_loss / total_neg if total_neg > 0 else zero
        main_loss = 0.5 * (pos_loss + neg_loss)
        precision = total_tp / torch.clamp(total_tp + total_fp, min=1e-8)
        recall = total_tp / torch.clamp(total_tp + total_fn, min=1e-8)
        f_score = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)
        log_dict = {
            "main_loss": main_loss.detach(),
            "pos_loss": pos_loss.detach(),
            "neg_loss": neg_loss.detach(),
            "prec": precision,
            "rec": recall,
            "fs": f_score,
            "1-fs": 1 - f_score,
            "1-rec": 1 - recall,
            "1-prec": 1 - precision,
        }
        total_loss = main_loss.clone()
        if self.use_dice_loss:
            dice_loss = 1 - total_dice_num / total_dice_den
            total_loss = total_loss + self.dice_weight * dice_loss
            log_dict["dice_loss"] = dice_loss.detach()
        if self.use_pred_balanced_loss:
            pred_pos_loss = total_pred_pos_loss / total_pred_pos if total_pred_pos > 0 else zero
            pred_neg_loss = total_pred_neg_loss / total_pred_neg if total_pred_neg > 0 else zero
            pred_balanced_loss = 0.5 * (pred_pos_loss + pred_neg_loss)
            total_loss = total_loss + self.pred_loss_weight * pred_balanced_loss
            log_dict.update({
                "pred_balanced_loss": pred_balanced_loss.detach(),
                "pred_pos_loss": pred_pos_loss.detach(),
                "pred_neg_loss": pred_neg_loss.detach(),
            })
        return total_loss, log_dict


class ReferenceSpacetimeAllPairEdgeLoss(SpacetimeEdgeLoss):
    """Copied current all-pair implementation, with chunked pair materialization."""

    max_generated_pairs = 268_435_456

    def _positive_keys(self, pair: torch.Tensor, max_nodes: int) -> torch.Tensor:
        if pair.numel() == 0:
            return pair.new_empty((0,), dtype=torch.long)
        pair = torch.sort(pair.long(), dim=1).values
        return pair[:, 0] * max_nodes + pair[:, 1]

    def _positive_mask(self, p0: torch.Tensor, p1: torch.Tensor, positive_keys: torch.Tensor, max_nodes: int) -> torch.Tensor:
        if positive_keys.numel() == 0:
            return torch.zeros(p0.shape[0], dtype=torch.bool, device=p0.device)
        pair_keys = p0.long() * max_nodes + p1.long()
        pos = torch.searchsorted(positive_keys, pair_keys)
        in_bounds = pos < positive_keys.numel()
        pos = pos.clamp(max=positive_keys.numel() - 1)
        return in_bounds & (positive_keys[pos] == pair_keys)

    def _spacetime_distance_indices(self, node_embed: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        pair = torch.stack([p0, p1], dim=-1)
        return self.spacetime_distance(node_embed, pair)

    def _chunk_sums(self, node_embed: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, pos_mask: torch.Tensor, pos_count: int):
        d = self._spacetime_distance_indices(node_embed, p0, p1)
        d_float = d.float()
        pos_logits = d_float[pos_mask]
        zero = d_float.new_zeros(())
        if not self.use_focal_loss:
            pos_bce = F.softplus(-pos_logits).sum() if pos_count > 0 else zero
            pos_as_neg_bce = F.softplus(pos_logits).sum() if pos_count > 0 else zero
            neg_bce = F.softplus(d_float).sum() - pos_as_neg_bce
        else:
            p = torch.sigmoid(d_float)
            eps = 1e-8
            pos_p = p[pos_mask]
            pos_focal = -self.alpha * (1 - pos_p).pow(self.gamma) * torch.log(pos_p + eps) if pos_count > 0 else zero
            neg_focal_all = -(1 - self.alpha) * p.pow(self.gamma) * torch.log(1 - p + eps)
            neg_focal_pos = -(1 - self.alpha) * pos_p.pow(self.gamma) * torch.log(1 - pos_p + eps) if pos_count > 0 else zero
            pos_bce = pos_focal.sum()
            neg_bce = neg_focal_all.sum() - neg_focal_pos.sum()

        pred_pos_loss = zero
        pred_neg_loss = zero
        pred_pos_count = zero
        pred_neg_count = zero
        if not self.use_focal_loss and self.use_pred_balanced_loss:
            probs = torch.sigmoid(d_float)
            pred_pos_mask = probs >= 0.5
            pred_pos_count = pred_pos_mask.sum()
            pred_neg_count = d_float.new_tensor(d_float.numel()) - pred_pos_count
            pos_probs = probs[pos_mask]
            pred_pos_probs = probs[pred_pos_mask]
            pred_neg_probs = probs[~pred_pos_mask]
            pred_pos_loss = F.binary_cross_entropy(pred_pos_probs, torch.zeros_like(pred_pos_probs), reduction="sum") if pred_pos_probs.numel() > 0 else zero
            pred_neg_loss = F.binary_cross_entropy(pred_neg_probs, torch.zeros_like(pred_neg_probs), reduction="sum") if pred_neg_probs.numel() > 0 else zero
            if pos_count > 0:
                pos_pred_pos_mask = pred_pos_mask[pos_mask]
                pos_pred_pos_probs = pos_probs[pos_pred_pos_mask]
                pos_pred_neg_probs = pos_probs[~pos_pred_pos_mask]
                pos_pos_delta = (
                    F.binary_cross_entropy(pos_pred_pos_probs, torch.ones_like(pos_pred_pos_probs), reduction="sum")
                    - F.binary_cross_entropy(pos_pred_pos_probs, torch.zeros_like(pos_pred_pos_probs), reduction="sum")
                    if pos_pred_pos_probs.numel() > 0
                    else zero
                )
                pos_neg_delta = (
                    F.binary_cross_entropy(pos_pred_neg_probs, torch.ones_like(pos_pred_neg_probs), reduction="sum")
                    - F.binary_cross_entropy(pos_pred_neg_probs, torch.zeros_like(pos_pred_neg_probs), reduction="sum")
                    if pos_pred_neg_probs.numel() > 0
                    else zero
                )
                pred_pos_loss = pred_pos_loss + pos_pos_delta
                pred_neg_loss = pred_neg_loss + pos_neg_delta

        dice_num = zero
        dice_den = zero
        if self.use_dice_loss:
            p_chunk = torch.sigmoid(d_float)
            dice_num = 2 * p_chunk[pos_mask].sum() if pos_count > 0 else zero
            dice_den = p_chunk.sum() + pos_count + 1e-8

        pred_label = d > 0
        tp = (pred_label & pos_mask).sum()
        pred_pos = pred_label.sum()
        fp = pred_pos - tp
        fn = d_float.new_tensor(pos_count) - tp
        return pos_bce, neg_bce, dice_num, dice_den, pred_pos_loss, pred_neg_loss, pred_pos_count, pred_neg_count, tp, fp, fn

    def _empty_state(self) -> dict[str, Any]:
        return {
            "total_pos_loss": 0.0,
            "total_neg_loss": 0.0,
            "total_pos": 0,
            "total_neg": 0,
            "total_dice_num": 0.0,
            "total_dice_den": 0.0,
            "total_pred_pos_loss": 0.0,
            "total_pred_neg_loss": 0.0,
            "total_pred_pos": 0,
            "total_pred_neg": 0,
            "total_tp": 0.0,
            "total_fp": 0.0,
            "total_fn": 0.0,
        }

    def _accumulate_index_chunk(self, node_embed: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, positive_keys: torch.Tensor, max_nodes: int, state: dict[str, Any]) -> None:
        pos_mask = self._positive_mask(p0, p1, positive_keys, max_nodes)
        pos_count = int(pos_mask.sum().item())
        neg_count = p0.numel() - pos_count

        def chunk_fn(node_embed_arg: torch.Tensor):
            return self._chunk_sums(node_embed_arg, p0, p1, pos_mask, pos_count)

        use_checkpoint = torch.is_grad_enabled() and node_embed.requires_grad
        sums = checkpoint(chunk_fn, node_embed, use_reentrant=False) if use_checkpoint else chunk_fn(node_embed)
        names = [
            "total_pos_loss",
            "total_neg_loss",
            "total_dice_num",
            "total_dice_den",
            "total_pred_pos_loss",
            "total_pred_neg_loss",
            "total_pred_pos",
            "total_pred_neg",
            "total_tp",
            "total_fp",
            "total_fn",
        ]
        for name, value in zip(names, sums):
            state[name] = state[name] + value
        state["total_pos"] += pos_count
        state["total_neg"] += neg_count

    def _finalize(self, state: dict[str, Any], device: torch.device):
        zero = torch.tensor(0.0, device=device)
        total_pos = state["total_pos"]
        total_neg = state["total_neg"]
        pos_loss = state["total_pos_loss"] / total_pos if total_pos > 0 else zero
        neg_loss = state["total_neg_loss"] / total_neg if total_neg > 0 else zero
        main_loss = 0.5 * (pos_loss + neg_loss)
        total_tp = state["total_tp"] if torch.is_tensor(state["total_tp"]) else zero
        total_fp = state["total_fp"] if torch.is_tensor(state["total_fp"]) else zero
        total_fn = state["total_fn"] if torch.is_tensor(state["total_fn"]) else zero
        precision = total_tp / torch.clamp(total_tp + total_fp, min=1e-8)
        recall = total_tp / torch.clamp(total_tp + total_fn, min=1e-8)
        f_score = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)
        log_dict = {
            "main_loss": main_loss.detach(),
            "pos_loss": pos_loss.detach(),
            "neg_loss": neg_loss.detach(),
            "prec": precision,
            "rec": recall,
            "fs": f_score,
            "1-prec": 1 - precision,
            "1-rec": 1 - recall,
            "1-fs": 1 - f_score,
        }
        total_loss = main_loss.clone()
        if self.use_dice_loss:
            dice_loss = 1 - state["total_dice_num"] / state["total_dice_den"] if torch.is_tensor(state["total_dice_den"]) and state["total_dice_den"] > 0 else zero
            total_loss = total_loss + self.dice_weight * dice_loss
            log_dict["dice_loss"] = dice_loss.detach()
        if self.use_pred_balanced_loss:
            pred_pos_loss = state["total_pred_pos_loss"] / state["total_pred_pos"] if state["total_pred_pos"] > 0 else zero
            pred_neg_loss = state["total_pred_neg_loss"] / state["total_pred_neg"] if state["total_pred_neg"] > 0 else zero
            pred_balanced_loss = 0.5 * (pred_pos_loss + pred_neg_loss)
            total_loss = total_loss + self.pred_loss_weight * pred_balanced_loss
            log_dict.update({
                "pred_balanced_loss": pred_balanced_loss.detach(),
                "pred_pos_loss": pred_pos_loss.detach(),
                "pred_neg_loss": pred_neg_loss.detach(),
            })
        return total_loss, log_dict

    def forward(self, node_embed: torch.Tensor, positive_pair: torch.Tensor, positive_pair_offsets: torch.Tensor, offsets: torch.Tensor, vertex_mask: torch.Tensor):
        state = self._empty_state()
        for mesh_idx in range(offsets.numel() - 1):
            node_start = int(offsets[mesh_idx].item())
            node_end = int(offsets[mesh_idx + 1].item())
            vertices = torch.nonzero(vertex_mask[node_start:node_end], as_tuple=False).flatten() + node_start
            if vertices.numel() < 2:
                continue
            pos_start = int(positive_pair_offsets[mesh_idx].item())
            pos_end = int(positive_pair_offsets[mesh_idx + 1].item())
            positive = positive_pair[pos_start:pos_end]
            max_nodes = node_embed.shape[0]
            positive_keys = torch.sort(self._positive_keys(positive, max_nodes)).values
            nv = vertices.numel()
            chunk_pair_limit = min(self.chunk_size, self.max_generated_pairs)
            chunk_start = 0
            while chunk_start < nv - 1:
                remaining = chunk_pair_limit
                row = chunk_start
                max_rows = 0
                while row < nv - 1 and remaining >= nv - row - 1:
                    remaining -= nv - row - 1
                    row += 1
                    max_rows += 1
                chunk_end = min(chunk_start + max(1, max_rows), nv - 1)
                row_counts = nv - torch.arange(chunk_start + 1, chunk_end + 1, device=node_embed.device, dtype=torch.long)
                num_pairs = int(row_counts.sum().item())
                if num_pairs == 0:
                    break
                row_local = torch.repeat_interleave(torch.arange(chunk_start, chunk_end, device=node_embed.device), row_counts)
                col_local = torch.arange(num_pairs, device=node_embed.device) - torch.repeat_interleave(torch.cumsum(row_counts, dim=0) - row_counts, row_counts) + row_local + 1
                self._accumulate_index_chunk(node_embed, vertices[row_local], vertices[col_local], positive_keys, max_nodes, state)
                chunk_start = chunk_end
        return self._finalize(state, node_embed.device)


if triton is not None:
    @triton.jit
    def _softplus_tl(x):
        return tl.log(1.0 + tl.exp(-tl.abs(x))) + tl.maximum(x, 0.0)

    @triton.jit
    def _pid_to_upper_tri(pid, num_col_tiles):
        """Convert linear pid to (row, col) in upper triangle (including diagonal).

        Maps pid in [0, num_col_tiles*(num_col_tiles+1)//2) to (tile_i, tile_j)
        where tile_j >= tile_i, using the inverse triangular number formula.
        Row i has (num_col_tiles - i) elements starting at column i.
        """
        tmp = num_col_tiles + 0.5
        tile_i = (tmp - tl.sqrt(tmp * tmp - 2.0 * pid)).to(tl.int32)
        row_start = tile_i * num_col_tiles - tile_i * (tile_i - 1) // 2
        tile_j = pid - row_start + tile_i
        return tile_i, tile_j

    @triton.jit
    def _find_mesh_idx(cum_ptr, pid, num_meshes: tl.constexpr):
        """Find mesh_idx such that cum_tiles[mesh_idx] <= pid < cum_tiles[mesh_idx+1].

        Linear search over meshes (typically B=8, so very fast).
        Returns index in [0, num_meshes-1].
        """
        mesh_idx = 0
        for i in tl.static_range(num_meshes):
            if tl.load(cum_ptr + i + 1) <= pid:
                mesh_idx = i + 1
        return mesh_idx

    # ---- Legacy kernels (full grid, kept for reference) ----

    @triton.jit
    def _allneg_forward_kernel(
        node_ptr,
        vertices_ptr,
        stats_ptr,
        pred_count_ptr,
        nv: tl.constexpr,
        total_nodes: tl.constexpr,
        dim: tl.constexpr,
        scale_ptr,
        num_col_tiles: tl.constexpr,
        BLOCK: tl.constexpr,
        HALF: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tile_i = pid // num_col_tiles
        tile_j = pid - tile_i * num_col_tiles

        offs_i = tile_i * BLOCK + tl.arange(0, BLOCK)
        offs_j = tile_j * BLOCK + tl.arange(0, BLOCK)
        valid_i = offs_i < nv
        valid_j = offs_j < nv
        idx_i = tl.load(vertices_ptr + offs_i, mask=valid_i, other=0)
        idx_j = tl.load(vertices_ptr + offs_j, mask=valid_j, other=0)
        offs_h = tl.arange(0, HALF)

        t_i = tl.load(node_ptr + idx_i[:, None] * dim + offs_h[None, :], mask=valid_i[:, None], other=0.0).to(tl.float32)
        t_j = tl.load(node_ptr + idx_j[:, None] * dim + offs_h[None, :], mask=valid_j[:, None], other=0.0).to(tl.float32)
        s_i = tl.load(node_ptr + idx_i[:, None] * dim + HALF + offs_h[None, :], mask=valid_i[:, None], other=0.0).to(tl.float32)
        s_j = tl.load(node_ptr + idx_j[:, None] * dim + HALF + offs_h[None, :], mask=valid_j[:, None], other=0.0).to(tl.float32)

        q_i = tl.sum(t_i * t_i, axis=1) - tl.sum(s_i * s_i, axis=1)
        q_j = tl.sum(t_j * t_j, axis=1) - tl.sum(s_j * s_j, axis=1)
        scale_weight = tl.load(scale_ptr)
        raw_d = q_i[:, None] + q_j[None, :] - 2.0 * tl.dot(t_i, tl.trans(t_j), input_precision="ieee") + 2.0 * tl.dot(s_i, tl.trans(s_j), input_precision="ieee")
        d = raw_d * scale_weight
        valid = (offs_i[:, None] < offs_j[None, :]) & valid_i[:, None] & valid_j[None, :]
        pred_pos = d >= 0.0
        probs = 1.0 / (1.0 + tl.exp(-d))
        bce0 = _softplus_tl(d)
        prob_bce0 = -tl.maximum(tl.log(1.0 - probs), -100.0)

        all_bce = tl.sum(tl.where(valid, bce0, 0.0))
        pred_pos_loss = tl.sum(tl.where(valid & pred_pos, prob_bce0, 0.0))
        pred_neg_loss = tl.sum(tl.where(valid & (~pred_pos), prob_bce0, 0.0))
        prob_sum = tl.sum(tl.where(valid, probs, 0.0))
        pred_pos_count = tl.sum(tl.where(valid & pred_pos, 1, 0))

        tl.store(stats_ptr + pid * 4 + 0, all_bce)
        tl.store(stats_ptr + pid * 4 + 1, pred_pos_loss)
        tl.store(stats_ptr + pid * 4 + 2, pred_neg_loss)
        tl.store(stats_ptr + pid * 4 + 3, prob_sum)
        tl.store(pred_count_ptr + pid, pred_pos_count)


    @triton.jit
    def _allneg_backward_kernel(
        node_ptr,
        vertices_ptr,
        grad_node_ptr,
        grad_scale_ptr,
        grad_stats_ptr,
        nv: tl.constexpr,
        total_nodes: tl.constexpr,
        dim: tl.constexpr,
        scale_ptr,
        num_col_tiles: tl.constexpr,
        BLOCK: tl.constexpr,
        HALF: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tile_i = pid // num_col_tiles
        tile_j = pid - tile_i * num_col_tiles

        offs_i = tile_i * BLOCK + tl.arange(0, BLOCK)
        offs_j = tile_j * BLOCK + tl.arange(0, BLOCK)
        valid_i = offs_i < nv
        valid_j = offs_j < nv
        idx_i = tl.load(vertices_ptr + offs_i, mask=valid_i, other=0)
        idx_j = tl.load(vertices_ptr + offs_j, mask=valid_j, other=0)
        offs_h = tl.arange(0, HALF)

        t_i = tl.load(node_ptr + idx_i[:, None] * dim + offs_h[None, :], mask=valid_i[:, None], other=0.0).to(tl.float32)
        t_j = tl.load(node_ptr + idx_j[:, None] * dim + offs_h[None, :], mask=valid_j[:, None], other=0.0).to(tl.float32)
        s_i = tl.load(node_ptr + idx_i[:, None] * dim + HALF + offs_h[None, :], mask=valid_i[:, None], other=0.0).to(tl.float32)
        s_j = tl.load(node_ptr + idx_j[:, None] * dim + HALF + offs_h[None, :], mask=valid_j[:, None], other=0.0).to(tl.float32)

        q_i = tl.sum(t_i * t_i, axis=1) - tl.sum(s_i * s_i, axis=1)
        q_j = tl.sum(t_j * t_j, axis=1) - tl.sum(s_j * s_j, axis=1)
        scale_weight = tl.load(scale_ptr)
        raw_d = q_i[:, None] + q_j[None, :] - 2.0 * tl.dot(t_i, tl.trans(t_j), input_precision="ieee") + 2.0 * tl.dot(s_i, tl.trans(s_j), input_precision="ieee")
        d = raw_d * scale_weight
        valid = (offs_i[:, None] < offs_j[None, :]) & valid_i[:, None] & valid_j[None, :]
        pred_pos = d >= 0.0
        probs = 1.0 / (1.0 + tl.exp(-d))

        g_all = tl.load(grad_stats_ptr + 0)
        g_pred_pos = tl.load(grad_stats_ptr + 1)
        g_pred_neg = tl.load(grad_stats_ptr + 2)
        g_prob = tl.load(grad_stats_ptr + 3)
        pred_prob_grad = probs
        pred_prob_grad = tl.where(probs >= 1.0 - 1.0e-12, 0.0, pred_prob_grad)
        g_bucket = tl.where(pred_pos, g_pred_pos, g_pred_neg) * pred_prob_grad
        grad_d_scaled = tl.where(valid, g_all * probs + g_bucket + g_prob * probs * (1.0 - probs), 0.0)
        grad_raw = grad_d_scaled * scale_weight

        row_sum = tl.sum(grad_raw, axis=1)
        col_sum = tl.sum(grad_raw, axis=0)
        row_t = 2.0 * (row_sum[:, None] * t_i - tl.dot(grad_raw, t_j, input_precision="ieee"))
        col_t = 2.0 * (col_sum[:, None] * t_j - tl.dot(tl.trans(grad_raw), t_i, input_precision="ieee"))
        row_s = -2.0 * (row_sum[:, None] * s_i - tl.dot(grad_raw, s_j, input_precision="ieee"))
        col_s = -2.0 * (col_sum[:, None] * s_j - tl.dot(tl.trans(grad_raw), s_i, input_precision="ieee"))

        tl.atomic_add(grad_node_ptr + idx_i[:, None] * dim + offs_h[None, :], row_t, sem="relaxed", mask=valid_i[:, None])
        tl.atomic_add(grad_node_ptr + idx_j[:, None] * dim + offs_h[None, :], col_t, sem="relaxed", mask=valid_j[:, None])
        tl.atomic_add(grad_node_ptr + idx_i[:, None] * dim + HALF + offs_h[None, :], row_s, sem="relaxed", mask=valid_i[:, None])
        tl.atomic_add(grad_node_ptr + idx_j[:, None] * dim + HALF + offs_h[None, :], col_s, sem="relaxed", mask=valid_j[:, None])
        tl.atomic_add(grad_scale_ptr, tl.sum(tl.where(valid, grad_d_scaled * raw_d, 0.0)), sem="relaxed")

    # ---- v2 kernels: upper-triangle-only dispatch ----

    @triton.jit
    def _allneg_forward_kernel_v2(
        node_ptr,
        vertices_ptr,
        vertex_offsets_ptr,
        cum_tiles_ptr,
        ntiles_ptr,
        stats_ptr,
        pred_count_ptr,
        total_tiles,
        dim: tl.constexpr,
        scale_ptr,
        num_meshes: tl.constexpr,
        BLOCK: tl.constexpr,
        HALF: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= total_tiles:
            return

        # Varlen dispatch: find which mesh and local tile this pid belongs to
        mesh_idx = _find_mesh_idx(cum_tiles_ptr, pid, num_meshes)
        local_pid = pid - tl.load(cum_tiles_ptr + mesh_idx)
        ncols_mesh = tl.load(ntiles_ptr + mesh_idx)
        tile_i, tile_j = _pid_to_upper_tri(local_pid, ncols_mesh)

        vertex_start = tl.load(vertex_offsets_ptr + mesh_idx)
        vertex_end = tl.load(vertex_offsets_ptr + mesh_idx + 1)
        nv = vertex_end - vertex_start

        offs_i = tile_i * BLOCK + tl.arange(0, BLOCK)
        offs_j = tile_j * BLOCK + tl.arange(0, BLOCK)
        valid_i = offs_i < nv
        valid_j = offs_j < nv
        base_i = tl.load(vertices_ptr + vertex_start + offs_i, mask=valid_i, other=0)
        base_j = tl.load(vertices_ptr + vertex_start + offs_j, mask=valid_j, other=0)
        offs_h = tl.arange(0, HALF)

        t_i = tl.load(node_ptr + base_i[:, None] * dim + offs_h[None, :], mask=valid_i[:, None], other=0.0).to(tl.float32)
        t_j = tl.load(node_ptr + base_j[:, None] * dim + offs_h[None, :], mask=valid_j[:, None], other=0.0).to(tl.float32)
        s_i = tl.load(node_ptr + base_i[:, None] * dim + HALF + offs_h[None, :], mask=valid_i[:, None], other=0.0).to(tl.float32)
        s_j = tl.load(node_ptr + base_j[:, None] * dim + HALF + offs_h[None, :], mask=valid_j[:, None], other=0.0).to(tl.float32)

        q_i = tl.sum(t_i * t_i, axis=1) - tl.sum(s_i * s_i, axis=1)
        q_j = tl.sum(t_j * t_j, axis=1) - tl.sum(s_j * s_j, axis=1)
        scale_weight = tl.load(scale_ptr)
        raw_d = q_i[:, None] + q_j[None, :] - 2.0 * tl.dot(t_i, tl.trans(t_j), input_precision="ieee") + 2.0 * tl.dot(s_i, tl.trans(s_j), input_precision="ieee")
        d = raw_d * scale_weight
        valid = (offs_i[:, None] < offs_j[None, :]) & valid_i[:, None] & valid_j[None, :]
        pred_pos = d >= 0.0
        probs = 1.0 / (1.0 + tl.exp(-d))
        bce0 = _softplus_tl(d)
        prob_bce0 = -tl.maximum(tl.log(1.0 - probs), -100.0)

        all_bce = tl.sum(tl.where(valid, bce0, 0.0))
        pred_pos_loss = tl.sum(tl.where(valid & pred_pos, prob_bce0, 0.0))
        pred_neg_loss = tl.sum(tl.where(valid & (~pred_pos), prob_bce0, 0.0))
        prob_sum = tl.sum(tl.where(valid, probs, 0.0))
        pred_pos_count = tl.sum(tl.where(valid & pred_pos, 1, 0))

        tl.store(stats_ptr + pid * 4 + 0, all_bce)
        tl.store(stats_ptr + pid * 4 + 1, pred_pos_loss)
        tl.store(stats_ptr + pid * 4 + 2, pred_neg_loss)
        tl.store(stats_ptr + pid * 4 + 3, prob_sum)
        tl.store(pred_count_ptr + pid, pred_pos_count)


    @triton.jit
    def _allneg_backward_kernel_v2(
        node_ptr,
        vertices_ptr,
        vertex_offsets_ptr,
        cum_tiles_ptr,
        ntiles_ptr,
        grad_node_ptr,
        grad_scale_ptr,
        grad_stats_ptr,
        total_tiles,
        dim: tl.constexpr,
        scale_ptr,
        num_meshes: tl.constexpr,
        BLOCK: tl.constexpr,
        HALF: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= total_tiles:
            return

        # Varlen dispatch: find which mesh and local tile this pid belongs to
        mesh_idx = _find_mesh_idx(cum_tiles_ptr, pid, num_meshes)
        local_pid = pid - tl.load(cum_tiles_ptr + mesh_idx)
        ncols_mesh = tl.load(ntiles_ptr + mesh_idx)
        tile_i, tile_j = _pid_to_upper_tri(local_pid, ncols_mesh)

        vertex_start = tl.load(vertex_offsets_ptr + mesh_idx)
        vertex_end = tl.load(vertex_offsets_ptr + mesh_idx + 1)
        nv = vertex_end - vertex_start

        offs_i = tile_i * BLOCK + tl.arange(0, BLOCK)
        offs_j = tile_j * BLOCK + tl.arange(0, BLOCK)
        valid_i = offs_i < nv
        valid_j = offs_j < nv
        base_i = tl.load(vertices_ptr + vertex_start + offs_i, mask=valid_i, other=0)
        base_j = tl.load(vertices_ptr + vertex_start + offs_j, mask=valid_j, other=0)
        offs_h = tl.arange(0, HALF)

        t_i = tl.load(node_ptr + base_i[:, None] * dim + offs_h[None, :], mask=valid_i[:, None], other=0.0).to(tl.float32)
        t_j = tl.load(node_ptr + base_j[:, None] * dim + offs_h[None, :], mask=valid_j[:, None], other=0.0).to(tl.float32)
        s_i = tl.load(node_ptr + base_i[:, None] * dim + HALF + offs_h[None, :], mask=valid_i[:, None], other=0.0).to(tl.float32)
        s_j = tl.load(node_ptr + base_j[:, None] * dim + HALF + offs_h[None, :], mask=valid_j[:, None], other=0.0).to(tl.float32)

        q_i = tl.sum(t_i * t_i, axis=1) - tl.sum(s_i * s_i, axis=1)
        q_j = tl.sum(t_j * t_j, axis=1) - tl.sum(s_j * s_j, axis=1)
        scale_weight = tl.load(scale_ptr)
        raw_d = q_i[:, None] + q_j[None, :] - 2.0 * tl.dot(t_i, tl.trans(t_j), input_precision="ieee") + 2.0 * tl.dot(s_i, tl.trans(s_j), input_precision="ieee")
        d = raw_d * scale_weight
        valid = (offs_i[:, None] < offs_j[None, :]) & valid_i[:, None] & valid_j[None, :]
        pred_pos = d >= 0.0
        probs = 1.0 / (1.0 + tl.exp(-d))

        g_all = tl.load(grad_stats_ptr + 0)
        g_pred_pos = tl.load(grad_stats_ptr + 1)
        g_pred_neg = tl.load(grad_stats_ptr + 2)
        g_prob = tl.load(grad_stats_ptr + 3)
        pred_prob_grad = probs
        pred_prob_grad = tl.where(probs >= 1.0 - 1.0e-12, 0.0, pred_prob_grad)
        g_bucket = tl.where(pred_pos, g_pred_pos, g_pred_neg) * pred_prob_grad
        grad_d_scaled = tl.where(valid, g_all * probs + g_bucket + g_prob * probs * (1.0 - probs), 0.0)
        grad_raw = grad_d_scaled * scale_weight

        row_sum = tl.sum(grad_raw, axis=1)
        col_sum = tl.sum(grad_raw, axis=0)
        row_t = 2.0 * (row_sum[:, None] * t_i - tl.dot(grad_raw, t_j, input_precision="ieee"))
        col_t = 2.0 * (col_sum[:, None] * t_j - tl.dot(tl.trans(grad_raw), t_i, input_precision="ieee"))
        row_s = -2.0 * (row_sum[:, None] * s_i - tl.dot(grad_raw, s_j, input_precision="ieee"))
        col_s = -2.0 * (col_sum[:, None] * s_j - tl.dot(tl.trans(grad_raw), s_i, input_precision="ieee"))

        tl.atomic_add(grad_node_ptr + base_i[:, None] * dim + offs_h[None, :], row_t, sem="relaxed", mask=valid_i[:, None])
        tl.atomic_add(grad_node_ptr + base_j[:, None] * dim + offs_h[None, :], col_t, sem="relaxed", mask=valid_j[:, None])
        tl.atomic_add(grad_node_ptr + base_i[:, None] * dim + HALF + offs_h[None, :], row_s, sem="relaxed", mask=valid_i[:, None])
        tl.atomic_add(grad_node_ptr + base_j[:, None] * dim + HALF + offs_h[None, :], col_s, sem="relaxed", mask=valid_j[:, None])
        tl.atomic_add(grad_scale_ptr, tl.sum(tl.where(valid, grad_d_scaled * raw_d, 0.0)), sem="relaxed")


def _build_varlen_metadata(vertex_offsets: torch.Tensor, tile_size: int):
    """Build varlen dispatch metadata for batch-level kernel launch.

    Args:
        vertex_offsets: [B+1] cumulative valid-vertex offsets
        tile_size: tile block size (64 or 128)

    Returns:
        cum_tiles: [B+1] cumulative upper-tri tile counts (starts at 0)
        ntiles: [B] number of column tiles per mesh
        total_tiles: int, total number of tiles across all meshes
    """
    mesh_sizes = vertex_offsets[1:] - vertex_offsets[:-1]  # [B]
    ntiles = torch.ceil(mesh_sizes.float() / tile_size).to(torch.int32)  # [B]
    upper_tiles = ntiles * (ntiles + 1) // 2  # [B]
    cum_tiles = torch.cat([torch.zeros(1, device=vertex_offsets.device, dtype=torch.int32),
                           upper_tiles.cumsum(dim=0)])  # [B+1]
    total_tiles = int(cum_tiles[-1].item())
    return cum_tiles, ntiles, total_tiles


def _masked_vertices_and_offsets(offsets: torch.Tensor, vertex_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Vectorized: a single device sync (the nonzero) replaces the per-mesh Python loop
    # of .item()/nonzero calls, which otherwise starves the GPU (CPU-bound stalls).
    device = offsets.device
    num_meshes = offsets.numel() - 1
    all_vertices = vertex_mask.nonzero(as_tuple=False).flatten().to(torch.long)
    node_counts = (offsets[1:] - offsets[:-1]).to(torch.long)
    node_mesh = torch.repeat_interleave(torch.arange(num_meshes, device=device), node_counts)
    per_mesh = torch.bincount(node_mesh[all_vertices], minlength=num_meshes)
    vertex_offsets = torch.cat([all_vertices.new_zeros(1), per_mesh.cumsum(0)])
    return all_vertices.contiguous(), vertex_offsets


class _TritonAllNegativeStats(torch.autograd.Function):
    @staticmethod
    def forward(ctx, node_embed: torch.Tensor, vertices: torch.Tensor, vertex_offsets: torch.Tensor, cum_tiles: torch.Tensor, ntiles: torch.Tensor, total_tiles: int, scale_weight: torch.Tensor, block_size: int):
        if triton is None:
            raise RuntimeError("Triton is required for _TritonAllNegativeStats")
        if node_embed.dim() != 2 or node_embed.shape[1] != 64:
            raise RuntimeError("Triton fast path currently expects node_embed shape [N, 64]")
        num_meshes = vertex_offsets.numel() - 1
        vertices_i64 = vertices.to(torch.int64).contiguous()
        vertex_offsets_i32 = vertex_offsets.to(torch.int32).contiguous()
        cum_tiles_i32 = cum_tiles.to(torch.int32).contiguous()
        ntiles_i32 = ntiles.to(torch.int32).contiguous()

        stats = torch.empty((total_tiles, 4), device=node_embed.device, dtype=torch.float32)
        pred_counts = torch.empty((total_tiles,), device=node_embed.device, dtype=torch.int64)

        num_warps = 8 if block_size >= 128 else 4

        _allneg_forward_kernel_v2[(total_tiles,)](
            node_embed,
            vertices_i64,
            vertex_offsets_i32,
            cum_tiles_i32,
            ntiles_i32,
            stats,
            pred_counts,
            total_tiles,
            node_embed.shape[1],
            scale_weight.float(),
            num_meshes,
            BLOCK=block_size,
            HALF=node_embed.shape[1] // 2,
            num_warps=num_warps,
        )
        out = torch.empty((5,), device=node_embed.device, dtype=torch.float32)
        out[:4] = stats.sum(dim=0)
        out[4] = pred_counts.sum().float()
        ctx.save_for_backward(node_embed, vertices_i64, vertex_offsets_i32, cum_tiles_i32, ntiles_i32, scale_weight)
        ctx.block_size = block_size
        ctx.total_tiles = total_tiles
        ctx.num_meshes = num_meshes
        ctx.num_warps = num_warps
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        node_embed, vertices_i64, vertex_offsets_i32, cum_tiles_i32, ntiles_i32, scale_weight = ctx.saved_tensors
        grad_node = torch.zeros(node_embed.shape, device=node_embed.device, dtype=torch.float32)
        grad_scale = torch.zeros((1,), device=node_embed.device, dtype=torch.float32)
        grad_stats = grad_out[:4].contiguous().float()
        _allneg_backward_kernel_v2[(ctx.total_tiles,)](
            node_embed,
            vertices_i64,
            vertex_offsets_i32,
            cum_tiles_i32,
            ntiles_i32,
            grad_node,
            grad_scale,
            grad_stats,
            ctx.total_tiles,
            node_embed.shape[1],
            scale_weight.float(),
            ctx.num_meshes,
            BLOCK=ctx.block_size,
            HALF=node_embed.shape[1] // 2,
            num_warps=ctx.num_warps,
        )
        if scale_weight.requires_grad:
            grad_scale_out = grad_scale.reshape_as(scale_weight).to(scale_weight.dtype)
        else:
            grad_scale_out = None
        return grad_node.to(node_embed.dtype), None, None, None, None, None, grad_scale_out, None


class FlashSpacetimeAllPairEdgeLoss(ReferenceSpacetimeAllPairEdgeLoss):
    """Tiled all-pair loss that avoids global pair/logit materialization.

    This implementation keeps the production forward signature. It uses the
    metric dot-product identity in tile matrices and sparse positive masks per
    tile. Backward is handled by autograd through checkpointed tile recompute.
    Unsupported focal mode delegates to the copied reference implementation.
    """

    def __init__(self, *args: Any, tile_size: int = 64, use_triton_fast_path: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tile_size = tile_size
        self.use_triton_fast_path = use_triton_fast_path

    def _can_use_triton_fast_path(self, node_embed: torch.Tensor, vertex_mask: torch.Tensor) -> bool:
        return (
            triton is not None
            and self.use_triton_fast_path
            and node_embed.is_cuda
            and node_embed.dtype in (torch.bfloat16, torch.float32)
            and node_embed.dim() == 2
            and node_embed.shape[-1] == 64
            and self.scale in ["no", "learn_nobias"]
            and not self.use_focal_loss
            and self.use_pred_balanced_loss
            and self.use_dice_loss
            and self.tile_size in [64, 128]
        )

    def _scale_weight_tensor(self, node_embed: torch.Tensor) -> torch.Tensor:
        if self.scale == "learn_nobias":
            return self.scale_d.weight.reshape(1).to(device=node_embed.device, dtype=torch.float32)
        return torch.ones((1,), device=node_embed.device, dtype=torch.float32)

    def _positive_corrections(self, node_embed: torch.Tensor, positive_pair: torch.Tensor, keep: torch.Tensor, scale_weight: torch.Tensor):
        if positive_pair.numel() == 0:
            zero = node_embed.new_zeros((), dtype=torch.float32)
            return zero, zero, zero, zero, zero, zero, zero
        # Compute distances in fp32 using the same dot-product identity as the Triton kernel:
        # d = q_i + q_j - 2*t_i@t_j + 2*s_i@s_j
        # where q_i = ||t_i||^2 - ||s_i||^2
        dim = node_embed.shape[-1]
        half = dim // 2
        p0 = positive_pair[:, 0]
        p1 = positive_pair[:, 1]
        t = node_embed[:, :half].float()
        s = node_embed[:, half:].float()
        q = (t * t).sum(dim=-1) - (s * s).sum(dim=-1)
        t_dot = (t[p0] * t[p1]).sum(dim=-1)
        s_dot = (s[p0] * s[p1]).sum(dim=-1)
        d = (q[p0] + q[p1] - 2.0 * t_dot + 2.0 * s_dot) * scale_weight
        # Masked (sync-free) reductions: keep excludes self-pairs and padded/non-vertex
        # endpoints. A multiplicative float mask avoids boolean-index/.any() device syncs.
        keep_f = keep.to(torch.float32)
        probs = torch.sigmoid(d)
        pred_pos = (d >= 0).to(torch.float32) * keep_f
        pred_neg = (d < 0).to(torch.float32) * keep_f
        pos_as_pos = (F.softplus(-d) * keep_f).sum()
        pos_as_neg = (F.softplus(d) * keep_f).sum()
        # BCE(p, 1) - BCE(p, 0) per element (PyTorch clamps log >= -100, matching the kernel).
        bce_delta = (
            F.binary_cross_entropy(probs, torch.ones_like(probs), reduction="none")
            - F.binary_cross_entropy(probs, torch.zeros_like(probs), reduction="none")
        )
        pred_pos_delta = (bce_delta * pred_pos).sum()
        pred_neg_delta = (bce_delta * pred_neg).sum()
        dice_num = 2.0 * (probs * keep_f).sum()
        tp = pred_pos.sum()
        fp_delta = -tp
        fn = keep_f.sum() - tp
        return pos_as_pos, pos_as_neg, pred_pos_delta, pred_neg_delta, dice_num, fp_delta, fn

    def _masked_positive_pairs(self, positive_pair: torch.Tensor, vertex_mask: torch.Tensor) -> torch.Tensor:
        if positive_pair.numel() == 0:
            return positive_pair
        p0 = positive_pair[:, 0]
        p1 = positive_pair[:, 1]
        keep = (p0 != p1) & vertex_mask[p0] & vertex_mask[p1]
        return positive_pair[keep]

    def _positive_keep_mask(self, positive_pair: torch.Tensor, vertex_mask: torch.Tensor) -> torch.Tensor:
        # Boolean mask of valid positive pairs (no self-pairs, both endpoints are vertices).
        if positive_pair.numel() == 0:
            return positive_pair.new_zeros((0,), dtype=torch.bool)
        p0 = positive_pair[:, 0]
        p1 = positive_pair[:, 1]
        return (p0 != p1) & vertex_mask[p0] & vertex_mask[p1]

    def _forward_triton_fast(self, node_embed: torch.Tensor, positive_pair: torch.Tensor, positive_pair_offsets: torch.Tensor, offsets: torch.Tensor, vertex_mask: torch.Tensor):
        scale_weight = self._scale_weight_tensor(node_embed)
        vertices, vertex_offsets = _masked_vertices_and_offsets(offsets, vertex_mask)
        keep = self._positive_keep_mask(positive_pair, vertex_mask)
        total_pos = keep.sum()  # int64 tensor kept on device (no sync)

        # Build varlen metadata once for all meshes — single kernel launch.
        # total_tiles is the one device->host read needed (it is the kernel grid size).
        cum_tiles, ntiles, total_tiles = _build_varlen_metadata(vertex_offsets, self.tile_size)

        # Compute total_pairs analytically from masked vertex counts (kept as a device tensor)
        mesh_sizes = vertex_offsets[1:] - vertex_offsets[:-1]
        total_pairs = (mesh_sizes.long() * (mesh_sizes.long() - 1) // 2).sum()

        if total_tiles == 0:
            zero = node_embed.new_zeros((), dtype=torch.float32)
            log_dict = {k: zero for k in ["main_loss", "pos_loss", "neg_loss", "prec", "rec", "fs", "1-prec", "1-rec", "1-fs", "dice_loss", "pred_balanced_loss", "pred_pos_loss", "pred_neg_loss"]}
            return zero, log_dict

        stats = _TritonAllNegativeStats.apply(node_embed, vertices, vertex_offsets, cum_tiles, ntiles, total_tiles, scale_weight, self.tile_size)
        all_bce = stats[0]
        all_pred_pos_loss = stats[1]
        all_pred_neg_loss = stats[2]
        all_prob_sum = stats[3]
        all_pred_pos_count = stats[4]

        pos_as_pos, pos_as_neg, pred_pos_delta, pred_neg_delta, dice_num_pos, _fp_delta, fn = self._positive_corrections(
            node_embed, positive_pair, keep, scale_weight
        )
        total_pos_f = total_pos.to(torch.float32)
        total_neg = (total_pairs - total_pos).clamp_min(1).to(torch.float32)
        neg_loss = (all_bce - pos_as_neg) / total_neg
        pos_loss = pos_as_pos / total_pos_f.clamp_min(1.0)
        main_loss = 0.5 * (pos_loss + neg_loss)

        pred_pos_count = all_pred_pos_count
        pred_neg_count = total_pairs.to(torch.float32) - pred_pos_count
        pred_pos_loss = (all_pred_pos_loss + pred_pos_delta) / pred_pos_count.clamp_min(1.0)
        pred_neg_loss = (all_pred_neg_loss + pred_neg_delta) / pred_neg_count.clamp_min(1.0)
        pred_balanced_loss = 0.5 * (pred_pos_loss + pred_neg_loss)

        dice_den = all_prob_sum + total_pos_f + 1e-8
        dice_loss = 1.0 - dice_num_pos / dice_den
        total_loss = main_loss + self.pred_loss_weight * pred_balanced_loss + self.dice_weight * dice_loss

        tp = total_pos_f - fn
        # pred_pos_count (Triton kernel, over ALL pairs) and tp (PyTorch positive-
        # corrections, over POSITIVE pairs) come from two different code paths. Near
        # the decision boundary (|d|~0, e.g. small learnable scale) they can disagree
        # on the sign of d for the positive pairs, making pred_pos_count < tp so
        # fp = pred_pos_count - tp goes negative. Since tp + fp == pred_pos_count, that
        # collapses the precision denominator to a tiny integer and blows precision up
        # to ~1e5 (e.g. prec=312500). Clamp fp >= 0 so precision stays in [0, 1].
        fp = (pred_pos_count - tp).clamp_min(0.0)
        precision = tp / torch.clamp(tp + fp, min=1e-8)
        recall = tp / torch.clamp(tp + fn, min=1e-8)
        f_score = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)

        log_dict = {
            "main_loss": main_loss.detach(),
            "pos_loss": pos_loss.detach(),
            "neg_loss": neg_loss.detach(),
            "prec": precision,
            "rec": recall,
            "fs": f_score,
            "1-prec": 1 - precision,
            "1-rec": 1 - recall,
            "1-fs": 1 - f_score,
            "dice_loss": dice_loss.detach(),
            "pred_balanced_loss": pred_balanced_loss.detach(),
            "pred_pos_loss": pred_pos_loss.detach(),
            "pred_neg_loss": pred_neg_loss.detach(),
        }
        return total_loss, log_dict

    def _tile_logits(self, node_embed: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        dim = node_embed.shape[-1]
        half = dim // 2
        t = node_embed[:, :half]
        s = node_embed[:, half:]
        rt = t[rows].float()
        rs = s[rows].float()
        ct = t[cols].float()
        cs = s[cols].float()
        d = q[rows, None] + q[cols][None, :] - 2.0 * (rt @ ct.T) + 2.0 * (rs @ cs.T)
        return self._scale_distance(d, dim)

    def _positive_mask_matrix(self, rows: torch.Tensor, cols: torch.Tensor, positive_keys: torch.Tensor, max_nodes: int) -> torch.Tensor:
        if positive_keys.numel() == 0:
            return torch.zeros((rows.numel(), cols.numel()), dtype=torch.bool, device=rows.device)
        keys = rows[:, None].long() * max_nodes + cols[None, :].long()
        pos = torch.searchsorted(positive_keys, keys.reshape(-1))
        in_bounds = pos < positive_keys.numel()
        pos = pos.clamp(max=positive_keys.numel() - 1)
        return (in_bounds & (positive_keys[pos] == keys.reshape(-1))).reshape(rows.numel(), cols.numel())

    def _accumulate_tile(self, node_embed: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, local_rows: torch.Tensor, local_cols: torch.Tensor, q: torch.Tensor, positive_keys: torch.Tensor, max_nodes: int, state: dict[str, Any]) -> None:
        upper_mask = local_rows[:, None] < local_cols[None, :]
        if not bool(upper_mask.any().item()):
            return
        pos_mask = self._positive_mask_matrix(rows, cols, positive_keys, max_nodes) & upper_mask
        pos_count = int(pos_mask.sum().item())
        pair_count = int(upper_mask.sum().item())
        neg_count = pair_count - pos_count

        def tile_fn(node_embed_arg: torch.Tensor, q_arg: torch.Tensor):
            logits = self._tile_logits(node_embed_arg, rows, cols, q_arg).float()
            valid_logits = logits[upper_mask]
            valid_pos_mask = pos_mask[upper_mask]
            pos_logits = valid_logits[valid_pos_mask]
            zero = valid_logits.new_zeros(())
            pos_bce = F.softplus(-pos_logits).sum() if pos_count > 0 else zero
            pos_as_neg_bce = F.softplus(pos_logits).sum() if pos_count > 0 else zero
            neg_bce = F.softplus(valid_logits).sum() - pos_as_neg_bce
            pred_pos_loss = zero
            pred_neg_loss = zero
            pred_pos_count = zero
            pred_neg_count = zero
            if self.use_pred_balanced_loss:
                probs = torch.sigmoid(valid_logits)
                pred_pos_mask = probs >= 0.5
                pred_pos_count = pred_pos_mask.sum()
                pred_neg_count = valid_logits.new_tensor(valid_logits.numel()) - pred_pos_count
                pred_pos_probs = probs[pred_pos_mask]
                pred_neg_probs = probs[~pred_pos_mask]
                pred_pos_loss = F.binary_cross_entropy(pred_pos_probs, torch.zeros_like(pred_pos_probs), reduction="sum") if pred_pos_probs.numel() > 0 else zero
                pred_neg_loss = F.binary_cross_entropy(pred_neg_probs, torch.zeros_like(pred_neg_probs), reduction="sum") if pred_neg_probs.numel() > 0 else zero
                if pos_count > 0:
                    pos_probs = probs[valid_pos_mask]
                    pos_pred_pos_mask = pred_pos_mask[valid_pos_mask]
                    pos_pred_pos_probs = pos_probs[pos_pred_pos_mask]
                    pos_pred_neg_probs = pos_probs[~pos_pred_pos_mask]
                    pred_pos_loss = pred_pos_loss + (
                        F.binary_cross_entropy(pos_pred_pos_probs, torch.ones_like(pos_pred_pos_probs), reduction="sum")
                        - F.binary_cross_entropy(pos_pred_pos_probs, torch.zeros_like(pos_pred_pos_probs), reduction="sum")
                        if pos_pred_pos_probs.numel() > 0
                        else zero
                    )
                    pred_neg_loss = pred_neg_loss + (
                        F.binary_cross_entropy(pos_pred_neg_probs, torch.ones_like(pos_pred_neg_probs), reduction="sum")
                        - F.binary_cross_entropy(pos_pred_neg_probs, torch.zeros_like(pos_pred_neg_probs), reduction="sum")
                        if pos_pred_neg_probs.numel() > 0
                        else zero
                    )
            dice_num = zero
            dice_den = zero
            if self.use_dice_loss:
                probs = torch.sigmoid(valid_logits)
                dice_num = 2 * probs[valid_pos_mask].sum() if pos_count > 0 else zero
                dice_den = probs.sum() + pos_count + 1e-8
            pred_label = valid_logits > 0
            tp = (pred_label & valid_pos_mask).sum()
            pred_pos = pred_label.sum()
            fp = pred_pos - tp
            fn = valid_logits.new_tensor(pos_count) - tp
            return pos_bce, neg_bce, dice_num, dice_den, pred_pos_loss, pred_neg_loss, pred_pos_count, pred_neg_count, tp, fp, fn

        use_checkpoint = torch.is_grad_enabled() and node_embed.requires_grad
        sums = checkpoint(tile_fn, node_embed, q, use_reentrant=False) if use_checkpoint else tile_fn(node_embed, q)
        names = [
            "total_pos_loss",
            "total_neg_loss",
            "total_dice_num",
            "total_dice_den",
            "total_pred_pos_loss",
            "total_pred_neg_loss",
            "total_pred_pos",
            "total_pred_neg",
            "total_tp",
            "total_fp",
            "total_fn",
        ]
        for name, value in zip(names, sums):
            state[name] = state[name] + value
        state["total_pos"] += pos_count
        state["total_neg"] += neg_count

    def forward(self, node_embed: torch.Tensor, positive_pair: torch.Tensor, positive_pair_offsets: torch.Tensor, offsets: torch.Tensor, vertex_mask: torch.Tensor):
        # The Triton kernel indexes node_embed via raw pointer arithmetic
        # (node_ptr + idx*dim + offs), assuming a contiguous [N, dim] row stride.
        # Callers often pass a non-contiguous column slice (e.g. edge_embed =
        # torch.split(pred, ...)[0], whose .float() is a no-op when pred is already
        # fp32), which would make the kernel read the wrong memory -> garbage logits
        # -> negative loss. Force contiguity so the stride matches the kernel's
        # assumption. (autograd routes the gradient back through .contiguous().)
        node_embed = node_embed.contiguous()
        if self._can_use_triton_fast_path(node_embed, vertex_mask):
            return self._forward_triton_fast(node_embed, positive_pair, positive_pair_offsets, offsets, vertex_mask)
        if self.use_focal_loss:
            return super().forward(node_embed, positive_pair, positive_pair_offsets, offsets, vertex_mask)

        state = self._empty_state()
        dim = node_embed.shape[-1]
        half = dim // 2
        q = (node_embed[:, :half].float().square().sum(dim=-1) - node_embed[:, half:].float().square().sum(dim=-1)).contiguous()
        max_nodes = node_embed.shape[0]
        for mesh_idx in range(offsets.numel() - 1):
            node_start = int(offsets[mesh_idx].item())
            node_end = int(offsets[mesh_idx + 1].item())
            vertices = torch.nonzero(vertex_mask[node_start:node_end], as_tuple=False).flatten() + node_start
            nv = int(vertices.numel())
            if nv < 2:
                continue
            pos_start = int(positive_pair_offsets[mesh_idx].item())
            pos_end = int(positive_pair_offsets[mesh_idx + 1].item())
            positive_keys = torch.sort(self._positive_keys(positive_pair[pos_start:pos_end], max_nodes)).values
            local = torch.arange(nv, device=node_embed.device)
            for row_start in range(0, nv, self.tile_size):
                row_end = min(row_start + self.tile_size, nv)
                for col_start in range(row_start, nv, self.tile_size):
                    col_end = min(col_start + self.tile_size, nv)
                    self._accumulate_tile(
                        node_embed,
                        vertices[row_start:row_end],
                        vertices[col_start:col_end],
                        local[row_start:row_end],
                        local[col_start:col_end],
                        q,
                        positive_keys,
                        max_nodes,
                        state,
                    )
        return self._finalize(state, node_embed.device)
