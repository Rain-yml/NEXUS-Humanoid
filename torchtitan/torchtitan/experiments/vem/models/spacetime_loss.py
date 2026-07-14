import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import math
from torchtitan.experiments.vem.models.transformer import (
    FP32LayerNorm, 
)

class SpacetimeEdgeLoss(nn.Module):
    """
    Computes BCE or Focal loss + F-score for spacetime edge classification
    using chunking for large pair tensors.
    """

    def __init__(
        self,
        use_focal_loss=False,
        use_pred_balanced_loss=False,
        use_dice_loss=False,
        dice_weight=1.0,
        pred_loss_weight=1.0,
        gamma=2.0,
        alpha=0.25,
        chunk_size=200000,
        extra_layer=False,
        scale='no',
    ):
        super().__init__()
        assert scale in ['no', 'std', 'learn', 'learn_nobias']
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
        if self.scale == 'learn':
            self.scale_d = nn.Linear(1, 1)
        elif self.scale == 'learn_nobias':
            self.scale_d = nn.Linear(1, 1, bias=False)
    
    def spacetime_distance(self, node_embed, pair):
        dim = node_embed.shape[-1]

        p0 = pair[:, 0]
        p1 = pair[:, 1]

        node_t = node_embed[..., : dim // 2]
        node_s = node_embed[..., dim // 2 :]

        dt = ((node_t[p0] - node_t[p1]) ** 2).sum(dim=-1)
        ds = ((node_s[p0] - node_s[p1]) ** 2).sum(dim=-1)
        d = dt - ds
        if self.scale == 'std':
            d = d / (2 * math.sqrt(dim))
        elif self.scale in ['learn', 'learn_nobias']:
            # Apply the learnable scale in fp32. scale_d.weight is bf16 under mixed
            # precision; the original `.to(weight.dtype)` downcast d to bf16, diverging
            # from the fp32 flash kernel. fp32 here keeps flash/non-flash equivalent.
            w = self.scale_d.weight.reshape(()).float()
            d = (d.float() * w)
            if self.scale_d.bias is not None:
                d = d + self.scale_d.bias.reshape(()).float()
        return d

    def forward(self, node_embed, pair, target):
        """
        node_embed: [V, H, D]
        pair:       [M, 2]
        target:     [M] (0/1)

        Returns:
            loss, log_dict with F-score, precision, recall, bce/focal loss
        """

        dim = node_embed.shape[-1]

        total_pos_loss = 0.0
        total_neg_loss = 0.0
        total_pos = 0
        total_neg = 0
    
        # Dice loss accumulators
        total_dice_num = 0.0
        total_dice_den = 0.0

        # NEW: predicted-balanced accumulators
        total_pred_pos_loss = 0.0
        total_pred_neg_loss = 0.0
        total_pred_pos = 0
        total_pred_neg = 0

        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0

        num_pairs = pair.size(0)
        chunk_size = self.chunk_size

        for start in range(0, num_pairs, chunk_size):
            end = min(start + chunk_size, num_pairs)

            d = self.spacetime_distance(node_embed, pair[start:end])

            t_chunk = target[start:end]

            # =============== LOSS ===============
            if not self.use_focal_loss:
                pos_mask = (t_chunk == 1)
                neg_mask = (t_chunk == 0)

                if pos_mask.any():
                    pos_loss = F.binary_cross_entropy_with_logits(
                        d[pos_mask].float(),
                        t_chunk[pos_mask].float(),
                        reduction="mean"
                    )
                    total_pos_loss += pos_loss * pos_mask.sum()
                    total_pos += pos_mask.sum()

                if neg_mask.any():
                    neg_loss = F.binary_cross_entropy_with_logits(
                        d[neg_mask].float(),
                        t_chunk[neg_mask].float(),
                        reduction="mean"
                    )
                    total_neg_loss += neg_loss * neg_mask.sum()
                    total_neg += neg_mask.sum()
                # # BCE (sum reduction for manual aggregation)
                # loss_chunk = F.binary_cross_entropy_with_logits(
                #     d.float(), t_chunk.float(), reduction="sum",
                # )
            else:
                # ---------- focal loss ----------
                p = torch.sigmoid(d.float())
                t = t_chunk.float()
                eps = 1e-8

                loss_pos = -self.alpha * (1 - p).pow(self.gamma) * torch.log(p + eps) * t
                loss_neg = -(1 - self.alpha) * p.pow(self.gamma) * torch.log(1 - p + eps) * (1 - t)
                loss_chunk = (loss_pos + loss_neg)

                # Split and aggregate (balanced focal is uncommon but this keeps same structure)
                pos_mask = (t_chunk == 1)
                neg_mask = (t_chunk == 0)

                if pos_mask.any():
                    total_pos_loss += loss_chunk[pos_mask].mean() * pos_mask.sum()
                    total_pos += pos_mask.sum()

                if neg_mask.any():
                    total_neg_loss += loss_chunk[neg_mask].mean() * neg_mask.sum()
                    total_neg += neg_mask.sum()

            # =================== NEW: PREDICTED BALANCED LOSS ===================
            if not self.use_focal_loss and self.use_pred_balanced_loss:  # only for BCE mode
                probs = torch.sigmoid(d.float())
                pred_pos_mask = (probs >= 0.5)
                pred_neg_mask = (probs < 0.5)

                if pred_pos_mask.any():
                    pred_pos_loss = F.binary_cross_entropy(
                        probs[pred_pos_mask],
                        t_chunk[pred_pos_mask].float(),
                        reduction="mean"
                    )
                    total_pred_pos_loss += pred_pos_loss * pred_pos_mask.sum()
                    total_pred_pos += pred_pos_mask.sum()

                if pred_neg_mask.any():
                    pred_neg_loss = F.binary_cross_entropy(
                        probs[pred_neg_mask],
                        t_chunk[pred_neg_mask].float(),
                        reduction="mean"
                    )
                    total_pred_neg_loss += pred_neg_loss * pred_neg_mask.sum()
                    total_pred_neg += pred_neg_mask.sum()
            
            # =================== DICE LOSS ===================
            if self.use_dice_loss:
                p_chunk = torch.sigmoid(d.float())
                t_float = t_chunk.float()

                dice_num = 2 * (p_chunk * t_float).sum()
                dice_den = (p_chunk + t_float).sum() + 1e-8

                total_dice_num += dice_num
                total_dice_den += dice_den

            # ================= METRICS ====================
            pred_label = (d > 0).long()
            total_tp += ((pred_label == 1) & (t_chunk == 1)).sum()
            total_fp += ((pred_label == 1) & (t_chunk == 0)).sum()
            total_fn += ((pred_label == 0) & (t_chunk == 1)).sum()

        # ======= Final Balanced BCE ========
        pos_loss = total_pos_loss / total_pos if total_pos > 0 else torch.tensor(0.0, device=d.device)
        neg_loss = total_neg_loss / total_neg if total_neg > 0 else torch.tensor(0.0, device=d.device)
        main_loss = 0.5 * (pos_loss + neg_loss)

        # ======= F-score =======
        precision = total_tp / max(total_tp + total_fp, 1e-8)
        recall = total_tp / max(total_tp + total_fn, 1e-8)
        f_score = 2 * precision * recall / max(precision + recall, 1e-8)

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

        # ======= Dice loss ========
        if self.use_dice_loss:
            dice_coeff = total_dice_num / total_dice_den
            dice_loss = 1 - dice_coeff
            total_loss += self.dice_weight * dice_loss
            log_dict.update({
                "dice_loss": dice_loss.detach(),
            })

        # ======= Pred-Balanced Loss ========
        if self.use_pred_balanced_loss:
            if total_pred_pos > 0:
                pred_pos_loss = total_pred_pos_loss / total_pred_pos
            else:
                pred_pos_loss = torch.tensor(0.0, device=d.device)

            if total_pred_neg > 0:
                pred_neg_loss = total_pred_neg_loss / total_pred_neg
            else:
                pred_neg_loss = torch.tensor(0.0, device=d.device)

            pred_balanced_loss = 0.5 * (pred_pos_loss + pred_neg_loss)
            total_loss += self.pred_loss_weight * pred_balanced_loss
            log_dict.update({
                "pred_balanced_loss": pred_balanced_loss.detach(),
                "pred_pos_loss": pred_pos_loss.detach(),
                "pred_neg_loss": pred_neg_loss.detach(),
            })

        return total_loss, log_dict

class SpacetimeAllPairEdgeLoss(SpacetimeEdgeLoss):
    """
    Edge loss that receives positive pairs only and recreates all unordered
    vertex pairs inside each mesh chunk on device.
    """

    # max_generated_pairs = 262_144
    # max_generated_pairs = 8388608 # 2 ** 23
    max_generated_pairs = 268_435_456 # 2 ** 28

    def _positive_keys(self, pair, max_nodes):
        if pair.numel() == 0:
            return pair.new_empty((0,), dtype=torch.long)
        pair = torch.sort(pair.long(), dim=1).values
        return pair[:, 0] * max_nodes + pair[:, 1]

    def _positive_mask(self, p0, p1, positive_keys, max_nodes):
        if positive_keys.numel() == 0:
            return torch.zeros(p0.shape[0], dtype=torch.bool, device=p0.device)
        pair_keys = p0.long() * max_nodes + p1.long()
        pos = torch.searchsorted(positive_keys, pair_keys)
        in_bounds = pos < positive_keys.numel()
        pos = pos.clamp(max=positive_keys.numel() - 1)
        return in_bounds & (positive_keys[pos] == pair_keys)

    def _spacetime_distance_indices(self, node_embed, p0, p1):
        from torch.distributed.tensor import DTensor
        dim = node_embed.shape[-1]
        node_t = node_embed[..., : dim // 2]
        node_s = node_embed[..., dim // 2 :]

        dt = ((node_t[p0] - node_t[p1]) ** 2).sum(dim=-1)
        ds = ((node_s[p0] - node_s[p1]) ** 2).sum(dim=-1)
        d = dt - ds
        if self.scale == 'std':
            d = d / (2 * math.sqrt(dim))
        elif self.scale in ['learn', 'learn_nobias']:
            # Apply the learnable scale in fp32. scale_d.weight is bf16 under mixed
            # precision; the original `.to(weight.dtype)` downcast d to bf16, diverging
            # from the fp32 flash kernel. fp32 here keeps flash/non-flash equivalent.
            w = self.scale_d.weight.reshape(()).float()
            d = (d.float() * w)
            if self.scale_d.bias is not None:
                d = d + self.scale_d.bias.reshape(()).float()
        return d

    def _chunk_sums(self, node_embed, p0, p1, pos_mask, pos_count):
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
            pos_focal = (
                -self.alpha * (1 - pos_p).pow(self.gamma) * torch.log(pos_p + eps)
                if pos_count > 0
                else zero
            )
            neg_focal_all = -(1 - self.alpha) * p.pow(self.gamma) * torch.log(1 - p + eps)
            neg_focal_pos = (
                -(1 - self.alpha) * pos_p.pow(self.gamma) * torch.log(1 - pos_p + eps)
                if pos_count > 0
                else zero
            )
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
            pred_pos_loss = (
                F.binary_cross_entropy(
                    pred_pos_probs,
                    torch.zeros_like(pred_pos_probs),
                    reduction="sum",
                )
                if pred_pos_probs.numel() > 0
                else zero
            )
            pred_neg_loss = (
                F.binary_cross_entropy(
                    pred_neg_probs,
                    torch.zeros_like(pred_neg_probs),
                    reduction="sum",
                )
                if pred_neg_probs.numel() > 0
                else zero
            )
            if pos_count > 0:
                pos_pred_pos_mask = pred_pos_mask[pos_mask]
                pos_pred_pos_probs = pos_probs[pos_pred_pos_mask]
                pos_pred_neg_probs = pos_probs[~pos_pred_pos_mask]
                pos_pos_delta = (
                    F.binary_cross_entropy(
                        pos_pred_pos_probs,
                        torch.ones_like(pos_pred_pos_probs),
                        reduction="sum",
                    )
                    - F.binary_cross_entropy(
                        pos_pred_pos_probs,
                        torch.zeros_like(pos_pred_pos_probs),
                        reduction="sum",
                    )
                    if pos_pred_pos_probs.numel() > 0
                    else zero
                )
                pos_neg_delta = (
                    F.binary_cross_entropy(
                        pos_pred_neg_probs,
                        torch.ones_like(pos_pred_neg_probs),
                        reduction="sum",
                    )
                    - F.binary_cross_entropy(
                        pos_pred_neg_probs,
                        torch.zeros_like(pos_pred_neg_probs),
                        reduction="sum",
                    )
                    if pos_pred_neg_probs.numel() > 0
                    else zero
                )
                pred_pos_loss = pred_pos_loss + (
                    pos_pos_delta
                )
                pred_neg_loss = pred_neg_loss + (
                    pos_neg_delta
                )

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

        return (
            pos_bce,
            neg_bce,
            dice_num,
            dice_den,
            pred_pos_loss,
            pred_neg_loss,
            pred_pos_count,
            pred_neg_count,
            tp,
            fp,
            fn,
        )

    def _accumulate_index_chunk(self, node_embed, p0, p1, positive_keys, max_nodes, state):
        pos_mask = self._positive_mask(p0, p1, positive_keys, max_nodes)
        pos_count = int(pos_mask.sum().item())
        neg_count = p0.numel() - pos_count

        def chunk_fn(node_embed_arg):
            return self._chunk_sums(node_embed_arg, p0, p1, pos_mask, pos_count)

        use_checkpoint = torch.is_grad_enabled() and node_embed.requires_grad
        if use_checkpoint:
            sums = checkpoint(chunk_fn, node_embed, use_reentrant=False)
        else:
            sums = chunk_fn(node_embed)

        (
            pos_loss,
            neg_loss,
            dice_num,
            dice_den,
            pred_pos_loss,
            pred_neg_loss,
            pred_pos_count,
            pred_neg_count,
            tp,
            fp,
            fn,
        ) = sums

        state["total_pos_loss"] += pos_loss
        state["total_neg_loss"] += neg_loss
        state["total_pos"] += pos_count
        state["total_neg"] += neg_count
        state["total_dice_num"] += dice_num
        state["total_dice_den"] += dice_den
        state["total_pred_pos_loss"] += pred_pos_loss
        state["total_pred_neg_loss"] += pred_neg_loss
        state["total_pred_pos"] += pred_pos_count
        state["total_pred_neg"] += pred_neg_count
        state["total_tp"] += tp
        state["total_fp"] += fp
        state["total_fn"] += fn

    def _finalize(self, state, device):
        zero = torch.tensor(0.0, device=device)
        total_pos = state["total_pos"]
        total_neg = state["total_neg"]
        pos_loss = (
            state["total_pos_loss"] / total_pos
            if total_pos > 0
            else zero
        )
        neg_loss = (
            state["total_neg_loss"] / total_neg
            if total_neg > 0
            else zero
        )
        main_loss = 0.5 * (pos_loss + neg_loss)

        total_tp = state["total_tp"]
        total_fp = state["total_fp"]
        total_fn = state["total_fn"]
        if not torch.is_tensor(total_tp):
            total_tp = zero
        if not torch.is_tensor(total_fp):
            total_fp = zero
        if not torch.is_tensor(total_fn):
            total_fn = zero
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
            if torch.is_tensor(state["total_dice_den"]) and state["total_dice_den"] > 0:
                dice_coeff = state["total_dice_num"] / state["total_dice_den"]
                dice_loss = 1 - dice_coeff
            else:
                dice_loss = zero
            total_loss += self.dice_weight * dice_loss
            log_dict.update({
                "dice_loss": dice_loss.detach(),
            })

        if self.use_pred_balanced_loss:
            total_pred_pos = state["total_pred_pos"]
            total_pred_neg = state["total_pred_neg"]
            pred_pos_loss = (
                state["total_pred_pos_loss"] / total_pred_pos
                if total_pred_pos > 0
                else zero
            )
            pred_neg_loss = (
                state["total_pred_neg_loss"] / total_pred_neg
                if total_pred_neg > 0
                else zero
            )

            pred_balanced_loss = 0.5 * (pred_pos_loss + pred_neg_loss)
            total_loss += self.pred_loss_weight * pred_balanced_loss
            log_dict.update({
                "pred_balanced_loss": pred_balanced_loss.detach(),
                "pred_pos_loss": pred_pos_loss.detach(),
                "pred_neg_loss": pred_neg_loss.detach(),
            })

        return total_loss, log_dict

    def forward(self, node_embed, positive_pair, positive_pair_offsets, offsets, vertex_mask):
        state = {
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

        for mesh_idx in range(offsets.numel() - 1):
            node_start = int(offsets[mesh_idx].item())
            node_end = int(offsets[mesh_idx + 1].item())
            local_vertex_mask = vertex_mask[node_start:node_end]
            vertices = torch.nonzero(local_vertex_mask, as_tuple=False).flatten() + node_start
            if vertices.numel() < 2:
                continue

            pos_start = int(positive_pair_offsets[mesh_idx].item())
            pos_end = int(positive_pair_offsets[mesh_idx + 1].item())
            positive = positive_pair[pos_start:pos_end]
            max_nodes = node_embed.shape[0]
            positive_keys = self._positive_keys(positive, max_nodes)
            positive_keys = torch.sort(positive_keys).values

            nv = vertices.numel()
            chunk_pair_limit = min(self.chunk_size, self.max_generated_pairs)
            chunk_start = 0
            while chunk_start < nv - 1:
                first_row_pairs = nv - chunk_start - 1
                max_rows = 1
                if first_row_pairs < chunk_pair_limit:
                    # Sum of row pair counts from chunk_start to chunk_end is
                    # bounded without creating all pairs first.
                    remaining = chunk_pair_limit
                    row = chunk_start
                    max_rows = 0
                    while row < nv - 1 and remaining >= nv - row - 1:
                        remaining -= nv - row - 1
                        row += 1
                        max_rows += 1
                chunk_end = min(chunk_start + max_rows, nv - 1)
                row_counts = nv - torch.arange(
                    chunk_start + 1,
                    chunk_end + 1,
                    device=node_embed.device,
                    dtype=torch.long,
                )
                num_pairs = row_counts.sum().item()
                if num_pairs == 0:
                    continue

                row_local = torch.repeat_interleave(
                    torch.arange(chunk_start, chunk_end, device=node_embed.device),
                    row_counts,
                )
                col_local = torch.arange(num_pairs, device=node_embed.device) - torch.repeat_interleave(
                    torch.cumsum(row_counts, dim=0) - row_counts,
                    row_counts,
                ) + row_local + 1
                p0 = vertices[row_local]
                p1 = vertices[col_local]
                self._accumulate_index_chunk(node_embed, p0, p1, positive_keys, max_nodes, state)
                chunk_start = chunk_end

        return self._finalize(state, node_embed.device)

class SpacetimeFaceLoss(nn.Module):
    """
    Balanced BCE + Pred-Balanced Loss for face classification.
    """

    def __init__(
        self,
        use_pred_balanced_loss=False,
        use_dice_loss=False,
        dice_weight=1.0,
        pred_loss_weight=1.0,
        chunk_size=200000,
        style='mink',
        scale='no',
    ):
        super().__init__()
        assert style in ['gram_diff', 'mink', 'tri_dot']
        assert scale in ['no', 'std', 'learn', 'learn_nobias']
        self.chunk_size = chunk_size
        self.use_pred_balanced_loss = use_pred_balanced_loss
        self.pred_loss_weight = pred_loss_weight
        self.use_dice_loss = use_dice_loss
        self.dice_weight = dice_weight
        self.style = style
        self.scale = scale
        if self.scale == 'learn':
            self.scale_d = nn.Linear(1, 1)
        elif self.scale == 'learn_nobias':
            self.scale_d = nn.Linear(1, 1, bias=False)

    def spacetime_area(self, node_embed, triplet):
        dim = node_embed.shape[-1]
        t = node_embed[..., : dim // 2]
        s = node_embed[..., dim // 2 :]
        # assert (triplet < node_embed.size(0)).all()
        # assert (triplet >= 0).all()
        t0, t1, t2 = t[triplet[:, 0]], t[triplet[:, 1]], t[triplet[:, 2]]
        s0, s1, s2 = s[triplet[:, 0]], s[triplet[:, 1]], s[triplet[:, 2]]
        if self.style == 'tri_dot':
            # Symmetric cubic on the RAW embeddings (NOT differences):
            #   D(x,y,z) = Σ_i x_i y_i z_i (time) − Σ_i x_i y_i z_i (space).
            # The elementwise triple product x_i y_i z_i is FULLY symmetric under vertex
            # permutation (no |·| needed) and indefinite. NOTE: not translation-invariant
            # (uses raw embeddings, not differences) — relies on the network to position the
            # face embedding. Parameter-free (only the single learnable temperature the edge
            # head uses). Overfit-verified to reach ~0 face loss (f_fs=1.0), far below the
            # area/Gram-form ~0.42 floor (the cap there is caused by translation-invariance).
            t0f, t1f, t2f = t0.float(), t1.float(), t2.float()
            s0f, s1f, s2f = s0.float(), s1.float(), s2.float()
            Dt = (t0f * t1f * t2f).sum(-1)
            Ds = (s0f * s1f * s2f).sum(-1)
            d = Dt - Ds  # time − space
            if self.scale == 'std':
                d = d / (2 * math.sqrt(dim))
            elif self.scale in ['learn', 'learn_nobias']:
                w = self.scale_d.weight.reshape(()).float()
                d = d.float() * w
                if self.scale == 'learn' and self.scale_d.bias is not None:
                    d = d + self.scale_d.bias.reshape(()).float()
            return d.float()
        else:
            if self.style == 'mink':
                u_t, u_s = t1 - t0, s1 - s0
                v_t, v_s = t2 - t0, s2 - s0

                uu = (u_t * u_t).sum(-1) - (u_s * u_s).sum(-1)
                vv = (v_t * v_t).sum(-1) - (v_s * v_s).sum(-1)
                uv = (u_t * v_t).sum(-1) - (u_s * v_s).sum(-1)

                area_sq = uu * vv - uv.pow(2)
                d = area_sq
            else:
                # vectors spanning triangles
                u_s, v_s = s1 - s0, s2 - s0
                u_t, v_t = t1 - t0, t2 - t0

                # squared spatial area
                uu_s = (u_s * u_s).sum(-1)
                vv_s = (v_s * v_s).sum(-1)
                uv_s = (u_s * v_s).sum(-1)
                S = uu_s * vv_s - uv_s**2

                # squared temporal area
                uu_t = (u_t * u_t).sum(-1)
                vv_t = (v_t * v_t).sum(-1)
                uv_t = (u_t * v_t).sum(-1)
                T = uu_t * vv_t - uv_t**2

                # spacetime triple score
                d = T - S
            if self.scale == 'std':
                d /= 2 * math.sqrt(2) * dim
            elif self.scale in ['learn']:
                d = self.scale_d(d.unsqueeze(-1).to(self.scale_d.weight.dtype)).squeeze(-1).float()
            elif self.scale in ['learn_nobias']:
                d = 1e-2 * self.scale_d(d.unsqueeze(-1).to(self.scale_d.weight.dtype)).squeeze(-1).float()
            return d

    def forward(self, node_embed, triplet, target):

        total_pos_loss, total_neg_loss = 0.0, 0.0
        total_pos, total_neg = 0, 0
        total_pred_pos_loss, total_pred_neg_loss = 0.0, 0.0
        total_pred_pos, total_pred_neg = 0, 0
        total_tp, total_fp, total_fn = 0.0, 0.0, 0.0

        # Dice loss accumulators
        total_dice_num = 0.0
        total_dice_den = 0.0

        for start in range(0, triplet.size(0), self.chunk_size):
            end = min(start + self.chunk_size, triplet.size(0))

            d = self.spacetime_area(node_embed, triplet[start:end])
            # print('face', d)

            t_chunk = target[start:end]

            pos_mask, neg_mask = (t_chunk == 1), (t_chunk == 0)
            if pos_mask.any():
                total_pos_loss += F.binary_cross_entropy_with_logits(d[pos_mask].float(), t_chunk[pos_mask].float(), reduction="mean") * pos_mask.sum()
                total_pos += pos_mask.sum()
            if neg_mask.any():
                total_neg_loss += F.binary_cross_entropy_with_logits(d[neg_mask].float(), t_chunk[neg_mask].float(), reduction="mean") * neg_mask.sum()
                total_neg += neg_mask.sum()

            if self.use_pred_balanced_loss:
                probs = torch.sigmoid(d.float())
                pred_pos, pred_neg = (probs >= 0.5), (probs < 0.5)
                if pred_pos.any():
                    total_pred_pos_loss += F.binary_cross_entropy(probs[pred_pos], t_chunk[pred_pos].float(), reduction="mean") * pred_pos.sum()
                    total_pred_pos += pred_pos.sum()
                if pred_neg.any():
                    total_pred_neg_loss += F.binary_cross_entropy(probs[pred_neg], t_chunk[pred_neg].float(), reduction="mean") * pred_neg.sum()
                    total_pred_neg += pred_neg.sum()
            
            if self.use_dice_loss:
                p_chunk = torch.sigmoid(d.float())
                t_float = t_chunk.float()

                dice_num = 2 * (p_chunk * t_float).sum()
                dice_den = p_chunk.sum() + t_float.sum()
                total_dice_num += dice_num
                total_dice_den += dice_den

            pred_label = (d > 0).long()
            total_tp += ((pred_label == 1) & (t_chunk == 1)).sum()
            total_fp += ((pred_label == 1) & (t_chunk == 0)).sum()
            total_fn += ((pred_label == 0) & (t_chunk == 1)).sum()
        
        # ======= Final Balanced BCE ========
        pos_loss = total_pos_loss / total_pos if total_pos > 0 else torch.tensor(0.0, device=node_embed.device)
        neg_loss = total_neg_loss / total_neg if total_neg > 0 else torch.tensor(0.0, device=node_embed.device)
        main_loss = 0.5 * (pos_loss + neg_loss)

        # ======= F-score =======
        precision = total_tp / max(total_tp + total_fp, 1e-8)
        recall = total_tp / max(total_tp + total_fn, 1e-8)
        f_score = 2 * precision * recall / max(precision + recall, 1e-8)

        log_dict = {
            "main": main_loss.detach(),
            "pos": pos_loss.detach(),
            "neg": neg_loss.detach(),
            "prec": precision,
            "rec": recall,
            "fs": f_score,
            "1-prec": 1 - precision,
            "1-rec": 1 - recall,
            "1-fs": 1 - f_score,
        }

        total_loss = main_loss.clone()

        # ======= Dice loss ========
        if self.use_dice_loss:
            dice_coeff = total_dice_num / total_dice_den
            dice_loss = 1 - dice_coeff
            total_loss += self.dice_weight * dice_loss
            log_dict.update({
                "dice": dice_loss.detach(),
            })

        # ======= Pred-Balanced Loss ========
        if self.use_pred_balanced_loss:
            if total_pred_pos > 0:
                pred_pos_loss = total_pred_pos_loss / total_pred_pos
            else:
                pred_pos_loss = torch.tensor(0.0, device=d.device)

            if total_pred_neg > 0:
                pred_neg_loss = total_pred_neg_loss / total_pred_neg
            else:
                pred_neg_loss = torch.tensor(0.0, device=d.device)

            pred_balanced_loss = 0.5 * (pred_pos_loss + pred_neg_loss)
            total_loss += self.pred_loss_weight * pred_balanced_loss
            log_dict.update({
                # "pred_balanced_loss": pred_balanced_loss.detach(),
                "pred_t": pred_pos_loss.detach(),
                "pred_f": pred_neg_loss.detach(),
            })

        return total_loss, log_dict


class SpacetimeMultiheadEdgeLoss(SpacetimeEdgeLoss):
    def __init__(
        self,
        use_focal_loss=False,
        use_pred_balanced_loss=False,
        use_dice_loss=False,
        dice_weight=1.0,
        pred_loss_weight=1.0,
        gamma=2.0,
        alpha=0.25,
        chunk_size=200000,
        extra_layer=False,
        scale='no',
        heads=1,
        dim=32,
    ):
        super(SpacetimeEdgeLoss, self).__init__()
        assert scale in ['learn', 'learn_nobias', 'norm_learn_nobias']
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
        self.heads = heads
        if self.scale == 'learn':
            self.scale_d = nn.Linear(self.heads, 1)
        elif self.scale == 'learn_nobias':
            self.scale_d = nn.Linear(self.heads, 1, bias=False)
        elif self.scale == 'norm_learn_nobias':
            self.norm = FP32LayerNorm(dim // (self.heads * 2), elementwise_affine=False)
            self.scale_d = nn.Linear(self.heads, 1, bias=False)
    
    def spacetime_distance(self, node_embed, pair):
        dim = node_embed.shape[-1]
        assert dim % (self.heads * 2) == 0
        node_embed = node_embed.view(-1, self.heads, dim // (self.heads * 2), 2)

        p0 = pair[:, 0]
        p1 = pair[:, 1]

        node_t = node_embed[..., 0]
        node_s = node_embed[..., 1]
        if self.scale == 'norm_learn_nobias':
            node_t = self.norm(node_t)
            node_s = self.norm(node_s)

        dt = ((node_t[p0] - node_t[p1]) ** 2).sum(dim=-1)
        ds = ((node_s[p0] - node_s[p1]) ** 2).sum(dim=-1)
        d = dt - ds
        if self.scale in ['learn', 'norm_learn_nobias']:
            d = self.scale_d(d.to(self.scale_d.weight.dtype)).squeeze(-1).float()
        elif self.scale in ['learn_nobias']:
            d = self.scale_d(d.to(self.scale_d.weight.dtype)).squeeze(-1).float()
        return d
    
class SpacetimeMultiheadFaceLoss(SpacetimeFaceLoss):
    def __init__(
        self,
        use_pred_balanced_loss=False,
        use_dice_loss=False,
        dice_weight=1.0,
        pred_loss_weight=1.0,
        chunk_size=200000,
        style='mink',
        scale='no',
        heads=1,
        dim=32,
    ):
        super(SpacetimeFaceLoss, self).__init__()
        assert style in ['gram_diff', 'mink']
        assert scale in ['learn', 'learn_nobias', 'norm_learn_nobias']
        self.chunk_size = chunk_size
        self.use_pred_balanced_loss = use_pred_balanced_loss
        self.pred_loss_weight = pred_loss_weight
        self.use_dice_loss = use_dice_loss
        self.dice_weight = dice_weight
        self.style = style
        self.scale = scale
        self.heads = heads
        if self.scale == 'learn':
            self.scale_d = nn.Linear(self.heads, 1)
        elif self.scale == 'learn_nobias':
            self.scale_d = nn.Linear(self.heads, 1, bias=False)
        elif self.scale == 'norm_learn_nobias':
            self.norm = FP32LayerNorm(dim // (self.heads * 2), elementwise_affine=False)
            self.scale_d = nn.Linear(self.heads, 1, bias=False)
    
    def spacetime_area(self, node_embed, triplet):
        dim = node_embed.shape[-1]
        assert dim % (self.heads * 2) == 0
        node_embed = node_embed.view(-1, self.heads, dim // (self.heads * 2), 2)
        t = node_embed[..., 0]
        s = node_embed[..., 1]
        if self.scale == 'norm_learn_nobias':
            t = self.norm(t)
            s = self.norm(s)
        t0, t1, t2 = t[triplet[:, 0]], t[triplet[:, 1]], t[triplet[:, 2]]
        s0, s1, s2 = s[triplet[:, 0]], s[triplet[:, 1]], s[triplet[:, 2]]
        if self.style == 'mink':
            u_t, u_s = t1 - t0, s1 - s0
            v_t, v_s = t2 - t0, s2 - s0

            uu = (u_t * u_t).sum(-1) - (u_s * u_s).sum(-1)
            vv = (v_t * v_t).sum(-1) - (v_s * v_s).sum(-1)
            uv = (u_t * v_t).sum(-1) - (u_s * v_s).sum(-1)

            area_sq = uu * vv - uv.pow(2)
            d = area_sq
        else:
            # vectors spanning triangles
            u_s, v_s = s1 - s0, s2 - s0
            u_t, v_t = t1 - t0, t2 - t0

            # squared spatial area
            uu_s = (u_s * u_s).sum(-1)
            vv_s = (v_s * v_s).sum(-1)
            uv_s = (u_s * v_s).sum(-1)
            S = uu_s * vv_s - uv_s**2

            # squared temporal area
            uu_t = (u_t * u_t).sum(-1)
            vv_t = (v_t * v_t).sum(-1)
            uv_t = (u_t * v_t).sum(-1)
            T = uu_t * vv_t - uv_t**2

            # spacetime triple score
            d = T - S
        if self.scale in ['learn', 'norm_learn_nobias']:
            d = self.scale_d(d.to(self.scale_d.weight.dtype)).squeeze(-1).float()
        elif self.scale in ['learn_nobias']:
            d = 1e-2 * self.scale_d(d.to(self.scale_d.weight.dtype)).squeeze(-1).float()
        return d 


class DeterminantOrientLoss(nn.Module):
    def __init__(self, orient_embed_dim):
        super().__init__()
        assert orient_embed_dim % 3 == 0
        self.orient_embed_dim = orient_embed_dim
        self.n_mat = orient_embed_dim // 3
        self.scale = nn.Linear(self.n_mat, 1, bias=False)
    
    def orient(self, orient_embed, face_index):
        face_embed = orient_embed[face_index] # F x 3 x orient_embed_dim
        b = face_embed.shape[0]
        face_determinant = torch.det(face_embed.view(b, 3, self.n_mat, 3).permute(0, 2, 1, 3).float()) # F x n_mat
        return self.scale(face_determinant.to(dtype=self.scale.weight.dtype)).squeeze(-1)
    
    def forward(self, orient_embed, face_index, label):
        '''
        orient_embed: N x orient_embed_dim
        face_index: F x 3
        label: F
        '''
        orient_logits = self.orient(orient_embed, face_index)
        loss = F.binary_cross_entropy_with_logits(orient_logits.float(), label.float(), reduction='mean')
        acc = ((orient_logits > 0).long() == label).float().mean()
        return loss, {
            'loss': loss.detach(),
            'acc': acc,
            '1-acc': 1 - acc
        }
