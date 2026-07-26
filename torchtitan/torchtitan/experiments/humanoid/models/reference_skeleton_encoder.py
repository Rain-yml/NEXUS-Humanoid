"""Graph and 3D-RoPE encoder for arbitrary reference skeletons."""

from __future__ import annotations

import torch
from torch import nn

from torchtitan.experiments.vem.models.mesh_gnn import MeshTransformerBlock
from torchtitan.experiments.vem.models.transformer import (
    RMSNorm,
    RotaryPosEmbed3D,
    VEM2DecoderLayer,
)


class ReferenceSkeletonEncoder(nn.Module):
    """Encode one feature per joint from geometry and parent-child edges."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        edge_hidden_dim: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int,
        intermediate_size: int,
        grid_size: int = 512,
        rope_theta: float = 2026.0,
        qk_norm: bool = True,
        attention_bias: bool = True,
        use_flash_attn_3: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_attention_heads:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")

        self.node_projection = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_projection = nn.Linear(10, edge_hidden_dim)
        self.graph_layers = nn.ModuleList(
            [
                MeshTransformerBlock(
                    node_feat_dim=hidden_dim,
                    edge_feat_dim=edge_hidden_dim,
                    hidden_dim=hidden_dim,
                    ffn_dim=intermediate_size,
                    heads=num_attention_heads,
                )
                for _ in range(num_blocks)
            ]
        )
        self.rope = RotaryPosEmbed3D(
            attention_head_dim=hidden_dim // num_attention_heads,
            max_seq_len=grid_size,
            theta=rope_theta,
        )
        self.global_layers = nn.ModuleList(
            [
                VEM2DecoderLayer(
                    layer_idx=layer_index,
                    hidden_size=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    rope=self.rope,
                    qk_norm=qk_norm,
                    attention_bias=attention_bias,
                    use_flash_attn_3=use_flash_attn_3,
                    contains_cross_attention=False,
                    attn_dtype="bf16",
                    is_causal=False,
                )
                for layer_index in range(num_blocks)
            ]
        )
        self.output_norm = RMSNorm(hidden_dim)

    @staticmethod
    def _edge_features(
        positions: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        source, target = edge_index
        source_positions = positions[source]
        target_positions = positions[target]
        vector = target_positions - source_positions
        length = vector.norm(dim=-1, keepdim=True)
        direction = vector / length.clamp_min(1e-8)
        midpoint = 0.5 * (source_positions + target_positions)
        return torch.cat([vector, length, direction, midpoint], dim=-1)

    def init_rope(self) -> None:
        self.rope.init_weights()

    def forward(
        self,
        positions: torch.Tensor,
        rope_positions: torch.Tensor,
        edge_index: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Reference positions must have shape (J, 3)")
        if rope_positions.shape != positions.shape:
            raise ValueError("Reference RoPE positions must match reference positions")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Reference edge_index must have shape (2, E)")
        if int(cu_seqlens[-1]) != len(positions):
            raise ValueError("Reference cu_seqlens do not cover all joints")

        dtype = self.node_projection[0].weight.dtype
        positions = positions.to(dtype=dtype)
        hidden_states = self.node_projection(positions)
        edge_features = None
        if edge_index.shape[1] > 0:
            edge_features = self.edge_projection(
                self._edge_features(positions, edge_index).to(dtype=dtype)
            )
        for graph_layer, global_layer in zip(
            self.graph_layers, self.global_layers, strict=True
        ):
            if edge_features is not None:
                hidden_states, edge_features = graph_layer(
                    hidden_states, edge_index, edge_features
                )
            hidden_states = global_layer(
                hidden_states,
                position_ids=rope_positions,
                cu_seqlens=cu_seqlens,
            )
        return self.output_norm(hidden_states)


__all__ = ["ReferenceSkeletonEncoder"]
