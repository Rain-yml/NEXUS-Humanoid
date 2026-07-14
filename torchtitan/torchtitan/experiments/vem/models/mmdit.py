import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from diffusers.utils import logging

from torchtitan.experiments.vem.models.transformer import (
    RMSNorm,
    MLP,
    FP32LayerNorm,
    RotaryPosEmbed3D,
    type_str_to_dtype,
)

logger = logging.get_logger(__name__)


class MMDiTBlock(nn.Module):
    """
    Dual-stream DiT block with joint attention.
    Both the input stream and the condition stream have their own
    norm / modulation / Q,K,V projections / MLP, but they share a single
    joint attention computation (Q/K/V concatenated along the sequence dim).
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_key_value_heads: Optional[int] = None,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        use_flash_attn_3: bool = False,
        rope_3d: Optional[RotaryPosEmbed3D] = None,
        attn_dtype: str = "bf16",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        )
        self.head_dim = hidden_size // num_attention_heads
        self.qk_norm = qk_norm
        self.rope_3d = rope_3d
        self.use_flash_attn_3 = use_flash_attn_3
        self.attn_dtype = type_str_to_dtype(attn_dtype)

        # ---- Input stream components ----
        self.norm1 = FP32LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = FP32LayerNorm(hidden_size, elementwise_affine=False)

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=attention_bias)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)

        self.mlp = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_size) / hidden_size ** 0.5)

        # ---- Condition stream components ----
        self.c_norm1 = FP32LayerNorm(hidden_size, elementwise_affine=False)
        self.c_norm2 = FP32LayerNorm(hidden_size, elementwise_affine=False)

        self.c_q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=attention_bias)
        self.c_k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.c_v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.c_o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=attention_bias)

        if qk_norm:
            self.c_q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
            self.c_k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)

        self.c_mlp = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.c_modulation = nn.Parameter(torch.randn(1, 6, hidden_size) / hidden_size ** 0.5)

        # ---- Flash attention ----
        if use_flash_attn_3:
            from flash_attn_interface import flash_attn_varlen_func, flash_attn_func
        else:
            from flash_attn import flash_attn_varlen_func, flash_attn_func
        self.flash_attn_func = flash_attn_func
        self.flash_attn_varlen_func = flash_attn_varlen_func

    def forward(
        self,
        hidden_states_cat: torch.Tensor,
        temb_modulation_cat: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        mask_x: Optional[torch.Tensor] = None,
        mask_c: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:  (L, C) for varlen — input stream
            temb_modulation: (L, 6, C) for varlen — input stream modulation
            cu_seqlens: cumulative sequence lengths for input stream (varlen mode)
            mask_x: (L) input stream token mask
            mask_c: (L) condition stream token mask
            positions:  (L1, 3) — 3D positions for RoPE (input stream only)

        Returns:
            (hidden_states, c_hidden_states) — both updated
        """
        idx_x = mask_x.nonzero(as_tuple=False).squeeze(1)
        idx_c = mask_c.nonzero(as_tuple=False).squeeze(1)

        temb_modulation = temb_modulation_cat.index_select(0, idx_x)
        c_temb_modulation = temb_modulation_cat.index_select(0, idx_c)

        hidden_states = hidden_states_cat.index_select(0, idx_x)
        c_hidden_states = hidden_states_cat.index_select(0, idx_c)

        dtype = hidden_states.dtype
        varlen = True

        # ================================================================
        # 1. AdaLN modulation — 6 values per stream (shift/scale/gate for attn & FFN)
        # ================================================================
        with amp.autocast(dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = (
                self.modulation + temb_modulation.float()
            ).unbind(dim=1)
            c_shift_msa, c_scale_msa, c_gate_msa, c_shift_ffn, c_scale_ffn, c_gate_ffn = (
                self.c_modulation + c_temb_modulation.float()
            ).unbind(dim=1)

        # ================================================================
        # 2. Project Q/K/V for both streams with AdaLN-modulated input
        # ================================================================
        # Input stream
        x_normed = (self.norm1(hidden_states).float() * (1 + scale_msa) + shift_msa).to(dtype)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        # Condition stream
        c_normed = (self.c_norm1(c_hidden_states).float() * (1 + c_scale_msa) + c_shift_msa).to(dtype)
        c_q = self.c_q_proj(c_normed)
        c_k = self.c_k_proj(c_normed)
        c_v = self.c_v_proj(c_normed)

        # ================================================================
        # 3. Reshape, QK norm, RoPE
        # ================================================================
        # Varlen mode
        q_len = hidden_states.size(0)
        c_len = c_hidden_states.size(0)

        q = q.view(q_len, self.num_attention_heads, self.head_dim)
        k = k.view(q_len, self.num_key_value_heads, self.head_dim)
        v = v.view(q_len, self.num_key_value_heads, self.head_dim)

        c_q = c_q.view(c_len, self.num_attention_heads, self.head_dim)
        c_k = c_k.view(c_len, self.num_key_value_heads, self.head_dim)
        c_v = c_v.view(c_len, self.num_key_value_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            c_q = self.c_q_norm(c_q)
            c_k = self.c_k_norm(c_k)

        if self.rope_3d is not None:
            q, k = self.rope_3d(q, k, positions)
        
        joint_q = torch.zeros(
            q_len + c_len, self.num_attention_heads, self.head_dim, dtype=q.dtype, device=q.device
        )
        joint_k = torch.zeros(
            q_len + c_len, self.num_key_value_heads, self.head_dim, dtype=q.dtype, device=q.device
        )
        joint_v = torch.zeros(
            q_len + c_len, self.num_key_value_heads, self.head_dim, dtype=q.dtype, device=q.device
        )

        idx_x = mask_x.nonzero(as_tuple=False).squeeze(1)
        idx_c = mask_c.nonzero(as_tuple=False).squeeze(1)
        inv_perm = torch.argsort(torch.cat([idx_x, idx_c], dim=0))

        joint_q = torch.cat([q, c_q], dim=0).index_select(0, inv_perm)
        joint_k = torch.cat([k, c_k], dim=0).index_select(0, inv_perm)
        joint_v = torch.cat([v, c_v], dim=0).index_select(0, inv_perm)

        max_seqlen = torch.diff(cu_seqlens).max().item()

        # Build joint varlen tensors
        if self.use_flash_attn_3:
            joint_out = self.flash_attn_varlen_func(
                joint_q.to(self.attn_dtype),
                joint_k.to(self.attn_dtype),
                joint_v.to(self.attn_dtype),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
            )[0]
        else:
            joint_out = self.flash_attn_varlen_func(
                joint_q.to(self.attn_dtype),
                joint_k.to(self.attn_dtype),
                joint_v.to(self.attn_dtype),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
            )

        # Split joint output back into input and condition streams
        attn_out = joint_out.index_select(0, idx_x).reshape(q_len, self.hidden_size).type_as(hidden_states)
        c_attn_out = joint_out.index_select(0, idx_c).reshape(c_len, self.hidden_size).type_as(c_hidden_states)

        # ================================================================
        # 7. Gate + O proj
        # ================================================================

        attn_out = self.o_proj(attn_out)
        c_attn_out = self.c_o_proj(c_attn_out)

        # ================================================================
        # 8. Residual (FP32)
        # ================================================================
        with amp.autocast(dtype=torch.float32):
            hidden_states = attn_out * gate_msa + hidden_states
            c_hidden_states = c_attn_out * c_gate_msa + c_hidden_states

        hidden_states = hidden_states.to(dtype)
        c_hidden_states = c_hidden_states.to(dtype)

        # ================================================================
        # 9. MLP with AdaLN
        # ================================================================
        residual = hidden_states
        c_residual = c_hidden_states

        mlp_in = (self.norm2(hidden_states).float() * (1 + scale_ffn) + shift_ffn).to(dtype)
        mlp_out = self.mlp(mlp_in)

        c_mlp_in = (self.c_norm2(c_hidden_states).float() * (1 + c_scale_ffn) + c_shift_ffn).to(dtype)
        c_mlp_out = self.c_mlp(c_mlp_in)

        with amp.autocast(dtype=torch.float32):
            hidden_states = residual + mlp_out * gate_ffn
            c_hidden_states = c_residual + c_mlp_out * c_gate_ffn

        hidden_states = hidden_states.to(dtype)
        c_hidden_states = c_hidden_states.to(dtype)

        hidden_states_cat = torch.zeros_like(hidden_states_cat)
        hidden_states_cat = hidden_states_cat.index_copy(0, idx_x, hidden_states)
        hidden_states_cat = hidden_states_cat.index_copy(0, idx_c, c_hidden_states)

        return hidden_states_cat
