import math
from typing import Any, Dict, Optional, Tuple, Union, Literal
import numpy as np
import math
import torch.cuda.amp as amp

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import logging

from torchtitan.experiments.vem.models.transformer import (
    Attention, 
    RMSNorm, 
    MLP, 
    FP32LayerNorm,
    RotaryPosEmbed3D,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class VEMDiTBlock(nn.Module):
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
        cross_attn_norm: bool = False,
        contain_cross_attention: bool = True,
        gated: bool = False,
        rope_3d: Optional[RotaryPosEmbed3D] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps
        self.attention_bias = attention_bias
        self.use_flash_attn_3 = use_flash_attn_3
        self.cross_attn_norm = cross_attn_norm
        self.conain_cross_attention = contain_cross_attention

        self.norm1 = FP32LayerNorm(hidden_size, elementwise_affine=False)

        self.self_attn = Attention(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            is_cross_attention=False,
            is_causal=False,
            rope=rope_3d,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            bias=attention_bias,
            use_flash_attn_3=use_flash_attn_3,
            gated=gated,
        )

        if self.conain_cross_attention:
            self.norm3 = FP32LayerNorm(hidden_size, elementwise_affine=True) if cross_attn_norm else nn.Identity()

            self.cross_attn = Attention(
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                is_cross_attention=True,
                is_causal=False,
                rope=None,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
                bias=attention_bias,
                use_flash_attn_3=use_flash_attn_3,
                gated=gated,
            )

        self.norm2 = FP32LayerNorm(hidden_size, elementwise_affine=False)

        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_size) / hidden_size ** 0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_modulation: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_encoder: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        max_seqlen_encoder: Optional[int] = None,
    ) -> torch.Tensor:
        '''
        hidden_states: B x L x C or L x C
        encoder_hidden_states: B x L2 x C
        temb: B x 6 x C or L x 6 x C
        positions: B x L x 3 or L x 3
        '''
        if cu_seqlens is None:
            with amp.autocast(dtype=torch.float32):
                shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = (self.modulation + temb_modulation.float()).chunk(6, dim=1)
        else:
            with amp.autocast(dtype=torch.float32):
                shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = (self.modulation + temb_modulation.float()).unbind(dim=1)
        
        dtype = hidden_states.dtype
        
        # self attention
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=(self.norm1(hidden_states).float() * (1 + scale_msa) + shift_msa).to(dtype=dtype),
            cu_seqlens_q=cu_seqlens,
            position_ids=positions,
            max_seqlen_q=max_seqlen,
        )
        with amp.autocast(dtype=torch.float32):
            hidden_states = hidden_states * gate_msa + residual

        hidden_states = hidden_states.to(dtype=dtype)

        if self.conain_cross_attention:
            # cross attention
            residual = hidden_states
            hidden_states = self.cross_attn(
                hidden_states=(self.norm3(hidden_states).float()).to(dtype=dtype),
                encoder_hidden_states=encoder_hidden_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens_encoder,
                max_seqlen_q=max_seqlen,
                max_seqlen_kv=max_seqlen_encoder,
            )

            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(
            (self.norm2(hidden_states).float() * (1 + scale_ffn) + shift_ffn).to(dtype=dtype)
        )

        with amp.autocast(dtype=torch.float32):
            hidden_states = residual + hidden_states * gate_ffn
        
        hidden_states = hidden_states.to(dtype=dtype)
        
        return hidden_states

class Head(nn.Module):
    def __init__(self, dim, out_dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.eps = eps

        # layers
        self.norm = FP32LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, hidden_states, temb, cu_seqlens=None):
        r"""
        Args:
            hidden_states: B x L x C
            temb: B x C
        """
        assert temb.dtype == torch.float32
        if cu_seqlens is not None:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                shift, scale = (self.modulation + temb.unsqueeze(1)).unbind(dim=1)
                hidden_states = self.head(self.norm(hidden_states) * (1 + scale) + shift)
        else:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                shift, scale = (self.modulation + temb.unsqueeze(1)).chunk(2, dim=1)
                hidden_states = self.head(self.norm(hidden_states) * (1 + scale) + shift)
        return hidden_states

class FaceMLPEncoder(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.linear = nn.Linear(1, dim_out, bias=True)
    
    def forward(self, face_num):
        min_f, max_f = 100, 20000
        shift = 2000
        mean = (math.log(min_f + shift) + math.log(max_f + shift)) * 0.5
        std = (math.log(max_f + shift) - math.log(min_f + shift)) * 0.5
        x = (torch.log(face_num + shift) - mean) / std
        bs = face_num.shape[0]
        dtype = self.linear.weight.dtype
        return self.linear(x.view(bs, 1, 1).to(dtype=dtype))

class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            FP32LayerNorm(in_dim), 
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), 
            torch.nn.Linear(in_dim, out_dim),
            FP32LayerNorm(out_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens