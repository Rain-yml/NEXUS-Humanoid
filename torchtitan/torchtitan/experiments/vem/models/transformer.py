import math
from typing import Any, Dict, Optional, Tuple, Union, Literal, List
import numpy as np
from functools import lru_cache
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
import torch.distributed as dist

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def type_str_to_dtype(type_str):
    if type_str == 'bf16':
        return torch.bfloat16
    elif type_str == 'fp16':
        return torch.float16
    raise NotImplementedError

def _init_norm(module):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.ones_(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)

def _basic_init(module):
    if isinstance(module, (nn.LayerNorm, FP32LayerNorm, nn.RMSNorm, RMSNorm)):
        _init_norm(module)

    if isinstance(module, nn.Linear):
        # nn.init.xavier_uniform_(module.weight)
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    """
    Simple RoPE as in LLaMA-style models.
    """
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len_cached = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Cache cos/sin
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def init_weights(self):
        # need for torchtitan training
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=self.inv_freq.device) / self.dim))
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, dim]
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor):
        # todo: B H T D -> B T H D or T H D
        # position_ids: B T or T
        """
        q, k: [B, T, H, D] or [T, H, D]
        position_ids: [B, T] or [T]
        """
        if q.dim() == 4:
            b, t, h, d = q.shape
            assert pos.dim() == 2
            cos = self.cos_cached.index_select(0, pos.view(-1)).view(b, t, d)
            sin = self.sin_cached.index_select(0, pos.view(-1)).view(b, t, d)
            cos = cos.unsqueeze(2)  # [B,T,1,D]
            sin = sin.unsqueeze(2)
            return self.apply_rotary_pos_emb(q, k, cos, sin)
        elif q.dim() == 3:
            t, h, d = q.shape
            assert pos.dim() == 1
            cos = self.cos_cached.index_select(0, pos).view(t, d)
            sin = self.sin_cached.index_select(0, pos).view(t, d)
            cos = cos.unsqueeze(1)  # [T,1,D]
            sin = sin.unsqueeze(1)
            return self.apply_rotary_pos_emb(q, k, cos, sin)
        else:
            raise NotImplementedError

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        # x_rot = x * cos + rotate(x) * sin
        q_type = q.dtype
        rope_dtype = cos.dtype
        assert rope_dtype == torch.float32
        q_embed = (q.to(rope_dtype) * cos) + (self.rotate_half(q.to(rope_dtype)) * sin)
        k_embed = (k.to(rope_dtype) * cos) + (self.rotate_half(k.to(rope_dtype)) * sin)
        q_embed = q_embed.to(q_type)
        k_embed = k_embed.to(q_type)
        return q_embed, k_embed


class Attention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        is_cross_attention: bool = False,
        is_causal: bool = False,
        rope: Optional[RotaryEmbedding] = None,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        bias: bool = False,
        use_flash_attn_3: bool = False,
        attn_dtype: str = "bf16",
        gated: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.qk_norm = qk_norm
        self.is_cross_attention = is_cross_attention

        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=bias)
        
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        
        self.rope = rope
        self.is_causal = is_causal

        if use_flash_attn_3:
            from flash_attn_interface import flash_attn_varlen_func, flash_attn_func
            self.flash_attn_varlen_func = flash_attn_varlen_func
            self.flash_attn_func = flash_attn_func
        else:
            from flash_attn import flash_attn_varlen_func, flash_attn_func
            self.flash_attn_varlen_func = flash_attn_varlen_func
            self.flash_attn_func = flash_attn_func
        
        self.use_flash_attn_3 = use_flash_attn_3
        self.attn_dtype = type_str_to_dtype(attn_dtype)
        self.gated = gated
        if self.gated:
            self.gate_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Cache] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert (encoder_hidden_states is not None) == self.is_cross_attention, "encoder_hidden_states must be provided for cross attention."
        assert (not (self.is_cross_attention and self.is_causal)), "Cross attention does not support causal masking."
        query_states = self.q_proj(hidden_states)

        if encoder_hidden_states is not None:
            assert not use_cache
            # Cross attention: use encoder_hidden_states for key and value
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        
        if cu_seqlens_q is None:
            # standard batched attention
            bsz, q_len, _ = hidden_states.size()
            kv_len = key_states.size(1)

            query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim)
            key_states = key_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim)
            value_states = value_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim)

            if self.qk_norm:
                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)

            if self.rope is not None:
                # Apply rotary embeddings
                query_states, key_states = self.rope(query_states, key_states, position_ids)
            
            if use_cache:
                key_states, value_states = past_key_values.update(key_states.transpose(1, 2), value_states.transpose(1, 2), self.layer_idx)
                key_states = key_states.transpose(1, 2) 
                value_states = value_states.transpose(1, 2) 

            if self.use_flash_attn_3:
                attn_output = self.flash_attn_func(
                    query_states.to(self.attn_dtype),
                    key_states.to(self.attn_dtype),
                    value_states.to(self.attn_dtype),
                    causal=self.is_causal,
                )[0]
            else:
                attn_output = self.flash_attn_func(
                    query_states.to(self.attn_dtype),
                    key_states.to(self.attn_dtype),
                    value_states.to(self.attn_dtype),
                    causal=self.is_causal,
                )

            attn_output = attn_output.view(bsz, q_len, self.hidden_size).type_as(hidden_states)
        else:
            if use_cache:
                raise NotImplementedError("use_cache is not supported for variable length attention.")
            assert (cu_seqlens_kv is not None) == self.is_cross_attention, "cu_seqlens_kv must be provided for cross attention with variable length."
            if not self.is_cross_attention:
                cu_seqlens_kv = cu_seqlens_q
            
            q_len, _ = query_states.size()
            kv_len, _ = key_states.size()
            
            # var len attention
            query_states = query_states.view(q_len, self.num_attention_heads, self.head_dim)
            key_states = key_states.view(kv_len, self.num_key_value_heads, self.head_dim)
            value_states = value_states.view(kv_len, self.num_key_value_heads, self.head_dim)

            if self.qk_norm:
                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)
            if self.rope is not None:
                # Apply rotary embeddings
                query_states, key_states = self.rope(query_states, key_states, position_ids)
            
            # max_seqlen for flash varlen. These must be the EXACT per-batch maxima
            # (an oversized constant is bit-identical but ~20% slower in flash-attn).
            # They are identical across every layer that shares these cu_seqlens, so
            # the caller computes them ONCE per step and passes them in, avoiding a
            # GPU->CPU sync (.item()) in each of the ~96 attention calls per step.
            # Fall back to the local sync if not provided (e.g. inference paths).
            if max_seqlen_q is None:
                max_seqlen_q = torch.diff(cu_seqlens_q).max().item()
            if max_seqlen_kv is None:
                max_seqlen_kv = (
                    max_seqlen_q if not self.is_cross_attention
                    else torch.diff(cu_seqlens_kv).max().item()
                )

            if self.use_flash_attn_3:
                attn_output = self.flash_attn_varlen_func(
                    query_states.to(self.attn_dtype),
                    key_states.to(self.attn_dtype),
                    value_states.to(self.attn_dtype),
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_kv,
                    causal=self.is_causal,
                )[0]
            else:
                attn_output = self.flash_attn_varlen_func(
                    query_states.to(self.attn_dtype),
                    key_states.to(self.attn_dtype),
                    value_states.to(self.attn_dtype),
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_kv,
                    causal=self.is_causal,
                )

            attn_output = attn_output.view(q_len, self.hidden_size).type_as(hidden_states)

        if self.gated:
            gate = self.gate_linear(hidden_states)
            attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)

        return attn_output


class VEM2DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_idx,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_key_value_heads: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        use_flash_attn_3: bool = False,
        contains_cross_attention: bool = False,
        attn_dtype: str = "bf16",
        gated: bool = False,
        is_causal: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.use_cross_attention = contains_cross_attention

        # Self attention
        self.self_attn = Attention(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rope=rope,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            bias=attention_bias,
            is_cross_attention=False,
            use_flash_attn_3=use_flash_attn_3,
            is_causal=is_causal,
            attn_dtype=attn_dtype,
            gated=gated,
        )
        self.self_attn_norm = RMSNorm(hidden_size, eps=1e-6)

        # Cross attention (optional)
        if self.use_cross_attention:
            self.cross_attn = Attention(
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rope=None, # no RoPE for cross attention
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
                bias=attention_bias,
                is_cross_attention=True,
                use_flash_attn_3=use_flash_attn_3,
                is_causal=False,
                attn_dtype=attn_dtype,
                gated=gated,
            )
            self.cross_attn_norm = RMSNorm(hidden_size, eps=1e-6)

        # MLP
        self.mlp = MLP(hidden_size, intermediate_size)
        self.mlp_norm = RMSNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Cache] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_encoder: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        residual = hidden_states

        # Self attention
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            encoder_hidden_states=None,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=None,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        # Cross attention (optional)
        if self.use_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_norm(hidden_states)
            hidden_states = self.cross_attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                encoder_hidden_states=encoder_hidden_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens_encoder,
            )
            hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs

class VEM2Decoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 81920,
        num_key_value_heads: Optional[int] = None,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        use_flash_attn_3: bool = False,
        cross_attention_interval: int = 1,
        abs_positional_embedding: bool = False,
        attn_dtype: str = "bf16",
    ):
        super().__init__()
        self.abs_positional_embedding = abs_positional_embedding
        if self.abs_positional_embedding:
            self.rope = None
        else:
            self.rope = RotaryEmbedding(
                dim=hidden_size // num_attention_heads,
                base=rope_theta,
                max_position_embeddings=max_position_embeddings,
            )
        
        self.layers = nn.ModuleList([
            VEM2DecoderLayer(
                layer_idx=i,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_key_value_heads=num_key_value_heads,
                rope=self.rope,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
                attention_bias=attention_bias,
                use_flash_attn_3=use_flash_attn_3,
                contains_cross_attention=(cross_attention_interval > 0 and (i % cross_attention_interval == 0)),
                attn_dtype=attn_dtype,
            )
            for i in range(num_hidden_layers)
        ])
        
        self.norm = RMSNorm(hidden_size, eps=1e-6)
    
    def init_weights(self):
        self.apply(_basic_init)
        if self.rope is not None:
            self.rope.init_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_encoder: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                encoder_hidden_states=encoder_hidden_states,
                cu_seqlens=cu_seqlens,
                cu_seqlens_encoder=cu_seqlens_encoder,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states

class VEM2PointEncoderLayer(nn.Module):
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
        # is_cross_attention: bool = False,
        contains_self_attn: bool = True,
        contains_cross_attn: bool = False,
        attn_dtype: str = "bf16",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        # self.is_cross_attention = is_cross_attention
        self.contains_self_attn = contains_self_attn
        self.contains_cross_attn = contains_cross_attn
        assert self.contains_self_attn or self.contains_cross_attn, "At least one of self attention or cross attention must be True."
        self.num_attn_layers = int(self.contains_self_attn) + int(self.contains_cross_attn)
        assert self.num_attn_layers > 0

        # Attention (cross or self)
        if self.num_attn_layers == 2:
            self.attn_type = "self"
            self.attn2_type = "cross"
        elif self.num_attn_layers == 1:
            self.attn_type = "cross" if self.contains_cross_attn else "self"
            self.attn2_type = None
        else:
            raise NotImplementedError
        
        self.attention = Attention(
            layer_idx=layer_idx, # not used
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            bias=attention_bias,
            use_flash_attn_3=use_flash_attn_3,
            is_cross_attention=(self.attn_type == "cross"),
            is_causal=False,
            attn_dtype=attn_dtype,
        )
        self.attention_norm = RMSNorm(hidden_size, eps=1e-6)

        if self.attn2_type is not None:
            self.attention2 = Attention(
                layer_idx=layer_idx, # not used
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
                bias=attention_bias,
                use_flash_attn_3=use_flash_attn_3,
                is_cross_attention=(self.attn2_type == "cross"),
                is_causal=False,
                attn_dtype=attn_dtype,
            )
            self.attention2_norm = RMSNorm(hidden_size, eps=1e-6)

        # MLP
        self.mlp = MLP(hidden_size, intermediate_size)
        self.mlp_norm = RMSNorm(hidden_size, eps=1e-6)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        residual = hidden_states

        # Attention (cross or self)
        hidden_states = self.attention_norm(hidden_states)
        if self.attn_type == "self":
            hidden_states = self.attention(
                hidden_states=hidden_states,
            )
        else:
            assert encoder_hidden_states is not None, "encoder_hidden_states must be provided for cross attention."
            hidden_states = self.attention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        hidden_states = residual + hidden_states

        if self.attn2_type is not None:
            # maybe another attention (cross)
            residual = hidden_states
            hidden_states = self.attention2_norm(hidden_states)
            if self.attn2_type == "self":
                hidden_states = self.attention2(
                    hidden_states=hidden_states,
                )
            else:
                assert encoder_hidden_states is not None, "encoder_hidden_states must be provided for cross attention."
                hidden_states = self.attention2(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                )
            hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs


class FrequencyPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32)

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.logspace = logspace
        self.include_pi = include_pi
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = self.get_dims(input_dim)
    
    def init_weights(self):
        if self.logspace:
            frequencies = 2.0 ** torch.arange(self.num_freqs, dtype=torch.float32, device=self.frequencies.device)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (self.num_freqs - 1), self.num_freqs, dtype=torch.float32, device=self.frequencies.device)

        if self.include_pi:
            frequencies *= torch.pi
        self.frequencies = frequencies
        print("[FrequencyPositionalEmbedding] weights initialized.", self.frequencies)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(dtype=torch.float32)
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                embed = torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                embed = torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            embed = x
        return embed.to(dtype=dtype)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[int, torch.Tensor],
    theta: float = 10000.0,
    rope_percentage: float = 1.0,
    use_real: bool = False,
    repeat_interleave_real: bool = False,
    freqs_dtype: torch.dtype = torch.float32,
    linear_factor: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    生成 1D 旋转位置编码的频率
    
    Args:
        dim: head_dim
        pos: 位置序列长度或位置 tensor
        theta: RoPE 基础频率
        rope_percentage: 应用 RoPE 的维度比例
        use_real: 是否返回实数形式 (cos, sin)
        repeat_interleave_real: 是否使用 repeat_interleave 方式
        freqs_dtype: 输出数据类型
        linear_factor: 线性缩放因子
        
    Returns:
        如果 use_real=True: (freqs_cos, freqs_sin) 各为 [S, D]
        如果 use_real=False: freqs_cis 复数形式 [S, D/2]
    """
    if isinstance(pos, int):
        pos = torch.arange(pos, dtype=freqs_dtype)
    else:
        pos = pos.to(freqs_dtype)
    
    rope_dim = int(dim * rope_percentage)
    nope_dim = dim - rope_dim
    
    freqs = (
        1.0
        / (theta ** (torch.arange(0, rope_dim, 2, dtype=freqs_dtype, device=pos.device)[: (rope_dim // 2)] / rope_dim))
        / linear_factor
    )  # [D/2]
    
    freqs = torch.cat([freqs, torch.zeros(nope_dim // 2, dtype=freqs.dtype, device=freqs.device)])
    freqs = torch.outer(pos, freqs)  # [S, D/2]
    
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox style
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro style
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina style - 返回复数形式
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64, [S, D/2]
        return freqs_cis

class RotaryPosEmbed3D(nn.Module):
    """
    3D 旋转位置编码，用于 Octree Diffusion 等 3D 场景
    将 head_dim 分成三部分分别编码 x, y, z 三个维度
    """
    def __init__(
        self,
        attention_head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
        rope_percentage: float = 1.0,
        # grid_size: int = 128,
    ):
        super().__init__()
        self.in_channels = 3
        self.attention_head_dim = attention_head_dim
        self.max_seq_len = max_seq_len
        # self.grid_size = grid_size
        self.theta = theta
        self.rope_percentage = rope_percentage
        
        # 将 head_dim 分成三部分: h_dim, w_dim 各占 2*(head_dim//6)，剩余给 t_dim
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        
        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta,
                rope_percentage=rope_percentage,
                use_real=False, 
                repeat_interleave_real=False, 
                freqs_dtype=torch.float64
            )
            freqs.append(freq)
        freqs = torch.cat(freqs, dim=1)  # [max_seq_len, head_dim//2]
        self.register_buffer("freqs", freqs, persistent=False)
        self.inited = False
    
    def init_weights(self):
        """torchtitan 训练需要的权重初始化"""
        h_dim = w_dim = 2 * (self.attention_head_dim // 6)
        t_dim = self.attention_head_dim - h_dim - w_dim
        
        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, self.max_seq_len, self.theta,
                rope_percentage=self.rope_percentage,
                use_real=False, 
                repeat_interleave_real=False, 
                freqs_dtype=torch.float64
            )
            freqs.append(freq.to(self.freqs.device))
        self.freqs = torch.cat(freqs, dim=1)
        self.inited = True
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor):
        if self.training:
            assert self.inited
        rope_embed = self.forward_rope(pos)
        assert q.shape[-1] == rope_embed.shape[-1] * 2
        q = self.apply_rotary_emb_3d(q, rope_embed)
        k = self.apply_rotary_emb_3d(k, rope_embed)
        return q, k
    
    def apply_rotary_emb_3d(self, hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        应用 3D 旋转位置编码 (RoPE)
        
        Args:
            hidden_states: (total_len, num_heads, head_dim) 或 (B, T, num_heads, head_dim)
            freqs: (total_len, head_dim // 2) 复数形式的频率
            
        Returns:
            旋转后的 hidden_states，形状与输入相同
        """
        # 将输入转换为复数形式
        # hidden_states: [N, H, D] -> [N, H, D/2, 2] -> [N, H, D/2] (Complex)
        # NOTE: float32/complex64 is used here instead of float64/complex128.
        # The rotation is by precomputed integer-indexed angles; float32 matches the
        # float64 result to cosine=1.0 (mean abs diff ~2e-8) once the output is cast
        # back to bf16, while the complex64 multiply is ~1.8x faster than complex128.
        x_complex = torch.view_as_complex(hidden_states.to(torch.float32).unflatten(-1, (-1, 2)))

        # 处理 freqs 的维度以支持广播
        # freqs: [total_len, head_dim // 2] -> [total_len, 1, head_dim // 2]
        if freqs.dim() == 2:
            freqs = freqs.unsqueeze(1)

        # 执行旋转 (复数乘法) - freqs buffer 为 complex128，下采到 complex64 保持单精度
        x_rotated = x_complex * freqs.to(torch.complex64)
        
        # 变回实数并恢复形状
        x_out = torch.view_as_real(x_rotated).flatten(-2, -1)
        
        return x_out.type_as(hidden_states)
    
    def forward_rope(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (N, 3) 3D 位置坐标，已离散化的整数索引 [0, grid_size]
            
        Returns:
            rope_embed: (N, head_dim//2) 复数形式的旋转编码
        """
        N, dim = positions.shape
        assert dim == self.in_channels, f"Expected 3D positions, got {dim}D"
        
        # positions 已经是离散化的整数坐标，直接使用
        # 确保是 long 类型并 clamp 到有效范围
        positions = positions.long().clamp(0, self.max_seq_len - 1)
        
        # 分割频率表
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )
        
        # 分别索引 x, y, z 三个维度的频率
        freqs_x = freqs[0][positions[:, 0]]
        freqs_y = freqs[1][positions[:, 1]]
        freqs_z = freqs[2][positions[:, 2]]
        
        # 拼接得到完整的 3D RoPE
        rope_embed = torch.cat([freqs_x, freqs_y, freqs_z], dim=-1)
        return rope_embed

# ============================================================================
# Fourier Position Embedding
# ============================================================================

class FourierEmbedding3D(nn.Module):
    """3D Fourier 位置编码
    
    接收离散化整数坐标 [0, grid_size]，内部使用 float32 进行归一化和计算，
    避免 bfloat16 精度问题。
    """
    def __init__(self, dim: int, num_freqs: int = 8, grid_size: int = 512):
        super().__init__()
        self.dim = dim
        self.num_freqs = num_freqs
        self.grid_size = grid_size
        input_dim = 3 * num_freqs * 2 + 3  # sin + cos + raw
        self.mlp = nn.Linear(input_dim, dim)
        freqs = 2.0 ** torch.arange(num_freqs).float() * math.pi
        self.register_buffer('freqs', freqs)
        self.inited = False
    
    def init_weights(self):
        """初始化 buffer (用于 meta device 初始化流程)"""
        freqs = 2.0 ** torch.arange(self.num_freqs, dtype=torch.float32, device=self.freqs.device) * math.pi
        self.freqs = freqs
        self.inited = True
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 3) 或 (..., 3) 离散化整数坐标 [0, grid_size]
        Returns:
            (N, dim) 或 (..., dim) 编码后的特征
        """
        if self.training:
            assert self.inited
        shape = coords.shape[:-1]
        coords_flat = coords.reshape(-1, 3)
        # 使用 float32 归一化到 [-1, 1]，避免 bfloat16 精度问题
        coords_flat = coords_flat.float() / self.grid_size * 2 - 1  # [0, grid_size] -> [-1, 1]
        # (N, 3, num_freqs)
        x = coords_flat[:, :, None] * self.freqs[None, None, :]
        # (N, 3 * num_freqs)
        x = x.reshape(-1, 3 * self.num_freqs)
        # sin, cos, raw
        enc = torch.cat([torch.sin(x), torch.cos(x), coords_flat], dim=-1)
        return self.mlp(enc.to(self.mlp.weight.dtype)).reshape(*shape, self.dim)

strict_fp32_modules = (
    FrequencyPositionalEmbedding,
    RotaryEmbedding,
    FourierEmbedding3D,
    RotaryPosEmbed3D,
)


def to_dtype_except_strict_fp32(dtype: torch.dtype, module: nn.Module):
    if isinstance(module, strict_fp32_modules):
        to_dtype = torch.float32
    else:
        to_dtype = dtype
    
    # Iterate over all parameters and buffers
    for param in module.parameters(recurse=False):
        param.data = param.data.to(dtype=to_dtype)
    for buffer in module.buffers(recurse=False):
        buffer.data = buffer.data.to(dtype=to_dtype)

keep_precision_modules = (
    FrequencyPositionalEmbedding,
    RotaryEmbedding,
    FourierEmbedding3D,
    RotaryPosEmbed3D,
)

def to_dtype_except_keep_precision(dtype: torch.dtype, module: nn.Module):
    if isinstance(module, keep_precision_modules):
        return
    
    for param in module.parameters(recurse=False):
        param.data = param.data.to(dtype=dtype)
    for buffer in module.buffers(recurse=False):
        buffer.data = buffer.data.to(dtype=dtype)


class VEM2PointEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_tokens: int,
        num_key_value_heads: Optional[int] = None,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        dim_in: int = 6,
        num_self_attention_layers: int = 8,
        num_freqs: int = 8,
        enable_dora_cross_attn: bool = False,
        dora_cross_attn_layer_idx: int = 0,
        norm_out: bool = False,
        use_flash_attn_3: bool = False,
        mode: str = "learnable",
        attn_dtype: bool = "bf16",
    ):
        super().__init__()
        assert mode in ['learnable', 'downsample']
        self.num_self_attention_layers = num_self_attention_layers
        self.dim_in = dim_in
        self.enable_dora_cross_attn = enable_dora_cross_attn
        self.dora_cross_attn_layer_idx = dora_cross_attn_layer_idx
        self.norm_out = norm_out
        self.mode = mode
        self.num_tokens = num_tokens
        assert not (self.enable_dora_cross_attn and self.dora_cross_attn_layer_idx <= 0)

        # Frequency positional embedding
        self.xyz_embedder = FrequencyPositionalEmbedding(
            num_freqs=num_freqs,
            logspace=True,
            input_dim=3,
            include_input=True,
            include_pi=False,
        )

        self.point_embedding_dim = self.xyz_embedder.out_dim + 3

        # Project point features to hidden size
        self.proj_in = nn.Linear(self.point_embedding_dim, hidden_size)

        if self.enable_dora_cross_attn:
            self.proj_in_dora = nn.Linear(self.xyz_embedder.out_dim, hidden_size)

        # 1 cross attention layer + N self attention layers
        self.layers = nn.ModuleList([
                VEM2PointEncoderLayer(
                    layer_idx=0,
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    num_key_value_heads=num_key_value_heads,
                    qk_norm=qk_norm,
                    qk_norm_eps=qk_norm_eps,
                    attention_bias=attention_bias,
                    contains_self_attn=False,
                    contains_cross_attn=True,
                    use_flash_attn_3=use_flash_attn_3,
                    attn_dtype=attn_dtype,
                )
            ] + 
            [
                VEM2PointEncoderLayer(
                    layer_idx=i + 1,
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    num_key_value_heads=num_key_value_heads,
                    qk_norm=qk_norm,
                    qk_norm_eps=qk_norm_eps,
                    attention_bias=attention_bias,
                    contains_self_attn=True,
                    contains_cross_attn=(self.enable_dora_cross_attn and (i + 1 == self.dora_cross_attn_layer_idx)),
                    use_flash_attn_3=use_flash_attn_3,
                    attn_dtype=attn_dtype,
                )
                for i in range(num_self_attention_layers)
            ]
        )

        if mode == 'learnable':
            self.learnable_query = nn.Parameter(torch.randn(1, num_tokens, hidden_size))
        if self.norm_out:
            self.norm = RMSNorm(hidden_size, eps=1e-6)

    def fps_sample(
        self,
        point_cloud: torch.Tensor,
        feat: torch.Tensor,
        num_tokens: int,
    ):
        from torch_cluster import fps
        # if self.training:
        supersample_ratio = 4
        ind = torch.randperm(point_cloud.shape[1])[:supersample_ratio * num_tokens]
        pre_pc = point_cloud[:, ind, :]
        B, N, d = pre_pc.shape
        pos = pre_pc.view(B*N, d)
        assert d == 3

        batch = torch.arange(B, device=point_cloud.device)
        batch = torch.repeat_interleave(batch, N)
        idx = fps(pos.float(), batch, ratio=1. / supersample_ratio, random_start=self.training)

        pre_feats = feat[:, ind, :]
        B, N, D = pre_feats.shape
        pos_feats = pre_feats.view(B*N, D)
        sampled_feats = pos_feats[idx].view(B, num_tokens, D)

        sampled_pos = pos[idx].view(B, num_tokens, d)

        return sampled_feats, sampled_pos

    def forward(
        self,
        point_cloud: torch.Tensor,
        point_cloud_salient: Optional[torch.Tensor] = None,
        sort_points: bool = False,
        cast_to_bf16: bool = False,
    ) -> torch.Tensor:
        assert not sort_points
        assert ((point_cloud_salient is not None) == self.enable_dora_cross_attn), "point_cloud_salient must be provided when enable_dora_cross_attn is True."
        if isinstance(self.num_tokens, list):
            idx = torch.zeros(1, dtype=torch.long, device=point_cloud.device)
            if dist.get_rank() == 0:
                idx[0] = torch.randint(0, len(self.num_tokens), (1,))
            dist.broadcast(idx, src=0)
            num_tokens = self.num_tokens[idx.item()]
            # num_tokens = self.num_tokens[torch.randint(0, len(self.num_tokens), (1,)).item()]
            max_num_tokens = max(self.num_tokens)
            pc_downsample_ratio = num_tokens / max_num_tokens
            # shuffle and downsample
            perm = torch.randperm(point_cloud.shape[1])
            point_cloud = point_cloud[:, perm[:int(pc_downsample_ratio * point_cloud.shape[1])], :]
            if point_cloud_salient is not None:
                perm = torch.randperm(point_cloud_salient.shape[1])
                point_cloud_salient = point_cloud_salient[:, perm[:int(pc_downsample_ratio * point_cloud_salient.shape[1])], :]
        else:
            num_tokens = self.num_tokens

        # Apply frequency positional embedding
        batch_size, num_points, num_channels = point_cloud.shape
        xyz, normals = point_cloud[..., :3], point_cloud[..., 3:]

        point_features1 = torch.cat([self.xyz_embedder(xyz), normals], dim=-1)
        # print("point feature", point_features1.dtype)
        if cast_to_bf16:
            point_features1 = point_features1.bfloat16()
        point_features = self.proj_in(point_features1)

        if self.enable_dora_cross_attn:
            batch_size, num_points, num_channels = point_cloud_salient.shape
            assert num_channels == 3
            if cast_to_bf16:
                point_features_salient = self.proj_in_dora(self.xyz_embedder(point_cloud_salient).bfloat16())
            else:
                point_features_salient = self.proj_in_dora(self.xyz_embedder(point_cloud_salient))

        if self.mode == 'learnable':
            hidden_states = self.learnable_query.expand(batch_size, -1, -1)
        else:
            hidden_states, _ = self.fps_sample(
                point_cloud[..., :3],
                point_features,
                num_tokens=num_tokens,
            )

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=point_features,
                )
            elif self.enable_dora_cross_attn and (idx == self.dora_cross_attn_layer_idx):
                hidden_states = layer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=point_features_salient,
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                )
        
        if self.norm_out:
            hidden_states = self.norm(hidden_states)
        
        return hidden_states

class VEM2DualCAEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_tokens: Union[int, List[int]],
        num_key_value_heads: Optional[int] = None,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        dim_in: int = 6,
        num_self_attention_layers: int = 8,
        num_freqs: int = 8,
        enable_dora_cross_attn: bool = False,
        dora_ratio: float = 0.0,
        norm_out: bool = False,
        mode: str = 'learnable',
        use_flash_attn_3: bool = False,
        attn_dtype: str = "bf16",
    ):
        super().__init__()
        assert mode in ['learnable', 'downsample']
        assert not (mode == 'learnable' and isinstance(num_tokens, list))
        assert enable_dora_cross_attn
        self.mode = mode
        self.num_self_attention_layers = num_self_attention_layers
        self.dim_in = dim_in
        self.enable_dora_cross_attn = enable_dora_cross_attn
        self.norm_out = norm_out
        self.dora_ratio = dora_ratio
        self.num_tokens = num_tokens

        # Frequency positional embedding
        self.xyz_embedder = FrequencyPositionalEmbedding(
            num_freqs=num_freqs,
            logspace=True,
            input_dim=3,
            include_input=True,
            include_pi=False,
        )

        self.point_embedding_dim = self.xyz_embedder.out_dim + 3

        # Project point features to hidden size
        self.proj_in = nn.Linear(self.point_embedding_dim, hidden_size)

        self.pre_layers = [
            VEM2PointEncoderLayer(
                layer_idx=0,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_key_value_heads=num_key_value_heads,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
                attention_bias=attention_bias,
                contains_self_attn=False,
                contains_cross_attn=True,
                use_flash_attn_3=use_flash_attn_3,
                attn_dtype=attn_dtype,
            ),
        ]
        if self.enable_dora_cross_attn:
            self.pre_layers.append(
                VEM2PointEncoderLayer(
                    layer_idx=1,
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    num_key_value_heads=num_key_value_heads,
                    qk_norm=qk_norm,
                    qk_norm_eps=qk_norm_eps,
                    attention_bias=attention_bias,
                    contains_self_attn=False,
                    contains_cross_attn=True,
                    use_flash_attn_3=use_flash_attn_3,
                    attn_dtype=attn_dtype,
                )
            )
        self.pre_layers = nn.ModuleList(self.pre_layers)
        # 1 cross attention layer + N self attention layers
        self.layers = nn.ModuleList([
            VEM2PointEncoderLayer(
                layer_idx=i + len(self.pre_layers),
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_key_value_heads=num_key_value_heads,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
                attention_bias=attention_bias,
                contains_self_attn=True,
                contains_cross_attn=False,
                use_flash_attn_3=use_flash_attn_3,
                attn_dtype=attn_dtype,
            )
            for i in range(num_self_attention_layers)
        ])

        if mode == 'learnable':
            self.learnable_query = nn.Parameter(torch.randn(1, num_tokens, hidden_size))
        
        if self.norm_out:
            self.norm = RMSNorm(hidden_size, eps=1e-6)
    
    def fps_sample(
        self,
        point_cloud: torch.Tensor,
        feat: torch.Tensor,
        num_tokens: int,
    ):
        from torch_cluster import fps
        # if self.training:
        supersample_ratio = 4
        ind = torch.randperm(point_cloud.shape[1])[:supersample_ratio * num_tokens]
        pre_pc = point_cloud[:, ind, :]
        B, N, d = pre_pc.shape
        pos = pre_pc.view(B*N, d)
        assert d == 3

        batch = torch.arange(B, device=point_cloud.device)
        batch = torch.repeat_interleave(batch, N)
        idx = fps(pos.float(), batch, ratio=1. / supersample_ratio, random_start=self.training)

        pre_feats = feat[:, ind, :]
        B, N, D = pre_feats.shape
        pos_feats = pre_feats.view(B*N, D)
        sampled_feats = pos_feats[idx].view(B, num_tokens, D)

        sampled_pos = pos[idx].view(B, num_tokens, d)

        return sampled_feats, sampled_pos

    def forward(
        self,
        point_cloud: torch.Tensor,
        point_cloud_salient: Optional[torch.Tensor] = None,
        sort_points: bool = False,
    ) -> torch.Tensor:
        assert ((point_cloud_salient is not None) == self.enable_dora_cross_attn), "point_cloud_salient must be provided when enable_dora_cross_attn is True."
        # Apply frequency positional embedding
        assert point_cloud.shape[-1] == 6 and point_cloud_salient.shape[-1] == 6
        xyz, normals = point_cloud[..., :3], point_cloud[..., 3:]
        point_features = self.proj_in(torch.cat([self.xyz_embedder(xyz), normals], dim=-1))

        xyz, normals = point_cloud_salient[..., :3], point_cloud_salient[..., 3:]
        point_features_salient = self.proj_in(torch.cat([self.xyz_embedder(xyz), normals], dim=-1))

        batch_size, _, _ = point_cloud.shape
        if self.mode == 'learnable':
            hidden_states = self.learnable_query.expand(batch_size, -1, -1)
        else:
            if isinstance(self.num_tokens, list):
                if self.training:
                    num_tokens = self.num_tokens[torch.randint(0, len(self.num_tokens), (1,)).item()]
                else:
                    num_tokens = max(self.num_tokens)
            else:
                num_tokens = self.num_tokens
            num_dora_tokens = int(num_tokens * self.dora_ratio)
            assert point_cloud_salient.shape[1] >= 4 * num_dora_tokens
            uniform_token, uniform_pos = self.fps_sample(point_cloud[..., :3], point_features, num_tokens=num_tokens - num_dora_tokens)
            dora_token, dora_pos = self.fps_sample(point_cloud_salient[..., :3], point_features_salient, num_tokens=num_dora_tokens)

            hidden_states = torch.cat([uniform_token, dora_token], dim=1)
            if sort_points:
                token_pos = torch.cat([uniform_pos, dora_pos], dim=1)
                sampled_points_fp32 = token_pos.float()
                sampled_points_shifted = sampled_points_fp32 + 1 # should be in [0, 2]
                sampled_points_hashed = sampled_points_shifted[...,2] * 1e6 + sampled_points_shifted[...,1] * 1e3 + sampled_points_shifted[...,0]
                _, indices = torch.sort(sampled_points_hashed, dim=1)
                batch_indices = torch.arange(batch_size, device=indices.device).view(-1, 1).expand(-1, num_tokens)
                hidden_states = hidden_states[batch_indices, indices]
        
        hidden_states = self.pre_layers[0](
            hidden_states=hidden_states,
            encoder_hidden_states=point_features,
        ) + self.pre_layers[1](
            hidden_states=hidden_states,
            encoder_hidden_states=point_features_salient,
        )

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
            )
        
        if self.norm_out:
            hidden_states = self.norm(hidden_states)
        
        return hidden_states