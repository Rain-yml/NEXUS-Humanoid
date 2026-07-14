from dataclasses import dataclass, asdict
from typing import Optional, Literal, Tuple, Dict, Any, List, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from diffusers.utils import logging
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from torchtitan.experiments.vem.models.transformer import (
    Attention,
    RMSNorm,
    MLP,
    FP32LayerNorm,
    RotaryPosEmbed3D,
    FourierEmbedding3D,
)
from torchtitan.experiments.vem.models.dit import (
    Head,
    MLPProj,
)
from torchtitan.experiments.vem.models.mmdit import MMDiTBlock
from torchtitan.experiments.vem.models.octree import (
    SinusoidalEmbedding,
    MLPEncoder,
    _to_int_list,
    get_octree_condition_lens,
    get_octree_condition_proj_tokens,
)
from torchtitan.experiments.vem.datasets.octree_utils import OctreeBatch
from torchtitan.tools.logging import logger

logger_diffusers = logging.get_logger(__name__)

def merge_cu_seqlens_with_masks(cu_seqlens_a, cu_seqlens_b):
    device = cu_seqlens_a.device

    # 1. lengths
    lens_a = cu_seqlens_a[1:] - cu_seqlens_a[:-1]   # (B,)
    lens_b = cu_seqlens_b[1:] - cu_seqlens_b[:-1]   # (B,)
    lens_m = lens_a + lens_b                        # (B,)

    B = lens_a.shape[0]
    total_tokens = lens_m.sum()

    # 2. merged cu_seqlens
    cu_seqlens_m = torch.zeros(B + 1, device=device, dtype=cu_seqlens_a.dtype)
    cu_seqlens_m[1:] = torch.cumsum(lens_m, dim=0)

    # 3. build batch indices
    batch_idx_a = torch.repeat_interleave(torch.arange(B, device=device), lens_a)
    batch_idx_b = torch.repeat_interleave(torch.arange(B, device=device), lens_b)

    # 4. relative positions
    rel_pos_a = torch.arange(lens_a.sum(), device=device) - cu_seqlens_a[batch_idx_a]
    rel_pos_b = torch.arange(lens_b.sum(), device=device) - cu_seqlens_b[batch_idx_b]

    # 5. absolute positions in merged layout
    pos_a = cu_seqlens_m[batch_idx_a] + rel_pos_a
    pos_b = cu_seqlens_m[batch_idx_b] + lens_a[batch_idx_b] + rel_pos_b

    # 6. masks
    mask_a = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    mask_b = torch.zeros(total_tokens, dtype=torch.bool, device=device)

    mask_a[pos_a] = True
    mask_b[pos_b] = True

    return cu_seqlens_m, mask_a, mask_b

class OctreeMMDiTModel(ModelMixin, ConfigMixin):
    """
    Octree MM-DiT diffusion model using dual-stream joint attention (MMDiTBlock)
    instead of cross-attention (OctreeDiTBlock).

    The input stream (octree nodes) and condition stream (image features) each have
    independent norm/Q/K/V/MLP, but share a single joint attention computation.
    3D RoPE is applied only to the input stream.
    """
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["condition_embedder", "norm", "cond_proj"]
    _no_split_modules = ["MMDiTBlock"]
    _keep_in_fp32_modules = ["time_embedder", "modulation", "c_modulation", "norm1", "norm2", "c_norm1", "c_norm2"]
    _keys_to_ignore_on_load_unexpected = []

    @register_to_config
    def __init__(
        self,
        # representation
        use_onehot_256: bool = True,
        # Transformer architecture
        num_layers: int = 12,
        dim: int = 512,
        freq_dim: int = 256,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        num_key_value_heads: Optional[int] = None,
        # Attention config
        attention_bias: bool = False,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        use_flash_attn_3: bool = False,
        # Position embedding type
        pos_embed_type: str = 'rotary',
        # 3D RoPE config
        use_3d_rope: bool = True,
        max_seq_len_3d: int = 10000,
        grid_size: int = 128,
        rope_theta: int = 10000,
        # Fourier position embedding
        num_freqs: int = 8,
        # Octree config
        max_depth: int = 7,
        # Image condition
        image_hidden_size: int = 1280,
        num_vertex_condition: bool = False,
        num_vertex_condition_config: List[int] = [24, 10000, 1024],
        quad_ratio_condition: bool = False,
        quad_ratio_condition_mapping: str = "identity",
        # Multiview conditioning
        mv_mode: bool = False,
        num_mv_views: int = 4,
        # MMDiT-specific
        attn_dtype: str = "bf16",
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        assert pos_embed_type in ['fourier', 'rotary']
        assert (pos_embed_type == 'rotary') == use_3d_rope

        self.dim = dim
        self.max_depth = max_depth
        self.grid_size = grid_size
        self.use_onehot_256 = use_onehot_256
        self.pos_embed_type = pos_embed_type
        self.use_3d_rope = use_3d_rope and (pos_embed_type == 'rotary')

        # Input/output dimensions
        self.token_dim = 256 if use_onehot_256 else 8

        # ==================== Input encoding ====================
        # 1. Center position embedding (Fourier mode only)
        if pos_embed_type == 'fourier':
            self.center_embed = FourierEmbedding3D(dim, num_freqs=num_freqs, grid_size=grid_size)
        else:
            self.center_embed = None

        # 2. Depth embedding
        self.depth_embed = nn.Embedding(32, dim)

        # 3. Noisy occupancy state projection
        self.noise_proj = nn.Linear(self.token_dim, dim)

        # ==================== Time embedding ====================
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(freq_dim),
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Input stream time modulation (6 params: shift/scale/gate for attn & FFN)
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        # Condition stream time modulation (6 params: shift/scale/gate for attn & FFN)
        self.c_time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        self.cond_proj = MLPProj(image_hidden_size, dim)
        if pos_embed_type == 'rotary':
            self.rope_3d = RotaryPosEmbed3D(
                attention_head_dim=dim // num_attention_heads,
                max_seq_len=max_seq_len_3d,
                theta=rope_theta,
            )
        else:
            self.rope_3d = None

        # ==================== Transformer Blocks (MMDiT) ====================
        self.layers = nn.ModuleList([
            MMDiTBlock(
                layer_idx=i,
                hidden_size=dim,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_key_value_heads=num_key_value_heads,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
                attention_bias=attention_bias,
                use_flash_attn_3=use_flash_attn_3,
                rope_3d=self.rope_3d,
                attn_dtype=attn_dtype,
            )
            for i in range(num_layers)
        ])

        self.head = Head(dim, self.token_dim)

        # Store config for time embedding
        self._freq_dim = freq_dim
        self.num_vertex_condition = num_vertex_condition
        if num_vertex_condition:
            self.num_embed = MLPEncoder(
                dim,
                scalar_min=num_vertex_condition_config[0],
                scalar_max=num_vertex_condition_config[1],
                scalar_shift=num_vertex_condition_config[2],
                scalar_mapping="log_shift_normalize",
            )
        self.quad_ratio_condition = quad_ratio_condition
        if quad_ratio_condition:
            self.quad_ratio_embed = MLPEncoder(dim, scalar_mapping=quad_ratio_condition_mapping)

        self.mv_mode = mv_mode
        if mv_mode:
            self.view_embed = nn.Embedding(num_mv_views, dim)

    def init_weights(self, buffer_device=None):
        def _init_norm(module):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

        def _basic_init(module):
            if isinstance(module, (nn.LayerNorm, FP32LayerNorm, nn.RMSNorm, RMSNorm)):
                _init_norm(module)
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # Zero init modulation for both streams
        for layer in self.layers:
            nn.init.zeros_(layer.modulation)
            nn.init.zeros_(layer.c_modulation)

        nn.init.zeros_(self.head.modulation)
        nn.init.zeros_(self.head.head.weight)

        if self.center_embed is not None:
            self.center_embed.init_weights()

        if self.rope_3d is not None:
            self.rope_3d.init_weights()

        pretrained_path = self.config.pretrained_path
        if pretrained_path is not None:
            from torchtitan.tools.logging import logger
            logger.info(f"Loading pretrained weights from {pretrained_path}")

            state_dict = torch.load(pretrained_path, map_location='cpu')
            state_dict = state_dict['ema']['model']
            # manually set view_embed.weight to zeros when resuming from single-view model
            state_dict['view_embed.weight'] = torch.zeros_like(self.view_embed.weight)

            self._load_pretrained_to_model(self, state_dict, "self")

    def _load_pretrained_to_model(self, model, state_dict, model_name):
        from torchtitan.tools.logging import logger
        from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions

        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())

        if model_keys and state_dict_keys:
            sample_model_key = next(iter(model_keys))
            sample_state_key = next(iter(state_dict_keys))

            if sample_model_key.startswith('_orig_mod.') and not sample_state_key.startswith('_orig_mod.'):
                state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
                logger.info(f"{model_name}: added _orig_mod. prefix to match compiled model")
            elif not sample_model_key.startswith('_orig_mod.') and sample_state_key.startswith('_orig_mod.'):
                state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
                logger.info(f"{model_name}: removed _orig_mod. prefix from checkpoint")

        set_model_state_dict(
            model,
            model_state_dict=state_dict,
            options=StateDictOptions(full_state_dict=True, strict=True),
        )
        logger.info(f"{model_name}: loaded pretrained weights using set_model_state_dict")

    def is_scalar_param(self, name, param):
        patterns = [
            'num_embed',
            'time_embed',
            'time_projection',
            'c_time_projection',
            'noise_proj',
            'depth_embed',
            'center_embed',
            'head',
            'cond_proj',
            'view_embed',
            'quad_ratio_embed',
        ]
        for p in patterns:
            if p in name:
                return True
        return False
    

    def forward(
        self,
        x_t: torch.Tensor,                   # (total_nodes, token_dim)
        t: torch.Tensor,                     # (total_nodes,) or (batch_size,)
        centers: torch.Tensor,               # (total_nodes, 3)
        depths: torch.Tensor,                # (total_nodes,)
        cu_seqlens_q: torch.Tensor,          # (batch_size + 1,)
        num_layers_per_mesh: list = None,
        encoder_hidden_states: torch.Tensor = None,
        condition_cu_seqlens: torch.Tensor = None,
        num_vertices: torch.Tensor = None,
        quad_ratios: torch.Tensor = None,
        view_indices: torch.Tensor = None,
        mv_cu_seqlens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass - Varlen mode

        Args:
            x_t: noisy occupancy state (256-dim one-hot or 8-bit)
            t: diffusion timestep [0, 1000]
            centers: parent node center coordinates (discretized integer, dtype=long, [0, grid_size])
            depths: node depth
            cu_seqlens_q: cumulative sequence lengths for Q
            num_layers_per_mesh: number of octree layers per mesh
            encoder_hidden_states: pre-encoded image features
            num_vertices: vertex counts per mesh
            view_indices: view ids (mv_mode only)
            mv_cu_seqlens: cumulative view counts (mv_mode only)

        Returns:
            predicted velocity field v (total_nodes, token_dim)
        """
        total_nodes = x_t.shape[0]
        batch_size = cu_seqlens_q.shape[0] - 1

        # ==================== Time embedding (float32 for numerical stability) ====================
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_emb = self.time_embed(t)  # (total_nodes, dim)
            # Input stream modulation (per-token)
            temb_modulation = self.time_projection(t_emb).view(total_nodes, 6, self.dim)

        # ==================== Condition encoding ====================
        condition_list = []

        if num_layers_per_mesh is not None and len(num_layers_per_mesh) > 0:
            num_unique_meshes = len(num_layers_per_mesh)
            unique_indices = []
            offset = 0
            for num_layers in num_layers_per_mesh:
                unique_indices.append(offset)
                offset += num_layers
        else:
            num_unique_meshes = batch_size
            unique_indices = None
        if num_layers_per_mesh is None:
            num_layers_per_mesh = [1] * num_unique_meshes
        if self.quad_ratio_condition and quad_ratios is not None:
            if quad_ratios.shape[0] != num_unique_meshes:
                raise ValueError(
                    f"Expected {num_unique_meshes} quad ratios, got {quad_ratios.shape[0]}"
                )

        # Project image features
        encoder_hidden_states = self.cond_proj(encoder_hidden_states)
        is_varlen_condition = condition_cu_seqlens is not None

        assert cu_seqlens_q is not None

        if self.mv_mode:
            assert view_indices is not None and mv_cu_seqlens is not None
            if is_varlen_condition:
                view_lens = torch.diff(condition_cu_seqlens)
                ve = self.view_embed(view_indices).to(encoder_hidden_states.dtype)
                encoder_hidden_states = encoder_hidden_states + ve.repeat_interleave(view_lens, dim=0)
                view_groups = [
                    encoder_hidden_states[condition_cu_seqlens[mv_cu_seqlens[i]]:condition_cu_seqlens[mv_cu_seqlens[i + 1]]]
                    for i in range(num_unique_meshes)
                ]
            else:
                # mv_mode: encoder_hidden_states is (total_views, num_tokens, dim)
                ve = self.view_embed(view_indices).unsqueeze(1).to(encoder_hidden_states.dtype)
                encoder_hidden_states = encoder_hidden_states + ve
                view_groups = [
                    encoder_hidden_states[mv_cu_seqlens[i]:mv_cu_seqlens[i + 1]].reshape(-1, self.dim)
                    for i in range(num_unique_meshes)
                ]

            if self.num_vertex_condition:
                assert num_vertices is not None
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    nv_embed = self.num_embed(num_vertices)
                view_groups = [
                    torch.cat([vg, nv.reshape(1, -1).to(vg.dtype)], dim=0)
                    for vg, nv in zip(view_groups, nv_embed)
                ]
            if self.quad_ratio_condition:
                assert quad_ratios is not None
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    qr_embed = self.quad_ratio_embed(quad_ratios)
                view_groups = [
                    torch.cat([vg, qr.reshape(1, -1).to(vg.dtype)], dim=0)
                    for vg, qr in zip(view_groups, qr_embed)
                ]

            encoder_hs_list = []
            kv_lens = []
            for hs, n_layers in zip(view_groups, num_layers_per_mesh):
                for _ in range(n_layers):
                    encoder_hs_list.append(hs)
                    kv_lens.append(hs.shape[0])
            encoder_hidden_states = torch.cat(encoder_hs_list, dim=0)
            cu_seqlens_kv = torch.zeros(len(kv_lens) + 1, dtype=torch.int32, device=cu_seqlens_q.device)
            cu_seqlens_kv[1:] = torch.cumsum(
                torch.tensor(kv_lens, dtype=torch.int32, device=cu_seqlens_q.device), dim=0
            )
        else:
            if is_varlen_condition:
                if self.num_vertex_condition:
                    assert num_vertices is not None
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        nv_embed = self.num_embed(num_vertices)
                if self.quad_ratio_condition:
                    assert quad_ratios is not None
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        qr_embed = self.quad_ratio_embed(quad_ratios)
                encoder_hs_list = []
                kv_lens = []
                for i, n_layers in enumerate(num_layers_per_mesh):
                    hs = encoder_hidden_states[condition_cu_seqlens[i]:condition_cu_seqlens[i + 1]]
                    if self.num_vertex_condition:
                        hs = torch.cat([hs, nv_embed[i].to(hs.dtype)], dim=0)
                    if self.quad_ratio_condition:
                        hs = torch.cat([hs, qr_embed[i].to(hs.dtype)], dim=0)
                    for _ in range(n_layers):
                        encoder_hs_list.append(hs)
                        kv_lens.append(hs.shape[0])
                encoder_hidden_states = torch.cat(encoder_hs_list, dim=0)
                cu_seqlens_kv = torch.zeros(len(kv_lens) + 1, dtype=torch.int32, device=cu_seqlens_q.device)
                cu_seqlens_kv[1:] = torch.cumsum(
                    torch.tensor(kv_lens, dtype=torch.int32, device=cu_seqlens_q.device), dim=0
                )
            else:
                # Standard single-view path
                if self.num_vertex_condition:
                    assert num_vertices is not None
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        nv_embed = self.num_embed(num_vertices)
                    assert nv_embed.dim() == 3
                    assert encoder_hidden_states.dim() == 3
                    encoder_hidden_states = torch.cat([encoder_hidden_states, nv_embed.to(encoder_hidden_states.dtype)], dim=1)
                if self.quad_ratio_condition:
                    assert quad_ratios is not None
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        qr_embed = self.quad_ratio_embed(quad_ratios)
                    assert qr_embed.dim() == 3
                    assert encoder_hidden_states.dim() == 3
                    encoder_hidden_states = torch.cat([encoder_hidden_states, qr_embed.to(encoder_hidden_states.dtype)], dim=1)

                repeat_counts = torch.tensor(num_layers_per_mesh, device=encoder_hidden_states.device)
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeat_counts, dim=0)

                num_condition_tokens = encoder_hidden_states.shape[1]
                encoder_hidden_states = encoder_hidden_states.reshape(batch_size * num_condition_tokens, -1)

                cu_seqlens_kv = torch.arange(
                    0, (batch_size + 1) * num_condition_tokens, num_condition_tokens,
                    device=cu_seqlens_q.device, dtype=torch.int32,
                )

        # ==================== Condition stream time modulation ====================
        with torch.amp.autocast('cuda', dtype=torch.float32):
            # Extract one t_emb per batch element and project for condition stream
            t_emb_per_sample = t_emb[cu_seqlens_q[:-1]]  # (batch_size, dim)
            c_temb_proj = self.c_time_projection(t_emb_per_sample).view(batch_size, 6, self.dim)
            # Expand to (total_condition_tokens, 6, dim)
            cond_lens = torch.diff(cu_seqlens_kv)  # (batch_size,)
            c_temb_modulation = c_temb_proj.repeat_interleave(cond_lens, dim=0)

        # ==================== Input encoding ====================
        noise_emb = self.noise_proj(x_t.to(self.noise_proj.weight.dtype))
        depth_emb = self.depth_embed(depths.clamp(0, self.max_depth)).to(noise_emb.dtype)

        if self.pos_embed_type == 'fourier':
            center_emb = self.center_embed(centers).to(noise_emb.dtype)
        else:
            center_emb = torch.zeros(total_nodes, self.dim, device=noise_emb.device, dtype=noise_emb.dtype)

        x = center_emb + depth_emb + noise_emb

        # ==================== token concatenation ====================
        cu_seqlens_all, mask_x, mask_cond = merge_cu_seqlens_with_masks(cu_seqlens_q, cu_seqlens_kv)
        x_cond = torch.zeros(cu_seqlens_all[-1], self.dim, device=x.device, dtype=x.dtype)
        x_cond[mask_x] = x
        x_cond[mask_cond] = encoder_hidden_states

        temb_modulation_concat = torch.zeros(cu_seqlens_all[-1], 6, self.dim, device=temb_modulation.device, dtype=temb_modulation.dtype)
        temb_modulation_concat[mask_x] = temb_modulation
        temb_modulation_concat[mask_cond] = c_temb_modulation

        # ==================== Transformer Blocks (MM-DiT) ====================
        for block in self.layers:
            x_cond = block(
                hidden_states_cat=x_cond,
                temb_modulation_cat=temb_modulation_concat,
                cu_seqlens=cu_seqlens_all,
                mask_x=mask_x,
                mask_c=mask_cond,
                positions=centers if self.use_3d_rope else None,
            )
        
        x = x_cond[mask_x]
        # ==================== Output (input stream only) ====================
        x = self.head(x, t_emb, cu_seqlens=cu_seqlens_q)

        return x

    def train_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        cu_seqlens: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        input_dict: OctreeBatch,
        dtype,
    ):
        pred = self(
            x_t=x_t,
            t=timesteps,
            centers=input_dict.layer_parent_centers_flat,
            depths=input_dict.layer_depths_flat,
            cu_seqlens_q=cu_seqlens,
            num_layers_per_mesh=input_dict.num_layers_per_mesh,
            encoder_hidden_states=conditions["encoder_hidden_states"].to(dtype),
            condition_cu_seqlens=conditions.get("condition_cu_seqlens"),
            num_vertices=input_dict.num_vertices,
            quad_ratios=input_dict.quad_ratios,
            view_indices=conditions.get("view_indices"),
            mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
        )

        return pred, {
            'num_tokens': x_t.shape[0],
            'num_flops': self.flops(
                cu_seqlens,
                get_octree_condition_lens(
                    conditions,
                    input_dict,
                    batch_size=cu_seqlens.shape[0] - 1,
                    num_vertex_condition=self.num_vertex_condition,
                    quad_ratio_condition=self.quad_ratio_condition,
                ),
                get_octree_condition_proj_tokens(conditions),
            ),
        }

    def _joint_attn_flops(self, s: int, s_enc: int, d: int, h: int, h_kv: int) -> float:
        kv_ratio = h_kv / h
        s_all = s + s_enc
        q_proj = 2 * s_all * d**2
        kv_proj = 4 * s_all * d**2 * kv_ratio
        attn = 4 * s_all**2 * d
        o_proj = 2 * s_all * d**2
        return q_proj + kv_proj + attn + o_proj

    def _ffn_flops(self, s: int, d: int, d_ffn: int) -> float:
        return 6 * s * d * d_ffn

    def flops(
        self,
        cu_seqlens: Optional[torch.Tensor] = None,
        condition_lens=None,
        condition_proj_tokens: Optional[int] = None,
    ) -> float:
        if cu_seqlens is None or condition_lens is None:
            return 100.0

        seqlen = _to_int_list(cu_seqlens[1:] - cu_seqlens[:-1])
        if isinstance(condition_lens, int):
            condition_lens = [condition_lens] * len(seqlen)
        else:
            condition_lens = _to_int_list(condition_lens)
        if len(condition_lens) != len(seqlen):
            raise ValueError(
                f"Expected {len(seqlen)} condition lengths, got {len(condition_lens)}"
            )

        dim = int(self.config.dim)
        dim_freq = int(self.config.freq_dim)
        dim_ffn = int(self.config.intermediate_size)
        dim_cond = int(self.config.image_hidden_size)
        h = int(self.config.num_attention_heads)
        h_kv = int(self.config.num_key_value_heads or h)
        token_dim = self.token_dim

        if condition_proj_tokens is None:
            condition_proj_tokens = sum(condition_lens)

        total = condition_proj_tokens * (dim_cond * dim + dim * dim) * 2
        for s, s_enc in zip(seqlen, condition_lens):
            transformer = self.config.num_layers * (
                self._joint_attn_flops(s, s_enc, dim, h, h_kv)
                + self._ffn_flops(s + s_enc, dim, dim_ffn)
            )
            time_embed_and_proj = (dim_freq * dim + dim * dim + dim * dim * 6) * 2 * s
            cond_time_proj = dim * dim * 6 * 2
            noise_proj = s * token_dim * dim * 2
            center_proj = (
                s * (3 * int(self.config.num_freqs) * 2 + 3) * dim * 2
                if self.config.pos_embed_type == "fourier"
                else 0
            )
            head = s * token_dim * dim * 2
            total += (
                transformer
                + time_embed_and_proj
                + cond_time_proj
                + noise_proj
                + center_proj
                + head
            )
        return total
