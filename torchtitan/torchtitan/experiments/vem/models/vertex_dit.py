import math
from typing import Any, Dict, Optional, Tuple, Union, Literal
import numpy as np
import torch.cuda.amp as amp

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import logging
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from torchtitan.experiments.vem.models.transformer import (
    Attention, 
    RMSNorm, 
    MLP, 
    FP32LayerNorm,
    FrequencyPositionalEmbedding,
    RotaryPosEmbed3D,
)
from torchtitan.experiments.vem.models.dit import (
    get_timestep_embedding, 
    VEMDiTBlock,
    Head,
    MLPProj,
)
from torchtitan.experiments.vem.models.octree import MLPEncoder

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class SpaceMeshDiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["condition_embedder", "norm"]
    _no_split_modules = ["VEMDiTBlock"]
    _keep_in_fp32_modules = ["time_embedder", "modulation", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = []

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        dim: int,
        freq_dim: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_key_value_heads: Optional[int] = None,
        attention_bias: bool = False,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        use_flash_attn_3: bool = False,
        num_freqs: int = 6,
        condition_dim: int = 256,
        gated: bool = False,
        pos_embed_type: str = "fourier",
        max_seq_len_rope: int = 10000,
        rope_theta: float = 10000.0,
        num_registers: int = 0,
        ape_scale_div: float = 1.0,
        mv_mode: bool = False,
        num_mv_views: int = 4,
        quad_ratio_condition: bool = False,
        quad_ratio_condition_mapping: str = "identity",
        quad_ratio_uncond: bool = False,
        symmetry_condition: bool = False,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        assert pos_embed_type in ["fourier", "rotary", "fourier_rotary"]
        self.pos_embed_type = pos_embed_type
        self.ape_scale_div = ape_scale_div
        self.mv_mode = mv_mode
        self.quad_ratio_condition = quad_ratio_condition
        self.symmetry_condition = symmetry_condition

        self.use_condition = condition_dim > 0
        if self.use_condition:
            self.cond_proj = MLPProj(condition_dim, dim)
            if mv_mode:
                self.view_embed = nn.Embedding(num_mv_views, dim)
        if quad_ratio_condition:
            if not self.use_condition:
                raise ValueError("quad_ratio_condition requires condition_dim > 0")
            self.quad_ratio_embed = MLPEncoder(dim, scalar_mapping=quad_ratio_condition_mapping)
            # Unconditional embedding used when quad_ratio < 0 (unknown quad ratio).
            # Named quad_ratio_uncond_embed to avoid clashing with the init flag.
            self.quad_ratio_uncond_enabled = quad_ratio_uncond
            if quad_ratio_uncond:
                self.quad_ratio_uncond_embed = nn.Embedding(1, dim)
        else:
            self.quad_ratio_uncond_enabled = False
        if symmetry_condition:
            if not self.use_condition:
                raise ValueError("symmetry_condition requires condition_dim > 0")
            # 4 directions (x=0 / y=0 / z=0 / any-of-xyz) x 2 states (symmetric / uncertain)
            # flat index = direction * 2 + state, state: 1=symmetric, 0=uncertain
            self.symmetry_embed = nn.Embedding(8, dim)

        # input projection
        if self.pos_embed_type == "rotary":
            self.rope_3d = RotaryPosEmbed3D(
                attention_head_dim=dim // num_attention_heads,
                max_seq_len=max_seq_len_rope,
                theta=rope_theta,
            )
            self.xyz_embedder = None
            self.proj_embedding_dim = 0
            self.proj = nn.Linear(in_channels, dim)
        elif self.pos_embed_type == "fourier_rotary":
            self.rope_3d = RotaryPosEmbed3D(
                attention_head_dim=dim // num_attention_heads,
                max_seq_len=max_seq_len_rope,
                theta=rope_theta,
            )
            self.xyz_embedder = FrequencyPositionalEmbedding(
                num_freqs=num_freqs,
                logspace=True,
                input_dim=3,
                include_input=True,
                include_pi=False,
            )
            self.proj_embedding_dim = self.xyz_embedder.out_dim
            self.proj = nn.Linear(self.proj_embedding_dim + in_channels, dim)
        else:
            self.rope_3d = None
            self.xyz_embedder = FrequencyPositionalEmbedding(
                num_freqs=num_freqs,
                logspace=True,
                input_dim=3,
                include_input=True,
                include_pi=False,
            )
            self.proj_embedding_dim = self.xyz_embedder.out_dim
            self.proj = nn.Linear(self.proj_embedding_dim + in_channels, dim)

        # transformer blocks
        self.layers = nn.ModuleList(
            [
                VEMDiTBlock(
                    layer_idx=i,
                    hidden_size=dim,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    num_key_value_heads=num_key_value_heads,
                    qk_norm=qk_norm,
                    qk_norm_eps=qk_norm_eps,
                    attention_bias=attention_bias,
                    use_flash_attn_3=use_flash_attn_3,
                    contain_cross_attention=self.use_condition,
                    gated=gated,
                    rope_3d=self.rope_3d,
                )
                for i in range(num_layers)
            ]
        )
        # output projection
        self.head = Head(dim, in_channels)
        
        # frequency to time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        self.num_registers = num_registers
        if self.num_registers > 0:
            self.registers = nn.Parameter(torch.zeros(self.num_registers, dim))

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
                # nn.init.xavier_uniform_(module.weight)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        self.apply(_basic_init)
        for layer in self.layers:
            nn.init.zeros_(layer.modulation)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.modulation)
        nn.init.zeros_(self.head.head.weight)
        if self.xyz_embedder is not None:
            self.xyz_embedder.init_weights()
        if self.rope_3d is not None:
            self.rope_3d.init_weights()
        if self.num_registers > 0:
            nn.init.normal_(self.registers, std=0.02)
        
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
        """
        Load pretrained weights to a model, handling DTensor case.
        
        Uses set_model_state_dict for proper DTensor handling with FSDP2/TP.
        Also handles _orig_mod. prefix mismatch when torch.compile is enabled.
        """
        from torchtitan.tools.logging import logger
        from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions

        # Handle _orig_mod. prefix mismatch (torch.compile adds this prefix)
        # Check if model has _orig_mod. prefix but state_dict doesn't
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        
        # Check if we need to add _orig_mod. prefix to state_dict
        if model_keys and state_dict_keys:
            sample_model_key = next(iter(model_keys))
            sample_state_key = next(iter(state_dict_keys))
            
            if sample_model_key.startswith('_orig_mod.') and not sample_state_key.startswith('_orig_mod.'):
                # Add _orig_mod. prefix to state_dict keys
                state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
                logger.info(f"{model_name}: added _orig_mod. prefix to match compiled model")
            elif not sample_model_key.startswith('_orig_mod.') and sample_state_key.startswith('_orig_mod.'):
                # Remove _orig_mod. prefix from state_dict keys
                state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
                logger.info(f"{model_name}: removed _orig_mod. prefix from checkpoint")
        

        # Use the distributed checkpoint API which handles DTensor properly
        set_model_state_dict(
            model,
            model_state_dict=state_dict,
            options=StateDictOptions(full_state_dict=True, strict=True),
        )
        logger.info(f"{model_name}: loaded pretrained weights using set_model_state_dict")
    
    def _embed_quad_ratio(self, quad_ratios: torch.Tensor) -> torch.Tensor:
        """Embed quad ratios, replacing negative entries with an uncond embedding.

        A negative quad_ratio means "quad ratio unknown": those entries use the
        learned unconditional embedding (quad_ratio_uncond_embed) instead of the
        MLPEncoder output, when enabled.

        Returns: (num_meshes, 1, dim)
        """
        with torch.amp.autocast('cuda', dtype=torch.float32):
            # Clamp negatives to 0 so the MLPEncoder never sees an out-of-range value;
            # those entries are overwritten by the uncond embedding below.
            qr_clamped = quad_ratios.clamp(min=0.0)
            qr_embed = self.quad_ratio_embed(qr_clamped)  # (num_meshes, 1, dim)
            if self.quad_ratio_uncond_enabled:
                # torch.where unconditionally (no .any() guard) to avoid a GPU->CPU sync.
                mask = (quad_ratios < 0).view(-1, 1, 1).to(qr_embed.device)
                uncond = self.quad_ratio_uncond_embed.weight.view(1, 1, -1).to(qr_embed.dtype)
                qr_embed = torch.where(mask, uncond, qr_embed)
        return qr_embed

    def _embed_symmetries(
        self,
        symmetries: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Embed the 4 symmetry-direction tokens per mesh.

        symmetries: (batch, 4) int64 with values in {1=symmetric, 0=uncertain} for
            directions [x=0, y=0, z=0, any-of-xyz]. When None, defaults to
            all-uncertain.

        Returns: (batch, 4, dim)
        """
        if symmetries is None:
            symmetries = torch.zeros(batch_size, 4, dtype=torch.long, device=device)
        else:
            symmetries = symmetries.to(device=device, dtype=torch.long)
        # flat index = direction * 2 + state, state: 1=symmetric, 0=uncertain
        directions = torch.arange(4, device=device, dtype=torch.long).view(1, 4)
        indices = directions * 2 + symmetries  # (batch, 4)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            sym_embed = self.symmetry_embed(indices)  # (batch, 4, dim)
        return sym_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        hidden_states_position: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor]=None,
        cu_seqlens: Optional[torch.Tensor]=None,
        cu_seqlens_encoder: Optional[torch.Tensor] = None,
        view_indices: Optional[torch.Tensor] = None,
        mv_cu_seqlens: Optional[torch.Tensor] = None,
        quad_ratios: Optional[torch.Tensor] = None,
        symmetries: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states (torch.FloatTensor): (total_vertices, in_channels) or (batch, num_tokens, in_channels).
            timesteps (torch.FloatTensor): (batch,) or (total_vertices,) Timesteps.
            hidden_states_position (torch.FloatTensor): (total_vertices, 3) Vertex positions.
            encoder_hidden_states (torch.FloatTensor): single-view: (batch, num_encoder_tokens, condition_dim);
                mv_mode: (total_views, num_encoder_tokens, condition_dim).
            cu_seqlens (torch.IntTensor): (batch+1,) cumulative sequence lengths for query.
            cu_seqlens_encoder (torch.IntTensor): (batch+1,) pre-built encoder cu_seqlens (optional).
            view_indices (torch.LongTensor): (total_views,) view IDs; only used when mv_mode=True.
            mv_cu_seqlens (torch.IntTensor): (batch+1,) cumulative view counts per mesh; only used when mv_mode=True.
            quad_ratios (torch.FloatTensor): (batch,) quad ratios; only used when quad_ratio_condition=True.
            symmetries (torch.LongTensor): (batch, 4) symmetry states (1=symmetric, 0=uncertain) for
                directions [x=0, y=0, z=0, any-of-xyz]; only used when symmetry_condition=True.
        """
        with torch.amp.autocast('cuda', dtype=torch.float32):
            temb = self.time_embedding(get_timestep_embedding(
                timesteps=timesteps,
                embedding_dim=self.config.freq_dim,
            )) # B x C

            temb_modulation = self.time_projection(temb).view(temb.shape[0], 6, self.config.dim)

        if self.mv_mode:
            assert view_indices is not None and mv_cu_seqlens is not None
            if self.quad_ratio_condition:
                assert quad_ratios is not None
            # encoder_hidden_states: (total_views, T, condition_dim)
            encoder_hidden_states = self.cond_proj(encoder_hidden_states)  # (total_views, T, dim)
            ve = self.view_embed(view_indices).unsqueeze(1).to(encoder_hidden_states.dtype)  # (total_views, 1, dim)
            encoder_hidden_states = encoder_hidden_states + ve
            batch_size = mv_cu_seqlens.shape[0] - 1
            kv_blocks = [
                encoder_hidden_states[mv_cu_seqlens[i]:mv_cu_seqlens[i + 1]].reshape(-1, self.config.dim)
                for i in range(batch_size)
            ]  # list of (V_i * T, dim)
            if self.quad_ratio_condition:
                if quad_ratios.shape[0] != batch_size:
                    raise ValueError(f"Expected {batch_size} quad ratios, got {quad_ratios.shape[0]}")
                qr_embed = self._embed_quad_ratio(quad_ratios)
                kv_blocks = [
                    torch.cat([b, qr.reshape(1, -1).to(b.dtype)], dim=0)
                    for b, qr in zip(kv_blocks, qr_embed)
                ]
            if self.symmetry_condition:
                sym_embed = self._embed_symmetries(
                    symmetries, batch_size, encoder_hidden_states.device
                )
                kv_blocks = [
                    torch.cat([b, sym.to(b.dtype)], dim=0)
                    for b, sym in zip(kv_blocks, sym_embed)
                ]
            kv_lens = [b.shape[0] for b in kv_blocks]
            encoder_hidden_states = torch.cat(kv_blocks, dim=0)  # (sum(V_i * T), dim)
            cu_seqlens_encoder = torch.zeros(batch_size + 1, dtype=torch.int32, device=encoder_hidden_states.device)
            cu_seqlens_encoder[1:] = torch.cumsum(
                torch.tensor(kv_lens, dtype=torch.int32, device=encoder_hidden_states.device), dim=0
            )
        else:
            encoder_hidden_states = self.cond_proj(encoder_hidden_states)

            if cu_seqlens is None:
                assert cu_seqlens_encoder is None
                if self.quad_ratio_condition:
                    assert quad_ratios is not None
                    bs = encoder_hidden_states.shape[0]
                    if quad_ratios.shape[0] != bs:
                        raise ValueError(f"Expected {bs} quad ratios, got {quad_ratios.shape[0]}")
                    qr_embed = self._embed_quad_ratio(quad_ratios)
                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states, qr_embed.to(encoder_hidden_states.dtype)],
                        dim=1,
                    )
                if self.symmetry_condition:
                    bs = encoder_hidden_states.shape[0]
                    sym_embed = self._embed_symmetries(symmetries, bs, encoder_hidden_states.device)
                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states, sym_embed.to(encoder_hidden_states.dtype)],
                        dim=1,
                    )
            else:
                if cu_seqlens_encoder is not None:
                    assert encoder_hidden_states.dim() == 2
                    bs = cu_seqlens_encoder.shape[0] - 1
                    # Rebuild kv_blocks when either condition is enabled
                    if self.quad_ratio_condition or self.symmetry_condition:
                        if self.quad_ratio_condition:
                            assert quad_ratios is not None
                            if quad_ratios.shape[0] != bs:
                                raise ValueError(f"Expected {bs} quad ratios, got {quad_ratios.shape[0]}")
                            qr_embed = self._embed_quad_ratio(quad_ratios)
                        if self.symmetry_condition:
                            sym_embed = self._embed_symmetries(symmetries, bs, encoder_hidden_states.device)
                        kv_blocks = []
                        for i in range(bs):
                            hs = encoder_hidden_states[cu_seqlens_encoder[i]:cu_seqlens_encoder[i + 1]]
                            if self.quad_ratio_condition:
                                hs = torch.cat([hs, qr_embed[i].to(hs.dtype)], dim=0)
                            if self.symmetry_condition:
                                hs = torch.cat([hs, sym_embed[i].to(hs.dtype)], dim=0)
                            kv_blocks.append(hs)
                        kv_lens = [b.shape[0] for b in kv_blocks]
                        encoder_hidden_states = torch.cat(kv_blocks, dim=0)
                        cu_seqlens_encoder = torch.zeros(
                            bs + 1,
                            dtype=torch.int32,
                            device=encoder_hidden_states.device,
                        )
                        cu_seqlens_encoder[1:] = torch.cumsum(
                            torch.tensor(kv_lens, dtype=torch.int32, device=encoder_hidden_states.device),
                            dim=0,
                        )
                else:
                    assert encoder_hidden_states.dim() == 3
                    bs = encoder_hidden_states.shape[0]
                    assert bs == cu_seqlens.shape[0] - 1
                    if self.quad_ratio_condition:
                        assert quad_ratios is not None
                        if quad_ratios.shape[0] != bs:
                            raise ValueError(f"Expected {bs} quad ratios, got {quad_ratios.shape[0]}")
                        qr_embed = self._embed_quad_ratio(quad_ratios)
                        encoder_hidden_states = torch.cat(
                            [encoder_hidden_states, qr_embed.to(encoder_hidden_states.dtype)],
                            dim=1,
                        )
                    if self.symmetry_condition:
                        sym_embed = self._embed_symmetries(symmetries, bs, encoder_hidden_states.device)
                        encoder_hidden_states = torch.cat(
                            [encoder_hidden_states, sym_embed.to(encoder_hidden_states.dtype)],
                            dim=1,
                        )
                    seqlen = encoder_hidden_states.shape[1]
                    cu_seqlens_encoder = torch.arange(0, (bs + 1) * seqlen, seqlen, dtype=torch.int32, device=cu_seqlens.device)
                    encoder_hidden_states = encoder_hidden_states.view(bs * seqlen, -1)
        
        if self.pos_embed_type == "fourier":
            point_feature = self.xyz_embedder(hidden_states_position / self.ape_scale_div).to(hidden_states.dtype)
            hidden_states = torch.cat([point_feature, hidden_states], dim=-1)
            positions = None
        elif self.pos_embed_type == "fourier_rotary":
            point_feature = self.xyz_embedder(hidden_states_position / self.ape_scale_div).to(hidden_states.dtype)
            hidden_states = torch.cat([point_feature, hidden_states], dim=-1)
            positions = hidden_states_position
        else:
            positions = hidden_states_position
            assert positions.dtype == torch.long
        
        hidden_states = self.proj(hidden_states)
        if self.num_registers > 0:
            # Add registers to each sequence
            batch_size = cu_seqlens.shape[0] - 1
            num_registers = self.num_registers

            # Create indices for inserting registers
            register_indices = (cu_seqlens[:-1].unsqueeze(1) + torch.arange(num_registers, device=cu_seqlens.device)).flatten()
            
            # Expand registers
            registers_expanded = self.registers.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.registers.shape[-1])
            
            # Create indices for original hidden_states (shifted by number of registers already inserted)
            total_tokens_original = hidden_states.shape[0]
            register_mask = torch.zeros(total_tokens_original + batch_size* num_registers, dtype=torch.bool, device=hidden_states.device)
            register_mask[register_indices + torch.arange(batch_size, device=cu_seqlens.device).repeat_interleave(num_registers) * num_registers] = True
            
            # Concatenate and scatter
            combined = torch.zeros(total_tokens_original + batch_size* num_registers, hidden_states.shape[-1], 
                                dtype=hidden_states.dtype, device=hidden_states.device)
            combined[register_mask] = registers_expanded
            combined[~register_mask] = hidden_states
            hidden_states = combined
            
            # Update cu_seqlens
            cu_seqlens_origin = cu_seqlens.clone()
            cu_seqlens = cu_seqlens + torch.arange(cu_seqlens.shape[0], dtype=cu_seqlens.dtype, device=cu_seqlens.device) * num_registers

            if positions is not None:
                positions_combined = torch.zeros(total_tokens_original + batch_size * num_registers, positions.shape[-1], dtype=positions.dtype, device=positions.device)
                positions_combined[~register_mask] = positions
                positions = positions_combined

            # also broadcast temb_modulation
            temb_modulation_combined = torch.zeros(total_tokens_original + batch_size * num_registers, 6, self.config.dim, dtype=temb_modulation.dtype, device=temb_modulation.device)
            pad_temb_modulation = temb_modulation[cu_seqlens_origin[:-1]].repeat_interleave(num_registers, dim=0)
            temb_modulation_combined[register_mask] = pad_temb_modulation
            temb_modulation_combined[~register_mask] = temb_modulation
            temb_modulation = temb_modulation_combined

        # Compute the flash-attn varlen maxima ONCE per forward (single GPU->CPU sync
        # each) rather than once per attention call. Identical across all layers since
        # every block shares cu_seqlens / cu_seqlens_encoder. Computed here (after the
        # register adjustment, which mutates cu_seqlens). Removes ~2 syncs/layer.
        max_seqlen = torch.diff(cu_seqlens).max().item()
        max_seqlen_encoder = (
            torch.diff(cu_seqlens_encoder).max().item()
            if cu_seqlens_encoder is not None else None
        )
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_modulation=temb_modulation,
                cu_seqlens=cu_seqlens,
                cu_seqlens_encoder=cu_seqlens_encoder,
                positions=positions,
                max_seqlen=max_seqlen,
                max_seqlen_encoder=max_seqlen_encoder,
            )
        
        if self.num_registers > 0:
            hidden_states = hidden_states[~register_mask]
            cu_seqlens = cu_seqlens_origin
        
        hidden_states = self.head(hidden_states, temb, cu_seqlens=cu_seqlens)
        return hidden_states

    def _attn_flops(self, s, s_enc, d, h, h_kv):
        qkv_proj = 2 * (s * d**2 + 2 * s_enc * d**2 * h_kv / h)
        attn = 4 * s * s_enc * d
        o_proj = 2 * s * d**2
        return qkv_proj + attn + o_proj
    
    def _ffn_flops(self, s, d, d_ffn):
        ffn = 6 * s * d * d_ffn
        return ffn

    def flops(self, cu_seqlens, s_enc) -> float:
        seqlen = cu_seqlens[1:] - cu_seqlens[:-1]
        if isinstance(s_enc, int):
            seqlen_enc = [s_enc] * len(seqlen)
        else:
            seqlen_enc = s_enc
        total = 0
        dim_freq = self.config.freq_dim
        dim = self.config.dim
        h = self.config.num_attention_heads
        dim_ffn = self.config.intermediate_size
        dim_cond = self.config.condition_dim
        dim_in = self.config.in_channels
        for s, s_enc in zip(seqlen, seqlen_enc):
            s = s.item()
            transformer = self.config.num_layers * (
                self._attn_flops(s, s_enc, dim, h, h) + 
                self._attn_flops(s, s, dim, h, h) + 
                self._ffn_flops(s, dim, dim_ffn)
            )

            time_embed_and_proj = (dim_freq * dim +  dim * dim + dim * dim * 6) * 2 * s
            image_proj = s_enc * (dim_cond * dim + dim * dim) * 2
            proj = s * (self.proj_embedding_dim + dim_in) * dim * 2
            head = s * dim_in * dim * 2
            total += transformer + time_embed_and_proj + image_proj + proj + head
        return total
    
    @torch.no_grad()
    def get_latents(self, encoder, input_dict, mean, std):
        if 'vertex_mask' in input_dict:
            enc = encoder.encode(
                pos=input_dict['nodes'],
                edge_index=input_dict['edges'].permute(1, 0),
                offsets=input_dict['encoder_cu_seqlens'],
                vertex_mask=input_dict['vertex_mask'],
                position=input_dict.get('encoder_position', None)
            )
            vertex_embed = enc['node_embed_mu'][input_dict['vertex_mask']].clone()
            latents = (vertex_embed - mean) / std
        else:
            enc = encoder.encode(
                pos=input_dict['nodes'],
                edge_index=input_dict['edges'].permute(1, 0),
                offsets=input_dict['encoder_cu_seqlens'],
                node_type=input_dict['node_type'],
                position=input_dict.get('encoder_position', None)
            )
            vertex_embed = enc['node_embed_mu'][input_dict['node_type'] == 0].clone()
            latents = (vertex_embed - mean) / std
        return latents, input_dict['cu_seqlens']
    
    def is_scalar_param(self, name, param):
        patterns = [
            'cond_proj',
            'view_embed',
            'quad_ratio_embed',
            'quad_ratio_uncond_embed',
            'symmetry_embed',
            'time_embedding',
            'time_projection',
            'registers',
            'head',
        ]
        for p in patterns:
            if p in name:
                return True
        if name.startswith('proj'):
            return True
        return False

    def train_step(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        cu_seqlens: torch.Tensor,
        conditions: dict,
        input_dict: dict,
        dtype: torch.dtype,
    ):
        model_pred = self(
            hidden_states=noisy_latents.to(dtype),
            timesteps=timesteps,
            cu_seqlens=cu_seqlens,
            encoder_hidden_states=conditions["encoder_hidden_states"].to(dtype),
            hidden_states_position=input_dict["vertices"],
            # hidden_states_position=torch.zeros(noisy_latents.shape[0], 3, dtype=torch.long, device=input_dict["vertices"].device),
            cu_seqlens_encoder=None,
            view_indices=conditions.get("view_indices"),
            mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
            quad_ratios=input_dict.get("quad_ratios"),
            symmetries=input_dict.get("symmetries"),
        )
        s_enc = conditions["encoder_hidden_states"].shape[1]
        if self.mv_mode and conditions.get("mv_cu_seqlens") is not None:
            mv_cu_seqlens = conditions["mv_cu_seqlens"]
            num_views = (mv_cu_seqlens[1:] - mv_cu_seqlens[:-1]).float().mean().item()
            s_enc = int(s_enc * num_views)
        if self.quad_ratio_condition:
            s_enc += 1
        if self.symmetry_condition:
            s_enc += 4
        flops = self.flops(cu_seqlens, s_enc)
        return model_pred, {
            "num_tokens": noisy_latents.shape[0],
            "num_flops": flops,
        }
