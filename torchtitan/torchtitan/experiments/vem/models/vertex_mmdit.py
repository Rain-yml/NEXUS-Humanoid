from typing import Optional

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from torchtitan.experiments.vem.models.dit import (
    Head,
    MLPProj,
    get_timestep_embedding,
)
from torchtitan.experiments.vem.models.mmdit import MMDiTBlock
from torchtitan.experiments.vem.models.transformer import (
    FP32LayerNorm,
    FrequencyPositionalEmbedding,
    RMSNorm,
    RotaryPosEmbed3D,
)
from torchtitan.tools.logging import logger


def merge_cu_seqlens_with_masks(cu_seqlens_a, cu_seqlens_b):
    device = cu_seqlens_a.device

    lens_a = cu_seqlens_a[1:] - cu_seqlens_a[:-1]
    lens_b = cu_seqlens_b[1:] - cu_seqlens_b[:-1]
    lens_m = lens_a + lens_b

    batch_size = lens_a.shape[0]
    total_tokens = lens_m.sum()

    cu_seqlens_m = torch.zeros(batch_size + 1, device=device, dtype=cu_seqlens_a.dtype)
    cu_seqlens_m[1:] = torch.cumsum(lens_m, dim=0)

    batch_idx_a = torch.repeat_interleave(torch.arange(batch_size, device=device), lens_a)
    batch_idx_b = torch.repeat_interleave(torch.arange(batch_size, device=device), lens_b)

    rel_pos_a = torch.arange(lens_a.sum(), device=device) - cu_seqlens_a[batch_idx_a]
    rel_pos_b = torch.arange(lens_b.sum(), device=device) - cu_seqlens_b[batch_idx_b]

    pos_a = cu_seqlens_m[batch_idx_a] + rel_pos_a
    pos_b = cu_seqlens_m[batch_idx_b] + lens_a[batch_idx_b] + rel_pos_b

    mask_a = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    mask_b = torch.zeros(total_tokens, dtype=torch.bool, device=device)

    mask_a[pos_a] = True
    mask_b[pos_b] = True

    return cu_seqlens_m, mask_a, mask_b


class SpaceMeshMMDiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["condition_embedder", "norm", "cond_proj"]
    _no_split_modules = ["MMDiTBlock"]
    _keep_in_fp32_modules = [
        "time_embedding",
        "time_projection",
        "c_time_projection",
        "modulation",
        "c_modulation",
        "norm1",
        "norm2",
        "c_norm1",
        "c_norm2",
    ]
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
        pos_embed_type: str = "fourier",
        max_seq_len_rope: int = 10000,
        rope_theta: float = 10000.0,
        num_registers: int = 0,
        ape_scale_div: float = 1.0,
        mv_mode: bool = False,
        num_mv_views: int = 4,
        attn_dtype: str = "bf16",
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        assert pos_embed_type in ["fourier", "rotary", "fourier_rotary"]
        assert condition_dim > 0, "SpaceMeshMMDiT requires condition tokens"

        self.dim = dim
        self.pos_embed_type = pos_embed_type
        self.ape_scale_div = ape_scale_div
        self.mv_mode = mv_mode

        self.cond_proj = MLPProj(condition_dim, dim)
        if mv_mode:
            self.view_embed = nn.Embedding(num_mv_views, dim)

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

        self.layers = nn.ModuleList(
            [
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
            ]
        )

        self.head = Head(dim, in_channels)

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )
        self.c_time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        self.num_registers = num_registers
        if self.num_registers > 0:
            self.registers = nn.Parameter(torch.zeros(self.num_registers, dim))

    def init_weights(self, buffer_device=None):
        def _init_norm(module):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
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
        for layer in self.layers:
            nn.init.zeros_(layer.modulation)
            nn.init.zeros_(layer.c_modulation)
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
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
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            state_dict = state_dict["ema"]["model"]
            if self.mv_mode and "view_embed.weight" not in state_dict:
                state_dict["view_embed.weight"] = torch.zeros_like(self.view_embed.weight)
            self._load_pretrained_to_model(self, state_dict, "self")

    def _load_pretrained_to_model(self, model, state_dict, model_name):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())

        if model_keys and state_dict_keys:
            sample_model_key = next(iter(model_keys))
            sample_state_key = next(iter(state_dict_keys))

            if sample_model_key.startswith("_orig_mod.") and not sample_state_key.startswith("_orig_mod."):
                state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
                logger.info(f"{model_name}: added _orig_mod. prefix to match compiled model")
            elif not sample_model_key.startswith("_orig_mod.") and sample_state_key.startswith("_orig_mod."):
                state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
                logger.info(f"{model_name}: removed _orig_mod. prefix from checkpoint")

        set_model_state_dict(
            model,
            model_state_dict=state_dict,
            options=StateDictOptions(full_state_dict=True, strict=True),
        )
        logger.info(f"{model_name}: loaded pretrained weights using set_model_state_dict")

    def _prepare_condition_stream(
        self,
        encoder_hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_encoder: Optional[torch.Tensor],
        view_indices: Optional[torch.Tensor],
        mv_cu_seqlens: Optional[torch.Tensor],
    ):
        batch_size = cu_seqlens.shape[0] - 1
        encoder_hidden_states = self.cond_proj(encoder_hidden_states)

        if self.mv_mode:
            assert view_indices is not None and mv_cu_seqlens is not None
            view_embed = self.view_embed(view_indices).unsqueeze(1).to(encoder_hidden_states.dtype)
            encoder_hidden_states = encoder_hidden_states + view_embed
            condition_blocks = [
                encoder_hidden_states[mv_cu_seqlens[i] : mv_cu_seqlens[i + 1]].reshape(-1, self.dim)
                for i in range(batch_size)
            ]
            condition_lens = [block.shape[0] for block in condition_blocks]
            encoder_hidden_states = torch.cat(condition_blocks, dim=0)
        else:
            if encoder_hidden_states.dim() == 3:
                assert cu_seqlens_encoder is None
                assert encoder_hidden_states.shape[0] == batch_size
                condition_lens = [encoder_hidden_states.shape[1]] * batch_size
                encoder_hidden_states = encoder_hidden_states.reshape(batch_size * condition_lens[0], -1)
            elif encoder_hidden_states.dim() == 2 and cu_seqlens_encoder is not None:
                assert cu_seqlens_encoder.shape[0] == batch_size + 1
                condition_lens = (cu_seqlens_encoder[1:] - cu_seqlens_encoder[:-1]).tolist()
            else:
                raise ValueError(
                    "Single-view SpaceMeshMMDiT expects encoder_hidden_states with shape "
                    "(B, T, C), or flattened (total_T, C) with cu_seqlens_encoder"
                )

        if cu_seqlens_encoder is not None and not self.mv_mode:
            cu_seqlens_cond = cu_seqlens_encoder.to(device=cu_seqlens.device, dtype=torch.int32)
        else:
            cu_seqlens_cond = torch.zeros(
                batch_size + 1,
                dtype=torch.int32,
                device=cu_seqlens.device,
            )
            cu_seqlens_cond[1:] = torch.cumsum(
                torch.tensor(condition_lens, dtype=torch.int32, device=cu_seqlens.device),
                dim=0,
            )
        return encoder_hidden_states, cu_seqlens_cond

    def _insert_registers(self, hidden_states, positions, temb_modulation, cu_seqlens):
        batch_size = cu_seqlens.shape[0] - 1
        num_registers = self.num_registers
        total_tokens_original = hidden_states.shape[0]

        register_mask = torch.zeros(
            total_tokens_original + batch_size * num_registers,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        register_indices = (
            cu_seqlens[:-1].unsqueeze(1)
            + torch.arange(num_registers, device=cu_seqlens.device)
        ).flatten()
        register_indices = register_indices + torch.arange(
            batch_size, device=cu_seqlens.device
        ).repeat_interleave(num_registers) * num_registers
        register_mask[register_indices] = True

        registers_expanded = self.registers.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.dim)
        hidden_states_with_registers = torch.zeros(
            total_tokens_original + batch_size * num_registers,
            hidden_states.shape[-1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_states_with_registers[register_mask] = registers_expanded
        hidden_states_with_registers[~register_mask] = hidden_states

        cu_seqlens_origin = cu_seqlens.clone()
        cu_seqlens = cu_seqlens + torch.arange(
            cu_seqlens.shape[0],
            dtype=cu_seqlens.dtype,
            device=cu_seqlens.device,
        ) * num_registers

        if positions is not None:
            positions_with_registers = torch.zeros(
                total_tokens_original + batch_size * num_registers,
                positions.shape[-1],
                dtype=positions.dtype,
                device=positions.device,
            )
            positions_with_registers[~register_mask] = positions
            positions = positions_with_registers

        temb_modulation_with_registers = torch.zeros(
            total_tokens_original + batch_size * num_registers,
            6,
            self.dim,
            dtype=temb_modulation.dtype,
            device=temb_modulation.device,
        )
        register_temb = temb_modulation[cu_seqlens_origin[:-1]].repeat_interleave(num_registers, dim=0)
        temb_modulation_with_registers[register_mask] = register_temb
        temb_modulation_with_registers[~register_mask] = temb_modulation

        return (
            hidden_states_with_registers,
            positions,
            temb_modulation_with_registers,
            cu_seqlens,
            cu_seqlens_origin,
            register_mask,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        hidden_states_position: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_encoder: Optional[torch.Tensor] = None,
        view_indices: Optional[torch.Tensor] = None,
        mv_cu_seqlens: Optional[torch.Tensor] = None,
    ):
        assert cu_seqlens is not None

        with torch.amp.autocast("cuda", dtype=torch.float32):
            temb = self.time_embedding(
                get_timestep_embedding(
                    timesteps=timesteps,
                    embedding_dim=self.config.freq_dim,
                )
            )
            temb_modulation = self.time_projection(temb).view(temb.shape[0], 6, self.dim)

        encoder_hidden_states, cu_seqlens_cond = self._prepare_condition_stream(
            encoder_hidden_states=encoder_hidden_states,
            cu_seqlens=cu_seqlens,
            cu_seqlens_encoder=cu_seqlens_encoder,
            view_indices=view_indices,
            mv_cu_seqlens=mv_cu_seqlens,
        )

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

        cu_seqlens_x = cu_seqlens
        cu_seqlens_origin = None
        register_mask = None
        if self.num_registers > 0:
            (
                hidden_states,
                positions,
                temb_modulation,
                cu_seqlens_x,
                cu_seqlens_origin,
                register_mask,
            ) = self._insert_registers(hidden_states, positions, temb_modulation, cu_seqlens)

        with torch.amp.autocast("cuda", dtype=torch.float32):
            sample_temb = temb[cu_seqlens[:-1]]
            c_temb = self.c_time_projection(sample_temb).view(cu_seqlens.shape[0] - 1, 6, self.dim)
            c_temb_modulation = c_temb.repeat_interleave(torch.diff(cu_seqlens_cond), dim=0)

        cu_seqlens_all, mask_x, mask_cond = merge_cu_seqlens_with_masks(cu_seqlens_x, cu_seqlens_cond)
        x_cond = torch.zeros(cu_seqlens_all[-1], self.dim, device=hidden_states.device, dtype=hidden_states.dtype)
        x_cond[mask_x] = hidden_states
        x_cond[mask_cond] = encoder_hidden_states

        temb_modulation_concat = torch.zeros(
            cu_seqlens_all[-1],
            6,
            self.dim,
            device=temb_modulation.device,
            dtype=temb_modulation.dtype,
        )
        temb_modulation_concat[mask_x] = temb_modulation
        temb_modulation_concat[mask_cond] = c_temb_modulation

        for layer in self.layers:
            x_cond = layer(
                hidden_states_cat=x_cond,
                temb_modulation_cat=temb_modulation_concat,
                cu_seqlens=cu_seqlens_all,
                mask_x=mask_x,
                mask_c=mask_cond,
                positions=positions,
            )

        hidden_states = x_cond[mask_x]
        if self.num_registers > 0:
            hidden_states = hidden_states[~register_mask]
            cu_seqlens_x = cu_seqlens_origin

        hidden_states = self.head(hidden_states, temb, cu_seqlens=cu_seqlens_x)
        return hidden_states

    def _attn_flops(self, s, s_enc, d):
        qkv_proj = 6 * (s + s_enc) * d**2
        attn = 4 * (s + s_enc) ** 2 * d
        o_proj = 2 * (s + s_enc) * d**2
        return qkv_proj + attn + o_proj

    def _ffn_flops(self, s, s_enc, d, d_ffn):
        return 6 * (s + s_enc) * d * d_ffn

    def flops(self, cu_seqlens, s_enc) -> float:
        seqlen = cu_seqlens[1:] - cu_seqlens[:-1]
        if isinstance(s_enc, int):
            seqlen_enc = [s_enc] * len(seqlen)
        else:
            seqlen_enc = s_enc
        total = 0
        dim_freq = self.config.freq_dim
        dim = self.config.dim
        dim_ffn = self.config.intermediate_size
        dim_cond = self.config.condition_dim
        dim_in = self.config.in_channels
        for s, s_enc_i in zip(seqlen, seqlen_enc):
            s = s.item()
            transformer = self.config.num_layers * (
                self._attn_flops(s, s_enc_i, dim)
                + self._ffn_flops(s, s_enc_i, dim, dim_ffn)
            )
            time_embed_and_proj = (dim_freq * dim + dim * dim + dim * dim * 12) * 2 * s
            image_proj = s_enc_i * (dim_cond * dim + dim * dim) * 2
            proj = s * (self.proj_embedding_dim + dim_in) * dim * 2
            head = s * dim_in * dim * 2
            total += transformer + time_embed_and_proj + image_proj + proj + head
        return total

    @torch.no_grad()
    def get_latents(self, encoder, input_dict, mean, std):
        enc = encoder.encode(
            pos=input_dict["nodes"],
            edge_index=input_dict["edges"].permute(1, 0),
            offsets=input_dict["encoder_cu_seqlens"],
            vertex_mask=input_dict["vertex_mask"],
            position=input_dict.get("encoder_position", None),
        )
        vertex_embed = enc["node_embed_mu"][input_dict["vertex_mask"]].clone()
        latents = (vertex_embed - mean) / std
        return latents, input_dict["cu_seqlens"]

    def is_scalar_param(self, name, param):
        patterns = [
            "cond_proj",
            "view_embed",
            "time_embedding",
            "time_projection",
            "c_time_projection",
            "registers",
            "head",
        ]
        for pattern in patterns:
            if pattern in name:
                return True
        if name.startswith("proj"):
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
            cu_seqlens_encoder=None,
            view_indices=conditions.get("view_indices"),
            mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
        )
        if self.mv_mode and conditions.get("mv_cu_seqlens") is not None:
            mv_cu_seqlens = conditions["mv_cu_seqlens"]
            num_views = mv_cu_seqlens[1:] - mv_cu_seqlens[:-1]
            s_enc = (conditions["encoder_hidden_states"].shape[1] * num_views).tolist()
        else:
            s_enc = conditions["encoder_hidden_states"].shape[1]
        flops = self.flops(cu_seqlens, s_enc)
        return model_pred, {
            "num_tokens": noisy_latents.shape[0],
            "num_flops": flops,
        }
