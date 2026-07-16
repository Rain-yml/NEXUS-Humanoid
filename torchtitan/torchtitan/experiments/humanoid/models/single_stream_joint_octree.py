"""Single-stream NEXUS octree model with persistent semantic-joint tokens.

This owned copy keeps the NEXUS transformer path intact. Mesh and joint tokens
share ordinary 3D-RoPE self-attention; training decides which tokens are clean
conditions and which tokens are diffusion targets.
"""
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
from torchtitan.experiments.humanoid.data.dataset import JointOctreeBatch
from torchtitan.tools.logging import logger

logger_diffusers = logging.get_logger(__name__)


def _to_int_list(values) -> List[int]:
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.detach().cpu().tolist()]
    return [int(v) for v in values]


def get_octree_condition_lens(
    conditions: Dict[str, torch.Tensor],
    input_dict: JointOctreeBatch,
    batch_size: int,
    num_vertex_condition: bool,
    quad_ratio_condition: bool = False,
    symmetry_condition: bool = False,
) -> List[int]:
    """Return per-expanded-octree-sample condition token counts."""
    encoder_hidden_states = conditions["encoder_hidden_states"]
    condition_cu_seqlens = conditions.get("condition_cu_seqlens")
    mv_cu_seqlens = conditions.get("mv_cu_seqlens")

    if input_dict.num_layers_per_mesh is not None and len(input_dict.num_layers_per_mesh) > 0:
        num_layers_per_mesh = _to_int_list(input_dict.num_layers_per_mesh)
        num_unique_meshes = len(num_layers_per_mesh)
    else:
        num_unique_meshes = batch_size
        num_layers_per_mesh = [1] * num_unique_meshes

    if mv_cu_seqlens is not None:
        mv_offsets = _to_int_list(mv_cu_seqlens)
        if condition_cu_seqlens is not None:
            condition_offsets = _to_int_list(condition_cu_seqlens)
            mesh_lens = [
                condition_offsets[mv_offsets[i + 1]] - condition_offsets[mv_offsets[i]]
                for i in range(num_unique_meshes)
            ]
        else:
            tokens_per_view = int(encoder_hidden_states.shape[1])
            mesh_lens = [
                (mv_offsets[i + 1] - mv_offsets[i]) * tokens_per_view
                for i in range(num_unique_meshes)
            ]
    else:
        if condition_cu_seqlens is not None:
            condition_offsets = _to_int_list(condition_cu_seqlens)
            mesh_lens = [
                condition_offsets[i + 1] - condition_offsets[i]
                for i in range(num_unique_meshes)
            ]
        else:
            mesh_lens = [int(encoder_hidden_states.shape[1])] * num_unique_meshes

    if num_vertex_condition:
        mesh_lens = [n + 1 for n in mesh_lens]
    if quad_ratio_condition:
        mesh_lens = [n + 1 for n in mesh_lens]
    if symmetry_condition:
        # 4 symmetry tokens per mesh (x=0 / y=0 / z=0 / any-of-xyz)
        mesh_lens = [n + 4 for n in mesh_lens]

    condition_lens: List[int] = []
    for n_tokens, n_layers in zip(mesh_lens, num_layers_per_mesh):
        condition_lens.extend([n_tokens] * n_layers)

    if len(condition_lens) != batch_size:
        raise ValueError(
            f"Expected {batch_size} condition lengths, got {len(condition_lens)}. "
            f"num_layers_per_mesh={num_layers_per_mesh}"
        )
    return condition_lens


def get_octree_condition_proj_tokens(conditions: Dict[str, torch.Tensor]) -> int:
    """Return the number of image condition tokens passed through cond_proj."""
    encoder_hidden_states = conditions["encoder_hidden_states"]
    if conditions.get("condition_cu_seqlens") is not None:
        return int(encoder_hidden_states.shape[0])
    return int(encoder_hidden_states.shape[0] * encoder_hidden_states.shape[1])


class OctreeDiTBlock(nn.Module):
    """
    Octree DiT Block with optional 3D RoPE for self-attention
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
        cross_attn_norm: bool = False,
        contain_cross_attention: bool = True,
        # 3D RoPE
        rope_3d: Optional[RotaryPosEmbed3D] = None,
        # use_3d_rope: bool = True,
        # max_seq_len_3d: int = 10000,
        # grid_size_3d: int = 128,
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
        self.contain_cross_attention = contain_cross_attention
        # self.use_3d_rope = use_3d_rope

        self.norm1 = FP32LayerNorm(hidden_size, elementwise_affine=False)

        # Self attention with 3D RoPE
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
            # 3D RoPE 参数
            # use_3d_rope=use_3d_rope,
            # max_seq_len_3d=max_seq_len_3d,
            # grid_size_3d=grid_size_3d,
        )

        if self.contain_cross_attention:
            self.norm3 = FP32LayerNorm(hidden_size, elementwise_affine=True) if cross_attn_norm else nn.Identity()

            # Cross attention 不使用 RoPE
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
                # use_3d_rope=False,
            )

        self.norm2 = FP32LayerNorm(hidden_size, elementwise_affine=False)

        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        # AdaLN modulation - 可学习参数
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_size) / hidden_size ** 0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb_modulation: torch.Tensor,
        positions: Optional[torch.Tensor] = None,  # 3D 位置 (total_len, 3) 离散化整数坐标，用于 3D RoPE
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_encoder: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        max_seqlen_encoder: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (total_len, C) or (B, L, C)
            encoder_hidden_states: (total_kv_len, C) or (B, L_kv, C)
            temb_modulation: (B, 6, C) or (total_len, 6, C) for varlen
            positions: (total_len, 3) 离散化整数坐标 (dtype=long)，用于 3D RoPE
            cu_seqlens: (B+1,) 用于 varlen attention
            cu_seqlens_encoder: (B+1,) 用于 cross attention varlen
        """
        # Compute modulation in float32 for numerical stability
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = self.modulation + temb_modulation
            if cu_seqlens is None:
                shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = modulation.chunk(6, dim=1)
            else:
                shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = modulation.unbind(dim=1)

        dtype = hidden_states.dtype

        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=((self.norm1(hidden_states).float() * (1 + scale_msa) + shift_msa).to(dtype)),
            cu_seqlens_q=cu_seqlens,
            position_ids=positions,  # 传入 3D 位置
            max_seqlen_q=max_seqlen,
        )
        with amp.autocast(dtype=torch.float32):
            hidden_states = hidden_states * gate_msa + residual

        # Cross attention (不使用 RoPE)
        if self.contain_cross_attention:
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



class SinusoidalEmbedding(nn.Module):
    """时间步的正弦位置编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        dim_out,
        scalar_min=24,
        scalar_max=10000,
        scalar_shift=1024,
        scalar_mapping: Literal["log_shift_normalize", "identity", "bins5"] = "log_shift_normalize",
    ):
        super().__init__()
        assert scalar_mapping in ["log_shift_normalize", "identity", "bins5"]
        self.dim_out = dim_out
        self.linear = nn.Sequential(
            nn.Linear(1, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )
        self.scalar_min = scalar_min
        self.scalar_max = scalar_max
        self.scalar_mapping = scalar_mapping
        self.scalar_shift = scalar_shift

    def forward(self, scalar):
        if self.scalar_mapping == "log_shift_normalize":
            min_f, max_f = self.scalar_min, self.scalar_max
            shift = self.scalar_shift
            mean = (math.log(min_f + shift) + math.log(max_f + shift)) * 0.5
            std = (math.log(max_f + shift) - math.log(min_f + shift)) * 0.5
            x = (torch.log(scalar.float() + shift) - mean) / std
        elif self.scalar_mapping == "identity":
            x = scalar.float()
        elif self.scalar_mapping == "bins5":
            x = torch.clamp((scalar.float() * 5).floor(), min=0, max=4) / 5
        else:
            raise NotImplementedError(f"Unsupported scalar mapping: {self.scalar_mapping}")

        bs = scalar.shape[0]
        dtype = self.linear[0].weight.dtype
        return self.linear(x.view(bs, 1, 1).to(dtype=dtype))

class SingleStreamJointOctreeDiffusionModel(ModelMixin, ConfigMixin):
    """
    八叉树逐层扩散模型

    训练目标：
    - 输入：父节点中心 (centers) + 深度 (depths) + 子节点ID (child_ids) + noisy 占用状态 (x_t)
    - 输出：预测 8 个子节点的占用状态 (8-bit 或 256-dim one-hot)

    支持两种位置编码方式:
    - pos_embed_type='fourier': 使用 FourierEmbedding3D
    - pos_embed_type='rotary': 使用 3D RoPE (在 attention 中处理)

    支持多种条件类型:
    - condition_type='pointcloud': 仅使用点云条件
    - condition_type='normal': 仅使用 normal 图像条件 (DINOv3)
    - condition_type='both': 同时使用点云和图像条件
    """
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["condition_embedder", "norm", "cond_proj"]
    _no_split_modules = ["OctreeDiTBlock"]
    _keep_in_fp32_modules = ["time_embedder", "modulation", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = []

    @register_to_config
    def __init__(
        self,
        # 表示模式
        use_onehot_256: bool = True,
        # Transformer 架构
        num_layers: int = 12,
        dim: int = 512,
        freq_dim: int = 256,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        num_key_value_heads: Optional[int] = None,
        # Attention 配置
        attention_bias: bool = False,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        use_flash_attn_3: bool = False,
        # 位置编码方式
        pos_embed_type: str = 'rotary',
        # 3D RoPE 配置
        use_3d_rope: bool = True,
        max_seq_len_3d: int = 10000,
        grid_size: int = 128,
        rope_theta: int = 10000,
        # Fourier 位置编码
        num_freqs: int = 8,
        # 八叉树配置
        max_depth: int = 7,
        # Cross attention
        contain_cross_attention: bool = True,
        # # PointCloudEncoder 配置
        # point_encoder_M: int = 2048,
        # point_encoder_layers: int = 4,
        # # VAE Bottleneck 配置
        # point_encoder_bottleneck_dim: int = 0,  # 0 表示不使用
        # point_encoder_use_vae: bool = False,
        # kl_weight: float = 1e-5,
        # vae_sample_warmup_steps: int = 0,
        # # 图像条件配置 (图像编码器在 trainer 中初始化)
        # condition_type: str = 'pointcloud',
        image_hidden_size: int = 1280,  # DINOv3-ViT-H/16+ 的 hidden_size
        num_vertex_condition: bool = False,
        quad_ratio_condition: bool = False,
        quad_ratio_uncond: bool = False,
        symmetry_condition: bool = False,
        # Multiview conditioning
        mv_mode: bool = False,    # Enable multiview conditioning
        num_mv_views: int = 4,    # Number of distinct view types (0=front,1=left,2=back,3=right)
        pretrained_path: Optional[str] = None,
        num_joint_tokens: int = 28,
    ):
        super().__init__()
        assert pos_embed_type in ['fourier', 'rotary']
        assert (pos_embed_type == 'rotary') == use_3d_rope

        self.dim = dim
        self.max_depth = max_depth
        # self.point_encoder_M = point_encoder_M
        self.grid_size = grid_size
        self.use_onehot_256 = use_onehot_256
        self.pos_embed_type = pos_embed_type
        self.use_3d_rope = use_3d_rope and (pos_embed_type == 'rotary')
        # self.condition_type = condition_type

        # VAE Bottleneck 配置
        # self.point_encoder_bottleneck_dim = point_encoder_bottleneck_dim
        # self.point_encoder_use_vae = point_encoder_use_vae
        # self.kl_weight = kl_weight
        # self.vae_sample_warmup_steps = int(vae_sample_warmup_steps)

        # 输入/输出维度
        self.token_dim = 256 if use_onehot_256 else 8

        # ==================== 输入编码 ====================
        # 1. 父节点中心位置编码 (仅 Fourier 模式使用)
        if pos_embed_type == 'fourier':
            self.center_embed = FourierEmbedding3D(dim, num_freqs=num_freqs, grid_size=grid_size)
        else:
            # Rotary 模式下，位置编码在 attention 中通过 3D RoPE 处理
            self.center_embed = None

        # 2. 深度编码
        self.depth_embed = nn.Embedding(32, dim)

        # 3. 子节点 ID 编码 (0-7)
        # self.child_id_embed = nn.Embedding(32, dim)

        # 4. Noisy 占用状态投影
        self.noise_proj = nn.Linear(self.token_dim, dim)
        self.joint_embed = nn.Embedding(num_joint_tokens, dim)
        self.joint_type_embed = nn.Embedding(1, dim)

        # ==================== 时间编码 ====================
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(freq_dim),
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # 时间调制投影 (用于 AdaLN)
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        self.cond_proj = MLPProj(image_hidden_size, dim)
        if pos_embed_type == 'rotary':
            self.rope_3d = RotaryPosEmbed3D(
                attention_head_dim=dim // num_attention_heads,
                max_seq_len=max_seq_len_3d,
                # grid_size=grid_size,
                theta=rope_theta,
            )
        else:
            self.rope_3d = None

        # ==================== Transformer Blocks ====================
        self.layers = nn.ModuleList([
            OctreeDiTBlock(
                layer_idx=i,
                hidden_size=dim,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_key_value_heads=num_key_value_heads,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
                attention_bias=attention_bias,
                use_flash_attn_3=use_flash_attn_3,
                contain_cross_attention=contain_cross_attention,
                rope_3d=self.rope_3d,
            )
            for i in range(num_layers)
        ])

        self.head = Head(dim, self.token_dim)

        # ==================== 输出 ====================
        # self.final_norm = RMSNorm(dim)
        # self.output_proj = nn.Linear(dim, self.token_dim)

        # Store config for time embedding
        self._freq_dim = freq_dim
        self.num_vertex_condition = num_vertex_condition
        if num_vertex_condition:
            self.num_embed = MLPEncoder(dim)
        self.quad_ratio_condition = quad_ratio_condition
        if quad_ratio_condition:
            self.quad_ratio_embed = MLPEncoder(dim, scalar_mapping="identity")
            # Unconditional embedding used when quad_ratio < 0 (unknown quad ratio).
            # Only created/used when quad_ratio_uncond is enabled.
            self.quad_ratio_uncond_enabled = quad_ratio_uncond
            if quad_ratio_uncond:
                self.quad_ratio_uncond_embed = nn.Embedding(1, dim)
        else:
            self.quad_ratio_uncond_enabled = False

        self.symmetry_condition = symmetry_condition
        if symmetry_condition:
            # 4 directions (x=0 / y=0 / z=0 / any-of-xyz) x 2 states (symmetric / uncertain)
            # flat index = direction * 2 + state, state: 0=symmetric, 1=uncertain
            self.symmetry_embed = nn.Embedding(8, dim)

        self.mv_mode = mv_mode
        if mv_mode:
            self.view_embed = nn.Embedding(num_mv_views, dim)

    def init_weights(self, buffer_device=None):
        """初始化权重"""
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

        # Zero init modulation
        for layer in self.layers:
            nn.init.zeros_(layer.modulation)

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
            # New semantic embeddings are the only parameters absent from NEXUS.
            state_dict["joint_embed.weight"] = self.joint_embed.weight.detach().cpu()
            state_dict["joint_type_embed.weight"] = self.joint_type_embed.weight.detach().cpu()
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

    def is_scalar_param(self, name, param):
        patterns = [
            'num_embed',
            'time_embed',
            'time_projection',
            'noise_proj',
            'depth_embed',
            'center_embed',
            'head',
            'cond_proj',
            'view_embed',
            'quad_ratio_embed',
            'quad_ratio_uncond_embed',
            'symmetry_embed',
            'joint_embed',
            'joint_type_embed',
        ]
        for p in patterns:
            if p in name:
                return True
        return False

    def _embed_quad_ratio(self, quad_ratios: torch.Tensor) -> torch.Tensor:
        """Embed quad ratios, replacing negative entries with an uncond embedding.

        A negative quad_ratio means "quad ratio unknown": those entries use the
        learned unconditional embedding instead of the MLPEncoder output.

        Returns: (num_meshes, 1, dim)
        """
        with torch.amp.autocast('cuda', dtype=torch.float32):
            # Clamp negatives to 0 so the MLPEncoder never sees an out-of-range value;
            # those entries are overwritten by the uncond embedding below.
            qr_clamped = quad_ratios.clamp(min=0.0)
            qr_embed = self.quad_ratio_embed(qr_clamped)  # (num_meshes, 1, dim)
            if self.quad_ratio_uncond_enabled:
                # Use torch.where unconditionally (no .any() guard) to avoid a GPU->CPU sync.
                mask = (quad_ratios < 0).view(-1, 1, 1).to(qr_embed.device)
                uncond = self.quad_ratio_uncond_embed.weight.view(1, 1, -1).to(qr_embed.dtype)
                qr_embed = torch.where(mask, uncond, qr_embed)
        return qr_embed

    def _embed_symmetries(
        self,
        symmetries: Optional[torch.Tensor],
        num_unique_meshes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Embed the 4 symmetry-direction tokens per mesh.

        symmetries: (num_meshes, 4) int64 with values in {1=symmetric, 0=uncertain}
            for directions [x=0, y=0, z=0, any-of-xyz]. When None, defaults to
            all-uncertain.

        Returns: (num_meshes, 4, dim)
        """
        if symmetries is None:
            symmetries = torch.zeros(num_unique_meshes, 4, dtype=torch.long, device=device)
        else:
            symmetries = symmetries.to(device=device, dtype=torch.long)
        # flat index = direction * 2 + state, state: 1=symmetric, 0=uncertain
        directions = torch.arange(4, device=device, dtype=torch.long).view(1, 4)
        indices = directions * 2 + symmetries  # (num_meshes, 4)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            sym_embed = self.symmetry_embed(indices)  # (num_meshes, 4, dim)
        return sym_embed

    def forward(
        self,
        x_t: torch.Tensor,                   # (total_nodes, token_dim) noisy 占用状态
        t: torch.Tensor,                     # (total_nodes,) 或 (batch_size,) 时间步
        centers: torch.Tensor,               # (total_nodes, 3) 父节点中心（离散化整数坐标，dtype=long，[0, grid_size]）
        depths: torch.Tensor,                # (total_nodes,) 深度
        # child_ids: torch.Tensor,             # (total_nodes,) 子节点 ID (0-7)
        cu_seqlens_q: torch.Tensor,          # (batch_size + 1,)
        num_layers_per_mesh: list = None,    # 每个 mesh 的层数（用于避免 point_encoder 重复计算）
        encoder_hidden_states: torch.Tensor = None,  # mv_mode=False: (num_unique_meshes, num_tokens, hidden_dim); mv_mode=True: (total_views, num_tokens, hidden_dim)
        condition_cu_seqlens: torch.Tensor = None,  # optional (num_conditions + 1,) for varlen condition tokens
        num_vertices: torch.Tensor = None,   # (num_unique_meshes,) 每个 mesh 的顶点数
        quad_ratios: torch.Tensor = None,   # (num_unique_meshes,) 每个 mesh 的 quad ratio
        symmetries: torch.Tensor = None,    # (num_unique_meshes, 4) int64, 1=symmetric 0=uncertain
        view_indices: torch.Tensor = None,   # (total_views,) int64, view id 0-3; only used when mv_mode=True
        mv_cu_seqlens: torch.Tensor = None,  # (num_unique_meshes+1,) int32, cumulative view counts; only used when mv_mode=True
        joint_ids: torch.Tensor = None,
        joint_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向传播 - Varlen 模式

        Args:
            x_t: noisy 占用状态 (256-dim one-hot 或 8-bit)
            t: 扩散时间步 [0, 1000]
            centers: 父节点中心坐标（离散化整数坐标，dtype=long，[0, grid_size]），rotary 模式下用于 3D RoPE 索引
            depths: 节点所在深度
            child_ids: 节点在父节点中的子节点 ID
            surface_points: 表面点云 (xyz + normal)，xyz 为 mesh 坐标系（通常 [-1, 1]），会通过 point_encoder 编码
            cu_seqlens_q: Q 的累积序列长度
            cu_seqlens_surface: surface_points 的累积序列长度
            num_layers_per_mesh: 每个 mesh 的层数（训练时使用，避免 point_encoder 重复计算）
            encoder_hidden_states: 预编码的图像特征 (来自 trainer 的 prepare_conditions)

        Returns:
            预测的速度场 v (total_nodes, token_dim)
        """
        total_nodes = x_t.shape[0]
        batch_size = cu_seqlens_q.shape[0] - 1

        # ==================== 时间编码 (float32 保证数值稳定) ====================
        with torch.amp.autocast('cuda', dtype=torch.float32):
            # 如果 t 是 (batch_size,)，需要扩展到每个 token
            # if t.shape[0] != total_nodes:
            #     t_expanded = []
            #     for i in range(batch_size):
            #         start, end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            #         t_expanded.append(t[i].expand(end - start))
            #     t = torch.cat(t_expanded)

            t_emb = self.time_embed(t)  # (total_nodes, dim)
            temb_modulation = self.time_projection(t_emb).view(total_nodes, 6, self.dim)

        # ==================== 条件编码 ====================
        # cast_to_bf16 = (x_t.dtype == torch.bfloat16)
        condition_list = []  # 收集所有条件

        # 确定唯一 mesh 数量 (用于图像条件)
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

        # Project image features: works for both single-view and mv flat formats
        encoder_hidden_states = self.cond_proj(encoder_hidden_states)  # (..., num_tokens, dim)
        is_varlen_condition = condition_cu_seqlens is not None

        assert cu_seqlens_q is not None

        if self.mv_mode:
            assert view_indices is not None and mv_cu_seqlens is not None
            if is_varlen_condition:
                # mv_mode varlen: encoder_hidden_states is (total_selected_tokens, dim)
                view_lens = torch.diff(condition_cu_seqlens)
                ve = self.view_embed(view_indices).to(encoder_hidden_states.dtype)
                encoder_hidden_states = encoder_hidden_states + ve.repeat_interleave(view_lens, dim=0)
                view_groups = [
                    encoder_hidden_states[condition_cu_seqlens[mv_cu_seqlens[i]]:condition_cu_seqlens[mv_cu_seqlens[i + 1]]]
                    for i in range(num_unique_meshes)
                ]
            else:
                # mv_mode fixed: encoder_hidden_states is (total_views, num_tokens, dim)
                ve = self.view_embed(view_indices).unsqueeze(1).to(encoder_hidden_states.dtype)  # (total_views, 1, dim)
                encoder_hidden_states = encoder_hidden_states + ve  # (total_views, num_tokens, dim)
                view_groups = [
                    encoder_hidden_states[mv_cu_seqlens[i]:mv_cu_seqlens[i + 1]].reshape(-1, self.dim)
                    for i in range(num_unique_meshes)
                ]  # list of (V_i * num_tokens, dim)

            # Optionally append vertex count token per mesh
            if self.num_vertex_condition:
                assert num_vertices is not None
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    nv_embed = self.num_embed(num_vertices)  # (num_unique_meshes, 1, dim)
                view_groups = [
                    torch.cat([vg, nv.reshape(1, -1).to(vg.dtype)], dim=0)
                    for vg, nv in zip(view_groups, nv_embed)
                ]
            if self.quad_ratio_condition:
                assert quad_ratios is not None
                qr_embed = self._embed_quad_ratio(quad_ratios)  # (num_unique_meshes, 1, dim)
                view_groups = [
                    torch.cat([vg, qr.reshape(1, -1).to(vg.dtype)], dim=0)
                    for vg, qr in zip(view_groups, qr_embed)
                ]
            if self.symmetry_condition:
                sym_embed = self._embed_symmetries(
                    symmetries, num_unique_meshes, cu_seqlens_q.device
                )  # (num_unique_meshes, 4, dim)
                view_groups = [
                    torch.cat([vg, sym.to(vg.dtype)], dim=0)
                    for vg, sym in zip(view_groups, sym_embed)
                ]

            # Repeat each mesh's KV sequence for each of its octree layers, then build variable cu_seqlens_kv
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
                # Single-view varlen: encoder_hidden_states is (total_selected_tokens, dim)
                if self.num_vertex_condition:
                    assert num_vertices is not None
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        nv_embed = self.num_embed(num_vertices)
                if self.quad_ratio_condition:
                    assert quad_ratios is not None
                    qr_embed = self._embed_quad_ratio(quad_ratios)
                if self.symmetry_condition:
                    sym_embed = self._embed_symmetries(
                        symmetries, num_unique_meshes, cu_seqlens_q.device
                    )
                encoder_hs_list = []
                kv_lens = []
                for i, n_layers in enumerate(num_layers_per_mesh):
                    hs = encoder_hidden_states[condition_cu_seqlens[i]:condition_cu_seqlens[i + 1]]
                    if self.num_vertex_condition:
                        hs = torch.cat([hs, nv_embed[i].to(hs.dtype)], dim=0)
                    if self.quad_ratio_condition:
                        hs = torch.cat([hs, qr_embed[i].to(hs.dtype)], dim=0)
                    if self.symmetry_condition:
                        hs = torch.cat([hs, sym_embed[i].to(hs.dtype)], dim=0)
                    for _ in range(n_layers):
                        encoder_hs_list.append(hs)
                        kv_lens.append(hs.shape[0])
                encoder_hidden_states = torch.cat(encoder_hs_list, dim=0)
                cu_seqlens_kv = torch.zeros(len(kv_lens) + 1, dtype=torch.int32, device=cu_seqlens_q.device)
                cu_seqlens_kv[1:] = torch.cumsum(
                    torch.tensor(kv_lens, dtype=torch.int32, device=cu_seqlens_q.device), dim=0
                )
            else:
                # Standard single-view path: encoder_hidden_states is (num_unique_meshes, num_tokens, dim)
                if self.num_vertex_condition:
                    assert num_vertices is not None
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        nv_embed = self.num_embed(num_vertices)
                    assert nv_embed.dim() == 3
                    assert encoder_hidden_states.dim() == 3
                    encoder_hidden_states = torch.cat([encoder_hidden_states, nv_embed.to(encoder_hidden_states.dtype)], dim=1)
                if self.quad_ratio_condition:
                    assert quad_ratios is not None
                    qr_embed = self._embed_quad_ratio(quad_ratios)
                    assert qr_embed.dim() == 3
                    assert encoder_hidden_states.dim() == 3
                    encoder_hidden_states = torch.cat([encoder_hidden_states, qr_embed.to(encoder_hidden_states.dtype)], dim=1)
                if self.symmetry_condition:
                    sym_embed = self._embed_symmetries(
                        symmetries, num_unique_meshes, cu_seqlens_q.device
                    )
                    assert sym_embed.dim() == 3
                    assert encoder_hidden_states.dim() == 3
                    encoder_hidden_states = torch.cat([encoder_hidden_states, sym_embed.to(encoder_hidden_states.dtype)], dim=1)

                repeat_counts = torch.tensor(num_layers_per_mesh, device=encoder_hidden_states.device)
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeat_counts, dim=0)

                # 转换为 varlen 格式: (B, M_total, dim) -> (B * M_total, dim)
                num_condition_tokens = encoder_hidden_states.shape[1]
                encoder_hidden_states = encoder_hidden_states.reshape(batch_size * num_condition_tokens, -1)

                # 构建 condition 的 cu_seqlens (uniform stride)
                cu_seqlens_kv = torch.arange(
                    0, (batch_size + 1) * num_condition_tokens, num_condition_tokens,
                    device=cu_seqlens_q.device, dtype=torch.int32,
                )

        noise_emb = self.noise_proj(x_t.to(self.noise_proj.weight.dtype))  # (total_nodes, dim)
        depth_emb = self.depth_embed(depths.clamp(0, self.max_depth)).to(noise_emb.dtype)  # (total_nodes, dim)

        if self.pos_embed_type == 'fourier':
            center_emb = self.center_embed(centers).to(noise_emb.dtype)  # (total_nodes, dim)
        else:
            # Rotary 模式：位置编码在 attention 层通过 3D RoPE 处理
            center_emb = torch.zeros(total_nodes, self.dim, device=noise_emb.device, dtype=noise_emb.dtype)

        # 组合
        x = center_emb + depth_emb + noise_emb
        if joint_ids is not None:
            if joint_mask is None:
                joint_mask = joint_ids >= 0
            safe_joint_ids = joint_ids.clamp_min(0)
            semantic = self.joint_embed(safe_joint_ids)
            token_type = self.joint_type_embed.weight[0].view(1, -1)
            x = x + (semantic + token_type).to(x.dtype) * joint_mask.unsqueeze(-1).to(x.dtype)

        # ==================== Transformer Blocks ====================
        # Compute the flash-attn varlen maxima ONCE per forward (single GPU->CPU sync
        # each) rather than once per attention call. These are identical across all
        # layers since every block shares cu_seqlens_q / cu_seqlens_kv. This removes
        # ~2 syncs/layer * num_layers (~96/step for a 24-layer model) from the hot loop.
        max_seqlen_q = torch.diff(cu_seqlens_q).max().item()
        max_seqlen_kv = (
            torch.diff(cu_seqlens_kv).max().item() if cu_seqlens_kv is not None else None
        )
        for block in self.layers:
            x = block(
                x,
                encoder_hidden_states=encoder_hidden_states,
                temb_modulation=temb_modulation,
                positions=centers if self.use_3d_rope else None,
                cu_seqlens=cu_seqlens_q,
                cu_seqlens_encoder=cu_seqlens_kv,
                max_seqlen=max_seqlen_q,
                max_seqlen_encoder=max_seqlen_kv,
            )

        # ==================== 输出 ====================
        x = self.head(x, t_emb, cu_seqlens=cu_seqlens_q)

        return x

    def train_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        cu_seqlens: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        input_dict: JointOctreeBatch,
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
            symmetries=input_dict.symmetries,
            view_indices=conditions.get("view_indices"),
            mv_cu_seqlens=conditions.get("mv_cu_seqlens"),
            joint_ids=input_dict.joint_ids_flat,
            joint_mask=input_dict.joint_mask_flat,
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
                    symmetry_condition=self.symmetry_condition,
                ),
                get_octree_condition_proj_tokens(conditions),
            ),
        }


    # def forward_with_cfg(
    #     self,
    #     x_t: torch.Tensor,
    #     t: torch.Tensor,
    #     centers: torch.Tensor,
    #     depths: torch.Tensor,
    #     child_ids: torch.Tensor,
    #     surface_points: torch.Tensor,
    #     cu_seqlens_q: torch.Tensor,
    #     cu_seqlens_surface: torch.Tensor,
    #     guidance_scale: float = 1.0,
    #     num_layers_per_mesh: list = None,
    #     encoder_hidden_states: torch.Tensor = None,
    # ) -> torch.Tensor:
    #     """
    #     带 Classifier-Free Guidance 的前向传播（用于推理）

    #     Args:
    #         guidance_scale: CFG scale，1.0 表示不使用 CFG
    #         其他参数同 forward

    #     Returns:
    #         CFG 组合后的预测速度场 v
    #     """
    #     if guidance_scale == 1.0:
    #         # 不使用 CFG，直接返回有条件预测
    #         pred, _ = self.forward(
    #             x_t, t, centers, depths, child_ids, surface_points,
    #             cu_seqlens_q, cu_seqlens_surface, num_layers_per_mesh,
    #             encoder_hidden_states=encoder_hidden_states,
    #         )
    #         return pred

    #     # normal/both 模式下，CFG 必须提供图像条件（否则 forward 会没有 condition 而直接报错）
    #     if self.condition_type in ['normal', 'both'] and encoder_hidden_states is None:
    #         raise ValueError(
    #             "forward_with_cfg requires encoder_hidden_states when condition_type is 'normal' or 'both'. "
    #             "Please pass pre-encoded image features: (num_meshes, num_tokens, hidden_size)."
    #         )

    #     # 有条件预测
    #     pred_cond, _ = self.forward(
    #         x_t, t, centers, depths, child_ids, surface_points,
    #         cu_seqlens_q, cu_seqlens_surface, num_layers_per_mesh,
    #         encoder_hidden_states=encoder_hidden_states,
    #     )

    #     # 无条件预测：
    #     # - pointcloud/both: 用 0 点云近似"无条件"
    #     # - normal/both: 用 0 特征近似"无条件"
    #     null_surface_points = surface_points
    #     if self.condition_type in ['pointcloud', 'both']:
    #         null_surface_points = torch.zeros_like(surface_points)

    #     null_encoder_hidden_states = encoder_hidden_states
    #     if self.condition_type in ['normal', 'both'] and encoder_hidden_states is not None:
    #         null_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

    #     pred_uncond, _ = self.forward(
    #         x_t, t, centers, depths, child_ids, null_surface_points,
    #         cu_seqlens_q, cu_seqlens_surface, num_layers_per_mesh,
    #         encoder_hidden_states=null_encoder_hidden_states,
    #     )

    #     # CFG 组合: pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    #     pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    #     return pred

    # def compute_loss(
    #     self,
    #     x_1: torch.Tensor,                   # (total_nodes, token_dim) GT 占用状态
    #     centers: torch.Tensor,
    #     depths: torch.Tensor,
    #     child_ids: torch.Tensor,
    #     condition: torch.Tensor,
    #     cu_seqlens_q: torch.Tensor,
    #     cu_seqlens_kv: torch.Tensor,
    #     max_seqlen_q: int,
    #     max_seqlen_kv: int,
    #     t: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     """
    #     Flow Matching 损失计算

    #     Args:
    #         x_1: GT 占用状态 (已转为 [-1, 1] 范围)
    #         其他参数同 forward

    #     Returns:
    #         MSE 损失
    #     """
    #     batch_size = cu_seqlens_q.shape[0] - 1
    #     device = x_1.device

    #     with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    #         # 采样时间 - 每个样本一个 t
    #         if t is None:
    #             t = torch.sigmoid(torch.randn(batch_size, device=device))

    #         # 采样噪声
    #         x_0 = torch.randn_like(x_1)

    #         # 扩展 t 到每个 token
    #         t_expanded = []
    #         for i in range(batch_size):
    #             start, end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
    #             t_expanded.append(t[i].expand(end - start))
    #         t_per_token = torch.cat(t_expanded)

    #         # 插值: x_t = t * x_1 + (1 - t) * x_0
    #         t_expand = t_per_token.unsqueeze(-1)
    #         x_t = t_expand * x_1 + (1 - t_expand) * x_0

    #         # 目标速度: v = x_1 - x_0
    #         target_v = x_1 - x_0

    #         # 预测速度
    #         pred_v = self.forward(
    #             x_t, t, centers, depths, child_ids, condition,
    #             cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
    #         )

    #     # MSE 损失
    #     loss = F.mse_loss(pred_v.float(), target_v.float())
    #     return loss

    def _attn_flops(self, s_q: int, s_kv: int, d: int, h: int, h_kv: int) -> float:
        kv_ratio = h_kv / h
        q_proj = 2 * s_q * d**2
        kv_proj = 4 * s_kv * d**2 * kv_ratio
        attn = 4 * s_q * s_kv * d
        o_proj = 2 * s_q * d**2
        return q_proj + kv_proj + attn + o_proj

    def _ffn_flops(self, s: int, d: int, d_ffn: int) -> float:
        return 6 * s * d * d_ffn

    def flops(
        self,
        cu_seqlens: Optional[torch.Tensor] = None,
        condition_lens=None,
        condition_proj_tokens: Optional[int] = None,
    ) -> float:
        """Estimate forward FLOPs for the current octree batch."""
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
            self_attn = self._attn_flops(s, s, dim, h, h_kv)
            cross_attn = (
                self._attn_flops(s, s_enc, dim, h, h_kv)
                if self.config.contain_cross_attention
                else 0
            )
            transformer = self.config.num_layers * (
                self_attn + cross_attn + self._ffn_flops(s, dim, dim_ffn)
            )
            time_embed_and_proj = (dim_freq * dim + dim * dim + dim * dim * 6) * 2 * s
            noise_proj = s * token_dim * dim * 2
            center_proj = (
                s * (3 * int(self.config.num_freqs) * 2 + 3) * dim * 2
                if self.config.pos_embed_type == "fourier"
                else 0
            )
            head = s * token_dim * dim * 2
            total += transformer + time_embed_and_proj + noise_proj + center_proj + head
        return total
