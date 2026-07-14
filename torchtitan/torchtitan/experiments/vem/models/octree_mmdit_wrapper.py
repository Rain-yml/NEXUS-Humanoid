from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Tuple, Dict, Any, List

import torch
import torch.nn as nn

from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol
from torchtitan.experiments.vem.models.octree_mmdit import OctreeMMDiTModel
from torchtitan.tools.logging import logger
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

@dataclass
class OctreeMMDiTArgs(BaseModelArgs):
    """Octree MM-DiT model arguments

    Uses dual-stream joint attention (MMDiTBlock) instead of cross-attention.
    """
    # Input/output
    use_onehot_256: bool = True

    # Transformer architecture
    num_layers: int = 12
    dim: int = 512
    freq_dim: int = 256
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    num_key_value_heads: Optional[int] = None

    # Attention config
    attention_bias: bool = False
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    use_flash_attn_3: bool = False

    # Position embedding type: 'fourier' or 'rotary'
    pos_embed_type: str = 'rotary'

    # 3D RoPE config (only when pos_embed_type='rotary')
    use_3d_rope: bool = True
    max_seq_len_3d: int = 10000
    grid_size: int = 128
    rope_theta: int = 10000

    # Fourier position embedding config
    num_freqs: int = 8

    # Octree config
    max_depth: int = 7

    # Image condition
    image_hidden_size: int = 1280

    num_vertex_condition: bool = False
    num_vertex_condition_config: List[int] = field(default_factory=lambda: [24, 10000, 1024])
    quad_ratio_condition: bool = False
    quad_ratio_condition_mapping: str = "identity"

    # Multiview conditioning
    mv_mode: bool = False
    num_mv_views: int = 4

    # MMDiT-specific
    attn_dtype: str = "bf16"

    pretrained_path: Optional[str] = None

    def get_nparams(self, model: nn.Module) -> int:
        nparams = sum(p.numel() for p in model.parameters())
        return nparams

    def __post_init__(self):
        device_name = torch.cuda.get_device_name(0)
        if "H" in device_name:
            self.use_flash_attn_3 = True
        else:
            self.use_flash_attn_3 = False

        logger.info(f"Using flash_attn_3: {self.use_flash_attn_3} according to device name: {device_name}")


class OctreeMMDiTWrapper(OctreeMMDiTModel, ModelProtocol):
    def __init__(self, model_args: OctreeMMDiTArgs):
        super().__init__(**asdict(model_args))

    @classmethod
    def from_model_args(cls, model_args: OctreeMMDiTArgs) -> "OctreeMMDiTWrapper":
        return cls(model_args)

octree_mmdit_configs = {
    'default-mv-2048res': OctreeMMDiTArgs(
        use_onehot_256=False,
        num_layers=24,
        dim=1024,
        freq_dim=256,
        num_attention_heads=16,
        intermediate_size=2048,
        num_key_value_heads=None,
        attention_bias=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        use_flash_attn_3=False,
        pos_embed_type='rotary',
        use_3d_rope=True,
        max_seq_len_3d=4096,
        grid_size=2048,
        rope_theta=4052,
        num_freqs=8,
        max_depth=11,
        image_hidden_size=1280,
        num_vertex_condition=True,
        mv_mode=True,
        num_mv_views=4,
    ),
    'default-2048res': OctreeMMDiTArgs(
        use_onehot_256=False,
        num_layers=24,
        dim=1024,
        freq_dim=256,
        num_attention_heads=16,
        intermediate_size=2048,
        num_key_value_heads=None,
        attention_bias=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        use_flash_attn_3=False,
        pos_embed_type='rotary',
        use_3d_rope=True,
        max_seq_len_3d=4096,
        grid_size=2048,
        rope_theta=4052,
        num_freqs=8,
        max_depth=11,
        image_hidden_size=1280,
        num_vertex_condition=True,
        mv_mode=False,
        num_mv_views=4,
    ),
    'default-2048res-qr': OctreeMMDiTArgs(
        use_onehot_256=False,
        num_layers=24,
        dim=1024,
        freq_dim=256,
        num_attention_heads=16,
        intermediate_size=2048,
        num_key_value_heads=None,
        attention_bias=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        use_flash_attn_3=False,
        pos_embed_type='rotary',
        use_3d_rope=True,
        max_seq_len_3d=4096,
        grid_size=2048,
        rope_theta=4052,
        num_freqs=8,
        max_depth=11,
        image_hidden_size=1280,
        num_vertex_condition=True,
        num_vertex_condition_config=[24, 30000, 1024],
        mv_mode=False,
        num_mv_views=4,
        quad_ratio_condition=True,
        quad_ratio_condition_mapping="bins5",
    ),
    '8B-2048res-qr': OctreeMMDiTArgs(
        use_onehot_256=False,
        num_layers=24,
        dim=3072,
        freq_dim=256,
        num_attention_heads=24,
        intermediate_size=14336,
        num_key_value_heads=None,
        attention_bias=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        use_flash_attn_3=False,
        pos_embed_type='rotary',
        use_3d_rope=True,
        max_seq_len_3d=4096,
        grid_size=2048,
        rope_theta=4052,
        num_freqs=8,
        max_depth=11,
        image_hidden_size=1280,
        num_vertex_condition=True,
        num_vertex_condition_config=[24, 30000, 1024],
        mv_mode=False,
        num_mv_views=4,
        quad_ratio_condition=True,
        quad_ratio_condition_mapping="bins5",
    ),
    '4B-2048res-qr': OctreeMMDiTArgs(
        use_onehot_256=False,
        num_layers=24,
        dim=1536,
        freq_dim=256,
        num_attention_heads=24,
        intermediate_size=7168,
        num_key_value_heads=None,
        attention_bias=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        use_flash_attn_3=False,
        pos_embed_type='rotary',
        use_3d_rope=True,
        max_seq_len_3d=4096,
        grid_size=2048,
        rope_theta=4052,
        num_freqs=8,
        max_depth=11,
        image_hidden_size=1280,
        num_vertex_condition=True,
        num_vertex_condition_config=[24, 30000, 1024],
        mv_mode=False,
        num_mv_views=4,
        quad_ratio_condition=True,
        quad_ratio_condition_mapping="bins5",
    ),
    '1.4B-2048res-qr': OctreeMMDiTArgs(
        use_onehot_256=False,
        num_layers=24,
        dim=1536,
        freq_dim=256,
        num_attention_heads=24,
        intermediate_size=4096,
        num_key_value_heads=None,
        attention_bias=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        use_flash_attn_3=False,
        pos_embed_type='rotary',
        use_3d_rope=True,
        max_seq_len_3d=4096,
        grid_size=2048,
        rope_theta=4052,
        num_freqs=8,
        max_depth=11,
        image_hidden_size=1280,
        num_vertex_condition=True,
        num_vertex_condition_config=[24, 30000, 1024],
        mv_mode=False,
        num_mv_views=4,
        quad_ratio_condition=True,
        quad_ratio_condition_mapping="bins5",
    ),
}
