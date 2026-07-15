"""TorchTitan wrapper for frozen-mesh dual-branch octree training."""
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Tuple, Dict, Any, List

import torch
import torch.nn as nn

from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol
from torchtitan.experiments.humanoid.models.dual_branch_octree import (
    DualBranchOctreeDiffusionModel,
)
from torchtitan.tools.logging import logger

@dataclass
class DualBranchOctreeDiffusionArgs(BaseModelArgs):
    """Octree Diffusion 模型参数

    训练目标：
    - 输入：父节点中心 (centers) + 深度 (depths) + 子节点ID (child_ids) + noisy 占用状态 (x_t)
    - 输出：预测 8 个子节点的占用状态 (8-bit 或 256-dim one-hot)
    """
    # 输入/输出
    use_onehot_256: bool = True     # True: 256-dim one-hot, False: 8-bit

    # Transformer 架构
    num_layers: int = 12
    dim: int = 512
    freq_dim: int = 256
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    num_key_value_heads: Optional[int] = None

    # Attention 配置
    attention_bias: bool = False
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    use_flash_attn_3: bool = False

    # 位置编码方式: 'fourier' 或 'rotary'
    pos_embed_type: str = 'rotary'

    # 3D RoPE 配置 (仅当 pos_embed_type='rotary' 时使用)
    use_3d_rope: bool = True
    max_seq_len_3d: int = 10000
    grid_size: int = 128
    rope_theta: int = 10000

    # Fourier 位置编码配置
    num_freqs: int = 8

    # 八叉树配置
    max_depth: int = 7

    # Cross attention (条件)
    contain_cross_attention: bool = True

    image_hidden_size: int = 1280  # DINOv3-ViT-H/16+ 的 hidden_size

    num_vertex_condition: bool = False
    quad_ratio_condition: bool = False
    quad_ratio_uncond: bool = False
    symmetry_condition: bool = False

    # Multiview conditioning
    mv_mode: bool = False      # Enable multiview conditioning
    num_mv_views: int = 4      # Number of distinct view types (0=front,1=left,2=back,3=right)

    pretrained_path: Optional[str] = None
    num_joint_tokens: int = 28

    def get_nparams(self, model: nn.Module) -> int:
        nparams = sum(p.numel() for p in model.parameters())
        return nparams

    def __post_init__(self):
        # change flash_attn_3 according to gpu
        device_name = torch.cuda.get_device_name(0)
        if "H" in device_name:
            self.use_flash_attn_3 = True
        else:
            self.use_flash_attn_3 = False

        logger.info(f"Using flash_attn_3: {self.use_flash_attn_3} according to device name: {device_name}")

    # @property
    # def token_dim(self) -> int:
    #     """输入/输出维度"""
    #     return 256 if self.use_onehot_256 else 8

class DualBranchOctreeDiffusionWrapper(DualBranchOctreeDiffusionModel, ModelProtocol):
    def __init__(self, model_args: DualBranchOctreeDiffusionArgs):
        super().__init__(**asdict(model_args))

    @classmethod
    def from_model_args(
        cls, model_args: DualBranchOctreeDiffusionArgs
    ) -> "DualBranchOctreeDiffusionWrapper":
        return cls(model_args)
