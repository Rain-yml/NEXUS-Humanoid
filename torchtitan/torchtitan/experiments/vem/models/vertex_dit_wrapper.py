from dataclasses import dataclass, asdict
from typing import Optional, Literal, Tuple, Dict, Any, List, Union

import torch
import torch.nn as nn

from torchtitan.experiments.vem.models.vertex_dit import SpaceMeshDiT
from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol
from torchtitan.tools.logging import logger

@dataclass
class SpaceMeshDiTArgs(BaseModelArgs):
    in_channels: int = 64
    num_layers: int = 24
    dim: int = 1024
    freq_dim: int = 256
    num_attention_heads: int = 64
    intermediate_size: int = 2816
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = True
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    use_flash_attn_3: bool = False
    num_freqs: int = 8
    condition_dim: int = 0
    gated: bool = False
    pos_embed_type: str = "fourier"
    max_seq_len_rope: int = 10000
    rope_theta: float = 10000.0
    num_registers: int = 0
    ape_scale_div: float = 1.0
    mv_mode: bool = False
    num_mv_views: int = 4
    quad_ratio_condition: bool = False
    quad_ratio_condition_mapping: str = "identity"
    quad_ratio_uncond: bool = False
    symmetry_condition: bool = False
    pretrained_path: Optional[str] = None

    def get_nparams(self, model: nn.Module) -> int:
        nparams = sum(p.numel() for p in model.parameters())
        return nparams
    
    def __post_init__(self):
        # change flash_attn_3 according to gpu
        device_name = torch.cuda.get_device_name(0)
        if "H" in device_name:
            self.use_flash_attn_3 = True
        if "L20Z" in device_name:
            self.use_flash_attn_3 = True
        else:
            self.use_flash_attn_3 = False
        
        logger.info(f"Using flash_attn_3: {self.use_flash_attn_3} according to device name: {device_name}")

    
class SpaceMeshDiTWrapper(SpaceMeshDiT, ModelProtocol):
    def __init__(self, model_args: SpaceMeshDiTArgs):
        super().__init__(**asdict(model_args))

    @classmethod
    def from_model_args(cls, model_args: SpaceMeshDiTArgs) -> "SpaceMeshDiTWrapper":
        return cls(model_args)
