from dataclasses import dataclass, asdict
from typing import Optional, Literal, Tuple, Dict, Any, List

import torch
import torch.nn as nn

from torchtitan.experiments.vem.models.mesh_gae import MeshGAEDeprecated, MeshGAE
from torchtitan.experiments.vem.models.mesh_fae import MeshFAE
from torchtitan.experiments.vem.models.mesh_fae_decoder import MeshFAEDecoder
from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol
from torchtitan.experiments.vem.mesh_tokenizers import TokenizerSpec
from torchtitan.tools.logging import logger

@dataclass
class MeshGAEArgsDeprecated(BaseModelArgs):
    input_node_dim: int = 3
    node_hidden_dim: int = 512
    edge_hidden_dim: int = 512
    attn_hidden_dim: int = 512
    ffn_dim: int = 1280
    num_layers: int = 8
    heads: int = 8
    num_freqs: int = 4
    undirected: bool = True
    bottleneck_dim: int = 64
    use_kl: bool = False
    kl_weight: float = 1e-4
    use_focal_loss: bool = False
    dice_weight: float = 0.0
    gated: bool = False
    full_attn_interval: int = 0
    spacetime_dim: int = 64
    chunk_size: int = 2_000_000
    spacetime_proj: bool = False

    def get_nparams(self, model: nn.Module) -> int:
        nparams = sum(p.numel() for p in model.parameters())
        return nparams

class MeshGAEWrapperDeprecated(MeshGAE, ModelProtocol):
    def __init__(self, model_args: MeshGAEArgsDeprecated):
        super().__init__(**asdict(model_args))

    @classmethod
    def from_model_args(cls, model_args: MeshGAEArgsDeprecated) -> "MeshGAEDeprecated":
        return cls(model_args)

@dataclass
class MeshGAEArgs(BaseModelArgs):
    input_node_dim: int = 3
    node_hidden_dim: int = 512
    ffn_dim: int = 1280
    bottleneck_dim: int = 64
    num_layers: int = 8
    heads: int = 8
    num_freqs: int = 4
    undirected: bool = True
    bottleneck_dim: int = 64
    use_kl: bool = False
    kl_weight: float = 1e-4
    dice_weight: float = 0.0
    gated: bool = False
    full_attn_interval: int = 0
    spacetime_dim: int = 64
    spacetime_proj: bool = False
    chunk_size: int = 2_000_000
    dist_scale: str = 'no'

    def get_nparams(self, model: nn.Module) -> int:
        nparams = sum(p.numel() for p in model.parameters())
        return nparams

class MeshGAEWrapper(MeshGAE, ModelProtocol):
    def __init__(self, model_args: MeshGAEArgs):
        super().__init__(**asdict(model_args))

    @classmethod
    def from_model_args(cls, model_args: MeshGAEArgs) -> "MeshGAE":
        return cls(model_args)


@dataclass
class MeshFAEArgs(BaseModelArgs):
    input_node_dim: int = 3
    node_hidden_dim: int = 512
    ffn_dim: int = 1280
    bottleneck_dim: int = 64
    num_layers: int = 8
    heads: int = 8
    num_freqs: int = 4
    undirected: bool = True
    bottleneck: str = 'kl'
    noise_interpolate: float = 0.0
    kl_weight: float = 1e-4
    dice_weight: float = 0.0
    gated: bool = False
    full_attn_interval: int = 0
    spacetime_dim: int = 64
    spacetime_proj: bool = False
    chunk_size: int = 2_000_000
    dist_scale: str = 'no'
    face_weight: float = 0.1
    face_loss_style: str = 'mink'
    face_dim_split: bool = False
    face_dim: int = 0
    num_decoder_layers: int = 0
    use_flash_attn_3: bool = False
    attn_bias: bool = False
    is_causal: bool = True # backward compatibility

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

class MeshFAEWrapper(MeshFAE, ModelProtocol):
    def __init__(self, model_args: MeshFAEArgs):
        super().__init__(**asdict(model_args))

    @classmethod
    def from_model_args(cls, model_args: MeshFAEArgs) -> "MeshFAE":
        return cls(model_args)

@dataclass
class MeshFAEDecoderArgs(BaseModelArgs):
    """Args for MeshFAE with full attention decoder after bottleneck."""
    input_node_dim: int = 3
    node_hidden_dim: int = 256
    ffn_dim: int = 512
    bottleneck_dim: int = 64
    num_layers: int = 12
    heads: int = 8
    num_freqs: int = 4
    undirected: bool = True
    bottleneck: str = 'kl'
    noise_interpolate: float = 0.0
    kl_weight: float = 1e-4
    dice_weight: float = 0.0
    gated: bool = False
    full_attn_interval: int = 0
    spacetime_dim: int = 64
    spacetime_proj: bool = False
    chunk_size: int = 2_000_000
    dist_scale: str = 'no'
    face_weight: float = 0.1
    face_loss_style: str = 'mink'
    face_dim_split: bool = False
    face_dim: int = 0
    use_pred_balanced_loss: bool = True
    # Decoder params
    decoder_dim: int = 1024
    decoder_ffn_dim: int = 4096
    decoder_layers: int = 16
    decoder_heads: int = 16
    attn_bias: bool = False
    pred_orient: bool = False
    orient_embed_dim: int = 3
    pos_embed_type: str = 'fourier'
    max_seq_len_rope: int = 10000
    rope_theta: float = 10000.0
    spacetime_heads: int = 1
    attn_dtype: str = "bf16"
    apply_face_loss: bool = True

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


class MeshFAEDecoderWrapper(MeshFAEDecoder, ModelProtocol):
    def __init__(self, model_args: MeshFAEDecoderArgs):
        super().__init__(**asdict(model_args))

    @classmethod
    def from_model_args(cls, model_args: MeshFAEDecoderArgs) -> "MeshFAEDecoder":
        return cls(model_args)
