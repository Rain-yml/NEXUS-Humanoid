from dataclasses import dataclass, asdict
from typing import Optional, Literal, Tuple, Dict, Any, List

import torch
import torch.nn as nn

from torchtitan.experiments.vem.models.mesh_gnn import MeshTransformer
from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol
from torchtitan.experiments.vem.mesh_tokenizers import TokenizerSpec
from torchtitan.tools.logging import logger

@dataclass
class MeshTransformerArgs(BaseModelArgs):
    input_node_dim: int = 3
    node_hidden_dim: int = 512
    edge_hidden_dim: int = 512
    attn_hidden_dim: int = 512
    ffn_dim: int = 1280
    output_dim: int = 1
    num_layers: int = 8
    heads: int = 8
    num_freqs: int = 4
    undirected: bool = True
    pos_weight: int = 1
    full_attn_interval: int = 0

    def get_nparams(self, model: nn.Module) -> int:
        nparams = sum(p.numel() for p in model.parameters())
        return nparams
    
class MeshTransformerWrapper(MeshTransformer, ModelProtocol):
    def __init__(self, model_args: MeshTransformerArgs):
        super().__init__(**asdict(model_args))

    @classmethod
    def from_model_args(cls, model_args: MeshTransformerArgs) -> "MeshTransformer":
        return cls(model_args)


