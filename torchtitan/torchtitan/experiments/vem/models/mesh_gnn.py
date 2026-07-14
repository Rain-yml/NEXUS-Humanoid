import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
import math
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torchtitan.experiments.vem.models.transformer import (
    RMSNorm, 
    FP32LayerNorm, 
    MLP, 
    FrequencyPositionalEmbedding,
    _basic_init,
    _init_norm,
    VEM2DecoderLayer as TransformerBlock,
)
from torch_geometric.nn import SAGEConv

class GraphConvBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.conv = SAGEConv(dim, dim)
        self.norm2 = RMSNorm(dim)

        self.ffn_node = MLP(
            hidden_size=dim,
            intermediate_size=ffn_dim,
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x + self.conv(self.norm1(x), edge_index)
        x = x + self.ffn_node(self.norm2(x))
        return x

class AttentionConv(MessagePassing):
    """Custom attention-based graph convolution for meshes"""
    
    def __init__(
        self, 
        node_feat_dim: int, 
        edge_feat_dim: int, 
        hidden_dim: int = 64, 
        heads: int = 4,
        bias: bool = True,
        gated: bool = False,
    ):
        super().__init__(aggr=None, node_dim=0)  # No built-in aggregation
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        
        # Edge update network
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_feat_dim + edge_feat_dim, hidden_dim),
            # nn.ReLU(),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, edge_feat_dim),
        )
        
        # Attention mechanism
        self.query_proj = nn.Linear(node_feat_dim, hidden_dim, bias=bias)
        self.key_proj = nn.Linear(node_feat_dim + edge_feat_dim, hidden_dim, bias=bias)
        self.value_proj = nn.Linear(node_feat_dim + edge_feat_dim, hidden_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, node_feat_dim)
        
        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * node_feat_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, node_feat_dim),
        )
        self.gated = gated

        if self.gated:
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention-based aggregation"""
        
        # Update edge features
        row, col = edge_index
        
        # Attention-based message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        
        # Update node features
        node_input = torch.cat([x, out], dim=1)
        updated_x = self.node_mlp(node_input)

        edge_input = torch.cat([updated_x[row], updated_x[col], edge_attr], dim=1)
        updated_edge_attr = self.edge_mlp(edge_input)
        
        return updated_x, updated_edge_attr
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor,
                index: torch.Tensor, ptr: Optional[torch.Tensor] = None,
                size_i: Optional[int] = None) -> torch.Tensor:
        """Create attention-weighted messages"""
        # Compute queries, keys, and values
        query = self.query_proj(x_i)  # [E, hidden_dim]
        key_value_input = torch.cat([x_j, edge_attr], dim=1)
        key = self.key_proj(key_value_input)    # [E, hidden_dim]
        value = self.value_proj(key_value_input) # [E, hidden_dim]
        
        # Reshape for multi-head attention
        batch_size = query.size(0)
        query = query.view(batch_size, self.heads, self.head_dim)
        key = key.view(batch_size, self.heads, self.head_dim)
        value = value.view(batch_size, self.heads, self.head_dim)
        
        # Compute attention scores
        scores = (query * key).sum(dim=-1) / (self.head_dim ** 0.5)  # [E, heads]
        
        # Apply softmax attention (grouped by target node)
        attention = softmax(scores, index, ptr, size_i)  # [E, heads]
        
        # Apply attention to values
        attention = attention.unsqueeze(-1)  # [E, heads, 1]
        attended_value = (attention * value).view(batch_size, -1)  # [E, hidden_dim]
        if self.gated:
            score = torch.sigmoid(self.gate_proj(x_i))
            attended_value *= score
        
        return self.out_proj(attended_value)
    
    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor,
                  ptr: Optional[torch.Tensor] = None,
                  dim_size: Optional[int] = None) -> torch.Tensor:
        """Aggregate messages (sum after attention weighting)"""
        # Use torch.scatter_add for GPU compatibility
        out = torch.zeros((dim_size or index.max() + 1, inputs.size(-1)), 
                         dtype=inputs.dtype, device=inputs.device)
        return out.scatter_add_(0, index.unsqueeze(-1).expand_as(inputs), inputs)
    
class MeshTransformerBlock(nn.Module):
    """Transformer-style block for meshes using AttentionConv"""
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 64,
        ffn_dim: int = 64,
        heads: int = 4,
        gated: bool = False,
    ):
        super().__init__()
        
        self.attn = AttentionConv(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            gated=gated,
        )
        
        # Pre-norm layers
        self.norm1_node = RMSNorm(node_feat_dim)
        self.norm1_edge = RMSNorm(edge_feat_dim)

        self.norm2_node = RMSNorm(node_feat_dim)
        self.norm2_edge = RMSNorm(edge_feat_dim)
        
        self.ffn_node = MLP(
            hidden_size=node_feat_dim,
            intermediate_size=ffn_dim,
        )
        self.ffn_edge = MLP(
            hidden_size=edge_feat_dim,
            intermediate_size=ffn_dim,
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: [N, node_feat_dim] node features
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_feat_dim] edge features
        """
        # --- Attention sublayer ---
        x_norm = self.norm1_node(x)
        edge_norm = self.norm1_edge(edge_attr)
        
        attn_out, updated_edge_attr = self.attn(x_norm, edge_index, edge_norm)
        x = x + attn_out
        edge_attr = edge_attr + updated_edge_attr  # Residual for edges
        
        # --- FFN sublayer ---
        x = x + self.ffn_node(self.norm2_node(x))
        edge_attr = edge_attr + self.ffn_edge(self.norm2_edge(edge_attr))
        
        return x, edge_attr


class MeshTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["node_norm", "edge_norm", "norm1_node", "norm1_edge", "norm2_node", "norm2_edge"]
    _no_split_modules = ["MeshTransformerBlock"]
    _keep_in_fp32_modules = ["node_norm", "edge_norm", "norm1_node", "norm1_edge", "norm2_node", "norm2_edge"]
    _keys_to_ignore_on_load_unexpected = []

    """Mesh GNN with various attention mechanisms"""
    
    @register_to_config
    def __init__(
        self, 
        input_node_dim: int = 3,
        node_hidden_dim: int = 64,
        edge_hidden_dim: int = 64,
        attn_hidden_dim: int = 64,
        ffn_dim: int = 128,
        output_dim: int = 32,
        num_layers: int = 3,
        heads: int = 4,
        num_freqs: int = 4,
        undirected: bool = True,
        pos_weight: int = 1,
        full_attn_interval: int = 0,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        input_edge_dim = 10
        
        # Input projection
        self.freq_emb = FrequencyPositionalEmbedding(
            num_freqs=num_freqs,
            logspace=True,
            input_dim=3,
            include_input=True,
            include_pi=False,
        )

        self.node_proj = nn.Linear(self.freq_emb.out_dim, node_hidden_dim)
        self.edge_proj = nn.Linear(input_edge_dim, edge_hidden_dim)
        
        # Graph convolution layers
        layers = []
        layers_type = []
        for i in range(num_layers):
            layers.append(
                MeshTransformerBlock(
                    node_feat_dim=node_hidden_dim, 
                    edge_feat_dim=edge_hidden_dim, 
                    hidden_dim=attn_hidden_dim, 
                    ffn_dim=ffn_dim, 
                    heads=heads,
                )
            )
            layers_type.append("GraphAttn")
            if full_attn_interval > 0 and (i + 1) % full_attn_interval == 0:
                layers.append(
                    TransformerBlock(
                        layer_idx=i,
                        hidden_size=node_hidden_dim,
                        num_attention_heads=heads,
                        intermediate_size=ffn_dim,
                        num_key_value_heads=None,
                        rope=None,
                        qk_norm=True,
                        use_flash_attn_3=False,
                        contains_cross_attention=False,
                        attn_dtype="bf16",
                        gated=False,
                        is_causal=False,
                        attention_bias=True,
                    )
                )
                layers_type.append("FullAttn")
            
        self.layers = nn.ModuleList(layers)
        self.layers_type = layers_type
        
        # self.node_norm = RMSNorm(node_hidden_dim)
        self.edge_norm = RMSNorm(edge_hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(edge_hidden_dim, output_dim)

        self.undirected = undirected
        self.pos_weight = pos_weight
    
    def get_edge_feature(self, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Extract edge features from vertex positions"""
        row, col = edge_index
        
        # Get vertex positions for each edge
        pos_i = pos[row]  # Source vertices
        pos_j = pos[col]  # Target vertices
        
        # Compute geometric features
        edge_vec = pos_j - pos_i  # Edge vectors
        edge_len = torch.norm(edge_vec, dim=1, keepdim=True)  # Edge lengths
        edge_unit = edge_vec / (edge_len + 1e-8)  # Unit edge vectors
        
        # Additional features
        midpoint = (pos_i + pos_j) / 2  # Edge midpoints
        
        # Concatenate all edge features
        edge_features = torch.cat([
            edge_vec,        # [3] - directional vector
            edge_len,        # [1] - length
            edge_unit,       # [3] - unit direction
            midpoint,        # [3] - midpoint
        ], dim=1)
        
        return edge_features
    
    def init_weights(self, buffer_device=None):
        self.apply(_basic_init)
        self.freq_emb.init_weights()
        
    def forward(self, pos: torch.Tensor, edge_index: torch.Tensor, cu_seqlens=None) -> dict:
        assert pos.shape[1] == self.config.input_node_dim
        
        if self.undirected:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Extract geometric edge features
        edge_attr = self.get_edge_feature(pos, edge_index)
        x = self.freq_emb(pos)
        
        # Project to hidden dimension
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        for layer_type, layer in zip(self.layers_type, self.layers):
            if layer_type == "GraphAttn":
                # Apply convolution
                x, edge_attr = layer(x, edge_index, edge_attr)
            else:
                assert cu_seqlens is not None, "cu_seqlens must be provided for full attention layers"
                # Apply full attention (treat nodes as sequence)
                x = layer(x, cu_seqlens=cu_seqlens)

        
        if self.undirected:
            # Average edge features for undirected edges
            E = edge_index.size(1) // 2
            edge_attr = (edge_attr[:E] + edge_attr[E:]) / 2
        
        # Output projection for nodes
        edge_output = self.output_proj(self.edge_norm(edge_attr))
        
        return edge_output

    def train_step(self, input_dict):
        pred = self(
            pos=input_dict['vertices'],
            edge_index=input_dict['edges'].permute(1, 0),
            cu_seqlens=input_dict.get('offsets', None),
        )
        loss, log_dict = self.loss_fn(pred, input_dict)
        return loss, log_dict

    def loss_fn(self, pred, input_dict):
        pred = pred.squeeze(-1) # E
        is_quad_diag = input_dict['is_quad_diag'] # E
        loss = F.binary_cross_entropy_with_logits(pred.float(), is_quad_diag.float(), reduction='mean', pos_weight=torch.tensor(self.pos_weight).to(pred.device))

        tn = ((pred < 0) & (is_quad_diag == 0)).sum().item()
        tp = ((pred >= 0) & (is_quad_diag == 1)).sum().item()
        fn = ((pred < 0) & (is_quad_diag == 1)).sum().item()
        fp = ((pred >= 0) & (is_quad_diag == 0)).sum().item()
        rec = tp / max(tp + fn, 1e-8)
        prec = tp / max(tp + fp, 1e-8)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1e-8)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        return loss, {
            "loss": loss.detach().cpu().item(),
            "rec": rec,
            "prec": prec,
            "acc": acc,
            "fs": f1,
        }

    def is_scalar_param(self, name, param):
        patterns = [
            'node_proj',
            'edge_proj',
            'output_proj',
        ]
        for p in patterns:
            if p in name:
                return True
        return False
