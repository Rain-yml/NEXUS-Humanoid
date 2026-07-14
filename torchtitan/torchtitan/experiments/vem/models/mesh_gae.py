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
from torchtitan.experiments.vem.models.mesh_gnn import (
    MeshTransformerBlock,
    GraphConvBlock,
)


class MeshGAEDeprecated(ModelMixin, ConfigMixin):
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
        bottleneck_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        num_freqs: int = 4,
        undirected: bool = True,
        pos_weight: int = 1,
        use_kl: bool = False,
        kl_weight: float = 1e-4,
        use_focal_loss: bool = False,
        dice_weight: float = 0.0,
        gated: bool = False,
        full_attn_interval: int = 0,
        spacetime_dim: int = 64,
        spacetime_proj: bool = False,
        chunk_size: int = 2_000_000,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.spacetime_dim = spacetime_dim
        self.spacetime_proj = spacetime_proj
        
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
                    heads=heads,
                    ffn_dim=ffn_dim,
                    gated=gated,
                )
            )
            layers_type.append("MeshTransformerBlock")
            if full_attn_interval > 0 and i % full_attn_interval == 0:
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
                        gated=gated,
                    )
                )
                layers_type.append("TransformerBlock")


        self.layers = nn.ModuleList(layers)
        self.layers_type = layers_type
        
        self.node_norm = RMSNorm(node_hidden_dim)
        
        # Output projection
        if not use_kl:
            self.to_out = nn.Linear(node_hidden_dim, bottleneck_dim)
        else:
            self.to_mu = nn.Linear(node_hidden_dim, bottleneck_dim)
            self.to_logvar = nn.Linear(node_hidden_dim, bottleneck_dim)

        self.undirected = undirected
        self.pos_weight = pos_weight

        if self.spacetime_proj:
            self.to_st = nn.Linear(bottleneck_dim, self.spacetime_dim)
        else:
            self.to_st = nn.Identity()

        self.loss_module = SpacetimeEdgeLoss(
            use_focal_loss=use_focal_loss, 
            gamma=2.0, 
            alpha=0.5, 
            use_pred_balanced_loss=True, 
            pred_loss_weight=1.0,
            use_dice_loss=dice_weight > 0.0, 
            dice_weight=dice_weight,
            chunk_size=chunk_size,
        )
    
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
        
    def forward(self, pos: torch.Tensor, edge_index: torch.Tensor, offsets: torch.Tensor) -> dict:
        assert pos.shape[1] == self.config.input_node_dim
        
        if self.undirected:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # print(self.node_proj.weight.dtype)
        
        # Extract geometric edge features
        edge_attr = self.get_edge_feature(pos, edge_index)
        x = self.freq_emb(pos)
        
        # Project to hidden dimension
        x = self.node_proj(x.to(dtype=self.node_proj.weight.dtype))
        edge_attr = self.edge_proj(edge_attr.to(dtype=self.edge_proj.weight.dtype))
        
        for layer_type, layer in zip(self.layers_type, self.layers):
            # Apply convolution
            if layer_type == "MeshTransformerBlock":
                x, edge_attr = layer(x, edge_index, edge_attr)
            else:
                x = layer(
                    x, 
                    cu_seqlens=offsets,
                )
        
        x_norm = self.node_norm(x)
        if not self.config.use_kl:
            node_feat = self.to_out(x_norm)
            return {
                'node_embed': node_feat
            }
        else:
            mu = self.to_mu(x_norm)
            logvar = self.to_logvar(x_norm)
            
            return {
                'node_feature': x_norm,
                'node_embed_mu': mu,
                'node_embed_logvar': logvar,
            }
    
    def loss_fn(self, pred_dict, input_dict):
        pair = input_dict['vertex_pair']
        target = input_dict['vertex_pair_label']
        log_dict = {}

        # --- sample KL or not ---
        if not self.config.use_kl:
            pred = pred_dict['node_embed'].float()
        else:
            mu = pred_dict['node_embed_mu'].float()
            logvar = pred_dict['node_embed_logvar'].float()
            noise = torch.randn_like(mu)
            pred = mu + noise * torch.exp(0.5 * logvar)
        
        st_feat = self.to_st(pred)
        loss, log_dict = self.loss_module(st_feat, pair, target)

        # KL part
        if self.config.use_kl:
            kl_loss = torch.mean(
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            )
            log_dict["kl_loss"] = kl_loss.detach()
            loss += kl_loss * self.config.kl_weight

        return loss, log_dict




class MeshGAE(ModelMixin, ConfigMixin):
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
        ffn_dim: int = 128,
        bottleneck_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        num_freqs: int = 4,
        undirected: bool = True,
        use_kl: bool = False,
        kl_weight: float = 1e-4,
        dice_weight: float = 0.0,
        gated: bool = False,
        full_attn_interval: int = 0,
        spacetime_dim: int = 64,
        spacetime_proj: bool = False,
        chunk_size: int = 2_000_000,
        dist_scale: str = 'no',
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.spacetime_dim = spacetime_dim
        self.spacetime_proj = spacetime_proj
        
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
        
        # Graph convolution layers
        layers = []
        layers_type = []
        for i in range(num_layers):
            layers.append(
                GraphConvBlock(dim=node_hidden_dim, ffn_dim=ffn_dim)
            )
            layers_type.append("GraphConvBlock")
            if full_attn_interval > 0 and i % full_attn_interval == 0:
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
                        gated=gated,
                    )
                )
                layers_type.append("TransformerBlock")


        self.layers = nn.ModuleList(layers)
        self.layers_type = layers_type
        
        self.node_norm = RMSNorm(node_hidden_dim)
        
        # Output projection
        if not use_kl:
            self.to_out = nn.Linear(node_hidden_dim, bottleneck_dim)
        else:
            self.to_mu = nn.Linear(node_hidden_dim, bottleneck_dim)
            self.to_logvar = nn.Linear(node_hidden_dim, bottleneck_dim)

        self.undirected = undirected

        if self.spacetime_proj:
            self.to_st = nn.Linear(bottleneck_dim, self.spacetime_dim)
        else:
            self.to_st = nn.Identity()

        self.loss_module = SpacetimeEdgeLoss(
            use_pred_balanced_loss=True, 
            pred_loss_weight=1.0,
            use_dice_loss=dice_weight > 0.0, 
            dice_weight=dice_weight,
            chunk_size=chunk_size,
            scale=dist_scale,
        )
    
    def _get_edge_feature(self, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
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
        
    def forward_gnn(self, x, edge_index, offsets):
        for layer_type, layer in zip(self.layers_type, self.layers):
            # Apply convolution
            if layer_type == "GraphConvBlock":
                x = layer(x, edge_index)
            else:
                x = layer(
                    x, 
                    cu_seqlens=offsets,
                )
        
        x = self.node_norm(x)
        return x

    def forward(
        self, 
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        offsets: torch.Tensor,
        vertex_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        '''
        nodes: N x 3
        edge_index: 2 x E
        vertex_mask: N
        offsets: B + 1
        '''
    
        if self.undirected:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        x = self.freq_emb(pos)
        if vertex_mask is not None:
            x = torch.cat([x, vertex_mask.unsqueeze(-1)], dim=-1)
        # Project to hidden dimension
        x = self.node_proj(x.to(dtype=self.node_proj.weight.dtype))

        x_norm = self.forward_gnn(x, edge_index, offsets)
        
        if not self.config.use_kl:
            node_feat = self.to_out(x_norm)
            return {
                'node_embed': node_feat
            }
        else:
            mu = self.to_mu(x_norm)
            logvar = self.to_logvar(x_norm)
            
            return {
                'node_feature': x_norm,
                'node_embed_mu': mu,
                'node_embed_logvar': logvar,
            }
    
    def loss_fn(self, pred_dict, input_dict):
        pair = input_dict['vertex_pair']
        target = input_dict['vertex_pair_label']
        log_dict = {}

        # --- sample KL or not ---
        if not self.config.use_kl:
            pred = pred_dict['node_embed'].float()
        else:
            mu = pred_dict['node_embed_mu'].float()
            logvar = pred_dict['node_embed_logvar'].float()
            noise = torch.randn_like(mu)
            pred = mu + noise * torch.exp(0.5 * logvar)
        
        st_feat = self.to_st(pred)
        loss, log_dict = self.loss_module(st_feat, pair, target)

        # KL part
        if self.config.use_kl:
            kl_loss = torch.mean(
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            )
            log_dict["kl_loss"] = kl_loss.detach()
            loss += kl_loss * self.config.kl_weight

        return loss, log_dict
    
    def train_step(self, input_dict):
        pred = self(
            pos=input_dict['nodes'],
            edge_index=input_dict['edges'].permute(1, 0),
            offsets=input_dict['offsets'],
        )
        loss, log_dict = self.loss_fn(pred, input_dict)
        return loss, log_dict, {
            "num_tokens": input_dict['edges'].shape[0],
            "num_flops": 100,
        }