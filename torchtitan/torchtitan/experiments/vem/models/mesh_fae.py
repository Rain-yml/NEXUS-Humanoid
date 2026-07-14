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
from torchtitan.experiments.vem.models.mesh_gnn import MeshTransformerBlock
from torchtitan.experiments.vem.models.mesh_gae import GraphConvBlock
from torchtitan.experiments.vem.models.spacetime_loss import SpacetimeEdgeLoss, SpacetimeFaceLoss

class MeshFAE(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["node_norm", "norm1", "norm2", "self_attn_norm", "mlp_norm"]
    _no_split_modules = []
    _keep_in_fp32_modules = ["node_norm", "norm1", "norm2", "self_attn_norm", "mlp_norm"]
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
        # use_kl: bool = False,
        bottleneck: str = "kl",
        noise_interpolate: float = 0.0,
        kl_weight: float = 1e-4,
        dice_weight: float = 0.0,
        gated: bool = False,
        full_attn_interval: int = 0,
        spacetime_dim: int = 64,
        spacetime_proj: bool = False,
        chunk_size: int = 2_000_000,
        dist_scale: str = 'no',
        face_weight: float = 0.1,
        face_loss_style: str = 'mink',
        face_dim_split: bool = False,
        face_dim: int = 0,
        num_decoder_layers: int = 0,
        use_flash_attn_3: bool = False,
        attn_bias: bool = False,
        is_causal: bool = True, # backward compatibility
    ):
        super().__init__()

        assert bottleneck in ["kl", 'ln+noise']
        if bottleneck == "ln+noise":
            assert noise_interpolate > 0
        
        self.num_layers = num_layers
        self.spacetime_dim = spacetime_dim
        self.spacetime_proj = spacetime_proj
        self.bottleneck = bottleneck
        
        input_edge_dim = 10
        # Input projection
        self.freq_emb = FrequencyPositionalEmbedding(
            num_freqs=num_freqs,
            logspace=True,
            input_dim=3,
            include_input=True,
            include_pi=False,
        )

        self.node_proj = nn.Linear(self.freq_emb.out_dim + 1, node_hidden_dim)
        
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
                        use_flash_attn_3=use_flash_attn_3,
                        contains_cross_attention=False,
                        attn_dtype="bf16",
                        gated=gated,
                        attention_bias=attn_bias,
                        is_causal=is_causal,
                    )
                )
                layers_type.append("TransformerBlock")
        
        self.layers = nn.ModuleList(layers)
        self.layers_type = layers_type

        if bottleneck_dim == spacetime_dim:
            self.to_out = nn.Identity()
        else:
            self.to_out = nn.Linear(bottleneck_dim, spacetime_dim)

        if self.bottleneck == "kl":
            self.bn_norm = FP32LayerNorm(node_hidden_dim)
            self.to_mu = nn.Linear(node_hidden_dim, bottleneck_dim)
            self.to_logvar = nn.Linear(node_hidden_dim, bottleneck_dim)
        elif self.bottleneck == "ln+noise":
            self.to_bn = nn.Linear(node_hidden_dim, bottleneck_dim)
            self.bn_norm = FP32LayerNorm(bottleneck_dim)
        else:
            raise NotImplementedError

        self.undirected = undirected

        self.edge_loss = SpacetimeEdgeLoss(
            use_pred_balanced_loss=True, 
            pred_loss_weight=1.0,
            use_dice_loss=dice_weight > 0.0, 
            dice_weight=dice_weight,
            chunk_size=chunk_size,
            scale=dist_scale,
        )

        self.face_loss = SpacetimeFaceLoss(
            use_pred_balanced_loss=True, 
            pred_loss_weight=1.0,
            use_dice_loss=dice_weight > 0.0, 
            dice_weight=dice_weight,
            chunk_size=chunk_size,
            scale=dist_scale,
            style=face_loss_style,
        )
        self.face_weight = face_weight
        self.face_dim_split = face_dim_split
        self.face_dim = face_dim

    def init_weights(self, buffer_device=None):
        self.apply(_basic_init)
        self.freq_emb.init_weights()
        if isinstance(self.to_out, nn.Linear):
            nn.init.kaiming_normal_(self.to_out.weight)
        
    def forward_encoder(self, x, edge_index, offsets):
        for layer_type, layer in zip(self.layers_type, self.layers):
            # Apply convolution
            if layer_type == "GraphConvBlock":
                x = layer(x, edge_index)
            else:
                x = layer(
                    x, 
                    cu_seqlens=offsets,
                )
        
        return x
    
    def encode(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        offsets: torch.Tensor,
        vertex_mask: Optional[torch.Tensor] = None,
    ):
        return self.forward(pos, edge_index, offsets, vertex_mask)
    
    def forward(
        self, 
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        offsets: torch.Tensor,
        vertex_mask: Optional[torch.Tensor] = None,
        noise_interpolate: float = 0.0,
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
            x = torch.cat([x, vertex_mask.to(dtype=x.dtype).unsqueeze(-1)], dim=-1)
        # Project to hidden dimension
        x = self.node_proj(x.to(dtype=self.node_proj.weight.dtype))

        x = self.forward_encoder(x, edge_index, offsets)
        if self.bottleneck == "kl":
            bn = self.bn_norm(x)
            
            mu = self.to_mu(bn)
            logvar = self.to_logvar(bn)

            noise = torch.randn_like(mu, dtype=torch.float32)
            x_sample = mu.float() + noise * torch.exp(0.5 * logvar.float())
            x_sample = x_sample.to(dtype=mu.dtype)

            pred = self.to_out(x_sample)

            return {
                'node_embed_mu': mu,
                'node_embed_logvar': logvar,
                'st_feat': pred,
            }
        elif self.bottleneck == "ln+noise":
            x = self.to_bn(x)
            mu = self.bn_norm(x)

            lengths = offsets[1:] - offsets[:-1]
            bs = offsets.shape[0] - 1
            noise_level = torch.rand(bs, dtype=torch.float32, device=x.device) * noise_interpolate
            noise_level = torch.repeat_interleave(noise_level, lengths).unsqueeze(1)

            noise = torch.randn_like(mu, dtype=torch.float32)
            x_sample = (1 - noise_level) * mu.float() + noise_level * noise
            x_sample = x_sample.to(dtype=mu.dtype)

            pred = self.to_out(x_sample)

            return {
                'node_embed_mu': mu,
                'st_feat': pred,
            }

    
    def loss_fn(self, pred_dict, input_dict):
        pair = input_dict['vertex_pair']
        target = input_dict['vertex_pair_label']
        triplet = input_dict['vertex_triplet']
        triplet_target = input_dict['vertex_triplet_label']
        vertex_mask = input_dict['vertex_mask']
        log_dict = {}

        # --- sample KL or not ---
        st_feat = pred_dict['st_feat']
        if self.face_dim_split:
            edge_feat = st_feat[:, :-self.face_dim]
            face_feat = st_feat[:, -self.face_dim:]
        else:
            edge_feat = st_feat
            face_feat = st_feat
        edge_loss, edge_log_dict = self.edge_loss(edge_feat, pair, target)
        face_loss, face_log_dict = self.face_loss(face_feat, triplet, triplet_target)

        loss = edge_loss + self.face_weight * face_loss
        log_dict.update({
            'edge': edge_loss.detach(),
            'face': face_loss.detach(),
        })
        log_dict.update(edge_log_dict)
        log_dict.update({'f_' + k: v for k, v in face_log_dict.items()})

        # KL part
        if self.bottleneck == "kl":
            mu = pred_dict['node_embed_mu'].float()
            logvar = pred_dict['node_embed_logvar'].float()
            logvar_v = logvar[vertex_mask]
            mu_v = mu[vertex_mask]
            kl_loss = torch.mean(
                -0.5 * torch.sum(1 + logvar_v - mu_v.pow(2) - logvar_v.exp(), dim=-1)
            )
            log_dict["kl_loss"] = kl_loss.detach()
            loss += kl_loss * self.config.kl_weight

        return loss, log_dict
    
    def train_step(self, input_dict):
        pred = self(
            pos=input_dict['nodes'],
            edge_index=input_dict['edges'].permute(1, 0),
            offsets=input_dict['offsets'],
            vertex_mask=input_dict['vertex_mask'],
            noise_interpolate=self.config.noise_interpolate,
        )
        loss, log_dict = self.loss_fn(pred, input_dict)
        return loss, log_dict, {
            "num_tokens": input_dict['edges'].shape[0],
            "num_flops": 100,
        }