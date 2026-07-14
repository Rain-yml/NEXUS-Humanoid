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
    RotaryPosEmbed3D,
)
from torchtitan.experiments.vem.models.mesh_gnn import MeshTransformerBlock
from torchtitan.experiments.vem.models.spacetime_loss import (
    SpacetimeEdgeLoss, 
    SpacetimeFaceLoss, 
    DeterminantOrientLoss, 
    SpacetimeMultiheadEdgeLoss,
    SpacetimeMultiheadFaceLoss,
)
from torchtitan.experiments.vem.models.mesh_gae import (
    GraphConvBlock, 
)


class MeshFAEDecoder(ModelMixin, ConfigMixin):
    """Mesh FAE with additional full attention decoder after bottleneck.
    
    The decoder operates only on vertex tokens, not face tokens.
    This requires computing separate cu_seqlens for vertex-only sequences.
    """
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["node_norm", "norm1", "norm2", "self_attn_norm", "mlp_norm"]
    _no_split_modules = []
    _keep_in_fp32_modules = ["node_norm", "norm1", "norm2", "self_attn_norm", "mlp_norm"]
    _keys_to_ignore_on_load_unexpected = []
    
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
        use_pred_balanced_loss: bool = True,
        # New decoder params
        decoder_dim: int = 1024,
        decoder_ffn_dim: int = 4096,
        decoder_layers: int = 16,
        decoder_heads: int = 16,
        attn_bias: bool = False,
        pred_orient: bool = False,
        orient_embed_dim: int = 3,
        pos_embed_type: str = 'fourier',
        max_seq_len_rope: int = 10000,
        rope_theta: float = 10000.0,
        spacetime_heads: int = 1,
        attn_dtype: str = "bf16",
        apply_face_loss: bool = True,
    ):
        super().__init__()
        assert bottleneck in ["kl", 'ln+noise']
        if bottleneck == "ln+noise":
            assert noise_interpolate > 0
        assert pos_embed_type in ["fourier", "rotary", "origin", "zero_rotary"]
        assert attn_dtype in ["bf16", "fp16"]
        
        self.num_layers = num_layers
        self.spacetime_dim = spacetime_dim
        self.spacetime_proj = spacetime_proj
        self.bottleneck = bottleneck
        self.pred_orient = pred_orient
        self.orient_embed_dim = orient_embed_dim
        self.pos_embed_type = pos_embed_type
        self.attn_dtype = attn_dtype
        self.apply_face_loss = apply_face_loss
        self.has_rope = pos_embed_type in ['rotary', 'zero_rotary']
        
        if self.pos_embed_type in ['rotary', 'zero_rotary']:
            self.rope_3d = RotaryPosEmbed3D(
                attention_head_dim=node_hidden_dim // heads,
                max_seq_len=max_seq_len_rope,
                theta=rope_theta,
            )
            self.rope_3d_decoder = RotaryPosEmbed3D(
                attention_head_dim=decoder_dim // decoder_heads,
                max_seq_len=max_seq_len_rope,
                theta=rope_theta,
            )
            self.freq_emb = None
            self.node_proj = nn.Linear(input_node_dim + 1, node_hidden_dim)
        elif self.pos_embed_type == "fourier":
            self.rope_3d = None
            self.rope_3d_decoder = None
            self.freq_emb = FrequencyPositionalEmbedding(
                num_freqs=num_freqs,
                logspace=True,
                input_dim=3,
                include_input=True,
                include_pi=False,
            )
            self.node_proj = nn.Linear(self.freq_emb.out_dim + 1 + input_node_dim - 3, node_hidden_dim)
        else:
            self.rope_3d = None
            self.rope_3d_decoder = None
            self.freq_emb = None
            self.node_proj = nn.Linear(input_node_dim + 1, node_hidden_dim)
        
        # Graph convolution layers (encoder)
        layers = []
        layers_type = []
        for i in range(num_layers):
            if full_attn_interval == -1:
                # No GraphConv, only TransformerBlock
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
                        attn_dtype=self.attn_dtype,
                        gated=gated,
                        is_causal=False,
                        attention_bias=attn_bias,
                    )
                )
                layers_type.append("TransformerBlock")
            else:
                # Normal: GraphConv + optional TransformerBlock
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
                            rope=self.rope_3d,
                            qk_norm=True,
                            use_flash_attn_3=False,
                            contains_cross_attention=False,
                            attn_dtype=self.attn_dtype,
                            gated=gated,
                            is_causal=False,  # Non-causal for mesh vertices (no ordering)
                            attention_bias=attn_bias,
                        )
                    )
                    layers_type.append("TransformerBlock")
        
        self.layers = nn.ModuleList(layers)
        self.layers_type = layers_type

        # Bottleneck
        if self.bottleneck == "kl":
            self.bn_norm = FP32LayerNorm(node_hidden_dim)
            self.to_mu = nn.Linear(node_hidden_dim, bottleneck_dim)
            self.to_logvar = nn.Linear(node_hidden_dim, bottleneck_dim)
        elif self.bottleneck == "ln+noise":
            self.to_bn = nn.Linear(node_hidden_dim, bottleneck_dim)
            # Use parameter-free LayerNorm so output is close to standard Gaussian
            # This enables proper interpolation with Gaussian noise
            self.bn_norm = FP32LayerNorm(bottleneck_dim, elementwise_affine=False)
        else:
            raise NotImplementedError

        # ========== NEW: Full attention decoder (vertex-only) ==========
        # Note: avoid storing integer attributes with prefixes that match ModuleList names
        # (e.g., decoder_dim vs decoder_layers) as it causes PyTorch distributed checkpoint issues
        
        self.use_decoder = decoder_layers > 0
        
        if self.use_decoder:
            # Project from bottleneck_dim to decoder hidden dim
            self.dec_proj_in = nn.Linear(bottleneck_dim, decoder_dim)
            
            # Full attention decoder layers (non-causal for vertex tokens)
            self.dec_blocks = nn.ModuleList([
                TransformerBlock(
                    layer_idx=i,
                    hidden_size=decoder_dim,
                    num_attention_heads=decoder_heads,
                    intermediate_size=decoder_ffn_dim,
                    num_key_value_heads=None,
                    qk_norm=True,
                    use_flash_attn_3=False,
                    attn_dtype=self.attn_dtype,
                    gated=gated,
                    is_causal=False,  # Non-causal for vertex tokens (no ordering)
                    attention_bias=attn_bias,
                    rope=self.rope_3d_decoder,
                )
                for i in range(decoder_layers)
            ])
            
            # Project from decoder hidden dim to spacetime_dim
            self.dec_proj_out = nn.Linear(decoder_dim, spacetime_dim)
            self.dec_norm = RMSNorm(decoder_dim, eps=1e-6)
            if self.pred_orient:
                self.dec_orient = nn.Linear(decoder_dim, orient_embed_dim)
                self.orient_loss = DeterminantOrientLoss(orient_embed_dim)

            self.to_out = None
        else:
            # to_out: used when decoder_layers=0, or as base path
            if bottleneck_dim == spacetime_dim:
                self.to_out = nn.Identity()
            else:
                self.to_out = nn.Linear(bottleneck_dim, spacetime_dim)

        self.undirected = undirected

        if spacetime_heads > 1:
            self.edge_loss = SpacetimeMultiheadEdgeLoss(
                use_pred_balanced_loss=use_pred_balanced_loss, 
                pred_loss_weight=1.0,
                use_dice_loss=dice_weight > 0.0, 
                dice_weight=dice_weight,
                chunk_size=chunk_size,
                scale=dist_scale,
                heads=spacetime_heads,
                dim=spacetime_dim - face_dim,
            )
            self.face_loss = SpacetimeMultiheadFaceLoss(
                use_pred_balanced_loss=use_pred_balanced_loss, 
                pred_loss_weight=1.0,
                use_dice_loss=dice_weight > 0.0, 
                dice_weight=dice_weight,
                chunk_size=chunk_size,
                scale=dist_scale,
                style=face_loss_style,
                heads=spacetime_heads,
                dim=face_dim,
            )
        else:
            self.edge_loss = SpacetimeEdgeLoss(
                use_pred_balanced_loss=use_pred_balanced_loss, 
                pred_loss_weight=1.0,
                use_dice_loss=dice_weight > 0.0, 
                dice_weight=dice_weight,
                chunk_size=chunk_size,
                scale=dist_scale,
            )
            self.face_loss = SpacetimeFaceLoss(
                use_pred_balanced_loss=use_pred_balanced_loss, 
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
        if self.freq_emb is not None:
            self.freq_emb.init_weights()
        if self.rope_3d is not None:
            self.rope_3d.init_weights()
        if self.rope_3d_decoder is not None:
            self.rope_3d_decoder.init_weights()
        if self.use_decoder:
            nn.init.kaiming_normal_(self.dec_proj_in.weight)
            nn.init.kaiming_normal_(self.dec_proj_out.weight)
            if self.pred_orient:
                nn.init.kaiming_normal_(self.dec_orient.weight)
        if isinstance(self.to_out, nn.Linear):
            nn.init.kaiming_normal_(self.to_out.weight)
        
    def forward_encoder(
        self, 
        x, 
        edge_index, 
        offsets,
        position: Optional[torch.Tensor] = None
    ):
        for layer_type, layer in zip(self.layers_type, self.layers):
            # Apply convolution
            if layer_type == "GraphConvBlock":
                x = layer(x, edge_index)
            else:
                x = layer(
                    x, 
                    cu_seqlens=offsets,
                    position_ids=position,
                )
        
        return x
    
    def compute_vertex_cu_seqlens(self, vertex_mask, offsets):
        """Compute cu_seqlens for vertex-only tokens.
        
        Args:
            vertex_mask: [N] boolean mask, True for vertices
            offsets: [B+1] cumulative sequence lengths for all tokens
            
        Returns:
            vertex_cu_seqlens: [B+1] cumulative sequence lengths for vertex-only tokens
        """
        B = offsets.shape[0] - 1
        vertex_counts = []
        
        for i in range(B):
            start = offsets[i].item()
            end = offsets[i + 1].item()
            # Count vertices in this batch element
            nv = vertex_mask[start:end].sum().item()
            vertex_counts.append(nv)
        
        # Build cumulative sums
        vertex_cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=offsets.device)
        for i, count in enumerate(vertex_counts):
            vertex_cu_seqlens[i + 1] = vertex_cu_seqlens[i] + count
        
        return vertex_cu_seqlens
    
    def forward_decoder(
        self, 
        x_vertex, 
        vertex_cu_seqlens,
        position: Optional[torch.Tensor] = None,
    ):
        assert (position is not None) == self.has_rope
        """Forward through the full attention decoder.
        
        Args:
            x_vertex: [Nv, bottleneck_dim] vertex-only features
            vertex_cu_seqlens: [B+1] cumulative sequence lengths
            
        Returns:
            decoded: [Nv, spacetime_dim] decoded features
        """
        # Project to decoder dimension
        x = self.dec_proj_in(x_vertex)
        
        # Apply decoder layers
        for layer in self.dec_blocks:
            x = layer(
                x, 
                cu_seqlens=vertex_cu_seqlens, 
                position_ids=position,
            )
        
        # Final norm and projection
        x = self.dec_norm(x)
        x_dec = self.dec_proj_out(x)
        
        return x_dec, x
    
    def encode(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        offsets: torch.Tensor,
        vertex_mask: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
    ):
        assert (position is not None) == self.has_rope, f"{position is not None} != {self.has_rope}"
        if self.undirected:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        if self.pos_embed_type == 'fourier':
            x = self.freq_emb(pos[:, :3])
            x = torch.cat([x, pos[:, 3:]], dim=-1)
        elif self.pos_embed_type in ['zero', 'zero_rotary']:
            x = torch.cat([torch.zeros_like(pos[:, :3]), pos[:, 3:]], dim=-1)
        else:
            x = pos
        if vertex_mask is not None:
            x = torch.cat([x, vertex_mask.to(dtype=x.dtype).unsqueeze(-1)], dim=-1)
        # Project to hidden dimension
        x = self.node_proj(x.to(dtype=self.node_proj.weight.dtype))

        # Encoder
        x = self.forward_encoder(x, edge_index, offsets, position=position)

        if self.bottleneck == "kl":
            bn = self.bn_norm(x)
            
            mu = self.to_mu(bn)
            logvar = self.to_logvar(bn)
            
        elif self.bottleneck == "ln+noise":
            x = self.to_bn(x)
            mu = self.bn_norm(x)
            logvar = None
        
        return {
            'node_embed_mu': mu,
            'node_embed_logvar': logvar,
        }
    
    def is_scalar_param(self, name, param):
        patterns = [
            'node_proj',
            'to_mu',
            'to_logvar',
            'dec_proj_in',
            'dec_proj_out',
            'dec_orient',
            'orient_loss.scale.',
            '.scale_d.',
        ]
        for p in patterns:
            if p in name:
                return True
        return False
    
    def forward(
        self, 
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        offsets: torch.Tensor,
        vertex_mask: Optional[torch.Tensor] = None,
        noise_interpolate: float = 0.0,
        position: Optional[torch.Tensor] = None,
    ) -> dict:
        '''
        nodes: N x 3
        edge_index: 2 x E
        vertex_mask: N (True for vertices, False for face centers)
        offsets: B + 1
        '''
        ret_encoder = self.encode(pos, edge_index, offsets, vertex_mask, position=position)

        mu = ret_encoder['node_embed_mu']
        logvar = ret_encoder['node_embed_logvar']
        # Bottleneck
        if self.bottleneck == "kl":
            if self.training:
                noise = torch.randn_like(mu, dtype=torch.float32)
                x_sample = mu.float() + noise * torch.exp(0.5 * logvar.float())
                x_sample = x_sample.to(dtype=mu.dtype)
            else:
                x_sample = mu.clone()
            
        elif self.bottleneck == "ln+noise":
            lengths = offsets[1:] - offsets[:-1]
            bs = offsets.shape[0] - 1
            noise_level = torch.rand(bs, dtype=torch.float32, device=mu.device) * noise_interpolate
            noise_level = torch.repeat_interleave(noise_level, lengths).unsqueeze(1)

            noise = torch.randn_like(mu, dtype=torch.float32)
            x_sample = (1 - noise_level) * mu.float() + noise_level * noise
            x_sample = x_sample.to(dtype=mu.dtype)
            logvar = None

        # ========== Decoder: only on vertex tokens ==========
        if self.use_decoder:
            # Extract vertex tokens only
            x_vertex = x_sample[vertex_mask]  # [Nv, bottleneck_dim]
            
            # Compute vertex-only cu_seqlens
            vertex_cu_seqlens = self.compute_vertex_cu_seqlens(vertex_mask, offsets)
            
            # Forward through decoder
            pred_vertex, h_norm = self.forward_decoder(x_vertex, vertex_cu_seqlens, position=position[vertex_mask] if position is not None else None)  # [Nv, spacetime_dim]
            
            # Create full prediction tensor (vertex features only matter for loss)
            # For face tokens, we can just use zeros or the original bottleneck output
            pred = torch.zeros(x_sample.shape[0], self.spacetime_dim, 
                              dtype=pred_vertex.dtype, device=pred_vertex.device)
            pred[vertex_mask] = pred_vertex
            if self.pred_orient:
                orient_embed = torch.zeros(x_sample.shape[0], self.orient_embed_dim, dtype=h_norm.dtype, device=h_norm.device)
                orient_embed[vertex_mask] = self.dec_orient(h_norm)
        else:
            # No decoder, just use to_out like MeshFAE
            pred = self.to_out(x_sample)

        ret = {
            'node_embed_mu': mu,
            'st_feat': pred,
        }
        if self.bottleneck == "kl":
            ret.update({
                'node_embed_logvar': logvar,
            })
        if self.pred_orient:
            ret.update({
                'orient_embed': orient_embed,
            })
        return ret

    
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
        if self.apply_face_loss:
            face_loss, face_log_dict = self.face_loss(face_feat, triplet, triplet_target)
            loss = edge_loss + self.face_weight * face_loss
            log_dict.update({
                'edge': edge_loss.detach(),
                'face': face_loss.detach(),
            })
        else:
            face_loss = 0.0
            face_log_dict = {}
            loss = edge_loss
            log_dict.update({
                'edge': edge_loss.detach(),
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
        
        if self.pred_orient:
            orient_embed = pred_dict['orient_embed']
            face_in_order = input_dict['face_in_order']
            orient_label = input_dict['orient_label']
            orient_loss, orient_log_dict = self.orient_loss(orient_embed, face_in_order, orient_label)
            loss += self.face_weight * orient_loss
            log_dict.update({'ori_' + k: v for k, v in orient_log_dict.items()})

        return loss, log_dict
    
    def train_step(self, input_dict):
        assert self.training
        pred = self(
            pos=input_dict['nodes'],
            edge_index=input_dict['edges'].permute(1, 0),
            offsets=input_dict['offsets'],
            vertex_mask=input_dict['vertex_mask'],
            noise_interpolate=self.config.noise_interpolate,
            position=input_dict.get('position', None),
        )
        loss, log_dict = self.loss_fn(pred, input_dict)
        return loss, log_dict, {
            "num_tokens": input_dict['edges'].shape[0],
            "num_flops": 100,
        }
