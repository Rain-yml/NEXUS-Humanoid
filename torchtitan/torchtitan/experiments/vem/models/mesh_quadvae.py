from dataclasses import dataclass, asdict, replace
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
from torchtitan.experiments.vem.models.mesh_gae import GraphConvBlock
from torchtitan.experiments.vem.models.spacetime_loss import (
    SpacetimeAllPairEdgeLoss,
    SpacetimeFaceLoss, 
    DeterminantOrientLoss, 
    SpacetimeMultiheadEdgeLoss,
    SpacetimeMultiheadFaceLoss,
)
from torchtitan.experiments.vem.models.spacetime_loss_fast import FlashSpacetimeAllPairEdgeLoss
from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol
from torchtitan.tools.logging import logger

class MeshQuadVAE(ModelMixin, ConfigMixin):
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
        # loss
        chunk_size: int = 2_000_000,
        dist_scale: str = 'no',
        face_weight: float = 0.6,
        diag_weight: float = 0.6,
        orient_weight: float = 0.6,
        face_loss_style: str = 'mink',
        face_embed_dim: int = 64,
        edge_embed_dim: int = 64,
        pred_diag: bool = True,
        diag_embed_dim: int = 64,
        pred_orient: bool = False,
        orient_embed_dim: int = 3,
        # New decoder params
        decoder_dim: int = 1024,
        decoder_ffn_dim: int = 4096,
        decoder_layers: int = 16,
        decoder_heads: int = 16,
        attn_bias: bool = False,
        pos_embed_type: str = 'fourier',
        max_seq_len_rope: int = 10000,
        rope_theta: float = 10000.0,
        attn_dtype: str = "bf16",
        node_type_embed_dim: int = 1,
        flash_loss: bool = False,
        use_flash_attn_3: bool = False,
    ):
        super().__init__()
        assert bottleneck in ["kl", 'ln+noise']
        if bottleneck == "ln+noise":
            assert noise_interpolate > 0
        assert pos_embed_type in ["zero_rotary", "rotary"]
        assert attn_dtype in ["bf16", "fp16"]
        assert node_type_embed_dim > 0
        
        self.num_layers = num_layers
        self.undirected = undirected
        self.bottleneck = bottleneck
        self.pred_diag = pred_diag
        self.pred_orient = pred_orient
        self.orient_embed_dim = orient_embed_dim
        self.pos_embed_type = pos_embed_type
        self.attn_dtype = attn_dtype
        self.has_rope = pos_embed_type in ['rotary', 'zero_rotary']

        self.edge_dim = edge_embed_dim
        self.diag_dim = diag_embed_dim if pred_diag else 0
        self.face_dim = face_embed_dim
        self.orient_dim = orient_embed_dim if pred_orient else 0
        self.output_dim = face_embed_dim + edge_embed_dim + self.diag_dim + self.orient_dim
        self.node_type_embed = nn.Embedding(32, node_type_embed_dim)
        
        if self.pos_embed_type in ['zero_rotary', 'rotary']:
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
            self.node_proj = nn.Linear(input_node_dim + node_type_embed_dim, node_hidden_dim)
        else:
            raise NotImplementedError
        
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
                        use_flash_attn_3=use_flash_attn_3,
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
                            use_flash_attn_3=use_flash_attn_3,
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
                use_flash_attn_3=use_flash_attn_3,
                attn_dtype=self.attn_dtype,
                gated=gated,
                is_causal=False,  # Non-causal for vertex tokens (no ordering)
                attention_bias=attn_bias,
                rope=self.rope_3d_decoder,
            )
            for i in range(decoder_layers)
        ])
        
        # Project from decoder hidden dim to spacetime_dim
        self.dec_proj_out = nn.Linear(decoder_dim, self.output_dim)
        self.dec_norm = RMSNorm(decoder_dim, eps=1e-6)
        if self.pred_orient:
            self.orient_loss = DeterminantOrientLoss(orient_embed_dim)

        if flash_loss:
            self.edge_loss = FlashSpacetimeAllPairEdgeLoss(
                use_pred_balanced_loss=True,
                pred_loss_weight=1.0,
                use_dice_loss=dice_weight > 0.0, 
                dice_weight=dice_weight,
                chunk_size=chunk_size,
                scale=dist_scale,
            )
        else:
            self.edge_loss = SpacetimeAllPairEdgeLoss(
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

        if self.pred_diag:
            if flash_loss:
                self.diag_loss = FlashSpacetimeAllPairEdgeLoss(
                    use_pred_balanced_loss=True,
                    pred_loss_weight=1.0,
                    use_dice_loss=dice_weight > 0.0,
                    dice_weight=dice_weight,
                    chunk_size=chunk_size,
                    scale=dist_scale,
                )
            else:
                self.diag_loss = SpacetimeAllPairEdgeLoss(
                    use_pred_balanced_loss=True,
                    pred_loss_weight=1.0,
                    use_dice_loss=dice_weight > 0.0,
                    dice_weight=dice_weight,
                    chunk_size=chunk_size,
                    scale=dist_scale,
                )


    def init_weights(self, buffer_device=None):
        self.apply(_basic_init)
        if self.freq_emb is not None:
            self.freq_emb.init_weights()
        if self.rope_3d is not None:
            self.rope_3d.init_weights()
        if self.rope_3d_decoder is not None:
            self.rope_3d_decoder.init_weights()
        nn.init.normal_(self.node_type_embed.weight, mean=0.0, std=0.02)
        # nn.init.kaiming_normal_(self.dec_proj_in.weight)
        # nn.init.kaiming_normal_(self.dec_proj_out.weight)
        
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

    @torch.no_grad()
    def decode_face(
        self,
        vertex_latents: torch.Tensor,
        vertex_positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        mode: str = "native_quad",
        decoder_positions: Optional[torch.Tensor] = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Decode vertex latents to recover mesh topology.

        Args:
            vertex_latents: [Nv, bottleneck_dim] vertex-only embeddings (post-bottleneck)
            vertex_positions: [Nv, 3] vertex 3D positions
            cu_seqlens: [B+1] cumulative vertex counts per mesh
            mode: Recovery mode - "tri", "tri_connect", "native_quad", or "native_quad_wireframe"
            decoder_positions: Optional [Nv, 3] precomputed positions for RoPE

        Returns:
            List of (vertices_np, triangles, quads) tuples, one per mesh in batch
            - vertices_np: [nv, 3] float32 array of vertex positions
            - triangles: [nt, 3] int64 array of triangle indices
            - quads: [nq, 4] int64 array of quad indices
        """
        # Lazy import to avoid circular dependency
        from torchtitan.experiments.vem.rewards.quad_decoder_recovery import recover_meshes_from_embeddings

        return recover_meshes_from_embeddings(
            model=self,
            vertex_latents=vertex_latents,
            vertex_positions=vertex_positions,
            cu_seqlens=cu_seqlens,
            mode=mode,
            decoder_positions=decoder_positions,
        )

    def encode(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        offsets: torch.Tensor,
        node_type: torch.Tensor,
        position: Optional[torch.Tensor] = None,
    ):
        assert (position is not None) == self.has_rope, f"{position is not None} != {self.has_rope}"
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        if self.pos_embed_type == 'zero_rotary':
            x = torch.cat([torch.zeros_like(pos[:, :3]), pos[:, 3:]], dim=-1)
        elif self.pos_embed_type == 'rotary':
            x = pos
        else:
            raise NotImplementedError
        
        node_type_embed = self.node_type_embed(node_type.long()).to(dtype=x.dtype)
        x = torch.cat([x, node_type_embed], dim=-1)
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
        node_type: torch.Tensor,
        noise_interpolate: float = 0.0,
        position: Optional[torch.Tensor] = None,
    ) -> dict:
        '''
        nodes: N x 3
        edge_index: 2 x E
        node_type: N (0 for vertices, 1 for triangle faces, 2 for quad faces)
        offsets: B + 1
        '''
        ret_encoder = self.encode(pos, edge_index, offsets, node_type, position=position)
        vertex_mask = node_type == 0

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
        # Extract vertex tokens only
        x_vertex = x_sample[vertex_mask]  # [Nv, bottleneck_dim]
        
        # Compute vertex-only cu_seqlens
        vertex_cu_seqlens = self.compute_vertex_cu_seqlens(vertex_mask, offsets)
        
        # Forward through decoder
        pred_vertex, _ = self.forward_decoder(x_vertex, vertex_cu_seqlens, position=position[vertex_mask] if position is not None else None)  # [Nv, spacetime_dim]
        
        # Create full prediction tensor (vertex features only matter for loss)
        # For face tokens, we can just use zeros or the original bottleneck output
        pred = torch.zeros(x_sample.shape[0], self.output_dim,
                            dtype=pred_vertex.dtype, device=pred_vertex.device)
        pred[vertex_mask] = pred_vertex
        edge_embed, diag_embed, face_embed, orient_embed = torch.split(pred, [self.edge_dim, self.diag_dim, self.face_dim, self.orient_dim], dim=-1)

        ret = {
            'node_embed_mu': mu,
            'st_feat': pred,
            'edge_embed': edge_embed,
            'diag_embed': diag_embed,
            'face_embed': face_embed,
            'orient_embed': orient_embed,
        }
        if self.bottleneck == "kl":
            ret.update({
                'node_embed_logvar': logvar,
            })
        return ret

    
    def loss_fn(self, pred_dict, input_dict):
        pair = input_dict['vertex_pair']
        triplet = input_dict['vertex_triplet']
        triplet_target = input_dict['vertex_triplet_label']
        vertex_mask = input_dict['node_type'] == 0

        # Compute the spacetime edge/diag losses in fp32. The bf16 gradient differs
        # between the flash kernel and the reference at the ~1e-3 level, which gets
        # amplified into divergent trajectories at the sharp loss transition; fp32 makes
        # the flash and non-flash losses numerically equivalent (grad cosine ~1.0).
        edge_feat = pred_dict['edge_embed'].float()
        face_feat = pred_dict['face_embed']
        orient_feat = pred_dict['orient_embed']

        log_dict = {}

        # --- sample KL or not ---
        edge_loss, edge_log_dict = self.edge_loss(
            edge_feat,
            pair,
            input_dict['pair_offsets'],
            input_dict['offsets'],
            vertex_mask,
        )
        face_loss, face_log_dict = self.face_loss(face_feat, triplet, triplet_target)

        loss = edge_loss + self.config.face_weight * face_loss
        log_dict.update({
            'edge': edge_loss.detach(),
            'face': face_loss.detach(),
        })

        log_dict.update(edge_log_dict)
        log_dict.update({'f_' + k: v for k, v in face_log_dict.items()})

        if self.pred_diag:
            diag_feat = pred_dict['diag_embed'].float()
            diag_loss, diag_log_dict = self.diag_loss(
                diag_feat,
                input_dict['vertex_pair_diag'],
                input_dict['diag_pair_offsets'],
                input_dict['offsets'],
                vertex_mask,
            )
            loss = loss + self.config.diag_weight * diag_loss
            log_dict['diag'] = diag_loss.detach()
            log_dict.update({'d_' + k: v for k, v in diag_log_dict.items()})

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
            face_in_order = input_dict['face_in_order']
            orient_label = input_dict['orient_label']
            orient_loss, orient_log_dict = self.orient_loss(orient_feat, face_in_order, orient_label)
            loss += self.config.orient_weight * orient_loss
            log_dict.update({'ori_' + k: v for k, v in orient_log_dict.items()})

        return loss, log_dict
    
    def train_step(self, input_dict):
        assert self.training
        pred = self(
            pos=input_dict['nodes'],
            edge_index=input_dict['edges'].permute(1, 0),
            offsets=input_dict['offsets'],
            node_type=input_dict['node_type'],
            noise_interpolate=self.config.noise_interpolate,
            position=input_dict.get('position', None),
        )
        loss, log_dict = self.loss_fn(pred, input_dict)
        return loss, log_dict, {
            "num_tokens": input_dict['edges'].shape[0],
            "num_flops": 100,
        }


@dataclass
class MeshQuadVAEArgs(BaseModelArgs):
    input_node_dim: int = 3
    node_hidden_dim: int = 64
    ffn_dim: int = 128
    bottleneck_dim: int = 64
    num_layers: int = 3
    heads: int = 4
    num_freqs: int = 4
    undirected: bool = True
    bottleneck: str = "kl"
    noise_interpolate: float = 0.0
    kl_weight: float = 1e-4
    dice_weight: float = 0.0
    gated: bool = False
    full_attn_interval: int = 0
    chunk_size: int = 2_000_000
    dist_scale: str = "no"
    face_weight: float = 0.6
    diag_weight: float = 0.6
    orient_weight: float = 0.6
    face_loss_style: str = "mink"
    face_embed_dim: int = 64
    edge_embed_dim: int = 64
    pred_diag: bool = True
    diag_embed_dim: int = 64
    pred_orient: bool = False
    orient_embed_dim: int = 3
    decoder_dim: int = 1024
    decoder_ffn_dim: int = 4096
    decoder_layers: int = 16
    decoder_heads: int = 16
    attn_bias: bool = False
    pos_embed_type: str = "zero_rotary"
    max_seq_len_rope: int = 10000
    rope_theta: float = 10000.0
    attn_dtype: str = "bf16"
    node_type_embed_dim: int = 1
    flash_loss: bool = False
    use_flash_attn_3: bool = False

    def get_nparams(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

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


class MeshQuadVAEWrapper(MeshQuadVAE, ModelProtocol):
    def __init__(self, model_args: MeshQuadVAEArgs):
        super().__init__(**asdict(model_args))

    @classmethod
    def from_model_args(cls, model_args: MeshQuadVAEArgs) -> "MeshQuadVAE":
        return cls(model_args)

def overrite_model_args(model_args: MeshQuadVAEArgs, **kwargs):
    return replace(model_args, **kwargs)

mesh_quadvae_configs = {
    'base-decoder-16-face0.6-ab-orient-12-kl-zerorope-nobias-res2048': MeshQuadVAEArgs(
        input_node_dim=6,
        node_hidden_dim=256,
        ffn_dim=1024,
        bottleneck_dim=16,
        num_layers=12,
        heads=8,
        num_freqs=4,
        undirected=True,
        bottleneck='kl',
        noise_interpolate=0.0,
        kl_weight=1e-6,
        gated=False,
        dice_weight=0.5,
        full_attn_interval=1,
        dist_scale='learn_nobias',
        face_weight=0.6,
        face_loss_style='gram_diff',
        face_embed_dim=64,
        edge_embed_dim=64,
        diag_embed_dim=64,
        # Decoder params
        decoder_dim=512,
        decoder_ffn_dim=2048,
        decoder_layers=16,
        decoder_heads=8,
        attn_bias=True,
        pred_orient=True,
        orient_embed_dim=12,
        pos_embed_type='zero_rotary',
        max_seq_len_rope=8192,
        rope_theta=4096 * 3,
        chunk_size=20_000_000,
        node_type_embed_dim=8,
    ),
    'base-decoder-16-face0.6-ab-orient-12-kl-zerorope-nobias-res2048-nodiag': MeshQuadVAEArgs(
        input_node_dim=6,
        node_hidden_dim=256,
        ffn_dim=1024,
        bottleneck_dim=16,
        num_layers=12,
        heads=8,
        num_freqs=4,
        undirected=True,
        bottleneck='kl',
        noise_interpolate=0.0,
        kl_weight=1e-6,
        gated=False,
        dice_weight=0.5,
        full_attn_interval=1,
        dist_scale='learn_nobias',
        face_weight=0.6,
        face_loss_style='gram_diff',
        face_embed_dim=64,
        edge_embed_dim=64,
        pred_diag=False,
        diag_embed_dim=64,
        # Decoder params
        decoder_dim=512,
        decoder_ffn_dim=2048,
        decoder_layers=16,
        decoder_heads=8,
        attn_bias=True,
        pred_orient=True,
        orient_embed_dim=12,
        pos_embed_type='zero_rotary',
        max_seq_len_rope=8192,
        rope_theta=4096 * 3,
        chunk_size=20_000_000,
        node_type_embed_dim=8,
    ),
}

mesh_quadvae_configs['base-decoder-8-face0.6-ab-orient-12-kl-zerorope-nobias-res2048'] = overrite_model_args(
    mesh_quadvae_configs['base-decoder-16-face0.6-ab-orient-12-kl-zerorope-nobias-res2048'],
    bottleneck_dim=8,
)

mesh_quadvae_configs['base-decoder-16-face0.6-ab-orient-12-kl-zerorope-nobias-res2048-kl1e-4'] = overrite_model_args(
    mesh_quadvae_configs['base-decoder-16-face0.6-ab-orient-12-kl-zerorope-nobias-res2048'],
    kl_weight=1e-4,
)

mesh_quadvae_configs['base-decoder-16-face0.6-ab-orient-12-kl-zerorope-nobias-res2048-flash'] = overrite_model_args(
    mesh_quadvae_configs['base-decoder-16-face0.6-ab-orient-12-kl-zerorope-nobias-res2048'],
    flash_loss=True,
)

mesh_quadvae_configs['base-decoder-16-face0.6tridot-ab-orient-12-kl-zerorope-nobias-res2048-flash'] = overrite_model_args(
    mesh_quadvae_configs['base-decoder-16-face0.6-ab-orient-12-kl-zerorope-nobias-res2048'],
    flash_loss=True,
    face_loss_style='tri_dot',
)
