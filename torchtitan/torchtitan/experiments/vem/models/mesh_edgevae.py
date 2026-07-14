from dataclasses import dataclass, asdict, replace
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torchtitan.experiments.vem.models.transformer import (
    RMSNorm,
    FP32LayerNorm,
    _basic_init,
    VEM2DecoderLayer as TransformerBlock,
    RotaryPosEmbed3D,
)
from torchtitan.experiments.vem.models.mesh_gae import GraphConvBlock
from torchtitan.experiments.vem.models.spacetime_loss import SpacetimeAllPairEdgeLoss
from torchtitan.experiments.vem.models.spacetime_loss_fast import FlashSpacetimeAllPairEdgeLoss
from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol
from torchtitan.tools.logging import logger

class MeshEdgeVAE(ModelMixin, ConfigMixin):
    """Edge VAE: encode a mesh *wireframe* into per-vertex latents and decode it back.

    The encoder is a general graph encoder; the input graph is built by the dataset in
    one of three modes (selected purely by ``node_type`` / ``edge_index`` and therefore
    transparent to this model):

      1. ``wireframe``      - nodes = vertices, edges = wireframe edges.
      2. ``bipartite``      - nodes = vertices + edge-nodes, edges = vertex-edge incidences
                              (the two node kinds are distinguished by ``node_type``).
      3. ``bipartite_diag`` - bipartite, plus quad-diagonal nodes as a third node type.

    Only the vertex tokens (``node_type == 0``) are pushed through the decoder and
    KL-regularized; the auxiliary edge/diagonal nodes only enrich the encoder.

    Supervision is edge-only (all-pair edge reconstruction over the wireframe), with an
    optional all-pair diagonal head.  There is no face or orientation supervision.
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
        node_hidden_dim: int = 256,
        ffn_dim: int = 1024,
        bottleneck_dim: int = 16,
        num_layers: int = 12,
        heads: int = 8,
        undirected: bool = True,
        bottleneck: str = "kl",
        noise_interpolate: float = 0.0,
        kl_weight: float = 1e-6,
        dice_weight: float = 0.5,
        gated: bool = False,
        full_attn_interval: int = 1,
        # loss
        chunk_size: int = 20_000_000,
        dist_scale: str = "learn_nobias",
        edge_embed_dim: int = 64,
        # optional diagonal head
        pred_diag: bool = False,
        diag_weight: float = 0.6,
        diag_embed_dim: int = 64,
        # decoder params
        decoder_dim: int = 512,
        decoder_ffn_dim: int = 2048,
        decoder_layers: int = 16,
        decoder_heads: int = 8,
        attn_bias: bool = True,
        pos_embed_type: str = "zero_rotary",
        max_seq_len_rope: int = 8192,
        rope_theta: float = 4096 * 3,
        attn_dtype: str = "bf16",
        node_type_embed_dim: int = 8,
        flash_loss: bool = True,
        use_flash_attn_3: bool = False,
    ):
        super().__init__()
        assert bottleneck in ["kl", "ln+noise"]
        if bottleneck == "ln+noise":
            assert noise_interpolate > 0
        assert pos_embed_type in ["zero_rotary", "rotary"]
        assert attn_dtype in ["bf16", "fp16"]
        assert node_type_embed_dim > 0

        self.num_layers = num_layers
        self.undirected = undirected
        self.bottleneck = bottleneck
        self.pred_diag = pred_diag
        self.pos_embed_type = pos_embed_type
        self.attn_dtype = attn_dtype
        self.has_rope = pos_embed_type in ["rotary", "zero_rotary"]

        self.edge_dim = edge_embed_dim
        self.diag_dim = diag_embed_dim if pred_diag else 0
        self.output_dim = self.edge_dim + self.diag_dim
        self.node_type_embed = nn.Embedding(32, node_type_embed_dim)

        if self.pos_embed_type in ["zero_rotary", "rotary"]:
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

        # ========== Encoder (graph conv + optional full attention) ==========
        layers = []
        layers_type = []
        for i in range(num_layers):
            if full_attn_interval == -1:
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
                layers.append(GraphConvBlock(dim=node_hidden_dim, ffn_dim=ffn_dim))
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
                            is_causal=False,  # Non-causal for mesh nodes (no ordering)
                            attention_bias=attn_bias,
                        )
                    )
                    layers_type.append("TransformerBlock")

        self.layers = nn.ModuleList(layers)
        self.layers_type = layers_type

        # ========== Bottleneck ==========
        if self.bottleneck == "kl":
            self.bn_norm = FP32LayerNorm(node_hidden_dim)
            self.to_mu = nn.Linear(node_hidden_dim, bottleneck_dim)
            self.to_logvar = nn.Linear(node_hidden_dim, bottleneck_dim)
        elif self.bottleneck == "ln+noise":
            self.to_bn = nn.Linear(node_hidden_dim, bottleneck_dim)
            self.bn_norm = FP32LayerNorm(bottleneck_dim, elementwise_affine=False)
        else:
            raise NotImplementedError

        # ========== Full attention decoder (vertex-only) ==========
        self.dec_proj_in = nn.Linear(bottleneck_dim, decoder_dim)
        self.dec_blocks = nn.ModuleList(
            [
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
            ]
        )
        self.dec_proj_out = nn.Linear(decoder_dim, self.output_dim)
        self.dec_norm = RMSNorm(decoder_dim, eps=1e-6)

        # ========== Losses ==========
        edge_loss_cls = FlashSpacetimeAllPairEdgeLoss if flash_loss else SpacetimeAllPairEdgeLoss
        self.edge_loss = edge_loss_cls(
            use_pred_balanced_loss=True,
            pred_loss_weight=1.0,
            use_dice_loss=dice_weight > 0.0,
            dice_weight=dice_weight,
            chunk_size=chunk_size,
            scale=dist_scale,
        )
        if self.pred_diag:
            self.diag_loss = edge_loss_cls(
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

    def forward_encoder(self, x, edge_index, offsets, position: Optional[torch.Tensor] = None):
        for layer_type, layer in zip(self.layers_type, self.layers):
            if layer_type == "GraphConvBlock":
                x = layer(x, edge_index)
            else:
                x = layer(x, cu_seqlens=offsets, position_ids=position)
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
            nv = vertex_mask[start:end].sum().item()
            vertex_counts.append(nv)

        vertex_cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=offsets.device)
        for i, count in enumerate(vertex_counts):
            vertex_cu_seqlens[i + 1] = vertex_cu_seqlens[i] + count
        return vertex_cu_seqlens

    def forward_decoder(self, x_vertex, vertex_cu_seqlens, position: Optional[torch.Tensor] = None):
        assert (position is not None) == self.has_rope
        x = self.dec_proj_in(x_vertex)
        for layer in self.dec_blocks:
            x = layer(x, cu_seqlens=vertex_cu_seqlens, position_ids=position)
        x = self.dec_norm(x)
        x_dec = self.dec_proj_out(x)
        return x_dec, x

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

        if self.pos_embed_type == "zero_rotary":
            x = torch.cat([torch.zeros_like(pos[:, :3]), pos[:, 3:]], dim=-1)
        elif self.pos_embed_type == "rotary":
            x = pos
        else:
            raise NotImplementedError

        node_type_embed = self.node_type_embed(node_type.long()).to(dtype=x.dtype)
        x = torch.cat([x, node_type_embed], dim=-1)
        x = self.node_proj(x.to(dtype=self.node_proj.weight.dtype))

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
            "node_embed_mu": mu,
            "node_embed_logvar": logvar,
        }

    def is_scalar_param(self, name, param):
        patterns = [
            "node_proj",
            "to_mu",
            "to_logvar",
            "dec_proj_in",
            "dec_proj_out",
            ".scale_d.",
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
        """
        pos: N x input_node_dim
        edge_index: 2 x E (message-passing graph edges)
        node_type: N (0 for vertices, 1 for edge nodes, 2 for diagonal nodes)
        offsets: B + 1
        """
        ret_encoder = self.encode(pos, edge_index, offsets, node_type, position=position)
        vertex_mask = node_type == 0

        mu = ret_encoder["node_embed_mu"]
        logvar = ret_encoder["node_embed_logvar"]

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
        x_vertex = x_sample[vertex_mask]  # [Nv, bottleneck_dim]
        vertex_cu_seqlens = self.compute_vertex_cu_seqlens(vertex_mask, offsets)
        pred_vertex, _ = self.forward_decoder(
            x_vertex,
            vertex_cu_seqlens,
            position=position[vertex_mask] if position is not None else None,
        )  # [Nv, output_dim]

        pred = torch.zeros(
            x_sample.shape[0], self.output_dim, dtype=pred_vertex.dtype, device=pred_vertex.device
        )
        pred[vertex_mask] = pred_vertex
        edge_embed, diag_embed = torch.split(pred, [self.edge_dim, self.diag_dim], dim=-1)

        ret = {
            "node_embed_mu": mu,
            "st_feat": pred,
            "edge_embed": edge_embed,
            "diag_embed": diag_embed,
        }
        if self.bottleneck == "kl":
            ret.update({"node_embed_logvar": logvar})
        return ret

    def loss_fn(self, pred_dict, input_dict):
        pair = input_dict["vertex_pair"]
        vertex_mask = input_dict["node_type"] == 0

        # Compute the all-pair edge/diag losses in fp32 (the flash kernel and the
        # reference diverge at the ~1e-3 level in bf16, which is amplified at the
        # sharp loss transition; fp32 makes them numerically equivalent).
        edge_feat = pred_dict["edge_embed"].float()

        log_dict = {}

        edge_loss, edge_log_dict = self.edge_loss(
            edge_feat,
            pair,
            input_dict["pair_offsets"],
            input_dict["offsets"],
            vertex_mask,
        )
        loss = edge_loss
        log_dict["edge"] = edge_loss.detach()
        log_dict.update(edge_log_dict)

        if self.pred_diag:
            diag_feat = pred_dict["diag_embed"].float()
            diag_loss, diag_log_dict = self.diag_loss(
                diag_feat,
                input_dict["vertex_pair_diag"],
                input_dict["diag_pair_offsets"],
                input_dict["offsets"],
                vertex_mask,
            )
            loss = loss + self.config.diag_weight * diag_loss
            log_dict["diag"] = diag_loss.detach()
            log_dict.update({"d_" + k: v for k, v in diag_log_dict.items()})

        if self.bottleneck == "kl":
            mu = pred_dict["node_embed_mu"].float()
            logvar = pred_dict["node_embed_logvar"].float()
            logvar_v = logvar[vertex_mask]
            mu_v = mu[vertex_mask]
            kl_loss = torch.mean(
                -0.5 * torch.sum(1 + logvar_v - mu_v.pow(2) - logvar_v.exp(), dim=-1)
            )
            log_dict["kl_loss"] = kl_loss.detach()
            loss = loss + kl_loss * self.config.kl_weight

        return loss, log_dict

    def train_step(self, input_dict):
        assert self.training
        pred = self(
            pos=input_dict["nodes"],
            edge_index=input_dict["edges"].permute(1, 0),
            offsets=input_dict["offsets"],
            node_type=input_dict["node_type"],
            noise_interpolate=self.config.noise_interpolate,
            position=input_dict.get("position", None),
        )
        loss, log_dict = self.loss_fn(pred, input_dict)
        return loss, log_dict, {
            "num_tokens": input_dict["edges"].shape[0],
            "num_flops": 100,
        }


@dataclass
class MeshEdgeVAEArgs(BaseModelArgs):
    input_node_dim: int = 3
    node_hidden_dim: int = 256
    ffn_dim: int = 1024
    bottleneck_dim: int = 16
    num_layers: int = 12
    heads: int = 8
    undirected: bool = True
    bottleneck: str = "kl"
    noise_interpolate: float = 0.0
    kl_weight: float = 1e-6
    dice_weight: float = 0.5
    gated: bool = False
    full_attn_interval: int = 1
    chunk_size: int = 20_000_000
    dist_scale: str = "learn_nobias"
    edge_embed_dim: int = 64
    pred_diag: bool = False
    diag_weight: float = 0.6
    diag_embed_dim: int = 64
    decoder_dim: int = 512
    decoder_ffn_dim: int = 2048
    decoder_layers: int = 16
    decoder_heads: int = 8
    attn_bias: bool = True
    pos_embed_type: str = "zero_rotary"
    max_seq_len_rope: int = 8192
    rope_theta: float = 4096 * 3
    attn_dtype: str = "bf16"
    node_type_embed_dim: int = 8
    flash_loss: bool = True
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


class MeshEdgeVAEWrapper(MeshEdgeVAE, ModelProtocol):
    def __init__(self, model_args: MeshEdgeVAEArgs):
        super().__init__(**asdict(model_args))

    @classmethod
    def from_model_args(cls, model_args: MeshEdgeVAEArgs) -> "MeshEdgeVAE":
        return cls(model_args)


def overrite_model_args(model_args: MeshEdgeVAEArgs, **kwargs):
    return replace(model_args, **kwargs)


mesh_edgevae_configs = {
    # Edge-only flavor (works with dataset modes ``wireframe`` and ``bipartite``).
    "base-edge-16-kl-zerorope-nobias-res2048-flash": MeshEdgeVAEArgs(
        input_node_dim=3,
        node_hidden_dim=256,
        ffn_dim=1024,
        bottleneck_dim=16,
        num_layers=12,
        heads=8,
        undirected=True,
        bottleneck="kl",
        noise_interpolate=0.0,
        kl_weight=1e-6,
        gated=False,
        dice_weight=0.5,
        full_attn_interval=1,
        dist_scale="learn_nobias",
        edge_embed_dim=64,
        pred_diag=False,
        decoder_dim=512,
        decoder_ffn_dim=2048,
        decoder_layers=16,
        decoder_heads=8,
        attn_bias=True,
        pos_embed_type="zero_rotary",
        max_seq_len_rope=8192,
        rope_theta=4096 * 3,
        chunk_size=20_000_000,
        node_type_embed_dim=8,
        flash_loss=True,
    ),
}

# Edge + diagonal flavor (pairs with dataset mode ``bipartite_diag``).
mesh_edgevae_configs["base-edge-16-diag-kl-zerorope-nobias-res2048-flash"] = overrite_model_args(
    mesh_edgevae_configs["base-edge-16-kl-zerorope-nobias-res2048-flash"],
    pred_diag=True,
    diag_weight=0.6,
    diag_embed_dim=64,
)

# Smaller-latent variant.
mesh_edgevae_configs["base-edge-8-kl-zerorope-nobias-res2048-flash"] = overrite_model_args(
    mesh_edgevae_configs["base-edge-16-kl-zerorope-nobias-res2048-flash"],
    bottleneck_dim=8,
)
