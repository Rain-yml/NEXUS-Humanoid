from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from torchtitan.experiments.vem.models.vertex_mmdit import SpaceMeshMMDiT
from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol
from torchtitan.tools.logging import logger


@dataclass
class SpaceMeshMMDiTArgs(BaseModelArgs):
    in_channels: int = 16
    num_layers: int = 24
    dim: int = 1024
    freq_dim: int = 256
    num_attention_heads: int = 16
    intermediate_size: int = 2048
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = True
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    use_flash_attn_3: bool = False
    num_freqs: int = 4
    condition_dim: int = 1280
    pos_embed_type: str = "rotary"
    max_seq_len_rope: int = 4096
    rope_theta: float = 4052.0
    num_registers: int = 0
    ape_scale_div: float = 2048.0
    mv_mode: bool = False
    num_mv_views: int = 4
    attn_dtype: str = "bf16"
    pretrained_path: Optional[str] = None

    def get_nparams(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def __post_init__(self):
        if not torch.cuda.is_available():
            logger.info("CUDA is not available; keeping configured flash_attn_3 setting")
            return

        device_name = torch.cuda.get_device_name(0)
        if "H" in device_name:
            self.use_flash_attn_3 = True
        else:
            self.use_flash_attn_3 = False

        logger.info(f"Using flash_attn_3: {self.use_flash_attn_3} according to device name: {device_name}")


class SpaceMeshMMDiTWrapper(SpaceMeshMMDiT, ModelProtocol):
    def __init__(self, model_args: SpaceMeshMMDiTArgs):
        super().__init__(**asdict(model_args))

    @classmethod
    def from_model_args(cls, model_args: SpaceMeshMMDiTArgs) -> "SpaceMeshMMDiTWrapper":
        return cls(model_args)


sm_mmdit_configs = {
    "0.5B-2048res": SpaceMeshMMDiTArgs(
        in_channels=16,
        num_layers=24,
        dim=1024,
        freq_dim=256,
        num_attention_heads=16,
        intermediate_size=2048,
        num_key_value_heads=None,
        attention_bias=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        use_flash_attn_3=False,
        num_freqs=4,
        condition_dim=1280,
        pos_embed_type="rotary",
        max_seq_len_rope=8192,
        rope_theta=4052,
        num_registers=0,
        ape_scale_div=2048,
        mv_mode=False,
        num_mv_views=4,
    ),
}
