from dataclasses import dataclass, asdict
from typing import Optional
import torch
import torch.nn as nn

from torchtitan.experiments.vem.models.vertex_dit import SpaceMeshDiT
from torchtitan.experiments.vem.models.vertex_dit_wrapper import SpaceMeshDiTArgs
from torchtitan.protocols.train_spec import ModelProtocol
from torchtitan.experiments.vem.models.dmd_wrapper import BaseDMDModel


@dataclass
class DMDModelArgs(SpaceMeshDiTArgs):
    # DMD-specific: path to pretrained checkpoint for teacher/student/fake_score init
    teacher_pretrained_path: str = ""
    
    def get_nparams(self, model: nn.Module) -> int:
        # Override: count only trainable params (student + fake_score, not teacher)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DMDModel(BaseDMDModel):
    """
    DMD Model wrapper containing teacher, student, and fake_score networks.
    
    Checkpoint behavior:
    - state_dict() returns only student and fake_score weights (teacher excluded)
    - load_state_dict() loads student and fake_score, teacher loaded from pretrained
    - On resume: student/fake_score from checkpoint, teacher from pretrained_path
    """
    
    def __init__(self, model_args: DMDModelArgs):
        super().__init__(SpaceMeshDiT, model_args)
    
    @classmethod
    def from_model_args(cls, model_args: DMDModelArgs) -> "DMDModel":
        return cls(model_args)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        hidden_states_position: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor]=None,
        cu_seqlens: Optional[torch.Tensor]=None,
        cu_seqlens_encoder: Optional[torch.Tensor] = None,
        view_indices: Optional[torch.Tensor] = None,
        mv_cu_seqlens: Optional[torch.Tensor] = None,
        quad_ratios: Optional[torch.Tensor] = None,
        use_model: str = "student",  # "student", "teacher", or "fake_score"
    ):
        """Forward through specified model."""
        model = getattr(self, use_model)
        return model(
            hidden_states=hidden_states,
            timesteps=timesteps,
            hidden_states_position=hidden_states_position,
            encoder_hidden_states=encoder_hidden_states,
            cu_seqlens=cu_seqlens,
            cu_seqlens_encoder=cu_seqlens_encoder,
            view_indices=view_indices,
            mv_cu_seqlens=mv_cu_seqlens,
            quad_ratios=quad_ratios,
        )

    @torch.no_grad()
    def get_latents(self, encoder, input_dict, mean, std):
        return self.teacher.get_latents(encoder, input_dict, mean, std)
    
    def is_scalar_param(self, name, param):
        patterns = [
            'cond_proj',
            'view_embed',
            'quad_ratio_embed',
            'time_embedding',
            'time_projection',
            'registers',
            'head',
        ]
        for p in patterns:
            if p in name:
                return True
        if name.startswith('student.proj') or name.startswith('fake_score.proj'):
            return True
        return False
