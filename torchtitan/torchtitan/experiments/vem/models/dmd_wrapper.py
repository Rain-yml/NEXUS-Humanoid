from dataclasses import dataclass, asdict
from typing import Optional
import torch
import torch.nn as nn

from torchtitan.experiments.vem.models.octree import OctreeDiffusionModel
from torchtitan.experiments.vem.models.octree_wrapper import OctreeDiffusionArgs
from torchtitan.protocols.train_spec import ModelProtocol


@dataclass
class DMDModelArgs(OctreeDiffusionArgs):
    """
    DMD model args - inherits ALL fields from OctreeDiffusionArgs.
    Only adds DMD-specific fields.
    """
    # DMD-specific: path to pretrained checkpoint for teacher/student/fake_score init
    teacher_pretrained_path: str = ""

    def get_nparams(self, model: nn.Module) -> int:
        # Override: count only trainable params (student + fake_score, not teacher)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

class BaseDMDModel(nn.Module, ModelProtocol):
    def __init__(self, model_cls, model_args):
        super().__init__()
        self.model_args = model_args
        
        # Get base model args (exclude DMD-specific fields like pretrained_path)
        base_args = asdict(model_args)
        base_args.pop('teacher_pretrained_path', None)
        
        # Create three models with identical architecture
        self.teacher = model_cls(**base_args)
        self.student = model_cls(**base_args)
        self.fake_score = model_cls(**base_args)
        
        # Teacher is always frozen
        self.teacher.requires_grad_(False)

    @classmethod
    def from_model_args(cls, model_args) -> "BaseDMDModel":
        raise NotImplementedError
    
    def init_weights(self, buffer_device=None):
        """
        Initialize all three models from pretrained weights.
        
        Efficient approach: load weights once, copy to all three models.
        Handles DTensor case (when model is parallelized before init_weights).
        """
        pretrained_path = self.model_args.teacher_pretrained_path

        # NOTE: must call init_weights before loading checkpoint to ensure correct PE
        self.teacher.init_weights(buffer_device)
        self.student.init_weights(buffer_device)
        self.fake_score.init_weights(buffer_device)
    
        if pretrained_path:
            # Load pretrained weights once
            from torchtitan.tools.logging import logger
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            
            # Load checkpoint (handles both DCP and regular checkpoints)
            if pretrained_path.endswith('.pt') or pretrained_path.endswith('.pth'):
                state_dict = torch.load(pretrained_path, map_location='cpu')
                # full checkpoint
                state_dict = state_dict['ema']['model']
                
                # Apply to all three models - handle DTensor case
                self._load_pretrained_to_model(self.teacher, state_dict, "teacher")
                self._load_pretrained_to_model(self.student, state_dict, "student")
                self._load_pretrained_to_model(self.fake_score, state_dict, "fake_score")
            else:
                # DCP checkpoint - need to load into a model with matching structure
                # The checkpoint was saved from OctreeDiffusionWrapper which wraps OctreeDiffusionModel
                import torch.distributed.checkpoint as dcp
                from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
                
                logger.info(f"Loading DCP checkpoint from {pretrained_path}")
                
                # For DCP, load into teacher first, then copy to student and fake_score
                # Use a temporary state dict to load
                # TODO: Handle _orig_mod. prefix if needed
                dcp.load(
                    {"ema": {"model": self.teacher}},
                    checkpoint_id=pretrained_path,
                )
                
                # Get the loaded state dict and copy to student and fake_score
                teacher_state = get_model_state_dict(self.teacher, options=StateDictOptions(full_state_dict=True))
                self._load_pretrained_to_model(self.student, teacher_state, "student")
                self._load_pretrained_to_model(self.fake_score, teacher_state, "fake_score")
                logger.info("teacher: loaded from DCP checkpoint")
            
            logger.info("Initialized teacher, student, fake_score from pretrained weights")
        else:
            # No pretrained path - initialize normally
            pass
        
        # Ensure teacher stays frozen
        self.teacher.requires_grad_(False)
        self.teacher.eval()
    
    def _load_pretrained_to_model(self, model, state_dict, model_name):
        """
        Load pretrained weights to a model, handling DTensor case.
        
        Uses set_model_state_dict for proper DTensor handling with FSDP2/TP.
        Also handles _orig_mod. prefix mismatch when torch.compile is enabled.
        """
        from torchtitan.tools.logging import logger
        from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions

        # Handle _orig_mod. prefix mismatch (torch.compile adds this prefix)
        # Check if model has _orig_mod. prefix but state_dict doesn't
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        
        # Check if we need to add _orig_mod. prefix to state_dict
        if model_keys and state_dict_keys:
            sample_model_key = next(iter(model_keys))
            sample_state_key = next(iter(state_dict_keys))
            
            if sample_model_key.startswith('_orig_mod.') and not sample_state_key.startswith('_orig_mod.'):
                # Add _orig_mod. prefix to state_dict keys
                state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
                logger.info(f"{model_name}: added _orig_mod. prefix to match compiled model")
            elif not sample_model_key.startswith('_orig_mod.') and sample_state_key.startswith('_orig_mod.'):
                # Remove _orig_mod. prefix from state_dict keys
                state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
                logger.info(f"{model_name}: removed _orig_mod. prefix from checkpoint")

        # Use the distributed checkpoint API which handles DTensor properly
        ret = set_model_state_dict(
            model,
            model_state_dict=state_dict,
            options=StateDictOptions(full_state_dict=True, strict=False),
        )
        if len(ret.missing_keys) > 0 or len(ret.unexpected_keys) > 0:
            logger.error(f"{model_name}: missing keys: {ret.missing_keys}")
            logger.error(f"{model_name}: unexpected keys: {ret.unexpected_keys}")
        
        logger.info(f"{model_name}: loaded pretrained weights using set_model_state_dict")
    
    def state_dict(self, *args, **kwargs):
        """
        Return state dict excluding teacher weights.
        
        Only student and fake_score are saved to checkpoint.
        Teacher is reloaded from pretrained_path on resume.
        """
        full_state = super().state_dict(*args, **kwargs)
        # Filter out teacher.* keys
        return {k: v for k, v in full_state.items() if not k.startswith('teacher.')}
    
    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """
        Load state dict for student and fake_score only.
        
        Teacher weights are NOT in the checkpoint - they're loaded
        from pretrained_path in init_weights() or _load_teacher().
        """
        # The incoming state_dict only has student.* and fake_score.* keys
        # Load with strict=False since teacher.* keys are missing
        super().load_state_dict(state_dict, strict=False)
        
        # Reload teacher from pretrained (in case this is a resume)
        self._load_teacher()
    
    def _load_teacher(self):
        """Reload teacher weights from pretrained checkpoint."""
        from torchtitan.tools.logging import logger
        pretrained_path = self.model_args.pretrained_path
        if pretrained_path:
            if pretrained_path.endswith('.pt') or pretrained_path.endswith('.pth'):
                state_dict = torch.load(pretrained_path, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                # Use the DTensor-safe loading method (handles _orig_mod. prefix)
                self._load_pretrained_to_model(self.teacher, state_dict, "teacher")
            else:
                # DCP checkpoint
                import torch.distributed.checkpoint as dcp
                dcp.load(
                    {"model": self.teacher},
                    checkpoint_id=pretrained_path,
                )
                logger.info("teacher: reloaded from DCP checkpoint")
        
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def setup_grad_requirements(self, update_student: bool):
        """Toggle requires_grad based on training phase."""
        if update_student:
            self.student.requires_grad_(True)
            self.fake_score.requires_grad_(False)
        else:
            self.student.requires_grad_(False)
            self.fake_score.requires_grad_(True)
        # Teacher always frozen
        self.teacher.requires_grad_(False)


class DMDModel(BaseDMDModel):
    """
    DMD Model wrapper containing teacher, student, and fake_score networks.
    
    Checkpoint behavior:
    - state_dict() returns only student and fake_score weights (teacher excluded)
    - load_state_dict() loads student and fake_score, teacher loaded from pretrained
    - On resume: student/fake_score from checkpoint, teacher from pretrained_path
    """
    
    def __init__(self, model_args: DMDModelArgs):
        super().__init__(OctreeDiffusionModel, model_args)
    
    @classmethod
    def from_model_args(cls, model_args: DMDModelArgs) -> "DMDModel":
        return cls(model_args)

    def is_scalar_param(self, name, param):
        patterns = [
            'num_embed',
            'quad_ratio_embed',
            'time_embed',
            'time_projection',
            'noise_proj',
            'depth_embed',
            'center_embed',
            'head',
            'cond_proj',
            'view_embed',
        ]
        for p in patterns:
            if p in name:
                return True
        return False
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        centers: torch.Tensor,
        depths: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        num_layers_per_mesh: list = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        num_vertices: Optional[torch.Tensor] = None,
        quad_ratios: Optional[torch.Tensor] = None,
        view_indices: Optional[torch.Tensor] = None,
        mv_cu_seqlens: Optional[torch.Tensor] = None,
        use_model: str = "student",  # "student", "teacher", or "fake_score"
    ):
        """Forward through specified model."""
        model = getattr(self, use_model)
        return model(
            x_t=x_t,
            t=t,
            centers=centers,
            depths=depths,
            cu_seqlens_q=cu_seqlens_q,
            num_layers_per_mesh=num_layers_per_mesh,
            encoder_hidden_states=encoder_hidden_states,
            num_vertices=num_vertices,
            quad_ratios=quad_ratios,
            view_indices=view_indices,
            mv_cu_seqlens=mv_cu_seqlens,
        )
