# Owned copy of experiments/vem/extra_args_octdiff.py.
from dataclasses import dataclass, field
from typing import Literal, List, Optional, Dict, Any
from torchtitan.config_manager import Model


@dataclass
class Scheduler:
    t_sampling_scheme: Literal["uniform", "logit_normal", "mixed_logit_normal"] = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    uniform_ratio: float = 0.0
    uniform_power: float = 1.0
    uniform_min: float = 0.0
    uniform_max: float = 1.0
    num_train_timesteps: int = 1000
    noise_std: float = 1.0

@dataclass
class Training:
    num_workers: int = 0
    drop_last: bool = True
    pin_memory: bool = True
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    image_encoder_type: str = "dinov2"
    image_encoder_model: str = "facebook/dinov2-with-registers-large"
    image_encoder_return_the_nth_hidden_states: int = -1
    use_foreground_condition_tokens: bool = False
    foreground_condition_token_dilation: int = 0
    loss_type: str = "vpred-vloss"
    joint_loss_sequence_capacity: int = 0
    pretrained_path: Optional[str] = None

@dataclass
class EMA:
    enabled: bool = False
    beta: float = 0.9999
    update_after_step: int = 100
    update_every: int = 10
    inv_gamma: float = 1.0
    power: float = 1.0
    min_value: float = 0.0


@dataclass
class Distill:
    """Distillation-specific training configuration (DMD)."""
    guidance_scale: Optional[float] = None  # CFG scale for teacher, None = disabled
    # student_update_freq=2: step 0,2,4... → student, step 1,3,5... → fake_score
    student_update_freq: int = 4  # Update student every N steps (step%N==0 → student)
    student_cfg_steps: int = 0  # Do CFG distill for first N steps (every iter), 0 = disabled
    # Multi-step distillation: 1 = single-step (default), >1 = multi-step
    student_sample_steps: int = 1
    # Timestep shift factor: shifted_t = shift * t / (1 + (shift - 1) * t)
    # 1.0 = no shift (uniform schedule), >1.0 = shift toward higher noise levels
    student_t_shift: float = 1.0


@dataclass
class JobConfig:
    """
    Extend the tyro parser with custom config classe for Flux model.
    """
    training: Training = field(default_factory=Training)
    scheduler: Scheduler = field(default_factory=Scheduler)
    ema: EMA = field(default_factory=EMA)
    distill: Distill = field(default_factory=Distill)
