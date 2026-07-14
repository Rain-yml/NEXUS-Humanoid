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
    dynamic_shift: bool = False
    shift_base: int = 256

@dataclass
class Training:
    num_workers: int = 0
    drop_last: bool = True
    pin_memory: bool = True
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    mean_std_path: str = ""
    encoder_flavor: str = ""
    encoder_pretrain_path: str = ""
    image_encoder_type: str = "dinov2"
    image_encoder_model: str = "facebook/dinov2-with-registers-large"
    image_encoder_return_the_nth_hidden_states: int = -1
    loss_type: str = "vpred-vloss"
    loss_calculation: str = 'token_avg'

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
    # student_update_freq=4: step 0,4,8... → student; others → fake_score
    student_update_freq: int = 4
    student_cfg_steps: int = 0  # CFG distill for first N steps (every iter), 0 = disabled
    student_sample_steps: int = 1  # 1 = single-step, >1 = multi-step
    student_t_shift: float = 1.0  # timestep shift factor, 1.0 = no shift


@dataclass
class RL:
    """RL finetuning controls for stage-2 smart mesh generation.

    Defaults are inert so existing stage-2 pretraining/distillation configs keep
    parsing without enabling the RL path.
    """

    enabled: bool = False
    pretrained_path: str = ""

    # Rollout collection
    num_rollout_steps: int = 8
    num_samples_per_input: int = 1
    num_batches_per_epoch: int = 1
    num_inner_epochs: int = 1
    sampler_type: Literal["euler", "dpm", "euler_sto"] = "euler"
    scheduler_shift: float = 1.0
    sampler_eta: float = 0.0
    save_rollout_interval: int = 0

    # DiffusionNFT / Flow-GRPO style objective
    beta: float = 1.0
    adv_clip_max: float = 5.0
    kl_coeff: float = 0.0
    use_per_input_advantage_norm: bool = True

    # Reward
    reward_name: str = "loop_simplicity"
    reward_names: List[str] = field(default_factory=list)
    reward_weights: List[float] = field(default_factory=list)
    reward_weighting_mode: Literal["raw", "batch_standardize"] = "raw"
    reward_workers: int = 4
    reward_min_loop_len: int = 2

    # Quad decoder recovery for reward computation
    quad_decoder_flavor: str = ""
    quad_decoder_ckpt_path: str = ""
    quad_decoder_mode: Literal[
        "tri",
        "tri_connect",
        "native_quad",
        "native_quad_wireframe",
    ] = "native_quad"
    quad_decoder_attn_dtype: str = "bf16"

    # PEFT LoRA. PEFT is imported lazily by train_rl_smgen.py.
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_path: str = ""
    lora_ema_enabled: bool = True
    lora_ema_beta: float = 0.9999
    lora_ema_update_after_step: int = 0
    lora_ema_update_every: int = 1

    # Old-policy adapter update
    old_policy_decay_type: Literal["constant", "none"] = "constant"
    old_policy_decay: float = 0.0
    old_policy_update_every: int = 1


@dataclass
class JobConfig:
    """
    Extend the tyro parser with custom config classe for Flux model.
    """
    training: Training = field(default_factory=Training)
    scheduler: Scheduler = field(default_factory=Scheduler)

    ema: EMA = field(default_factory=EMA)

    distill: Distill = field(default_factory=Distill)

    rl: RL = field(default_factory=RL)
