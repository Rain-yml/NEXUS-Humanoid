"""Train-spec registration for standalone humanoid experiments."""

from dataclasses import asdict

from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.experiments.humanoid.data.dataloader import build_humanoid_dataloader
from torchtitan.experiments.humanoid.models import (
    DualBranchOctreeDiffusionArgs,
    DualBranchOctreeDiffusionWrapper,
    JointOctreeDiffusionArgs,
    JointOctreeDiffusionWrapper,
    SingleStreamJointOctreeDiffusionArgs,
    SingleStreamJointOctreeDiffusionWrapper,
)
from torchtitan.experiments.vem.parallelize import parallelize
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec


joint_octree_configs = {
    "5b-mv": JointOctreeDiffusionArgs(
        use_onehot_256=False,
        num_layers=24,
        dim=3072,
        freq_dim=256,
        num_attention_heads=24,
        intermediate_size=14336,
        num_key_value_heads=None,
        attention_bias=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        use_flash_attn_3=False,
        pos_embed_type="rotary",
        use_3d_rope=True,
        max_seq_len_3d=2048,
        grid_size=512,
        rope_theta=2026,
        num_freqs=8,
        max_depth=9,
        contain_cross_attention=True,
        image_hidden_size=1280,
        num_vertex_condition=True,
        mv_mode=True,
        num_mv_views=4,
        num_joint_tokens=28,
    )
}

register_train_spec(
    TrainSpec(
        name="humanoid-joint-octree",
        cls=JointOctreeDiffusionWrapper,
        config=joint_octree_configs,
        parallelize_fn=parallelize,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_humanoid_dataloader,
        build_tokenizer_fn=None,
        build_loss_fn=None,
        pipelining_fn=None,
    )
)

dual_branch_configs = {
    flavor: DualBranchOctreeDiffusionArgs(**asdict(config))
    for flavor, config in joint_octree_configs.items()
}

register_train_spec(
    TrainSpec(
        name="humanoid-dual-branch-octree",
        cls=DualBranchOctreeDiffusionWrapper,
        config=dual_branch_configs,
        parallelize_fn=parallelize,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_humanoid_dataloader,
        build_tokenizer_fn=None,
        build_loss_fn=None,
        pipelining_fn=None,
    )
)


single_stream_configs = {
    flavor: SingleStreamJointOctreeDiffusionArgs(**asdict(config))
    for flavor, config in joint_octree_configs.items()
}

register_train_spec(
    TrainSpec(
        name="humanoid-single-stream-joint-octree",
        cls=SingleStreamJointOctreeDiffusionWrapper,
        config=single_stream_configs,
        parallelize_fn=parallelize,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_humanoid_dataloader,
        build_tokenizer_fn=None,
        build_loss_fn=None,
        pipelining_fn=None,
    )
)

__all__ = [
    "joint_octree_configs",
    "dual_branch_configs",
    "single_stream_configs",
]
