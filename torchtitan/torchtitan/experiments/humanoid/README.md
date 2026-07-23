# Humanoid Experiment Package

This package owns the humanoid-specific NEXUS path. Files that alter upstream
behavior are local copies, not subclasses or monkey patches.

- `data`: SSOT Parquet loading, BOS access, runtime NEXUS octree construction,
  canonical semantics, and combined mesh/joint octree batches.
- `models`: copied NEXUS octree DiT with semantic joint-token inputs.
- `pipelines`: copied layerwise inference with one-child joint traversal.
- `trainer.py`: copied flow-matching trainer with the original token-average loss;
  mesh/joint averages are diagnostics only.
- `dual_branch_trainer.py`: frozen pretrained mesh stream plus a trainable joint
  stream. Each joint block updates through one zero-initialized
  `joint_attend_mesh` adapter; the mesh stream never attends to joints and is
  excluded from the loss and optimizer.
- `single_stream_trainer.py`: one pretrained NEXUS stream containing clean mesh
  occupancy tokens at `t=0` and noisy semantic-joint tokens at sampled `t`.
  Joint loss is the only direct objective; mesh and joint tokens interact in
  the original 3D-RoPE self-attention blocks.
- `train_spec.py`: explicit GPU-side registration, imported only by training.
- `configs/dual_branch`: production and historical smoke records for the
  dual-branch path.
- `configs/single_stream`: smoke and production records for clean-mesh,
  joint-only diffusion.

The production dual-branch data path enumerates one `(asset, octree layer)`
sequence per rank directly from the Parquet manifest. It applies the pretrained
model's 11k limit after NEXUS discretization and vertex merging, then splits the
combined batch immediately before noising. Mesh and joints receive independent
noise at the same per-layer timestep. With one fixed-size joint sequence per
rank, the ordinary token mean is also the exact distributed token mean.

```bash
torchrun --nproc-per-node=8 \
  -m torchtitan.experiments.humanoid.dual_branch_trainer \
  --job.config_file \
  torchtitan/experiments/humanoid/configs/dual_branch/front_qem20k_vroid_train10k.toml
```

`scripts/humanoid/dual_branch_i2v_v2f_validate.py` rolls out both stage-1
streams and feeds the generated mesh points to the unchanged NEXUS stage 2.
For the rigging task, `scripts/humanoid/dual_branch_teacher_forced_validate.py`
instead loads every GT mesh-octree layer through the training dataset path,
constructs its timestep-matched noisy state, discards mesh predictions, and
denoises only the semantic joints.

The single-stream experiment uses the dataset's existing per-layer token order
`[mesh, joints]`. Mesh occupancy labels remain clean and receive timestep zero;
only joint occupancy labels are diffused and supervised. This gives joint tokens
direct 3D-RoPE self-attention over the GT mesh without a second model or an
additional cross-attention module.

```bash
torchrun --nproc-per-node=8 \
  -m torchtitan.experiments.humanoid.single_stream_trainer \
  --job.config_file \
  torchtitan/experiments/humanoid/configs/single_stream/front_qem20k_vroid_train10k.toml
```

`scripts/humanoid/single_stream_teacher_forced_validate.py` performs the matching
layerwise rollout: GT mesh tokens remain fixed while the scheduler updates only
the 28 semantic-joint tokens.

Dataset configs may set `joint_selection = "available"` for mixed skeleton
conventions. The schema remains one fixed semantic embedding vocabulary, but
each sample contributes only the unambiguous canonical joints in its NPZ.
Global schema IDs travel with those tokens through collation, so missing eyes or
finger bases do not renumber any other joint.

New architectural experiments should normally be sibling model or pipeline
files with their own model flavor and TOML. Shared behavior should only be
factored out after two real variants need it.
