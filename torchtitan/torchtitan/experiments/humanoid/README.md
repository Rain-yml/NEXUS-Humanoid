# Humanoid Experiment Package

This package owns the humanoid-specific NEXUS path. Files that alter upstream
behavior are local copies, not subclasses or monkey patches.

- `data`: SSOT Parquet loading, BOS access, NEXUS-compatible layer packing,
  canonical semantics, and combined mesh/joint octree batches.
- `models`: copied NEXUS octree DiT with semantic joint-token inputs.
- `pipelines`: copied layerwise inference with one-child joint traversal.
- `trainer.py`: copied flow-matching trainer with the original token-average loss;
  mesh/joint averages are diagnostics only.
- `dual_branch_trainer.py`: frozen pretrained mesh stream plus a trainable joint
  stream. Each joint block updates through one zero-initialized
  `joint_attend_mesh` adapter; the mesh stream never attends to joints and is
  excluded from the loss and optimizer.
- `train_spec.py`: explicit GPU-side registration, imported only by training.
- `configs/dual_branch`: overfit and smoke-100 records for the dual-branch path.

The dual-branch data path reuses the existing combined packed batch and splits
it immediately before noising. Mesh and joints receive independent noise at the
same per-layer timestep. Semantic joint IDs and one-child-per-depth trajectories
are unchanged. The smoke pack caps layer sequences at eight, and its joint MSE
is scaled by the actual sequence count over that capacity so every asset-depth
pair has equal weight across optimizer steps.

```bash
torchrun --nproc-per-node=8 \
  -m torchtitan.experiments.humanoid.dual_branch_trainer \
  --job.config_file \
  torchtitan/experiments/humanoid/configs/dual_branch/front_qem20k_smoke100.toml
```

`scripts/humanoid/dual_branch_i2v_v2f_validate.py` rolls out both stage-1
streams and feeds the generated mesh points to the unchanged NEXUS stage 2.
For the rigging task, `scripts/humanoid/dual_branch_teacher_forced_validate.py`
instead loads every GT mesh-octree layer through the training dataset path,
constructs its timestep-matched noisy state, discards mesh predictions, and
denoises only the semantic joints.

New architectural experiments should normally be sibling model or pipeline
files with their own model flavor and TOML. Shared behavior should only be
factored out after two real variants need it.
