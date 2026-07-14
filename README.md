# NEXUS-Humanoid

Standalone NEXUS development repository for joint generation and other
humanoid-specific octree experiments.

The repository owns a complete NEXUS source snapshot. There is no upstream
submodule, remote import, or launch-time patching. See [UPSTREAM.md](UPSTREAM.md)
for the imported revision.

## Layout

- `torchtitan/torchtitan/experiments/vem`: locally tracked NEXUS foundation.
- `torchtitan/torchtitan/experiments/humanoid`: humanoid datasets, models,
  trainers, pipelines, schemas, and experiment configurations.
- `torchtitan/scripts/humanoid`: dataset-manifest and inspection tools.
- `docs`: project data and experiment contracts.

## Training

Run commands from `torchtitan` so package-relative paths in the checked-in
configuration remain portable.

Install the lightweight humanoid data dependencies with `uv sync --extra
humanoid`. The CUDA/PyTorch stack is supplied by the training image.

```bash
uv run torchrun --nproc-per-node=1 \
  -m torchtitan.experiments.humanoid.trainer \
  --job.config_file torchtitan/experiments/humanoid/configs/smoke.toml
```

The smoke, overfit, and production configurations exercise the same model and
dataset code. Their manifests, step counts, and output locations differ.
