# Humanoid Experiment Package

This package owns the humanoid-specific NEXUS path. Files that alter upstream
behavior are local copies, not subclasses or monkey patches.

- `data`: SSOT Parquet loading, BOS access, NEXUS-compatible layer packing,
  canonical semantics, and combined mesh/joint octree batches.
- `models`: copied NEXUS octree DiT with semantic joint-token inputs.
- `pipelines`: copied layerwise inference with one-child joint traversal.
- `trainer.py`: copied flow-matching trainer with separate mesh/joint averages.
- `train_spec.py`: explicit GPU-side registration, imported only by training.
- `configs`: complete smoke, overfit, and production experiment records.

New architectural experiments should normally be sibling model or pipeline
files with their own model flavor and TOML. Shared behavior should only be
factored out after two real variants need it.
