# Experiments

Checked-in configurations are complete experiment records. Paths and behavior
must be declared in TOML rather than injected by launch scripts.

- `smoke.toml`: 50 steps on the smoke Parquet.
- `overfit10.toml`: 10,000 steps on a ten-asset Parquet.
- `train_5b_mv.toml`: full production training Parquet.

All three use `humanoid-joint-octree`, the `5b-mv` flavor, the same pretrained
checkpoint, four ordered views, the same joint schema, and the same losses.
Only manifest, run length, warmup, checkpoint cadence, and output folder may
differ for smoke or overfit experiments.
