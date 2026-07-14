# Experiments

Checked-in configurations are complete experiment records. Paths and behavior
must be declared in TOML rather than injected by launch scripts.

`front_overfit_small12.toml` is the current runnable experiment. It trains for
10,000 steps on eight training assets, with two validation and two test assets
reserved in the manifest. It uses canonical view 0 and the tracked
`smoke_small12_v1_packed.json` layer packing.

The checkpoint-sensitive optimizer, flow target, compile, activation
checkpointing, and EMA settings match the recorded NEXUS 5B MV run. Future
experiments must provide both an SSOT Parquet manifest and a NEXUS-format
packed-batch JSON; they should be copied from this config and change only the
manifest, packing, run length, checkpoint cadence, and output folder.
