# Experiments

Checked-in configurations are complete experiment records. Paths and behavior
must be declared in TOML rather than injected by launch scripts.

`front_overfit_small12.toml` is the current runnable experiment. It trains for
10,000 steps on eight training assets, with two validation and two test assets
reserved in the manifest. It uses canonical view 0 and the tracked
`smoke_small12_v1_packed.json` layer packing.

`front_qem20k_smoke100.toml` is the larger front-view smoke experiment. Its
tracked manifest contains 100 training, 10 validation, and 10 test assets from
`rig_npz_qem_20000` and `glb_qem_20000`. The manifest builder verifies every
required BOS object, validates the canonical joint schema, and records all
nine octree layer lengths. `pack_manifest.py` turns those lengths into the
same packed-batch tuples consumed by NEXUS, with an explicit token budget.

The dual-branch smoke packs at most eight asset-depth sequences per optimizer
batch. Its joint MSE is scaled by `sequences / 8`, giving each fixed-length
joint sequence the same coefficient even though NEXUS packs by mesh-token
count. The packed manifest is generated divisible by the configured worker and
data-parallel count, so the unchanged NEXUS loader does not discard a tail.

The checkpoint-sensitive optimizer, flow target, compile, activation
checkpointing, and EMA settings match the recorded NEXUS 5B MV run. Future
experiments must provide both an SSOT Parquet manifest and a NEXUS-format
packed-batch JSON; they should be copied from this config and change only the
manifest, packing, run length, checkpoint cadence, and output folder.

To build a QEM smoke manifest in an environment with BOS access:

```bash
PYTHONPATH=. uv run python scripts/humanoid/build_manifest.py \
  --accepted /mnt/pfs/users/liyumeng/data/rigged_humanoid/resave/humanoid-body-eyes-finger-bases-2spine-v1/accepted.parquet \
  --output ../manifests/humanoid_joint_octree/qem20k_smoke120_v1.parquet \
  --joint-schema torchtitan/experiments/humanoid/data/humanoid_28_v1.json \
  --rig-prefix rig_npz_qem_20000 --mesh-prefix glb_qem_20000 \
  --split-counts 100,10,10 --max-layer-tokens 10000

PYTHONPATH=. uv run python scripts/humanoid/pack_manifest.py \
  --manifest ../manifests/humanoid_joint_octree/qem20k_smoke120_v1.parquet \
  --output ../manifests/humanoid_joint_octree/qem20k_smoke100_v1_packed.json \
  --token-budget 12000 --max-sequences-per-batch 8 --pad-to-multiple 40
```

For the tracked eight-rank smoke config, the divisor is 40: eight ranks times
five workers times batch size one. Regenerate the packed manifest with the
matching divisor when any of those three values changes.
