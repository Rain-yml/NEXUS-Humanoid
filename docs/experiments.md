# Experiments

Checked-in configurations are complete experiment records. Paths and behavior
must be declared in TOML rather than injected by launch scripts.

`front_overfit_small12.toml` is the current runnable experiment. It trains for
10,000 steps on eight training assets, with two validation and two test assets
reserved in the manifest. It uses canonical view 0 and the tracked
`smoke_small12_v1_packed.json` layer packing.

`front_qem20k_smoke100.toml` is the larger legacy front-view smoke experiment.
Its tracked manifest contains 100 training, 10 validation, and 10 test assets
from `rig_npz_qem_20000` and `glb_qem_20000`.

The dual-branch smoke packs at most eight asset-depth sequences per optimizer
batch. Its joint MSE is scaled by `sequences / 8`, giving each fixed-length
joint sequence the same coefficient even though NEXUS packs by mesh-token
count. The packed manifest is generated divisible by the configured worker and
data-parallel count, so the unchanged NEXUS loader does not discard a tail.

The checkpoint-sensitive optimizer, flow target, compile, activation
checkpointing, and EMA settings match the recorded NEXUS 5B MV run. Future
production experiments provide only an SSOT Parquet manifest. The loader
deterministically enumerates all nine octree layers, rejects meshes above 11k
merged/discretized vertices at runtime, and emits one sequence per rank. No
geometry-derived packing metadata is stored in the manifest.

`single_stream/front_no_vroid_dynamic_joints.toml` is the current no-VRoid
front-view experiment. It keeps one global 28-joint embedding vocabulary while
loading only the canonical joints available in each asset. It runs for 100,000
steps, saves every 1,000 steps, and retains the latest three checkpoints.

To build its lightweight color-conditioned manifest:

```bash
PYTHONPATH=. uv run python scripts/humanoid/build_manifest.py \
  --accepted /mnt/pfs/users/liyumeng/data/rigged_humanoid/resave/humanoid-hands-toe-bases-head-2spine-no-vroid-v1/accepted.parquet \
  --output /mnt/pfs/users/liyumeng/data/rigged_humanoid/datasets/humanoid-hands-toe-bases-head-2spine-no-vroid-v1.parquet \
  --dataset-prefix datasets/humanoid-hands-toe-bases-head-2spine-no-vroid-v1
```

This step does not contact BOS or preprocess geometry. The runtime dataset
loads and discretizes each selected rig using the same path for every
experiment.
