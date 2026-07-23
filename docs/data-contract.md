# Humanoid Data Contract

The experiment Parquet is the single source of truth. Dataset code does not
derive buckets or keys from UUIDs and does not discover samples dynamically.

Required columns are `uuid`, `split`, `joint_schema`, `rig_npz_uri`,
`mesh_glb_uri`, `render_meta_uri`, and four explicit `color_view_N_uri`
columns. Normal-view URIs are retained for future experiments.

Accepted source manifests are immutable inventories produced by the resave
pipeline. The current dynamic-joint source is:

```text
/mnt/pfs/users/liyumeng/data/rigged_humanoid/resave/
humanoid-hands-toe-bases-head-2spine-no-vroid-v1/accepted.parquet
```

Materialized experiment manifests live under:

```text
/mnt/pfs/users/liyumeng/data/rigged_humanoid/datasets/
```

`scripts/humanoid/build_manifest.py` is intentionally metadata-only. It reads
the accepted inventory, filters untextured rows for color-conditioned
experiments, assigns stable SHA-256 splits, and materializes explicit BOS URIs.
It does not download rigs, inspect geometry, or compute octree packing
metadata. Runtime loading applies NEXUS discretization and rejects meshes above
the configured merged-point limit. `meta.json` remains the render pipeline's
completion marker and the experiment manifest should be built only after that
pipeline completes.

Smoke and overfit manifests are deterministic row subsets produced by
`scripts/humanoid/subset_manifest.py`. They preserve every source column and
never reinterpret or reconstruct an artifact URI.

The small smoke manifest is checked into the repository at
`manifests/humanoid_joint_octree/smoke_v1.parquet`; production manifests remain
on PFS.

The `humanoid-28-v1` schema is the global semantic vocabulary and embedding ID
space. In `joint_selection = "available"` mode, each asset emits only
unambiguous semantics present in its rig NPZ. Their IDs remain their global
schema indices, so conventions share embeddings and omitted joints never shift
the IDs of later tokens. The two canonical spine IDs are emitted only when the
source contains exactly two `spine` semantics that can be ordered by ancestry.

Mesh vertices define the NEXUS bounding-box transform; the identical transform
is applied to selected joint positions before discretization.
