# Humanoid Data Contract

The experiment Parquet is the single source of truth. Dataset code does not
derive buckets or keys from UUIDs and does not discover samples dynamically.

Required columns are `uuid`, `split`, `joint_schema`, `rig_npz_uri`,
`mesh_glb_uri`, `render_meta_uri`, and four explicit `color_view_N_uri`
columns. Normal-view URIs are retained for future experiments.

The canonical source manifest is:

```text
/mnt/pfs/users/liyumeng/data/rigged_humanoid/resave/
humanoid-body-eyes-finger-bases-2spine-v1/accepted.parquet
```

Materialized experiment manifests live under:

```text
/mnt/pfs/users/liyumeng/data/rigged_humanoid/datasets/
```

`meta.json` is the render completion marker. The manifest builder checks that
marker before admitting a row. Splits are stable SHA-256 partitions of UUIDs.
Evaluation and test data always use the split recorded in the same Parquet.

Smoke and overfit manifests are deterministic row subsets produced by
`scripts/humanoid/subset_manifest.py`. They preserve every source column and
never reinterpret or reconstruct an artifact URI.

The small smoke manifest is checked into the repository at
`manifests/humanoid_joint_octree/smoke_v1.parquet`; production manifests remain
on PFS.

The `humanoid-28-v1` schema contains 26 required singleton semantics and two
ordered spine joints. Mesh vertices define the NEXUS bounding-box transform;
the identical transform is applied to joint positions before discretization.
