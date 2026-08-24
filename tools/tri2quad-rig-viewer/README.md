# Tri2Quad rig viewer

This is the dataset adapter between strict Tri2Quad rig-transfer artifacts and
the convention-free `rigviz` display library.

- Result Parquet parts are discovered incrementally from PFS.
- Only accepted rows appear in the index.
- Selected NPZ artifacts are loaded lazily from BOS and retained in a bounded
  in-memory cache.
- Vertices, topology, joints, hierarchy, names, and sparse weights are passed
  through unchanged.

The Kubernetes deployment serves the live run on port `8769`.
