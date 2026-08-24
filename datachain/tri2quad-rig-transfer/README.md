# Strict Tri2Quad rig transfer

This DataChain job transfers source bone names, rest positions, hierarchy, and
skinning weights onto the exact final Tri2Quad vertex array.

## Acceptance contract

1. Import the source mesh, rest bones, hierarchy, and skinning weights as one
   bind-pose rig. Force Blender's armatures to rest pose before static cleanup.
2. Create joint markers in Blender world space, then parent those markers and
   every mesh root to the same normalization root. The producer transform runs
   once on that shared hierarchy; joint positions are read directly from the
   transformed markers and are never reconstructed with a fitted affine.
3. Merge vertices using Tri2Quad's fixed `digits_vertex=6` identity rule,
   remove the same degenerate/duplicate faces, and remove unreferenced vertices.
   Temporary `[-1, 1]` coordinates are used only to form merge keys; they never
   replace the shared Blender-space mesh or joint coordinates.
4. Evaluate only the producer's two recorded front modes (native and the
   VRoid `+Y` mode); final geometry must select exactly one.
5. Accept only when the normalized source gives an equal-cardinality, complete
   one-to-one correspondence to the final vertex array. Coordinate error is
   bounded by the two producer-owned six-decimal merge operations: `2e-6` per
   coordinate. This constant is not configurable or tuned for failures.
6. Reject coincident source vertices whose skinning weights disagree. Nothing
   is interpolated, guessed, or matched by a tunable distance threshold.

Accepted NPZ artifacts contain the final vertices and tri/quad connectivity,
joint node names/positions/parents, per-vertex sparse weights reordered by the
proven bijection, and the shared scene transform recorded from probe markers.
`accepted.parquet` is the sole downstream manifest;
`rejected.parquet` retains stable failure classes for deterministic follow-up.

The source manifest is built by joining `rigged-assets.parquet` from the prior
header audit to the exact Tri2Quad training manifest with `build_manifest.py`.
Run parameters select manifests and output locations; the correspondence rule
is code-owned and intentionally has no threshold configuration.
