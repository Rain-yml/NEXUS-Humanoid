# Joint Octree Architecture

The implementation is an owned copy of the native NEXUS octree path. The
24-layer 5B multiview DiT, image conditioning, timestep modulation, 3D RoPE,
flow schedule, and 8-channel output head are preserved.

At every depth, mesh occupancy tokens and 28 semantic joint tokens share one
attention sequence. Mesh targets remain 8-bit child occupancy vectors. Every
joint target is one-hot over the same eight children. A learned semantic
embedding identifies each joint; a learned type embedding distinguishes joint
tokens from mesh occupancy tokens.

Inference applies the native thresholding rule to mesh occupancy. Each joint
uses `argmax` and therefore follows exactly one child while retaining its
semantic identity. Mesh and joint flow losses are averaged independently and
combined with the configured `joint_loss_weight`.
