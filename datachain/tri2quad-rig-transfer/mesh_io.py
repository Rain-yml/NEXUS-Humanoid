from __future__ import annotations

import tempfile
from pathlib import Path

import meshio
import numpy as np

from contract import Rejection


def read_ply(payload: bytes):
    with tempfile.TemporaryDirectory(prefix="tri2quad-rig-") as directory:
        path = Path(directory) / "mesh.ply"
        path.write_bytes(payload)
        mesh = meshio.read(path)
    triangles = []
    quads = []
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangles.append(np.asarray(cell.data, dtype=np.int32))
        elif cell.type == "quad":
            quads.append(np.asarray(cell.data, dtype=np.int32))
        elif len(cell.data):
            raise Rejection("unsupported_final_cells", f"final PLY contains {cell.type} cells")
    return (
        np.asarray(mesh.points[:, :3], dtype=np.float64),
        np.concatenate(triangles) if triangles else np.empty((0, 3), dtype=np.int32),
        np.concatenate(quads) if quads else np.empty((0, 4), dtype=np.int32),
    )
