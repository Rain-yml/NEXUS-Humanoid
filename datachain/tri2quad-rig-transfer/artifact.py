from __future__ import annotations

import io

import numpy as np

from contract import SCHEMA
from correspondence import Transfer


def encode_artifact(uuid: str, transfer: Transfer) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        schema=np.asarray(SCHEMA),
        uuid=np.asarray(uuid),
        vertices=transfer.final_vertices,
        triangles=transfer.final_triangles,
        quads=transfer.final_quads,
        joint_names=np.asarray(transfer.joint_names, dtype=np.str_),
        joint_positions=transfer.joint_positions,
        parents=transfer.joint_parents,
        weight_indices=transfer.weight_indices,
        weight_values=transfer.weight_values,
        source_to_final=transfer.source_to_final,
        max_coordinate_error=np.asarray(transfer.max_coordinate_error),
    )
    return stream.getvalue()
