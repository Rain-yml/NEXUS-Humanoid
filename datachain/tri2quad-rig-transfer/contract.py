from __future__ import annotations

import numpy as np


SCHEMA = "nexus.tri2quad-rig.v1"
VERTEX_DIGITS = 6
# Source normalization and Tri2Quad each perform a six-decimal vertex merge.
MAX_COORDINATE_ERROR = 2.0 / 10**VERTEX_DIGITS
WEIGHT_ATOL = 1e-7


class Rejection(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code

def vertex_keys(vertices: np.ndarray) -> np.ndarray:
    scaled = np.rint(np.asarray(vertices, dtype=np.float64) * 10**VERTEX_DIGITS)
    if not np.isfinite(scaled).all():
        raise Rejection("non_finite_geometry", "vertices contain non-finite values")
    return scaled.astype(np.int64)


def structured_keys(vertices: np.ndarray) -> np.ndarray:
    keys = np.ascontiguousarray(vertex_keys(vertices))
    return keys.view(np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])).reshape(-1)
