from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiggedMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    weight_joint_nodes: np.ndarray
    weight_values: np.ndarray
    joint_nodes: np.ndarray
    joint_names: np.ndarray
    joint_positions: np.ndarray
    joint_parents: np.ndarray
