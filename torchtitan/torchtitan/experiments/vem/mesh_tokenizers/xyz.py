from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Tuple
import numpy as np
import trimesh
import torch
from .base import TokenizerSpec, LogitsProcessor


def clear_mesh(mesh: trimesh.Trimesh, orient_sensitive_merge=False):
    mesh.merge_vertices(digits_vertex=0)
    if orient_sensitive_merge:
        mask = np.zeros(len(mesh.faces), dtype=bool)
        mask[trimesh.grouping.unique_rows(mesh.faces)[0]] = True
        mesh.update_faces(mesh.nondegenerate_faces(height=1.e-8) & mask)
    else:
        mesh.update_faces(mesh.nondegenerate_faces(height=1.e-8) & mesh.unique_faces())

    mesh.remove_unreferenced_vertices()
    # mesh.fix_normals()
    assert np.all(mesh.area_faces > 0)
    return mesh

def discretize(
    t,
    num_discrete: int,
    continuous_range = (-1, 1),
):
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().astype(np.int32).clip(min = 0, max = num_discrete - 1)

def undiscretize(
    t,
    num_discrete: int,
    continuous_range = (-1, 1),
):
    lo, hi = continuous_range
    assert hi > lo

    t = t.astype(np.float32)

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo

class XYZTokenizer(TokenizerSpec):
    def __init__(
        self, 
        bins: int,
        split: str = 'no',
        vertex_range: Tuple[float, float] = (-1, 1),
        order: str = 'xyz',
        orient_sensitive_merge: bool = False,
    ):
        super().__init__()
        assert split in ['no', 'coord', 'vertex_coord']
        assert order in ['xyz', 'zyx']
        self.split = split
        self.bins = bins
        self.vertex_range = vertex_range
        if self.split == 'no':
            self._vocab_size = bins + 3
            self.vertex_shift = np.array([0, 0, 0])
            self.coord_shift = np.array([0, 0, 0])
        elif self.split == 'coord':
            self._vocab_size = 3 * bins + 3
            self.vertex_shift = np.array([0, 0, 0])
            self.coord_shift = np.array([0, bins, 2 * bins])
        elif self.split == 'vertex_coord':
            self._vocab_size = 9 * bins + 3
            self.vertex_shift = np.array([0, 3 * bins, 6 * bins])
            self.coord_shift = np.array([0, bins, 2 * bins])
        self.order = order
        self.orient_sensitive_merge = orient_sensitive_merge
    
    
    @property
    def vocab_size(self):
        """The vocabulary size"""
        return self._vocab_size

    @property
    def pad(self):
        """The PAD token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        return self._vocab_size - 1

    @property
    def bos(self):
        """The BOS token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        return self._vocab_size - 3

    @property
    def eos(self):
        """The EOS token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        return self._vocab_size - 2

    @property
    def bos_token_id(self):
        return self.bos
    
    @property
    def eos_token_id(self):
        return self.eos
        
    @property
    def pad_token_id(self):
        return self.pad
    
    def _reorder_vertics_faces(self, v, f):
        # assume v is xyz and order by z then y then x
        index_to_v = np.lexsort(v.T)
        v = v[index_to_v]
        f = np.argsort(index_to_v)[f]

        fi = np.argmin(f, axis=1)
        se = (fi + 1) % 3
        th = (fi + 2) % 3

        f = np.take_along_axis(f, np.concatenate([fi[:, None], se[:, None], th[:, None]], axis=1), axis=1)
        index_to_f = np.lexsort(f.T[::-1])
        f = f[index_to_f]

        return v, f

    def _discretize(self, t, continuous_range, bins):
        lo, hi = continuous_range

        t = (t - lo) / (hi - lo) * (bins - 1)
        t = (t + 0.5).astype(np.int32)

        assert( (t >= 0).all() and (t < bins).all() )
        return t
    
    def _undiscretize(self, t, continuous_range, bins):
        lo, hi = continuous_range
        # t in [0, bins - 1]
        t = t.astype(np.float32) / (bins - 1)
        return t * (hi - lo) + lo
    
    def reorder_vertices(self, v, from_order, to_order):
        assert set(from_order) == set(to_order)
        index = [
            from_order.index(c) for c in to_order
        ]
        return v[..., index]
    
    def tokenize(self, vertices: np.ndarray, faces: np.ndarray, return_mesh=False) -> np.ndarray:
        vertices = self._discretize(vertices, continuous_range=self.vertex_range, bins=self.bins)
        quant_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
        quant_mesh = clear_mesh(quant_mesh, orient_sensitive_merge=self.orient_sensitive_merge)
        v, f = self._reorder_vertics_faces(quant_mesh.vertices, quant_mesh.faces)
        tokens = v[f] # F x 3 x 3
        tokens += self.vertex_shift.reshape(1, 3, 1)
        tokens += self.coord_shift.reshape(1, 1, 3)
        tokens = self.reorder_vertices(tokens, from_order='xyz', to_order=self.order)
        tokens = tokens.flatten()
        if return_mesh:
            return np.array(tokens, dtype=np.int32), self._undiscretize(v, self.vertex_range, self.bins), f
        else:
            return np.array(tokens, dtype=np.int32)
    
    def detokenize(self, tokens: np.array):
        # remove trailing PAD tokens
        start = 0
        end = len(tokens)
        for i in range(len(tokens)):
            if tokens[i] == self.bos:
                start = i + 1
            if tokens[i] in [self.eos, self.pad]:
                end = i
                break
        
        n_faces = (end - start) // 9
        
        vertices = tokens[start:start + n_faces * 9].reshape(-1, 3, 3)
        vertices = self.reorder_vertices(vertices, from_order=self.order, to_order='xyz')
        vertices -= self.vertex_shift.reshape(1, 3, 1)
        vertices -= self.coord_shift.reshape(1, 1, 3)
        vertices = vertices.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        vertices = self._undiscretize(vertices, continuous_range=self.vertex_range, bins=self.bins)
        return vertices, faces
    

class XYZTokenizerCheck(XYZTokenizer):
    def detokenize(self, tokens: np.array):
        v, f = super().detokenize(tokens)
        return v, f, None