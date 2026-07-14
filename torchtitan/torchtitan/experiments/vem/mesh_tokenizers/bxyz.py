from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Tuple
import numpy as np
import trimesh
import torch
from .base import TokenizerSpec, LogitsProcessor

def clear_mesh(mesh: trimesh.Trimesh):
    mesh.merge_vertices(digits_vertex=0)
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

class BlockXYZTokenizer(TokenizerSpec):
    def __init__(
        self, 
        bins: int,
        block_size: int,
        vertex_range: Tuple[float, float] = (-1, 1),
    ):
        super().__init__()
        self.bins = bins
        self.block_size = block_size
        assert self.bins % self.block_size == 0, "bins must be divisible by block_size"
        self.vertex_range = vertex_range
        self.offset_size = self.bins // self.block_size
        self._vocab_size = self.block_size**3 + self.offset_size**3 + 3
    
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
    
    def xyz_to_bo(self, coords: np.ndarray) -> np.ndarray:
        # coords: n x 3
        # block_offset: n x 2
        assert coords.ndim == 2 and coords.shape[1] == 3
        bins, block_size, offset_size = self.bins, self.block_size, self.offset_size
        block_coords = coords // offset_size
        offset_coords = coords % offset_size
        block_index = (block_coords[:, 0] * block_size ** 2 +
                   block_coords[:, 1] * block_size +
                   block_coords[:, 2])
        offset_index = (offset_coords[:, 0] * offset_size ** 2 +
                    offset_coords[:, 1] * offset_size +
                    offset_coords[:, 2])
        offset_index += block_size ** 3
        return np.stack([block_index, offset_index], axis=-1)

    def bo_to_xyz(self, block_offset: np.ndarray) -> np.ndarray:
        # block_offset: n x 2
        # coords: n x 3
        assert block_offset.ndim == 2 and block_offset.shape[1] == 2
        block_size, offset_size = self.block_size, self.offset_size
        block_index = block_offset[:, 0]
        offset_index = block_offset[:, 1] - block_size ** 3

        # unflatten indices
        bx = block_index // (block_size ** 2)
        by = (block_index // block_size) % block_size
        bz = block_index % block_size

        ox = offset_index // (offset_size ** 2)
        oy = (offset_index // offset_size) % offset_size
        oz = offset_index % offset_size

        # combine
        x = bx * offset_size + ox
        y = by * offset_size + oy
        z = bz * offset_size + oz

        return np.stack([x, y, z], axis=-1)
    
    def tokenize(self, vertices: np.ndarray, faces: np.ndarray, return_mesh=False) -> np.ndarray:
        vertices = self._discretize(vertices, continuous_range=self.vertex_range, bins=self.bins)
        quant_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
        quant_mesh = clear_mesh(quant_mesh)
        v, f = self._reorder_vertics_faces(quant_mesh.vertices, quant_mesh.faces)
        tokens = v[f] # F x 3 x 3
        nf = tokens.shape[0]
        tokens = self.xyz_to_bo(tokens.reshape(nf * 3, 3))
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
        
        n_faces = (end - start) // 6
        vertices = tokens[start:start + n_faces * 6].reshape(n_faces * 3, 2)
        vertices = self.bo_to_xyz(vertices)
        faces = np.arange(n_faces * 3).reshape(-1, 3)
        vertices = self._undiscretize(vertices, continuous_range=self.vertex_range, bins=self.bins)
        return vertices, faces
    

class BlockXYZTokenizerCheck(BlockXYZTokenizer):
    def detokenize(self, tokens: np.array):
        v, f = super().detokenize(tokens)
        return v, f, None