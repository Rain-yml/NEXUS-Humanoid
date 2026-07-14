from .base import TokenizerSpec
from .xyz import XYZTokenizer, XYZTokenizerCheck
from .bxyz import BlockXYZTokenizer, BlockXYZTokenizerCheck

def build_mesh_tokenizer(
    tn,
    check: bool = False,
):
    if not check:
        if tn.name == "xyz":
            return XYZTokenizer(bins=tn.bins, split=tn.split, vertex_range=(-1, 1))
        elif tn.name == "bxyz":
            return BlockXYZTokenizer(bins=tn.bins, block_size=tn.block_size, vertex_range=(-1, 1)) 
        else:
            raise NotImplementedError
    else:
        if tn.name == "xyz":
            return XYZTokenizerCheck(bins=tn.bins, split=tn.split, vertex_range=(-1, 1))
        elif tn.name == "bxyz":
            return BlockXYZTokenizerCheck(bins=tn.bins, block_size=tn.block_size, vertex_range=(-1, 1)) 
        else:
            raise NotImplementedError



__all__ = [
    "build_mesh_tokenizer", 
    "TokenizerSpec"
]


