from dataclasses import asdict
from typing import Any, Optional, List, Union, Dict
import json
import random
import os
import numpy as np
import trimesh
from PIL import Image
import traceback
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.experiments.vem.dataloader import ParallelAwareDataloader
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger
from torchtitan.experiments.vem.datasets.bos import BOSClient
from torchtitan.experiments.vem.datasets.json_utils import load_json
from scipy.spatial.transform import Rotation
from torchtitan.experiments.vem.datasets.mesh_utils import (
    MeshProcessor,
    rand_int_with_pt,
    rand_with_pt,
)
import math
from torchtitan.experiments.vem.datasets.renderer.moderngl_rasterizer import FaceNormalRenderer
from torchtitan.experiments.vem.datasets.octree_utils import (
    discretize,
    undiscretize,
)
from torchtitan.experiments.vem.datasets.mesh_stae import (
    nx_all_triangles,
    shard,
    shard_interleave,
    rows_in_A_not_in_B,
    random_non_face_triplet_with_edge,
    random_non_face_triplet_with_edge_fast,
    sample_non_face_triplets,
    sample_non_face_triplets_fast,
    SpaceTimeAEDataset,
)

class SpaceTimeAEPackDataset(SpaceTimeAEDataset):
    worker_shard_data = ['batches']
    
    def __init__(
        self,
        batches_packed: str,
        repeats: int = 1,
        shuffle_seed: int = 0,
        use_bos: bool = False,
        bos_bucket: Optional[str] = None,
        bos_path_prefix: str = "",
        aug_flip: bool = True,
        aug_rotate_all: bool = False,
        aug_rotate_z: bool = True,
        aug_scale: bool = False,
        aug_scale_range: List[float] = [0.8, 1.2],
        yup_to_zup: bool = False,
        vertex_noise: float = 0.0,
        random_negative: bool = False,
        extra_feat: str = 'none',
        include_face: bool = False,
        include_face_orient: bool = False,
        vertex_resolutions: List[int] = [-1], 
        vertex_position_type: str = 'none',
        face_negative: str = 'random',
        # auto assigned args
        infinite: bool = True,
        force_divisible_by: int = 1,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        self.repeats = repeats
        self.shuffle_seed = shuffle_seed
        self.infinite = infinite
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.use_bos = use_bos
        assert vertex_position_type in ['none', 'int']
        assert face_negative in ['none', 'random']
        self.vertex_position_type = vertex_position_type

        if self.use_bos:
            self.bos_client = BOSClient()
            self.bos_bucket = bos_bucket
        else:
            self.bos_client = None
            self.bos_bucket = None

        self.sample_idx = 0

        # Load batches from packed JSON
        batches_packed_json = load_json(batches_packed)
        batches = batches_packed_json['batches']
        batches = batches * repeats
        if force_divisible_by > 1:
            batches = batches[:len(batches) // force_divisible_by * force_divisible_by]
            logger.info(f"Batch number after dropping: {len(batches)}")        

        batches = shard_interleave(batches, dp_world_size, dp_rank)

        rng = random.Random(shuffle_seed)
        rng.shuffle(batches)

        print("rank", dp_rank, batches[:10])

        self.batches = batches

        self.mp = MeshProcessor()
        self.aug_flip = aug_flip
        self.aug_rotate_all = aug_rotate_all
        self.aug_rotate_z = aug_rotate_z
        self.aug_scale = aug_scale
        self.aug_scale_range = aug_scale_range
        self.yup_to_zup = yup_to_zup
        self.vertex_noise = vertex_noise
        self.random_negative = random_negative
        self.include_face = include_face
        self.vertex_resolutions = vertex_resolutions
        self.include_face_orient = include_face_orient
        self.extra_feat = extra_feat
        self.face_negative = face_negative
        self.bos_path_prefix = bos_path_prefix
        assert self.extra_feat in ['none', 'normal', 'face_normal']
        print("Augmentations", "flip", aug_flip, "rotate_all", aug_rotate_all, "rotate_z", aug_rotate_z, "scale", aug_scale, "scale_range", aug_scale_range)
        print("dp_rank", dp_rank, "dp_world_size", dp_world_size)
    
    def __iter__(self):
        while True:
            sample_idx = self.sample_idx

            while True:
                try:
                    yield [self.get_data(f"{self.bos_path_prefix}{uid[:2]}/{uid}.ply") for (uid, _, _) in self.batches[sample_idx]]
                    break
                except GeneratorExit:
                    raise
                except:
                    logger.warning(f"Failed to load data for instance {self.batches[sample_idx]}: {traceback.format_exc()}")
                    sample_idx = random.randint(0, len(self.batches) - 1)

            self.sample_idx += 1

            if self.sample_idx >= len(self.batches):
                if not self.infinite:
                    logger.warning(f"Dataset has run out of data.")
                    break
                else:
                    self.sample_idx = 0
                    logger.warning(f"Dataset is being re-looped.")

    def load_state_dict(self, state_dict):
        self.sample_idx = state_dict["sample_idx"]
        self.batches = state_dict["batches"]
    
    def state_dict(self):
        return {
            "batches": self.batches,
            "sample_idx": self.sample_idx,
        }