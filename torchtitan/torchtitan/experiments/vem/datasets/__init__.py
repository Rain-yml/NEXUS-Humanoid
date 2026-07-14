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
from torch.utils.data import IterableDataset, DataLoader
from torch.distributed.checkpoint.stateful import Stateful
from torchvision.transforms import v2 as transforms_v2

from torchtitan.experiments.vem.dataloader import ParallelAwareDataloader
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger
from torchtitan.experiments.vem.datasets.bos import BOSClient
from torchtitan.experiments.vem.datasets.json_utils import load_json
from torchtitan.experiments.vem.datasets.mesh_stae import SpaceTimeAEDataset
from torchtitan.experiments.vem.datasets.mesh_stae_quad import SpaceTimeQuadAEDataset
from torchtitan.experiments.vem.datasets.mesh_stae_quad_pack import SpaceTimeQuadAEPackDataset
from torchtitan.experiments.vem.datasets.mesh_edgevae_pack import SpaceTimeEdgeAEPackDataset
from torchtitan.experiments.vem.datasets.mesh_stae_pack import SpaceTimeAEPackDataset
from torchtitan.experiments.vem.datasets.mesh_gen import SpaceTimeGenDataset, SpaceTimeRGBGenDataset, SpaceTimeRGBGenPackDataset
from torchtitan.experiments.vem.datasets.mesh_gen_rl import ImageVerticesRLDataset, SpaceTimeRGBGenRLPackDataset
from torchtitan.experiments.vem.datasets.mesh_gen_quad import SpaceTimeRGBGenQuadDataset, SpaceTimeRGBGenQuadPackDataset
from torchtitan.experiments.vem.datasets.vertex_oct import VertexOctreeRGBGen   
from torchtitan.experiments.vem.datasets.vertex_oct_pack import VertexOctreePackRGBGen   
from torchtitan.experiments.vem.datasets.vertex_oct_v2 import VertexOctreeRGBGenV2, VertexOctreePackRGBGenV2
from torchtitan.experiments.vem.datasets.tri2quad import Tri2QuadDataset

def build_mesh_stae_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    **kwargs,
):
    dataset_name = job_config.training.dataset

    if dataset_name == "stae":
        ds = SpaceTimeAEDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "stae-pack":
        ds = SpaceTimeAEPackDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "stae-quad":
        ds = SpaceTimeQuadAEDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "stae-quad-pack":
        ds = SpaceTimeQuadAEPackDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "edgevae-pack":
        ds = SpaceTimeEdgeAEPackDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "smgen":
        ds = SpaceTimeGenDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "smgen-rgb":
        ds = SpaceTimeRGBGenDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "smgen-rgb-pack":
        ds = SpaceTimeRGBGenPackDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "smgen-rgb-rl-pack":
        ds = SpaceTimeRGBGenRLPackDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "image-vertices-rl":
        ds = ImageVerticesRLDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "smgen-rgb-quad":
        ds = SpaceTimeRGBGenQuadDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "smgen-rgb-quad-pack":
        ds = SpaceTimeRGBGenQuadPackDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    else:
        raise NotImplementedError
    
    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.batch_size,
        collate_fn=ds.collate_fn,
        num_workers=job_config.training.num_workers,
        drop_last=job_config.training.drop_last,
        pin_memory=job_config.training.pin_memory,
    )

def build_oct_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    **kwargs,
):
    dataset_name = job_config.training.dataset

    if dataset_name == "vertex-rgb":
        ds = VertexOctreeRGBGen(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "vertex-rgb-pack":
        ds = VertexOctreePackRGBGen(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "vertex-rgb-v2":
        ds = VertexOctreeRGBGenV2(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    elif dataset_name == "vertex-rgb-pack-v2":
        ds = VertexOctreePackRGBGenV2(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    else:
        raise NotImplementedError
    
    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.batch_size,
        collate_fn=ds.collate_fn,
        num_workers=job_config.training.num_workers,
        drop_last=job_config.training.drop_last,
        pin_memory=job_config.training.pin_memory,
    )

def build_tri_to_quad_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    **kwargs,
):
    dataset_name = job_config.training.dataset

    if dataset_name == "tri2quad":
        ds = Tri2QuadDataset(
            **job_config.training.dataset_kwargs,
            force_divisible_by=job_config.training.batch_size * job_config.training.num_workers * dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
    else:
        raise NotImplementedError
    
    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.batch_size,
        collate_fn=ds.collate_fn,
        num_workers=job_config.training.num_workers,
        drop_last=job_config.training.drop_last,
        pin_memory=job_config.training.pin_memory,
    )

__all__ = [
    "build_mesh_stae_dataloader",
    "build_oct_dataloader",
    "build_tri_to_quad_dataloader",
]
