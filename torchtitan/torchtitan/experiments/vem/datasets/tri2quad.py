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
    # read_mesh,
    # read_mesh_from_bos,
    # normalize_vertices,
    # process,
    # sample,
    # sample_with_dora,
    MeshProcessor,
    rand_int_with_pt,
)


def shard(size: int, num_shards: int, index: int):
    if not 0 <= index < num_shards:
        raise ValueError("index should be in [0, num_shards-1]")
    
    div = size // num_shards
    mod = size % num_shards
    start = div * index + min(index, mod)
    end = start + div + (1 if index < mod else 0)
    
    return start, end

def shard_interleave(l, num_shards: int, index: int):
    if not 0 <= index < num_shards:
        raise ValueError("index should be in [0, num_shards-1]")

    return [l[i] for i in range(len(l)) if i % num_shards == index]

class Tri2QuadDataset(IterableDataset, Stateful):
    worker_shard_data = ['instance_ids']
    
    def __init__(
        self,
        instance_list: Union[str, List[str]],
        repeats: int = 1,
        shuffle_seed: int = 0,
        # specify_instance: Optional[str] = None,
        use_bos: bool = False,
        bos_bucket: Optional[str] = None,
        aug_flip: bool = True,
        aug_rotate_all: bool = False,
        aug_rotate_z: bool = True,
        aug_scale: bool = False,
        aug_scale_range: List[float] = [0.8, 1.2],
        aug_diag: bool = True,
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

        if self.use_bos:
            self.bos_client = BOSClient()
            self.bos_bucket = bos_bucket
        else:
            self.bos_client = None
            self.bos_bucket = None

        self.sample_idx = 0

        if isinstance(instance_list, str):
            instance_list = [instance_list]
        
        instance_ids = []
        for jp in instance_list:
            instance_ids.extend(load_json(jp))

        instance_ids = instance_ids * repeats
        if force_divisible_by > 1:
            instance_ids = instance_ids[len(instance_ids) % force_divisible_by:]
            logger.info(f"Instance number after dropping: {len(instance_ids)}")        

        rng = random.Random(shuffle_seed)
        rng.shuffle(instance_ids)

        print("rank", dp_rank, instance_ids[:10])

        self.instance_ids = shard_interleave(instance_ids, dp_world_size, dp_rank)

        self.packed = False

        self.mp = MeshProcessor()
        self.aug_flip = aug_flip
        self.aug_rotate_all = aug_rotate_all
        self.aug_rotate_z = aug_rotate_z
        self.aug_scale = aug_scale
        self.aug_scale_range = aug_scale_range
        self.aug_diag = aug_diag
        print("Augmentations", "flip", aug_flip, "rotate_all", aug_rotate_all, "rotate_z", aug_rotate_z, "scale", aug_scale, "scale_range", aug_scale_range, "aug_diag", aug_diag)
        print("dp_rank", dp_rank, "dp_world_size", dp_world_size)
    
    def get_data(self, instance_id: str):
        # print("Loading instance", instance_id)
        if self.use_bos:
            mesh_mixed = self.mp.read_mixed_mesh_from_bos(instance_id, self.bos_client, self.bos_bucket)
            vertices = mesh_mixed.vertices
            faces_tri = mesh_mixed.faces[~mesh_mixed.is_quad][:, :3]
            faces_quad = mesh_mixed.faces[mesh_mixed.is_quad]
        else:
            mesh_mixed = self.mp.read_mixed_mesh(instance_id)
            vertices = mesh_mixed.vertices
            faces_tri = mesh_mixed.faces[~mesh_mixed.is_quad][:, :3]
            faces_quad = mesh_mixed.faces[mesh_mixed.is_quad]
            # vertices, faces_tri, faces_quad = self.mp.read_quad_mesh(instance_id)
        
        vertices, faces_tri, faces_quad = self.mp.process_quad(vertices, faces_tri, faces_quad, z_up=True)

        vertices, faces_tri, faces_quad = self.mp.augment_quad(
            vertices=vertices,
            faces=faces_tri,
            faces_quad=faces_quad,
            aug_flip=self.aug_flip,
            aug_rotate_all=self.aug_rotate_all,
            aug_rotate_z=self.aug_rotate_z,
            aug_scale=self.aug_scale,
            aug_scale_range=self.aug_scale_range,
        )
        faces_tri = torch.from_numpy(faces_tri).long()
        faces_quad = torch.from_numpy(faces_quad).long()
        vertices = torch.from_numpy(vertices).float()
        edges = [
            faces_tri[:, [0,1]],
            faces_tri[:, [1,2]],
            faces_tri[:, [2,0]],
            faces_quad[:, [0,1]],
            faces_quad[:, [1,2]],
            faces_quad[:, [2,3]],
            faces_quad[:, [3,0]],
        ]
        edges = torch.cat(edges, dim=0)
        edges = torch.sort(edges, dim=1).values
        edges = torch.unique(edges, dim=0)
        # add diagonal edges for quads
        if self.aug_diag:
            # random [0, 2] or [1, 3]
            ri = rand_int_with_pt([0, 3])
            if ri == 0:
                diag_edges = faces_quad[:, [0,2]]
            elif ri == 1:
                diag_edges = faces_quad[:, [1,3]]
            else:
                # random for each quad
                m = torch.rand(faces_quad.shape[0]) < 0.5
                diag_edges_02 = faces_quad[m][:, [0,2]]
                diag_edges_13 = faces_quad[~m][:, [1,3]]
                diag_edges = torch.cat([diag_edges_02, diag_edges_13], dim=0)
            
        else:
            diag_edges = faces_quad[:, [0,2]]
        diag_edges = torch.sort(diag_edges, dim=1).values
        diag_edges = torch.unique(diag_edges, dim=0)
        
        all_edges = torch.cat([diag_edges, edges], dim=0)
        is_quad_diag = torch.zeros(all_edges.shape[0], dtype=torch.long)
        is_quad_diag[:diag_edges.shape[0]] = 1

        ret = {
            'vertices': vertices,
            'edges': all_edges,
            'is_quad_diag': is_quad_diag,
            'instance_id': instance_id,
        }
        # print("Loaded instance", instance_id)

        return ret

    def __iter__(self):
        while True:
            sample_idx = self.sample_idx

            while True:
                try:
                    if not self.packed:
                        yield [self.get_data(self.instance_ids[sample_idx])]
                    else:
                        yield [self.get_data(idx) for idx in self.instance_ids[sample_idx]]
                    break
                except GeneratorExit:
                    raise
                except:
                    logger.warning(f"Failed to load data for instance {self.instance_ids[sample_idx]}: {traceback.format_exc()}")
                    sample_idx = random.randint(0, len(self.instance_ids) - 1)

            self.sample_idx += 1

            if self.sample_idx >= len(self.instance_ids):
                if not self.infinite:
                    logger.warning(f"Dataset has run out of data.")
                    break
                else:
                    self.sample_idx = 0
                    logger.warning(f"Dataset is being re-looped.")

    def load_state_dict(self, state_dict):
        self.sample_idx = state_dict["sample_idx"]
        self.instance_ids = state_dict["instance_ids"]
    
    def state_dict(self):
        return {
            "instance_ids": self.instance_ids,
            "sample_idx": self.sample_idx,
        }
        
    def _collate_fn(self, batch):
        # flatten batches
        def flatten_list(l):
            return [item for sublist in l for item in sublist]
        
        batch = flatten_list(batch)

        # batch_collated = {
        #     "input_ids": torch.nested.nested_tensor([b["input_ids"] for b in batch], dtype=torch.long, layout=torch.jagged),
        #     "position_ids": torch.nested.nested_tensor([b["position_ids"] for b in batch], dtype=torch.long, layout=torch.jagged),
        #     "seq_mask": torch.cat([b["seq_mask"] for b in batch], dim=0),
        # }

        # merge the mesh into a large graph
        batch_collated = {}
        offset = [0]
        edge_offset = [0]
        vertices = []
        edges = []
        is_quad_diag = []

        for b in batch:
            vertices.append(b['vertices'])
            edges.append(b['edges'] + offset[-1])
            is_quad_diag.append(b['is_quad_diag'])
            offset.append(offset[-1] + b['vertices'].shape[0])
            edge_offset.append(edge_offset[-1] + b['edges'].shape[0])
        
        batch_collated['vertices'] = torch.cat(vertices, dim=0)
        batch_collated['edges'] = torch.cat(edges, dim=0)
        batch_collated['is_quad_diag'] = torch.cat(is_quad_diag, dim=0)
        batch_collated['offsets'] = torch.tensor(offset, dtype=torch.int32)
        batch_collated['edge_offsets'] = torch.tensor(edge_offset, dtype=torch.long)
        batch_collated['instance_ids'] = [b['instance_id'] for b in batch]

        del batch
        gc.collect()
        return batch_collated

    def collate_fn(self, batch):
        return self._collate_fn(batch)
