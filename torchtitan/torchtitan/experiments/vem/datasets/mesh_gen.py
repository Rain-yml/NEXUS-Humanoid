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
    rand_with_pt,
)
import math
from torchtitan.experiments.vem.datasets.renderer.moderngl_rasterizer import FaceNormalRenderer
from torchtitan.experiments.vem.datasets.octree_utils import (
    discretize,
    undiscretize,
)
from itertools import chain
import pandas as pd
import networkx as nx
from torchvision.transforms import v2 as transforms_v2


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

class SpaceTimeRGBGenDataset(Stateful, IterableDataset):
    worker_shard_data = ["instance_ids"]

    def __init__(
        self,
        instance_list: Union[str, List[str]],
        repeats: int = 1,
        shuffle_seed: int = 0,
        use_bos: bool = False,
        bos_bucket: Optional[str] = None,
        image_bucket: Optional[str] = None,
        image_resolution: int = 518,
        drop_image_rate: float = 0.0,
        num_face_range: List[int] = [0, 20000],
        num_vertex_range: List[int] = [0, 100000],
        yup_to_zup: bool = False,
        vertex_noise: float = 0.0,
        vertex_resolution: int = -1,
        vertex_position_type: str = 'int',
        encoder_vertex_position_type: str = 'none',
        extra_feat: str = 'none',
        alignment: str = 'none',
        # reweight_by_face: bool = False,
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
        # self.use_cos = use_cos
        self.drop_image_rate = drop_image_rate
        self.image_resolution = image_resolution
        self.vertex_noise = vertex_noise
        self.num_face_range = num_face_range
        self.vertex_position_type = vertex_position_type
        self.extra_feat = extra_feat
        self.alignment = alignment
        assert self.extra_feat in ['none', 'normal', 'face_normal']
        assert self.alignment in ['none', 'near_align']
        assert vertex_position_type in ['int', 'float']
        assert not (vertex_resolution <= 0 and vertex_position_type == 'int')
        assert encoder_vertex_position_type in ['none', 'int']
        self.vertex_resolution = vertex_resolution
        self.encoder_vertex_position_type = encoder_vertex_position_type

        if self.use_bos:
            self.bos_client = BOSClient()
            self.bos_bucket = bos_bucket
            self.image_bucket = image_bucket
        else:
            self.bos_client = None
            self.bos_bucket = None
            self.image_bucket = None

        self.sample_idx = 0

        if isinstance(instance_list, str):
            instance_list = [instance_list]

        instance_ids = []
        for jp in instance_list:
            if jp.endswith(".json") or jp.endswith(".json.gz"):
                instance_ids.extend(load_json(jp))
            elif jp.endswith('.parquet'):
                df = pd.read_parquet(jp)
                # df = df[df["num_faces"].between(num_face_range[0], num_face_range[1])]
                df = df[(df['num_faces'].between(num_face_range[0], num_face_range[1])) & (df['num_vertices'].between(num_vertex_range[0], num_vertex_range[1]))]
                instance_ids.extend(df["instance_id"].tolist())
            elif jp.endswith('.ply') or jp.endswith('.obj'):
                instance_ids.append(jp)
        
        instance_ids = list(enumerate(instance_ids))

        instance_ids = instance_ids * repeats
        if force_divisible_by > 1:
            instance_ids = instance_ids[len(instance_ids) % force_divisible_by:]
            logger.info(f"Instance number after dropping: {len(instance_ids)}")        

        print("Instance first 10:", instance_ids[:10])
        print("Perform shard:", dp_rank, dp_world_size)
        instance_ids = shard_interleave(instance_ids, dp_world_size, dp_rank)
        print("Instance first 10 after shard:", instance_ids[:10])

        rng = random.Random(shuffle_seed)
        rng.shuffle(instance_ids)

        self.instance_ids = instance_ids

        self.packed = False

        self.dp_rank = dp_rank

        # self.use_condition = use_condition
        self.rasterizer = None
        self.yup_to_zup = yup_to_zup
        self.mp = MeshProcessor()
        print("dp_rank", dp_rank, "dp_world_size", dp_world_size)

        identity = transforms_v2.Lambda(lambda x: x)
        self.transform = transforms_v2.Compose([
            transforms_v2.RandomApply([
                transforms_v2.ColorJitter(hue=0.3),
                transforms_v2.RandomChoice([
                    transforms_v2.GaussianBlur(kernel_size=(3, 7)),
                    identity,
                ]),
                transforms_v2.RandomChoice([
                    transforms_v2.JPEG(quality=(20, 80)),
                    identity,
                ]),
            ], p=0.5)
        ])
    
    def process_image(self, image, image_size):
        # random select color / normal
        image_np = np.array(image).astype(np.float32) / 255.
        mask = image_np[..., 3]
        height, width = mask.shape

        pixels_y, pixels_x = np.where(mask > 0)

        if len(pixels_y) > 0:
            valid = True

            h0, h1 = max(pixels_y.min() - 1, 0), min(pixels_y.max() + 1, height)
            w0, w1 = max(pixels_x.min() - 1, 0), min(pixels_x.max() + 1, width)

            # random edge crop
            crop_edge_prob = 0.05
            crop_h0, crop_h1, crop_w0, crop_w1 = (np.random.random(4) < crop_edge_prob).tolist()
            height_fg, width_fg = h1 - h0, w1 - w0
            crop_edge_max_ratio = 0.02
            crop_h0_pixels, crop_h1_pixels = ((np.random.random(2) * height_fg * crop_edge_max_ratio)).astype(np.int32).tolist()
            crop_w0_pixels, crop_w1_pixels = ((np.random.random(2) * width_fg * crop_edge_max_ratio)).astype(np.int32).tolist()
            if crop_h0 > 0:
                h0 = h0 + crop_h0_pixels
            if crop_h1 > 0:
                h1 = h1 - crop_h1_pixels
            if crop_w0 > 0:
                w0 = w0 + crop_w0_pixels
            if crop_w1 > 0:
                w1 = w1 - crop_w1_pixels
            
            height_fg, width_fg = h1 - h0, w1 - w0

            # random padding
            pad_ratio = random.uniform(0.05, 0.2)
            if height_fg > width_fg:
                size_padded = int(height_fg / (1 - pad_ratio))
            else:
                size_padded = int(width_fg / (1 - pad_ratio))

            image_np_padded = np.zeros((size_padded, size_padded, 4), dtype=np.float32)
            start_h = (size_padded - height_fg) // 2
            start_w = (size_padded - width_fg) // 2
            image_np_padded[start_h : start_h + height_fg, start_w : start_w + width_fg] = image_np[h0:h1, w0:w1]
            image_np = image_np_padded        

            # random grayscale background (avoid background color being too similar to foreground color)
            fg_grayscale = image_np[..., :3][image_np[..., 3] > 0].mean()
            bg_color = np.random.rand(1)
            while bg_color > fg_grayscale - 0.2 and bg_color < fg_grayscale + 0.2:
                bg_color = np.random.rand(1)
            
            bg_color = np.concatenate([bg_color, bg_color, bg_color])
            image_np = image_np[..., :3] * image_np[..., 3:4] + bg_color[None, None] * (1 - image_np[..., 3:4])
        else:
            valid = False

            image_np = np.zeros_like(image_np[..., :3])

        # convert back to PIL.Image
        image = Image.fromarray((image_np * 255.).clip(0, 255).astype(np.uint8))

        # resize to image_size
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)

        # apply various torchvision transforms
        image = self.transform(image)

        # convert to torch tensor
        image_pt = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1)   

        return image, image_pt

    def _get_image(
        self,
        instance_id,
    ):
        image_dir = instance_id.split('.')[0]
        meta_bos_uri = f"{image_dir}/meta.json"
        meta = json.load(self.bos_client.get_file(self.image_bucket, meta_bos_uri))

        valid_image_indices = []
        for i, loc in enumerate(meta['locations']):
            transform_matrix = np.array(loc['transform_matrix'])
            camera_loc = transform_matrix[:3, 3]
            elevation_deg = np.rad2deg(np.arctan2(camera_loc[2], np.sqrt(camera_loc[0]**2 + camera_loc[1]**2)))
            azimuth_deg = np.rad2deg(np.arctan2(camera_loc[1], camera_loc[0]))
            valid_image_indices.append(i)
        image_idx = np.random.choice(valid_image_indices)
        transform_matrix = np.array(meta['locations'][image_idx]['transform_matrix'])
        camera_loc = transform_matrix[:3, 3]
        # elevation_deg in [-90, 90]
        elevation_deg = np.rad2deg(np.arctan2(camera_loc[2], np.sqrt(camera_loc[0]**2 + camera_loc[1]**2)))
        # azimuth_deg in [-180, 180]
        azimuth_deg = np.rad2deg(np.arctan2(camera_loc[1], camera_loc[0]))
        rotation_idx = int(((azimuth_deg + 90 + 45) % 360) // 90)
        # object rotation, not camera
        # rotation_idx = (4 - rotation_idx) % 4 
        # rotation: 0,1,2,3 -> front, right, back, left view

        image_fn = meta['locations'][image_idx]['frames'][0]['name'] # not use normal
        image_bos_uri = f"{image_dir}/{image_fn}"
        image = Image.open(self.bos_client.get_file(self.image_bucket, image_bos_uri)).convert('RGBA')

        image, image_pt = self.process_image(image, self.image_resolution)   

        return {
            'image_pil': image,
            'image_pt': image_pt,
            'rotation_idx': rotation_idx,
        }
    

    def _get_mesh(
        self,
        instance_id,
        transform=None,
    ):
        if self.use_bos:
            assert self.bos_client is not None
            mesh, pc = self.mp.read_mesh_from_bos(instance_id, self.bos_client, self.bos_bucket, process=True)
        else:
            mesh, pc = self.mp.read_mesh(instance_id)
        
        vertices, faces = self.mp.process(mesh, z_up=self.yup_to_zup)
        if transform is not None:
            vertices = trimesh.transformations.transform_points(vertices, transform)

        if self.vertex_noise > 0:
            if rand_with_pt([0, 1]) < 0.7:
                noise_level = np.random.uniform(0, self.vertex_noise)
                vertices += np.random.randn(*vertices.shape) * noise_level

        if self.vertex_resolution > 0:
            vertices_dis = discretize(vertices, self.vertex_resolution)
            mesh_process = trimesh.Trimesh(vertices_dis, faces)
            mesh_process = self.mp.clear_mesh(mesh_process, digits_vertex=0)
            vertices, faces = np.copy(mesh_process.vertices), np.copy(mesh_process.faces)
            vertex_position = np.copy(vertices) * 3
            if self.vertex_position_type == 'int':
                token_pos = vertices.astype(np.int32)
                vertices = undiscretize(vertices, self.vertex_resolution)
                token_pos = torch.from_numpy(token_pos).long()
            else:
                vertices = undiscretize(vertices, self.vertex_resolution)
                token_pos = np.copy(vertices)
                token_pos = torch.from_numpy(token_pos).float()
        else:
            mesh_process = trimesh.Trimesh(vertices, faces)
            mesh_process = self.mp.clear_mesh(mesh_process, digits_vertex=6)
            vertices, faces = np.copy(mesh_process.vertices), np.copy(mesh_process.faces)
            token_pos = np.copy(vertices)
            token_pos = torch.from_numpy(token_pos).float()
            vertex_position = None
        
        # if faces.shape[0] > self.num_face_range[1]:
        #     logger.warning("Too many faces", instance_id, faces.shape[0])

        nv = vertices.shape[0]
        nf = faces.shape[0]
        # face centers as node features (3D)
        face_center = vertices[faces].mean(axis=1)  # (nf, 3)
        # concatenate vertex positions and face centers -> node features
        all_nodes = np.concatenate([vertices, face_center], axis=0)  # (nv+nf, 3)
        if self.extra_feat == 'normal':
            face_normals = np.copy(mesh_process.face_normals)
            vertex_normals = np.copy(mesh_process.vertex_normals)
            extra_feats = np.concatenate([vertex_normals, face_normals], axis=0)
            all_nodes = np.concatenate([all_nodes, extra_feats], axis=1) # (nv+nf, 6)
        elif self.extra_feat == 'face_normal':
            face_normals = np.copy(mesh_process.face_normals)
            vertex_normals = np.zeros((nv, 3))
            extra_feats = np.concatenate([vertex_normals, face_normals], axis=0)
            all_nodes = np.concatenate([all_nodes, extra_feats], axis=1) # (nv+nf, 6)
        vertex_mask = np.zeros(nv+nf, dtype=bool)
        vertex_mask[:nv] = True

        # face node indices in the concatenated node array
        face_indices = np.arange(nv, nv + nf, dtype=np.int64)  # (nf,)

        # For each face, create edges (face_idx <-> each vertex)
        # faces is (nf,3) containing vertex indices per face
        # Create directed edges both ways to represent undirected face-vertex relationship
        # First: face -> vertex (repeat face index 3 times)
        face_repeat = np.repeat(face_indices, 3)                      # (nf*3,)
        verts_flat = faces.reshape(-1)                                # (nf*3,)
        edges_fv = np.stack([face_repeat, verts_flat], axis=1)       # (nf*3, 2)

        ret = {
            # 'vertices': torch.from_numpy(vertices).float(),          # (nv, 3)
            'vertices': token_pos,
            'nodes': torch.from_numpy(all_nodes).float(),
            'edges': torch.from_numpy(edges_fv).long(),
            # vertex_pair should refer to vertex indices only (0..nv-1) — unchanged
            'vertex_mask': torch.from_numpy(vertex_mask).bool(),
        }

        if self.encoder_vertex_position_type == 'int':
            face_position = vertex_position[faces].mean(axis=1)
            encoder_position = np.concatenate([vertex_position, face_position], axis=0)
            ret['encoder_position'] = torch.from_numpy(encoder_position).long()

        return ret

    def get_data(self, instance_id: tuple):
        i, instance_id = instance_id

        ret = {
            'index': i,
            'instance_id': instance_id,
        }
        rotation_transform = None

        if self.drop_image_rate > 0 and rand_with_pt([0, 1]) < self.drop_image_rate:
            ret["image"] = torch.full((3, self.image_resolution, self.image_resolution), fill_value=0.5, dtype=torch.float32)
        else:
            image_dict = self._get_image(instance_id)
            ret["image"] = image_dict["image_pt"]
            # ret['rotation_idx'] = image_dict['rotation_idx']
            if self.alignment == 'near_align' and image_dict['rotation_idx'] != 0:
                rotation_transform = trimesh.transformations.rotation_matrix(- np.pi / 2 * image_dict['rotation_idx'], [0, 0, 1]) # rendering coordinate: z up, -y front, x right

        encoder_inputs = self._get_mesh(instance_id, transform=rotation_transform)
        ret.update(encoder_inputs)
            
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

        # merge the mesh into a large graph
        batch_collated = {}
        node_offset = [0]
        vertex_offset = [0]
        nodes = []
        edges = []
        vertices = []
        edges_offset = [0]

        for b in batch:
            nodes.append(b['nodes'])
            vertices.append(b['vertices'])
            edges.append(b['edges'] + node_offset[-1])
            edges_offset.append(edges_offset[-1] + b['edges'].shape[0])
            node_offset.append(node_offset[-1] + b['nodes'].shape[0])
            vertex_offset.append(vertex_offset[-1] + b['vertices'].shape[0])
        
        batch_collated['nodes'] = torch.cat(nodes, dim=0)
        batch_collated['vertices'] = torch.cat(vertices, dim=0)
        batch_collated['edges'] = torch.cat(edges, dim=0)
        batch_collated['encoder_cu_seqlens'] = torch.tensor(node_offset, dtype=torch.int32)
        batch_collated['cu_seqlens'] = torch.tensor(vertex_offset, dtype=torch.int32)
        batch_collated['instance_ids'] = [b['instance_id'] for b in batch]
        batch_collated['vertex_mask'] = torch.cat([b['vertex_mask'] for b in batch], dim=0)
        batch_collated['image'] = torch.stack([b['image'] for b in batch], dim=0)
        batch_collated['edges_offset'] = torch.tensor(edges_offset, dtype=torch.int32)
        if self.encoder_vertex_position_type == 'int':
            batch_collated['encoder_position'] = torch.cat([b['encoder_position'] for b in batch], dim=0)

        del batch
        gc.collect()
        return batch_collated

    def collate_fn(self, batch):
        return self._collate_fn(batch)

class SpaceTimeGenDataset(IterableDataset, Stateful):
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
        yup_to_zup: bool = False,
        up='z',
        front='-y',
        vertex_noise: float = 0.0,
        num_face_range: List[int] = [0, 20000],
        condition: str = 'no',
        elevation_deg_range: List[float] = [-40, 40],
        azimuth_deg_range: List[float] = [0, 360],
        camera_fovy_deg_range: List[float] = [30, 50],
        image_resolution: int = 518,
        drop_image_rate: float = 0.0,
        # auto assigned args
        infinite: bool = True,
        force_divisible_by: int = 1,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        assert condition in ['no', 'normal']
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
            if jp.endswith(".json") or jp.endswith(".json.gz"):
                instance_ids.extend(load_json(jp))
            elif jp.endswith('.parquet'):
                df = pd.read_parquet(jp)
                df = df[df["num_faces"].between(num_face_range[0], num_face_range[1])]
                instance_ids.extend(df["instance_id"].tolist())
            elif jp.endswith('.ply') or jp.endswith('.obj'):
                instance_ids.append(jp)

        instance_ids = instance_ids * repeats
        if force_divisible_by > 1:
            instance_ids = instance_ids[len(instance_ids) % force_divisible_by:]
            logger.info(f"Instance number after dropping: {len(instance_ids)}")        

        instance_ids = shard_interleave(instance_ids, dp_world_size, dp_rank)

        rng = random.Random(shuffle_seed)
        rng.shuffle(instance_ids)

        print("rank", dp_rank, instance_ids[:10])

        self.instance_ids = instance_ids

        self.packed = False

        self.mp = MeshProcessor()
        self.aug_flip = aug_flip
        self.aug_rotate_all = aug_rotate_all
        self.aug_rotate_z = aug_rotate_z
        self.aug_scale = aug_scale
        self.aug_scale_range = aug_scale_range
        self.yup_to_zup = yup_to_zup
        self.up = up
        self.front = front
        self.vertex_noise = vertex_noise
        self.condition = condition

        self.elevation_deg_range = elevation_deg_range
        self.azimuth_deg_range = azimuth_deg_range
        self.camera_fovy_deg_range = camera_fovy_deg_range
        self.image_resolution = image_resolution
        self.drop_image_rate = drop_image_rate
        self.rasterizer = None
        self.num_face_range = num_face_range
        print("Augmentations", "flip", aug_flip, "rotate_all", aug_rotate_all, "rotate_z", aug_rotate_z, "scale", aug_scale, "scale_range", aug_scale_range)
        print("dp_rank", dp_rank, "dp_world_size", dp_world_size)
    
    def get_data(self, instance_id: str):
        if self.use_bos:
            assert self.bos_client is not None
            mesh, pc = self.mp.read_mesh_from_bos(instance_id, self.bos_client, self.bos_bucket, process=True)
        else:
            mesh, pc = self.mp.read_mesh(instance_id)
        
        vertices, faces = self.mp.process(mesh, z_up=self.yup_to_zup, up=self.up, front=self.front)

        vertices, faces = self.mp.augment(
            vertices=vertices,
            faces=faces,
            aug_flip=self.aug_flip,
            aug_rotate_all=self.aug_rotate_all,
            aug_rotate_z=self.aug_rotate_z,
            aug_scale=self.aug_scale,
            aug_scale_range=self.aug_scale_range,
        )

        mesh_process = trimesh.Trimesh(vertices, faces)
        mesh_process = self.mp.clear_mesh(mesh_process, digits_vertex=6)

        vertices, faces = np.copy(mesh_process.vertices), np.copy(mesh_process.faces)

        if self.vertex_noise > 0:
            noise_level = np.random.uniform(0, self.vertex_noise)
            vertices += np.random.randn(*vertices.shape) * noise_level
        
        if faces.shape[0] > self.num_face_range[1]:
            logger.warning("Too many faces", instance_id, faces.shape[0])

        nv = vertices.shape[0]
        nf = faces.shape[0]
        # face centers as node features (3D)
        face_center = vertices[faces].mean(axis=1)  # (nf, 3)
        # concatenate vertex positions and face centers -> node features
        all_nodes = np.concatenate([vertices, face_center], axis=0)  # (nv+nf, 3)
        vertex_mask = np.zeros(nv+nf, dtype=bool)
        vertex_mask[:nv] = True

        # face node indices in the concatenated node array
        face_indices = np.arange(nv, nv + nf, dtype=np.int64)  # (nf,)

        # For each face, create edges (face_idx <-> each vertex)
        # faces is (nf,3) containing vertex indices per face
        # Create directed edges both ways to represent undirected face-vertex relationship
        # First: face -> vertex (repeat face index 3 times)
        face_repeat = np.repeat(face_indices, 3)                      # (nf*3,)
        verts_flat = faces.reshape(-1)                                # (nf*3,)
        edges_fv = np.stack([face_repeat, verts_flat], axis=1)       # (nf*3, 2)

        ret = {
            'vertices': torch.from_numpy(vertices).float(),          # (nv, 3)
            'nodes': torch.from_numpy(all_nodes).float(),            # (nv+nf, 3)
            'edges': torch.from_numpy(edges_fv).long(),
            # vertex_pair should refer to vertex indices only (0..nv-1) — unchanged
            'instance_id': instance_id,
            'vertex_mask': torch.from_numpy(vertex_mask).bool(),
        }
        if self.condition == 'normal':
            if self.drop_image_rate > 0 and rand_with_pt([0, 1]) < self.drop_image_rate:
                ret["image"] = torch.full((3, self.image_resolution, self.image_resolution), 0.5, dtype=torch.float32)
            else:
                if self.rasterizer is None:
                    self.rasterizer = FaceNormalRenderer((self.image_resolution, self.image_resolution), 4)
                    print("Rasterizer initialized in rank", self.dp_rank)

                fovy = rand_with_pt(self.camera_fovy_deg_range)
                elevation_deg = rand_with_pt(self.elevation_deg_range)
                azimuth_deg = rand_with_pt(self.azimuth_deg_range) - 90 # model looks at -y
                camera_distance = 1 / math.tan(math.radians(fovy / 2)) + 1

                normal = self.rasterizer.render_normal_vf(
                    vertices, 
                    faces, 
                    camera_distance=camera_distance, 
                    camera_fovy_deg=fovy, 
                    camera_elevation_deg=elevation_deg,
                    camera_azimuth_deg=azimuth_deg,
                    smooth=True,
                ) # H x W x 4
                # make background gray
                normal = normal.astype(np.float32) / 255.
                normal = normal[:, :, :3] * normal[:, :, 3:4] + 0.5 * (1 - normal[:, :, 3:4])
                normal = torch.from_numpy(normal).permute(2, 0, 1).clamp(0, 1)

                ret["image"] = normal
        
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
                    random.shuffle(self.instance_ids)
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

        # merge the mesh into a large graph
        batch_collated = {}
        node_offset = [0]
        vertex_offset = [0]
        nodes = []
        edges = []
        vertices = []
        edges_offset = [0]

        for b in batch:
            nodes.append(b['nodes'])
            vertices.append(b['vertices'])
            edges.append(b['edges'] + node_offset[-1])
            edges_offset.append(edges_offset[-1] + b['edges'].shape[0])
            node_offset.append(node_offset[-1] + b['nodes'].shape[0])
            vertex_offset.append(vertex_offset[-1] + b['vertices'].shape[0])
        
        batch_collated['nodes'] = torch.cat(nodes, dim=0)
        batch_collated['vertices'] = torch.cat(vertices, dim=0)
        batch_collated['edges'] = torch.cat(edges, dim=0)
        batch_collated['encoder_cu_seqlens'] = torch.tensor(node_offset, dtype=torch.int32)
        batch_collated['cu_seqlens'] = torch.tensor(vertex_offset, dtype=torch.int32)
        batch_collated['instance_ids'] = [b['instance_id'] for b in batch]
        batch_collated['vertex_mask'] = torch.cat([b['vertex_mask'] for b in batch], dim=0)
        batch_collated['image'] = torch.stack([b['image'] for b in batch], dim=0)
        batch_collated['edges_offset'] = torch.tensor(edges_offset, dtype=torch.int32)

        del batch
        gc.collect()
        return batch_collated

    def collate_fn(self, batch):
        return self._collate_fn(batch)

class SpaceTimeRGBGenPackDataset(SpaceTimeRGBGenDataset):
    worker_shard_data = ["batches"]

    def __init__(
        self,
        batches_packed: str,
        repeats: int = 1,
        shuffle_seed: int = 0,
        use_bos: bool = False,
        # use_cos: bool = False,
        bos_bucket: Optional[str] = None,
        image_bucket: Optional[str] = None,
        image_resolution: int = 518,
        drop_image_rate: float = 0.0,
        yup_to_zup: bool = False,
        vertex_noise: float = 0.0,
        vertex_resolution: int = -1,
        vertex_position_type: str = 'int',
        encoder_vertex_position_type: str = 'none',
        extra_feat: str = 'none',
        alignment: str = 'none',
        # multiview conditioning params
        mv_uuids_json: Optional[str] = None,
        mv_bucket: str = 'mesh-data-shapediff-render-mv',
        mv_prob: float = 0.5,
        mv_num_views_probs: List[float] = None,
        # reweight_by_face: bool = False,
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
        self.drop_image_rate = drop_image_rate
        self.image_resolution = image_resolution
        self.vertex_noise = vertex_noise
        self.vertex_position_type = vertex_position_type
        self.extra_feat = extra_feat
        self.alignment = alignment
        assert self.extra_feat in ['none', 'normal', 'face_normal']
        assert self.alignment in ['none', 'near_align']

        self.mv_prob = mv_prob
        self.mv_bucket = mv_bucket
        self.mv_num_views_probs = mv_num_views_probs if mv_num_views_probs is not None else [0.2, 0.4, 0.4]
        assert len(self.mv_num_views_probs) == 3, "mv_num_views_probs must have 3 elements: P(2), P(3), P(4)"
        if mv_uuids_json is not None:
            mv_uuids_data = load_json(mv_uuids_json)
            self.mv_uuids = set(mv_uuids_data if isinstance(mv_uuids_data, list) else mv_uuids_data['uuids'])
            logger.info(f"Loaded {len(self.mv_uuids)} UUIDs with mv renders from {mv_uuids_json}")
        else:
            self.mv_uuids = None
        assert vertex_position_type in ['int', 'float']
        assert not (vertex_resolution <= 0 and vertex_position_type == 'int')
        self.vertex_resolution = vertex_resolution
        self.encoder_vertex_position_type = encoder_vertex_position_type

        if self.use_bos:
            self.bos_client = BOSClient()
            self.bos_bucket = bos_bucket
            self.image_bucket = image_bucket
        else:
            self.bos_client = None
            self.bos_bucket = None
            self.image_bucket = None

        self.sample_idx = 0

        batches_packed_json = load_json(batches_packed)
        batches = batches_packed_json['batches']
        batches = batches * repeats
        if force_divisible_by > 1:
            batches = batches[:len(batches) // force_divisible_by * force_divisible_by]
            logger.info(f"Batch number after dropping: {len(batches)}")        

        batches = shard_interleave(batches, dp_world_size, dp_rank)

        self.batches = batches

        self.dp_rank = dp_rank

        # self.use_condition = use_condition
        self.rasterizer = None
        self.yup_to_zup = yup_to_zup
        self.mp = MeshProcessor()
        print("dp_rank", dp_rank, "dp_world_size", dp_world_size)

        identity = transforms_v2.Lambda(lambda x: x)
        self.transform = transforms_v2.Compose([
            transforms_v2.RandomApply([
                transforms_v2.ColorJitter(hue=0.3),
                transforms_v2.RandomChoice([
                    transforms_v2.GaussianBlur(kernel_size=(3, 7)),
                    identity,
                ]),
                transforms_v2.RandomChoice([
                    transforms_v2.JPEG(quality=(20, 80)),
                    identity,
                ]),
            ], p=0.5)
        ])
    
    def _get_mv_images(self, uuid: str):
        """Load N multiview renders for a uuid.

        Files live at bos://mv_bucket/uuid[:2]/uuid/color_{NNNN}.webp.
        Groups: 0000-0003 (group 0) and 0004-0007 (group 1).
        View index within a group: 0=front, 1=left, 2=back, 3=right (object centered)
        """
        group = random.randint(0, 1)
        n_views = np.random.choice([2, 3, 4], p=self.mv_num_views_probs)
        view_indices = sorted(random.sample([0, 1, 2, 3], n_views))

        images = []
        for vi in view_indices:
            filename = f"color_{group * 4 + vi:04d}.webp"
            bos_uri = f"{uuid[:2]}/{uuid}/{filename}"
            img = Image.open(self.bos_client.get_file(self.mv_bucket, bos_uri)).convert('RGBA')
            _, img_pt = self.process_image(img, self.image_resolution)
            images.append(img_pt)

        return {
            'images': torch.stack(images, dim=0),          # (N, 3, H, W)
            'view_indices': torch.tensor(view_indices, dtype=torch.long),  # (N,)
        }

    def get_data(self, instance_id: tuple):
        i, instance_id = instance_id

        ret = {
            'index': i,
            'instance_id': instance_id,
        }
        rotation_transform = None

        uid = os.path.basename(instance_id).replace('.ply', '')
        use_mv = (
            self.mv_uuids is not None
            and uid in self.mv_uuids
            and random.random() < self.mv_prob
        )

        if use_mv:
            mv = self._get_mv_images(uid)
            if self.drop_image_rate > 0 and rand_with_pt([0, 1]) < self.drop_image_rate:
                gray = torch.full((3, self.image_resolution, self.image_resolution), fill_value=0.5, dtype=torch.float32)
                ret['mv_images'] = gray.unsqueeze(0).expand(mv['images'].shape[0], -1, -1, -1).clone()
            else:
                ret['mv_images'] = mv['images']  # (N, 3, H, W)
            ret['mv_view_indices'] = mv['view_indices']  # (N,) int64
            if self.alignment == 'near_align':
                rotate_idx = rand_int_with_pt([0, 4])
                ret['mv_view_indices'] = (ret['mv_view_indices'] + rotate_idx) % 4
                rotation_transform = trimesh.transformations.rotation_matrix(np.pi / 2 * rotate_idx, [0, 0, 1])
        else:
            if self.drop_image_rate > 0 and rand_with_pt([0, 1]) < self.drop_image_rate:
                ret["image"] = torch.full((3, self.image_resolution, self.image_resolution), fill_value=0.5, dtype=torch.float32)
            else:
                image_dict = self._get_image(instance_id)
                ret["image"] = image_dict["image_pt"]
                if self.alignment == 'near_align' and image_dict['rotation_idx'] != 0:
                    rotation_transform = trimesh.transformations.rotation_matrix(- np.pi / 2 * image_dict['rotation_idx'], [0, 0, 1])

        encoder_inputs = self._get_mesh(instance_id, transform=rotation_transform)
        ret.update(encoder_inputs)

        return ret

    def __iter__(self):
        while True:
            sample_idx = self.sample_idx

            while True:
                try:
                    yield [self.get_data((i, f"{uid[:2]}/{uid}.ply")) for (uid, i, _) in self.batches[sample_idx]]
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

    def collate_fn(self, batch):
        def flatten_list(l):
            return [item for sublist in l for item in sublist]

        batch = flatten_list(batch)

        batch_collated = {}
        node_offset = [0]
        vertex_offset = [0]
        nodes = []
        edges = []
        vertices = []
        edges_offset = [0]

        for b in batch:
            nodes.append(b['nodes'])
            vertices.append(b['vertices'])
            edges.append(b['edges'] + node_offset[-1])
            edges_offset.append(edges_offset[-1] + b['edges'].shape[0])
            node_offset.append(node_offset[-1] + b['nodes'].shape[0])
            vertex_offset.append(vertex_offset[-1] + b['vertices'].shape[0])

        batch_collated['nodes'] = torch.cat(nodes, dim=0)
        batch_collated['vertices'] = torch.cat(vertices, dim=0)
        batch_collated['edges'] = torch.cat(edges, dim=0)
        batch_collated['encoder_cu_seqlens'] = torch.tensor(node_offset, dtype=torch.int32)
        batch_collated['cu_seqlens'] = torch.tensor(vertex_offset, dtype=torch.int32)
        batch_collated['instance_ids'] = [b['instance_id'] for b in batch]
        batch_collated['vertex_mask'] = torch.cat([b['vertex_mask'] for b in batch], dim=0)
        batch_collated['edges_offset'] = torch.tensor(edges_offset, dtype=torch.int32)
        if self.encoder_vertex_position_type == 'int':
            batch_collated['encoder_position'] = torch.cat([b['encoder_position'] for b in batch], dim=0)

        images_flat_list = []
        view_idx_list = []
        for b in batch:
            if 'mv_images' in b:
                images_flat_list.append(b['mv_images'])           # (N_i, 3, H, W)
                view_idx_list.append(b['mv_view_indices'])        # (N_i,) int64
            else:
                images_flat_list.append(b['image'].unsqueeze(0))  # (1, 3, H, W)
                view_idx_list.append(torch.zeros(1, dtype=torch.long))
        num_views_per_mesh = [x.shape[0] for x in images_flat_list]
        batch_collated['image'] = torch.cat(images_flat_list, dim=0)
        batch_collated['view_indices'] = torch.cat(view_idx_list, dim=0)
        mv_cu_seqlens = torch.zeros(len(num_views_per_mesh) + 1, dtype=torch.int32)
        mv_cu_seqlens[1:] = torch.cumsum(torch.tensor(num_views_per_mesh, dtype=torch.int32), dim=0)
        batch_collated['mv_cu_seqlens'] = mv_cu_seqlens

        del batch
        gc.collect()
        return batch_collated