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
from torchtitan.experiments.vem.datasets.bos import BOSClient, COSClient
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
import itertools
import pandas as pd
import math
from torchtitan.experiments.vem.datasets.renderer.moderngl_rasterizer import FaceNormalRenderer

from torchtitan.experiments.vem.datasets.octree_utils import (
    build_octree_by_layer,
    # discretize_points,
    discretize,
    undiscretize,
    normalize_points,
    OctreeData,
    OctreeBatch,
)

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


class VertexOctreeRGBGen(IterableDataset, Stateful):
    """
    Octree Diffusion 数据集
    
    加载 mesh 数据，构建八叉树表示用于扩散模型训练
    支持可选的图像条件（在线渲染 normal 图像）
    """
    worker_shard_data = ['instance_ids']
    
    def __init__(
        self,
        instance_list: Union[str, List[str]],
        repeats: int = 1,
        shuffle_seed: int = 0,
        use_bos: bool = False,
        use_cos: bool = False,
        bos_bucket: Optional[str] = None,
        image_bucket: Optional[str] = None,
        # 八叉树参数
        grid_size: int = 128,
        max_depth: int = 7,
        yup_to_zup: bool = False,
        # Overfit helper: cache processed instances to avoid rebuilding octrees
        cache_instances: bool = False,
        # 数据过滤
        num_face_range: List[int] = [0, 20000],
        num_vertex_range: List[int] = [0, 100000],
        # 图像条件参数 (normal 渲染)
        image_resolution: int = 518,  # 图像分辨率
        drop_image_rate: float = 0.0,  # 训练时随机 drop 图像的概率
        alignment: str = 'none',
        # 自动分配参数
        infinite: bool = True,
        force_divisible_by: int = 1,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        self.alignment = alignment
        assert self.alignment in ['none', 'near_align']
        self.repeats = repeats
        self.shuffle_seed = shuffle_seed
        self.infinite = infinite
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.use_bos = use_bos
        self.use_cos = use_cos
        self.cache_instances = cache_instances
        self._data_cache: dict[str, dict] = {}
        
        # 八叉树参数
        self.grid_size = grid_size
        self.max_depth = max_depth
        assert grid_size == 2**max_depth

        if self.use_bos:
            self.bos_client = BOSClient()
            self.bos_bucket = bos_bucket
            self.image_bucket = image_bucket
        elif self.use_cos:
            self.bos_client = COSClient()
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
                json_data = load_json(jp)
                instance_ids.extend(json_data)
            elif jp.endswith('.parquet'):
                df = pd.read_parquet(jp)
                if "num_faces" in df.columns and num_face_range is not None:
                    df = df[df["num_faces"].between(num_face_range[0], num_face_range[1])]
                if num_vertex_range is not None:
                    df = df[df["num_vertices"].between(num_vertex_range[0], num_vertex_range[1])]
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

        self.mp = MeshProcessor()
        self.yup_to_zup = yup_to_zup
        
        # 图像条件参数
        self.image_resolution = image_resolution
        self.drop_image_rate = drop_image_rate
        
        print("OctreeDiffusionDataset initialized")
        print("grid_size", grid_size, "max_depth", max_depth)
        print("image_resolution:", image_resolution, "drop_image_rate:", drop_image_rate)
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
    
    def get_data(self, instance_id: str):
        ret = {
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

        """加载单个 mesh 并构建八叉树，可选渲染 normal 图像"""
        if self.cache_instances and instance_id in self._data_cache:
            octree_data = self._data_cache[instance_id]
        else:
            if self.use_bos:
                assert self.bos_client is not None
                mesh, pc = self.mp.read_mesh_from_bos(instance_id, self.bos_client, self.bos_bucket, process=True)
            elif self.use_cos:
                mesh, pc = self.mp.read_mesh_from_bos(instance_id, self.bos_client, self.bos_bucket, process=True)
            else:
                mesh, pc = self.mp.read_mesh(instance_id)
            
            vertices, faces = self.mp.process(mesh, z_up=self.yup_to_zup)
            mesh_process = trimesh.Trimesh(vertices, faces)
            mesh_process = self.mp.clear_mesh(mesh_process, digits_vertex=6)
            if rotation_transform is not None:
                mesh_process.apply_transform(rotation_transform)
            vertices = np.copy(mesh_process.vertices)
            faces = np.copy(mesh_process.faces)

            # 离散化顶点
            discrete_points = discretize(vertices, self.grid_size)
            discrete_points = np.unique(discrete_points, axis=0)
            discrete_points = torch.from_numpy(discrete_points).long()
            
            octree_data = build_octree_by_layer(
                discrete_points=discrete_points,
                grid_size=self.grid_size,
                max_depth=self.max_depth,
            )

            if self.cache_instances:
                self._data_cache[instance_id] = ret
        
        ret['octree'] = octree_data
        ret['mesh'] = mesh_process
    
        return ret

    def __iter__(self):
        while True:
            sample_idx = self.sample_idx

            while True:
                try:
                    yield [self.get_data(self.instance_ids[sample_idx])]
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
        
    def collate_fn(self, batch, train_all_layers=True):
        """
        Collate function for OctreeDiffusionDataset
        
        将多个 OctreeData 合并为一个 batch，返回 OctreeBatch 对象
        
        Args:
            batch: list of samples
            use_onehot_256: 是否使用 256 维 one-hot (否则使用 8-bit)
            train_all_layers: 是否训练所有层（每层作为独立 sample），
                             若为 False 则随机选一层
        """
        # flatten batches
        def flatten_list(l):
            return [item for sublist in l for item in sublist]
        
        batch = flatten_list(batch)
        
        octrees: List[OctreeData] = [b['octree'] for b in batch]
        instance_ids = [b['instance_id'] for b in batch]
        
        # # 检查有效性
        min_layers = min(o.num_layers for o in octrees)
        # if min_layers == 0:
        #     logger.warning("Found octree with 0 layers, skipping batch")
        #     return None
        
        # 收集数据
        occupancy_list = []
        centers_list = []
        depths_list = []
        seqlens = []
        batch_instance_ids = []
        num_layers_per_mesh = []  # 记录每个 mesh 的层数，用于避免 point_encoder 重复计算
        num_vertices = []
        
        if train_all_layers:
            # 每个样本的每一层都作为独立的 batch sample
            for sample_idx, octree in enumerate(octrees):
                num_layers = octree.num_layers
                num_layers_per_mesh.append(num_layers)  # 记录这个 mesh 有多少层
                
                for layer_idx in range(num_layers):
                    # 8-bit 模式，转换为 [-1, 1] 范围
                    occ = octree.layer_occupancy[layer_idx] * 2 - 1  # (num_nodes, 8)
                    
                    centers = octree.layer_parent_centers[layer_idx]
                    # child_ids = octree.layer_child_ids[layer_idx]
                    depth = octree.layer_depths[layer_idx]
                    
                    num_nodes = occ.shape[0]
                    
                    occupancy_list.append(occ)
                    centers_list.append(centers)
                    depths_list.append(torch.full((num_nodes,), depth, dtype=torch.long))
                    seqlens.append(num_nodes)
                    batch_instance_ids.append(instance_ids[sample_idx])
        else:
            # 随机选择一层
            layer_idx = random.randint(0, min_layers - 1)
            
            for sample_idx, octree in enumerate(octrees):
                occ = octree.layer_occupancy[layer_idx] * 2 - 1
                
                centers = octree.layer_parent_centers[layer_idx]
                depth = octree.layer_depths[layer_idx]
                
                num_nodes = occ.shape[0]
                
                occupancy_list.append(occ)
                centers_list.append(centers)
                depths_list.append(torch.full((num_nodes,), depth, dtype=torch.long))
                seqlens.append(num_nodes)
                batch_instance_ids.append(instance_ids[sample_idx])
                
        # 拼接
        occupancy_flat = torch.cat(occupancy_list, dim=0).float()
        centers_flat = torch.cat(centers_list, dim=0)
        depths_flat = torch.cat(depths_list, dim=0)
        
        # 构建 cu_seqlens
        total_seqs = len(seqlens)
        cu_seqlens = torch.zeros(total_seqs + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), dim=0)
        
        # 处理图像条件 (每个唯一 mesh 一张图像)
        images = None
        uncond_images = None
        if 'image' in batch[0]:
            images = torch.stack([b['image'] for b in batch], dim=0)  # (num_unique_meshes, C, H, W)
            uncond_images = torch.full_like(images, 0.5)
        
        num_vertices = torch.tensor([o.num_vertices for o in octrees], dtype=torch.int32)
        
        result = OctreeBatch(
            layer_occupancy_flat=occupancy_flat,
            layer_parent_centers_flat=centers_flat,
            layer_depths_flat=depths_flat,
            cu_seqlens=cu_seqlens,
            max_seqlen=max(seqlens),
            layer_idx=-1 if train_all_layers else layer_idx,  # -1 表示所有层
            batch_size=total_seqs,
            instance_ids=batch_instance_ids,
            num_layers_per_mesh=num_layers_per_mesh if train_all_layers else None,
            images=images,
            uncond_images=uncond_images,
            num_vertices=num_vertices,
            meshes=[b['mesh'] for b in batch] if 'mesh' in batch[0] else None,
        )
        
        del batch
        gc.collect()
        
        return result
