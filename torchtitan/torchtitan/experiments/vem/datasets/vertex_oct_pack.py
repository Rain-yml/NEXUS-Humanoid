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
    build_octree_specific_layer,
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


class VertexOctreePackRGBGen(IterableDataset, Stateful):
    worker_shard_data = ['batches']
    
    def __init__(
        self,
        batches_packed: str,
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
        cache_instances: bool = False,
        image_resolution: int = 518,  # 图像分辨率
        drop_image_rate: float = 0.0,  # 训练时随机 drop 图像的概率
        alignment: str = 'none',
        # 多视图条件参数
        mv_uuids_json: Optional[str] = None,  # JSON file with list of UUIDs that have mv renders
        mv_bucket: str = 'mesh-data-shapediff-render-mv',  # BOS bucket for mv renders
        mv_prob: float = 0.5,  # probability of using mv renders for eligible UUIDs
        mv_num_views_probs: List[float] = None,  # P(2 views), P(3 views), P(4 views)
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

        # 多视图条件参数
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
        batches_packed_json = load_json(batches_packed)
        batches = batches_packed_json['batches']
        batches = batches * repeats
        if force_divisible_by > 1:
            # trim tailing batches as the flops are in a descending order 
            batches = batches[:len(batches) // force_divisible_by * force_divisible_by]
            logger.info(f"Batch number after dropping: {len(batches)}")        

        batches = shard_interleave(batches, dp_world_size, dp_rank)

        rng = random.Random(shuffle_seed)
        rng.shuffle(batches)

        print("rank", dp_rank, batches[:10])

        self.batches = batches 

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
            mask = image_np[..., 3] > 0

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
            mask = np.zeros(mask.shape, dtype=bool)

        # convert back to PIL.Image
        image = Image.fromarray((image_np * 255.).clip(0, 255).astype(np.uint8))
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255))

        # resize to image_size
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        mask_image = mask_image.resize((image_size, image_size), Image.Resampling.NEAREST)

        # apply various torchvision transforms
        image = self.transform(image)

        # convert to torch tensor
        image_pt = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1)   
        mask_pt = torch.from_numpy(np.array(mask_image) > 0)

        return image, image_pt, mask_pt
    
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

        image, image_pt, image_mask = self.process_image(image, self.image_resolution)

        return {
            'image_pil': image,
            'image_pt': image_pt,
            'image_mask': image_mask,
            'rotation_idx': rotation_idx,
        }

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
        masks = []
        for vi in view_indices:
            filename = f"color_{group * 4 + vi:04d}.webp"
            bos_uri = f"{uuid[:2]}/{uuid}/{filename}"
            img = Image.open(self.bos_client.get_file(self.mv_bucket, bos_uri)).convert('RGBA')
            _, img_pt, mask_pt = self.process_image(img, self.image_resolution)
            images.append(img_pt)
            masks.append(mask_pt)

        return {
            'images': torch.stack(images, dim=0),          # (N, 3, H, W)
            'image_masks': torch.stack(masks, dim=0),      # (N, H, W)
            'view_indices': torch.tensor(view_indices, dtype=torch.long),  # (N,)
        }

    def get_data(self, uid: str, layer_id: int):
        ret = {
            'instance_id': uid,
        }
        instance_id = f'{uid[:2]}/{uid}.ply'

        rotation_transform = None
        use_mv = (
            self.mv_uuids is not None
            and uid in self.mv_uuids
            and random.random() < self.mv_prob
        )

        if use_mv:
            mv = self._get_mv_images(uid)
            if self.drop_image_rate > 0 and rand_with_pt([0, 1]) < self.drop_image_rate:
                # Keep same number of views but replace all images with gray
                gray = torch.full((3, self.image_resolution, self.image_resolution), fill_value=0.5, dtype=torch.float32)
                ret['mv_images'] = gray.unsqueeze(0).expand(mv['images'].shape[0], -1, -1, -1).clone()
            else:
                ret['mv_images'] = mv['images']  # (N, 3, H, W)
            ret['mv_image_masks'] = mv['image_masks']  # (N, H, W)
            ret['mv_view_indices'] = mv['view_indices']  # (N,) int64
            if self.alignment == 'near_align':
                # random rotate and let the front view be the first view
                rotate_idx = rand_int_with_pt([0, 4])
                ret['mv_view_indices'] = (ret['mv_view_indices'] + rotate_idx) % 4
                # the original front view becomes the rotate_idx view
                # so we need to rotate the mesh around axis z for rotate_idx * 90 degrees (positive direction)
                rotation_transform = trimesh.transformations.rotation_matrix(np.pi / 2 * rotate_idx, [0, 0, 1])
        else:
            image_dict = self._get_image(instance_id)
            if self.drop_image_rate > 0 and rand_with_pt([0, 1]) < self.drop_image_rate:
                ret["image"] = torch.full((3, self.image_resolution, self.image_resolution), fill_value=0.5, dtype=torch.float32)
            else:
                ret["image"] = image_dict["image_pt"]
            ret["image_mask"] = image_dict["image_mask"]
            # ret['rotation_idx'] = image_dict['rotation_idx']
            if self.alignment == 'near_align' and image_dict['rotation_idx'] != 0:
                # the returned view (viewed as front view) is the rotate_idx view, so rotate the mesh around axis z for -rotate_idx * 90 degrees (negative direction)
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
            
            octree_data = build_octree_specific_layer(
                discrete_points=discrete_points,
                depth=layer_id,
                grid_size=self.grid_size,
                max_depth=self.max_depth,
            )

            if self.cache_instances:
                self._data_cache[instance_id] = ret
        
        ret['octree'] = octree_data
        # ret['mesh'] = mesh_process
    
        return ret

    def __iter__(self):
        while True:
            sample_idx = self.sample_idx

            while True:
                try:
                    yield [self.get_data(uid, layer_id) for (uid, sid, layer_id) in self.batches[sample_idx]]
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
        
        # 处理图像条件 — 统一使用 flat 格式，支持单视图和多视图混合
        # 每个 mesh 贡献 1 张（单视图）或 N 张（多视图）图像
        # images: (total_views, C, H, W), view_indices: (total_views,), mv_cu_seqlens: (B+1,)
        images = None
        image_masks = None
        view_indices = None
        mv_cu_seqlens = None
        has_any_image = 'image' in batch[0] or 'mv_images' in batch[0]
        if has_any_image:
            images_flat_list = []
            image_masks_flat_list = []
            view_idx_list = []
            for b in batch:
                if 'mv_images' in b:
                    images_flat_list.append(b['mv_images'])           # (N_i, 3, H, W)
                    image_masks_flat_list.append(b['mv_image_masks']) # (N_i, H, W)
                    view_idx_list.append(b['mv_view_indices'])        # (N_i,) int64
                else:
                    images_flat_list.append(b['image'].unsqueeze(0))  # (1, 3, H, W)
                    image_masks_flat_list.append(b['image_mask'].unsqueeze(0))  # (1, H, W)
                    view_idx_list.append(torch.zeros(1, dtype=torch.long))
            num_views_per_mesh = [x.shape[0] for x in images_flat_list]
            images = torch.cat(images_flat_list, dim=0)               # (total_views, 3, H, W)
            image_masks = torch.cat(image_masks_flat_list, dim=0)      # (total_views, H, W)
            view_indices = torch.cat(view_idx_list, dim=0)            # (total_views,)
            mv_cu_seqlens = torch.zeros(len(num_views_per_mesh) + 1, dtype=torch.int32)
            mv_cu_seqlens[1:] = torch.cumsum(torch.tensor(num_views_per_mesh, dtype=torch.int32), dim=0)
        
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
            image_masks=image_masks,
            uncond_images=uncond_images,
            view_indices=view_indices,
            mv_cu_seqlens=mv_cu_seqlens,
            num_vertices=num_vertices,
            meshes=[b['mesh'] for b in batch] if 'mesh' in batch[0] else None,
        )
        
        del batch
        gc.collect()
        
        return result
