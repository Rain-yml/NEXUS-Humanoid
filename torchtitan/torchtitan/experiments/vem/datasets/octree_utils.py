from typing import List, Tuple
import torch
from dataclasses import dataclass
from typing import List, Optional
import torch.nn.functional as F
import numpy as np

# ============================================================================
# 编码转换工具
# ============================================================================
@dataclass
class OctreeData:
    """
    按层组织的八叉树数据结构
    
    Attributes:
        layer_occupancy: List[Tensor] 每层的 8-bit 占用状态
            - layer_occupancy[i]: (num_nodes_at_layer_i, 8) 每个节点的 8 个子节点占用
        layer_occupancy_onehot: List[Tensor] 每层的 256-dim one-hot 表示
            - layer_occupancy_onehot[i]: (num_nodes_at_layer_i, 256)
        layer_parent_centers: List[Tensor] 每层节点的中心坐标（离散化整数坐标）
            - layer_parent_centers[i]: (num_nodes_at_layer_i, 3) dtype=long, 范围 [0, grid_size]
            - 使用整数坐标避免 bfloat16 精度问题
        layer_child_ids: List[Tensor] 每层节点在父节点中的子节点 ID (0-7)
            - layer_child_ids[i]: (num_nodes_at_layer_i,)
        layer_depths: List[int] 每层的深度值
        surface_points: Tensor 表面采样点（通常为 xyz 或 xyz+normal，例如 (N, 6)）
            - 本项目里 surface_points.xyz 通常处于 mesh 坐标系（由 MeshProcessor 归一化到 [-1, 1]）
        discrete_points: Tensor 离散化后的顶点坐标
            - shape: (M, 3) 整数坐标 [0, grid_size-1]
    """
    layer_occupancy: List[torch.Tensor]
    # layer_occupancy_onehot: List[torch.Tensor]
    layer_parent_centers: List[torch.Tensor]
    # layer_child_ids: List[torch.Tensor]
    layer_depths: List[int]
    num_vertices: int
    # surface_points: torch.Tensor
    # discrete_points: Optional[torch.Tensor] = None
    
    # def to(self, device: torch.device) -> 'OctreeData':
    #     """将所有张量移动到指定设备"""
    #     return OctreeData(
    #         layer_occupancy=[t.to(device) for t in self.layer_occupancy],
    #         layer_occupancy_onehot=[t.to(device) for t in self.layer_occupancy_onehot],
    #         layer_parent_centers=[t.to(device) for t in self.layer_parent_centers],
    #         layer_child_ids=[t.to(device) for t in self.layer_child_ids],
    #         layer_depths=self.layer_depths,
    #         surface_points=self.surface_points.to(device),
    #         discrete_points=self.discrete_points.to(device) if self.discrete_points is not None else None,
    #     )
    
    @property
    def num_layers(self) -> int:
        """返回八叉树的层数"""
        return len(self.layer_occupancy)
    
    def get_layer_num_nodes(self, layer_idx: int) -> int:
        """返回指定层的节点数"""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return 0
        return self.layer_occupancy[layer_idx].shape[0]
    
    @property
    def total_nodes(self) -> int:
        """返回所有层的节点总数"""
        return sum(occ.shape[0] for occ in self.layer_occupancy)
    
    def __repr__(self) -> str:
        layer_info = [f"L{d}:{occ.shape[0]}" for d, occ in zip(self.layer_depths, self.layer_occupancy)]
        return (
            f"OctreeData(num_layers={self.num_layers}, "
            f"layers=[{', '.join(layer_info)}], "
            f"total_nodes={self.total_nodes}"
        )


@dataclass
class OctreeBatch:
    """
    批量 OctreeData，用于训练
    
    Attributes:
        layer_occupancy_flat: Tensor 所有层所有 batch 的占用状态（展平）
            - shape: (total_nodes_all_batches, 8) 或 (total_nodes_all_batches, 256)
        layer_parent_centers_flat: Tensor 所有节点中心（离散化整数坐标，展平）
            - shape: (total_nodes_all_batches, 3), dtype=long, 范围 [0, grid_size]
        layer_depths_flat: Tensor 所有节点的深度（展平）
            - shape: (total_nodes_all_batches,)
        layer_child_ids_flat: Tensor 所有节点的子节点 ID（展平）
            - shape: (total_nodes_all_batches,)
        cu_seqlens: Tensor 累积序列长度，用于 Flash Attention varlen
            - shape: (batch_size + 1,)
        max_seqlen: int 最大序列长度
        surface_points_flat: Tensor 所有表面点（展平）
            - shape: (total_surface_points, 6) xyz + normal（xyz 通常为 mesh 坐标系 [-1, 1]）
        cu_seqlens_surface: Tensor 表面点的累积序列长度
            - shape: (batch_size + 1,)
        max_seqlen_surface: int 表面点最大序列长度
        layer_idx: int 当前训练的层索引
        batch_size: int 批大小
        instance_ids: List[str] 样本 ID 列表
        num_layers_per_mesh: List[int] 每个 mesh 的层数（用于避免 point_encoder 重复计算）
            - 当 train_all_layers=True 时，每个 mesh 会展开成多个 batch sample
            - 此字段记录每个原始 mesh 对应多少个 batch sample
        images: Optional[Tensor] 渲染的 normal 图像，用于图像条件
            - shape: (num_unique_meshes, C, H, W)
        dora_points_flat: Optional[Tensor] Dora 点（展平），给 VEM2DualCAEncoder 使用
            - shape: (total_dora_points, 3) 只有 xyz，从顶点均匀采样
        cu_seqlens_dora: Optional[Tensor] Dora 点的累积序列长度
            - shape: (batch_size + 1,)
        max_seqlen_dora: Optional[int] Dora 点最大序列长度
    """
    layer_occupancy_flat: torch.Tensor  # (total_nodes, 8) 或 (total_nodes, 256)
    layer_parent_centers_flat: torch.Tensor  # (total_nodes, 3)
    layer_depths_flat: torch.Tensor  # (total_nodes,)
    # layer_child_ids_flat: torch.Tensor  # (total_nodes,)
    cu_seqlens: torch.Tensor  # (batch_size + 1,)
    max_seqlen: int
    # surface_points_flat: torch.Tensor  # (total_surface_points, 6)
    # cu_seqlens_surface: torch.Tensor  # (batch_size + 1,)
    # max_seqlen_surface: int
    layer_idx: int
    batch_size: int
    instance_ids: Optional[List[str]] = None
    num_layers_per_mesh: Optional[List[int]] = None  # 每个 mesh 的层数
    images: Optional[torch.Tensor] = None  # (total_views, C, H, W) flat image conditions; use mv_cu_seqlens to split per mesh
    image_masks: Optional[torch.Tensor] = None  # (total_views, H, W) foreground masks aligned with images
    uncond_images: Optional[torch.Tensor] = None  # (num_unique_meshes, C, H, W) unconditional 图像条件
    num_vertices: Optional[torch.Tensor] = None # (num_unique_meshes,)
    num_faces: Optional[torch.Tensor] = None # (num_unique_meshes,)
    quad_ratios: Optional[torch.Tensor] = None # (num_unique_meshes,)
    symmetries: Optional[torch.Tensor] = None # (num_unique_meshes, 4) int64, 1=symmetric 0=uncertain
    meshes: Optional[List] = None
    view_indices: Optional[torch.Tensor] = None  # (total_views,) int64, view id 0-3 (front/left/back/right)
    mv_cu_seqlens: Optional[torch.Tensor] = None  # (num_unique_meshes+1,) int32, cumulative view counts per mesh
    # dora_points_flat: Optional[torch.Tensor] = None  # (total_dora_points, 3) dora 点，只有 xyz
    # cu_seqlens_dora: Optional[torch.Tensor] = None  # (batch_size + 1,)
    # max_seqlen_dora: Optional[int] = None
    
    def to(self, device: torch.device) -> 'OctreeBatch':
        """将所有张量移动到指定设备"""
        return OctreeBatch(
            layer_occupancy_flat=self.layer_occupancy_flat.to(device),
            layer_parent_centers_flat=self.layer_parent_centers_flat.to(device),
            layer_depths_flat=self.layer_depths_flat.to(device),
            # layer_child_ids_flat=self.layer_child_ids_flat.to(device),
            cu_seqlens=self.cu_seqlens.to(device),
            max_seqlen=self.max_seqlen,
            # surface_points_flat=self.surface_points_flat.to(device),
            # cu_seqlens_surface=self.cu_seqlens_surface.to(device),
            # max_seqlen_surface=self.max_seqlen_surface,
            layer_idx=self.layer_idx,
            batch_size=self.batch_size,
            instance_ids=self.instance_ids,
            num_layers_per_mesh=self.num_layers_per_mesh,
            images=self.images.to(device) if self.images is not None else None,
            image_masks=self.image_masks.to(device) if self.image_masks is not None else None,
            uncond_images=self.uncond_images.to(device) if self.uncond_images is not None else None,
            num_vertices=self.num_vertices.to(device) if self.num_vertices is not None else None,
            num_faces=self.num_faces.to(device) if self.num_faces is not None else None,
            quad_ratios=self.quad_ratios.to(device) if self.quad_ratios is not None else None,
            symmetries=self.symmetries.to(device) if self.symmetries is not None else None,
            meshes=self.meshes,
            view_indices=self.view_indices.to(device) if self.view_indices is not None else None,
            mv_cu_seqlens=self.mv_cu_seqlens.to(device) if self.mv_cu_seqlens is not None else None,
            # dora_points_flat=self.dora_points_flat.to(device) if self.dora_points_flat is not None else None,
            # cu_seqlens_dora=self.cu_seqlens_dora.to(device) if self.cu_seqlens_dora is not None else None,
            # max_seqlen_dora=self.max_seqlen_dora,
        )

def bits_to_index(bits: torch.Tensor) -> torch.Tensor:
    """
    将 8-bit 占用转为 0-255 整数索引
    
    Args:
        bits: (..., 8) 每个值是 0 或 1
        
    Returns:
        (...,) 整数索引 0-255
    """
    # bits[..., 0] 是最低位
    weights = 2 ** torch.arange(8, device=bits.device, dtype=bits.dtype)
    return (bits * weights).sum(dim=-1).long()


def index_to_bits(idx: torch.Tensor) -> torch.Tensor:
    """
    将 0-255 整数索引转为 8-bit
    
    Args:
        idx: (...,) 整数索引 0-255
        
    Returns:
        (..., 8) 每个值是 0 或 1
    """
    bits = []
    for i in range(8):
        bits.append((idx >> i) & 1)
    return torch.stack(bits, dim=-1).float()


def bits_to_onehot(bits: torch.Tensor) -> torch.Tensor:
    """
    将 8-bit 占用转为 256 维 one-hot
    
    Args:
        bits: (..., 8) 每个值是 0 或 1
        
    Returns:
        (..., 256) one-hot 向量
    """
    idx = bits_to_index(bits)
    return F.one_hot(idx, num_classes=256).float()


def onehot_to_bits(onehot: torch.Tensor) -> torch.Tensor:
    """
    将 256 维 one-hot/logits 转为 8-bit（用 argmax）
    
    Args:
        onehot: (..., 256) logits 或 one-hot
        
    Returns:
        (..., 8) 每个值是 0 或 1
    """
    idx = onehot.argmax(dim=-1)
    return index_to_bits(idx)


# ============================================================================
# 八叉树构建
# ============================================================================

def build_octree_by_layer(
    discrete_points: torch.LongTensor,
    grid_size: int = 128,
    max_depth: int = 7,
):
    """
    Build an octree from discrete 3D points.

    Args:
        discrete_points: (N, 3) LongTensor, values in [0, grid_size-1]
        grid_size: size of the grid, must satisfy 2^max_depth == grid_size
        max_depth: number of octree layers

    Returns:
        dict with keys:
            'voxel_occupancy_layers': list of (Ni, 8) tensors
            'voxel_position_layers': list of (Ni, 3) tensors
            'layer_depths': list of ints
    """
    assert 2 ** max_depth == grid_size
    assert discrete_points.ndim == 2 and discrete_points.shape[1] == 3

    device = discrete_points.device

    voxel_occupancy_layers = []
    voxel_position_layers = []
    layer_depths = list(range(max_depth))

    # ---- Level 0 initialization ----
    root_center = torch.tensor(
        [[grid_size // 2, grid_size // 2, grid_size // 2]],
        device=device,
        dtype=torch.long,
    )

    # For each voxel, keep the indices of points it contains
    voxel_centers = root_center
    voxel_point_indices = [torch.arange(discrete_points.shape[0], device=device)]

    # ---- Build layers ----
    for level in range(max_depth):
        voxel_size = grid_size // (2 ** level)
        half = voxel_size // 2

        num_voxels = len(voxel_point_indices)

        # Record voxel positions for this level
        voxel_position_layers.append(voxel_centers.clone())

        # Occupancy tensor for this layer
        occupancy = torch.zeros((num_voxels, 8), dtype=torch.long, device=device)

        next_voxel_centers = []
        next_voxel_point_indices = []

        for i in range(num_voxels):
            center = voxel_centers[i]
            point_ids = voxel_point_indices[i]
            pts = discrete_points[point_ids]

            # Generate 8 child voxels
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        child_idx = dx * 4 + dy * 2 + dz

                        offset = torch.tensor(
                            [
                                (-half if dx == 0 else 0),
                                (-half if dy == 0 else 0),
                                (-half if dz == 0 else 0),
                            ],
                            device=device,
                        )
                        min_corner = center + offset
                        # Axis-aligned bounds of child voxel
                        max_corner = min_corner + half

                        mask = (
                            (pts[:, 0] >= min_corner[0]) & (pts[:, 0] < max_corner[0]) &
                            (pts[:, 1] >= min_corner[1]) & (pts[:, 1] < max_corner[1]) &
                            (pts[:, 2] >= min_corner[2]) & (pts[:, 2] < max_corner[2])
                        )
                        if mask.any():
                            occupancy[i, child_idx] = 1
                            if half > 1:
                                child_center = min_corner + half // 2
                                next_voxel_centers.append(child_center)
                                next_voxel_point_indices.append(point_ids[mask])

        voxel_occupancy_layers.append(occupancy)

        # Prepare for next level
        if level < max_depth - 1:
            if len(next_voxel_centers) == 0:
                break

            voxel_centers = torch.stack(next_voxel_centers, dim=0)
            voxel_point_indices = next_voxel_point_indices

    return OctreeData(
        layer_occupancy=voxel_occupancy_layers,
        layer_parent_centers=voxel_position_layers,
        layer_depths=layer_depths,
        num_vertices=discrete_points.shape[0],
    )

def build_octree_specific_layer(
    discrete_points: torch.LongTensor,
    depth: int,
    grid_size: int = 128,
    max_depth: int = 7,
):
    """
    Directly compute a single octree layer without building the full tree.

    Args:
        discrete_points: (N,3) LongTensor in [0, grid_size-1]
        depth: desired octree depth (0 = root)
        grid_size: must satisfy 2^max_depth == grid_size
        max_depth: total tree depth

    Returns:
        occupancy: (Ni,8) tensor
        voxel_centers: (Ni,3) tensor
        layer_depth: int
    """
    assert 0 <= depth < max_depth
    assert 2 ** max_depth == grid_size
    assert discrete_points.ndim == 2 and discrete_points.shape[1] == 3

    device = discrete_points.device

    # ---- compute voxel grid at this depth ----
    voxel_size = grid_size // (2 ** depth)
    half = voxel_size // 2

    # integer voxel coordinates
    voxel_coords = discrete_points // voxel_size

    # unique voxels + inverse map
    unique_coords, inverse = torch.unique(
        voxel_coords, dim=0, return_inverse=True
    )
    num_voxels = unique_coords.shape[0]

    # voxel centers
    voxel_centers = unique_coords * voxel_size + voxel_size // 2

    # ---- occupancy ----
    occupancy = torch.zeros((num_voxels, 8), dtype=torch.long, device=device)

    # relative position inside voxel
    local_pts = discrete_points - (voxel_coords * voxel_size)

    if half > 0:
        # compute child index for each point
        child_bits = (local_pts >= half).long()
        child_idx = child_bits[:, 0] * 4 + child_bits[:, 1] * 2 + child_bits[:, 2]

        # mark occupancy
        occupancy[inverse, child_idx] = 1

    return OctreeData(
        layer_occupancy=[occupancy],
        layer_parent_centers=[voxel_centers],
        layer_depths=[depth],
        num_vertices=discrete_points.shape[0],
    )

def octree_layers_to_points(
    layer_occupancy: List[torch.Tensor],
    grid_size: int = 128,
) -> torch.Tensor:
    """
    从分层 occupancy 重建点云（离散坐标）
    
    Args:
        layer_occupancy: List[Tensor] 每层的占用状态 (num_nodes, 8)
        grid_size: 网格大小
        
    Returns:
        points: (num_points, 3) 离散化的点坐标
    """
    device = layer_occupancy[0].device if layer_occupancy else torch.device('cpu')
    
    # BFS 遍历八叉树
    queue = [(
        torch.zeros(3, dtype=torch.long, device=device),
        torch.full((3,), grid_size, dtype=torch.long, device=device)
    )]
    
    points = []
    layer_idx = 0
    node_idx_in_layer = 0
    
    while queue and layer_idx < len(layer_occupancy):
        next_queue = []
        occupancy = layer_occupancy[layer_idx]
        
        for min_xyz, max_xyz in queue:
            if node_idx_in_layer >= occupancy.shape[0]:
                break
                
            voxel_size = max_xyz - min_xyz
            # 如果是最小单元，记录点
            if voxel_size[0] == 1:
                points.append(min_xyz)
                continue
            
            occ = occupancy[node_idx_in_layer]
            mid = (min_xyz + max_xyz) // 2
            
            for cid in range(8):
                if occ[cid] > 0.5:
                    ox, oy, oz = (cid >> 0) & 1, (cid >> 1) & 1, (cid >> 2) & 1
                    cmin = torch.stack([
                        min_xyz[0] if ox == 0 else mid[0],
                        min_xyz[1] if oy == 0 else mid[1],
                        min_xyz[2] if oz == 0 else mid[2]
                    ])
                    cmax = torch.stack([
                        mid[0] if ox == 0 else max_xyz[0],
                        mid[1] if oy == 0 else max_xyz[1],
                        mid[2] if oz == 0 else max_xyz[2]
                    ])
                    
                    # 如果子节点是最小单元，记录点
                    if (cmax - cmin)[0] == 1:
                        points.append(cmin)
                    else:
                        next_queue.append((cmin, cmax))
            
            node_idx_in_layer += 1
        
        queue = next_queue
        layer_idx += 1
        node_idx_in_layer = 0
    
    if not points:
        return torch.empty(0, 3, dtype=torch.long, device=device)
    return torch.stack(points)


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


def discretize_points(
    points: torch.Tensor,
    grid_size: int = 128,
) -> torch.LongTensor:
    """
    将归一化点坐标 [0, 1] 离散化为网格索引 [0, grid_size-1]
    
    Args:
        points: (N, 3) 归一化点坐标
        grid_size: 网格大小
        
    Returns:
        (N, 3) 离散化的网格索引
    """
    discrete = (points * (grid_size - 1)).round().long()
    discrete = discrete.clamp(0, grid_size - 1)
    return discrete


def normalize_points(
    points: torch.Tensor,
    center: bool = True,
) -> torch.Tensor:
    """
    将点云归一化到 [0, 1] 范围
    
    Args:
        points: (N, 3) 点坐标
        center: 是否先将点云中心化
        
    Returns:
        (N, 3) 归一化后的点坐标
    """
    if center:
        centroid = points.mean(dim=0, keepdim=True)
        points = points - centroid
    
    # 缩放到 [-0.5, 0.5] 然后平移到 [0, 1]
    max_dist = points.abs().max()
    if max_dist > 0:
        points = points / (max_dist * 2)  # 现在在 [-0.5, 0.5]
    points = points + 0.5  # 现在在 [0, 1]
    
    return points.clamp(0, 1)
