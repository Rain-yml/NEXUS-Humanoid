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
from torch.utils.data import IterableDataset, Dataset
from torch.distributed.checkpoint.stateful import Stateful
from torchtitan.experiments.vem.datasets.json_utils import load_json


class ImageDataset(Dataset):

    def __init__(
        self,
        image_list: Union[str, List[str]],
        size: int = 518,
    ) -> None:
        if isinstance(image_list, str):
            image_list = [image_list]
        
        image_paths = []
        for jp in image_list:
            if jp.endswith(".json") or jp.endswith(".json.gz"):
                image_paths.extend(load_json(jp))
            elif jp.endswith(".png") or jp.endswith(".jpg") or jp.endswith(".jpeg") or jp.endswith(".webp"):
                image_paths.append(jp)
            else:
                raise ValueError(f"Invalid image list: {image_list}")
        
        self.size = size
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGBA")
        image = image.resize((self.size, self.size), resample=Image.BICUBIC)

        image = np.asarray(image).astype(np.float32) / 255.0
        # normal = normal[:, :, :3] * normal[:, :, 3:4] + 0.5 * (1 - normal[:, :, 3:4])
        image = image[:, :, :3] * image[:, :, 3:4] + 0.5 * (1 - image[:, :, 3:4])
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).clamp(0, 1)
        return {
            'image': image_tensor,
        }

    