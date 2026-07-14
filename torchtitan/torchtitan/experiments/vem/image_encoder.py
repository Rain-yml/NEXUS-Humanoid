from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import AutoModel


class DINOv2ImageEncoder(nn.Module):
    def __init__(self, model_name: Literal[
        "facebook/dinov2-with-registers-large",
        "facebook/dinov2-large",
    ], return_the_nth_hidden_states: int = -1):
        super().__init__()
        self.dtype = torch.bfloat16
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype)
        self.model.requires_grad_(False)
        self.model.eval()

        DINOv2_INPUT_MEAN = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32)[
            None, :, None, None
        ]
        DINOv2_INPUT_STD = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32)[
            None, :, None, None
        ]
        self.register_buffer("DINOv2_INPUT_MEAN", DINOv2_INPUT_MEAN, persistent=False)
        self.register_buffer("DINOv2_INPUT_STD", DINOv2_INPUT_STD, persistent=False)
        self.max_size = 518
        self.hidden_size = self.model.config.hidden_size
        self.return_the_nth_hidden_states = return_the_nth_hidden_states

    def preprocess(self, image: torch.Tensor):
        B, C, H, W = image.shape
        assert C == 3 and H <= self.max_size and W <= self.max_size
        image = (image - self.DINOv2_INPUT_MEAN.to(image)) / self.DINOv2_INPUT_STD.to(image)
        return image
    
    def forward(self, image: torch.Tensor):
        dtype = image.dtype
        if self.return_the_nth_hidden_states >= 0:
            out = self.model(image.to(self.dtype), output_hidden_states=True)
            features = out.hidden_states[self.return_the_nth_hidden_states]
        else:
            features = self.model(image.to(self.dtype)).last_hidden_state
        return features.to(dtype)


class DINOv3ImageEncoder(DINOv2ImageEncoder):
    def __init__(self, model_name: str, return_the_nth_hidden_states: int = -1):
        super().__init__(model_name, return_the_nth_hidden_states)
        self.max_size = 1024


class DINOv2ImageEncoderWithoutPooler(DINOv2ImageEncoder):
    def forward(self, image: torch.Tensor):
        dtype = image.dtype
        if self.return_the_nth_hidden_states >= 0:
            out = self.model(image.to(self.dtype), output_hidden_states=True)
            features = out.hidden_states[self.return_the_nth_hidden_states]
        else:
            features = self.model(image.to(self.dtype)).last_hidden_state
        return features[:,1:,:].to(dtype)    


class SigLIP2ImageEncoder(nn.Module):
    def __init__(self, model_name: Literal[
        "google/siglip2-so400m-patch14-224",
        "google/siglip2-so400m-patch16-512",
        "google/siglip2-large-patch16-512"
    ], return_the_nth_hidden_states: int = -1):
        super().__init__()
        self.dtype = torch.bfloat16
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype)
        del self.model.text_model
        self.model.requires_grad_(False)
        self.model.eval()

        SIGLIP2_INPUT_MEAN = torch.as_tensor([0.5, 0.5, 0.5], dtype=torch.float32)[
            None, :, None, None
        ]
        SIGLIP2_INPUT_STD = torch.as_tensor([0.5, 0.5, 0.5], dtype=torch.float32)[
            None, :, None, None
        ]
        self.register_buffer("SIGLIP2_INPUT_MEAN", SIGLIP2_INPUT_MEAN, persistent=False)
        self.register_buffer("SIGLIP2_INPUT_STD", SIGLIP2_INPUT_STD, persistent=False)
        self.max_size = 224
        self.hidden_size = self.model.config.vision_config.hidden_size
        self.return_the_nth_hidden_states = return_the_nth_hidden_states

    def preprocess(self, image: torch.Tensor):
        B, C, H, W = image.shape
        assert C == 3 and H <= self.max_size and W <= self.max_size
        image = (image - self.SIGLIP2_INPUT_MEAN.to(image)) / self.SIGLIP2_INPUT_STD.to(image)
        return image
    
    def forward(self, image: torch.Tensor):
        dtype = image.dtype
        out = self.model.vision_model(
            pixel_values=image.to(self.dtype),
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True
        )
        if self.return_the_nth_hidden_states >= 0:
            features = out.hidden_states[self.return_the_nth_hidden_states]
        else:
            features = out.last_hidden_state
        return features.to(dtype)
