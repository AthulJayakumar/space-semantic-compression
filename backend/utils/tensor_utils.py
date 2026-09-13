"""backend.utils.tensor_utils

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import torch
import torchvision.transforms as T
from PIL import Image


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def image_to_tensor(image: Image.Image, device: torch.device, stride: int) -> torch.Tensor:
    image = resize_to_stride(image.convert("RGB"), stride)
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    return transform(image).unsqueeze(0).to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = (tensor.detach().clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1).cpu()[0]
    array = (tensor.numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(array)


def image_to_metric_tensor(image: Image.Image, device: torch.device | str = "cpu") -> torch.Tensor:
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    return transform(image.convert("RGB")).unsqueeze(0).to(device)


def resize_to_stride(image: Image.Image, stride: int) -> Image.Image:
    width, height = image.size
    target_width = max(stride, (width // stride) * stride)
    target_height = max(stride, (height // stride) * stride)
    if (target_width, target_height) == (width, height):
        return image
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
