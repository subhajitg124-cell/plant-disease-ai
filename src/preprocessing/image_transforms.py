"""
Image Transformations and Normalization Module.

Provides standard image transformations, resizing to 224x224, 
normalization using ImageNet mean/std, data augmentations for training,
and conversions between PIL, OpenCV (NumPy BGR/RGB), and PyTorch Tensors.
"""

from typing import Union, Tuple, Optional, List
import numpy as np
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageTransformer:
    """
    Handles image resizing, normalization, color space conversions, and data augmentations.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        mean: Union[Tuple[float, float, float], np.ndarray] = IMAGENET_MEAN,
        std: Union[Tuple[float, float, float], np.ndarray] = IMAGENET_STD,
        augment: bool = False
    ):
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.augment = augment

    def load_image(self, input_source: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """
        Loads an image from file path, PIL Image, or NumPy array into a PIL RGB Image.
        """
        if isinstance(input_source, str):
            img = Image.open(input_source).convert("RGB")
        elif isinstance(input_source, Image.Image):
            img = input_source.convert("RGB")
        elif isinstance(input_source, np.ndarray):
            # Check if BGR (e.g. from cv2) or RGB
            if input_source.ndim == 3 and input_source.shape[2] == 3:
                # Assume RGB if loaded normally, convert array to PIL
                img = Image.fromarray(input_source.astype(np.uint8)).convert("RGB")
            else:
                raise ValueError(f"Invalid NumPy image shape: {input_source.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(input_source)}")
        return img

    def resize(self, img: Image.Image) -> Image.Image:
        """Resizes PIL image to target resolution using BICUBIC interpolation."""
        return img.resize(self.target_size, Image.Resampling.BICUBIC)

    def apply_augmentations(self, img: Image.Image) -> Image.Image:
        """Applies training augmentations (random flip, slight rotation)."""
        if not self.augment:
            return img

        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # Random rotation (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC)

        return img

    def to_array(self, img: Image.Image) -> np.ndarray:
        """Converts PIL RGB image to normalized float32 array of shape (3, H, W)."""
        arr = np.array(img, dtype=np.float32) / 255.0  # Scale to [0, 1]
        
        # Apply normalization: (x - mean) / std
        arr = (arr - self.mean) / self.std
        
        # Transpose from (H, W, C) to (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    def to_tensor(self, arr: np.ndarray):
        """Converts (C, H, W) numpy array to PyTorch Tensor or returns array if torch unavailable."""
        if HAS_TORCH:
            tensor = torch.from_numpy(arr).float()
            return tensor
        return arr

    def transform(self, input_source: Union[str, Image.Image, np.ndarray], return_tensor: bool = True):
        """
        Executes full preprocessing sequence: Load -> Resize -> Augment -> Normalize -> Tensor/Array.
        """
        img = self.load_image(input_source)
        img = self.resize(img)
        if self.augment:
            img = self.apply_augmentations(img)

        arr = self.to_array(img)

        if return_tensor and HAS_TORCH:
            return self.to_tensor(arr)
        return arr


def get_default_transforms(target_size: Tuple[int, int] = (224, 224), augment: bool = False) -> ImageTransformer:
    """Helper function to instantiate default ImageTransformer."""
    return ImageTransformer(target_size=target_size, augment=augment)
