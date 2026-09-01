"""
Image Validation and Plant Integrity Verification Module.

Checks file integrity, corrupted images, resolution boundaries, and HSV green 
plant coverage ratio to filter out non-plant images (PredictionStatus.NOT_A_PLANT).
"""

import os
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from src.contracts import PredictionStatus


class ImageValidator:
    """
    Validates uploaded images before deep neural network inference.
    """
    def __init__(
        self,
        min_width: int = 32,
        min_height: int = 32,
        foliage_green_threshold: float = 0.05
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.foliage_green_threshold = foliage_green_threshold

    def validate_file_integrity(self, input_source: Union[str, Image.Image, np.ndarray]) -> Tuple[bool, Optional[Image.Image], str]:
        """
        Validates file existence, format, and decodability.
        """
        if isinstance(input_source, str):
            if not os.path.exists(input_source):
                return False, None, f"File not found: '{input_source}'"
            try:
                img = Image.open(input_source)
                img.verify()  # Verify image integrity
                img = Image.open(input_source).convert("RGB")  # Reopen to read pixel data
            except Exception as e:
                return False, None, f"Corrupted or invalid image file: {str(e)}"
        elif isinstance(input_source, Image.Image):
            try:
                img = input_source.convert("RGB")
            except Exception as e:
                return False, None, f"Invalid PIL image object: {str(e)}"
        elif isinstance(input_source, np.ndarray):
            try:
                if input_source.ndim != 3 or input_source.shape[2] != 3:
                    return False, None, f"Invalid NumPy image dimensions: {input_source.shape}"
                img = Image.fromarray(input_source.astype(np.uint8)).convert("RGB")
            except Exception as e:
                return False, None, f"Failed to decode NumPy array: {str(e)}"
        else:
            return False, None, f"Unsupported input type: {type(input_source)}"

        return True, img, "Image file decoded successfully."

    def validate_dimensions(self, img: Image.Image) -> Tuple[bool, str]:
        """Validates that width and height meet minimum threshold requirements."""
        width, height = img.size
        if width < self.min_width or height < self.min_height:
            return False, f"Image dimensions ({width}x{height}) below minimum required ({self.min_width}x{self.min_height})."
        return True, "Dimensions valid."

    def evaluate_plant_foliage_ratio(self, img: Image.Image) -> float:
        """
        Calculates the proportion of pixels matching plant foliage (green/chlorophyll spectrum).
        Uses HSV color space thresholding:
        - Hue range: [25, 95] (greenish tones) or expanded [15, 100] (yellow-green to dark green leaves)
        - Saturation range: [25, 255]
        - Value range: [25, 255]
        """
        img_arr = np.array(img)
        
        if HAS_CV2:
            hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
            # Define HSV boundaries for plant leaf spectrum (green, yellow-green, brown-green)
            lower_green = np.array([20, 20, 20])
            upper_green = np.array([100, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_ratio = np.count_nonzero(mask) / float(img_arr.shape[0] * img_arr.shape[1])
        else:
            # Fallback green channel dominance check
            r = img_arr[:, :, 0].astype(np.float32)
            g = img_arr[:, :, 1].astype(np.float32)
            b = img_arr[:, :, 2].astype(np.float32)
            
            # Plant leaves typically have green > red and green > blue or high chlorophyll ratio
            green_mask = (g > r * 0.8) & (g > b * 0.8) & (g > 30)
            green_ratio = np.count_nonzero(green_mask) / float(img_arr.shape[0] * img_arr.shape[1])

        return float(green_ratio)

    def validate(self, input_source: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Executes full validation pipeline.
        
        Returns:
            Dict containing:
            - is_valid (bool)
            - status (str): 'supported' or 'not_a_plant'
            - reason (str)
            - foliage_ratio (float)
            - image (PIL Image if valid)
        """
        ok, img, msg = self.validate_file_integrity(input_source)
        if not ok or img is None:
            return {
                "is_valid": False,
                "status": PredictionStatus.NOT_A_PLANT.value,
                "reason": msg,
                "foliage_ratio": 0.0,
                "image": None
            }

        ok_dim, dim_msg = self.validate_dimensions(img)
        if not ok_dim:
            return {
                "is_valid": False,
                "status": PredictionStatus.NOT_A_PLANT.value,
                "reason": dim_msg,
                "foliage_ratio": 0.0,
                "image": None
            }

        foliage_ratio = self.evaluate_plant_foliage_ratio(img)
        if foliage_ratio < self.foliage_green_threshold:
            return {
                "is_valid": False,
                "status": PredictionStatus.NOT_A_PLANT.value,
                "reason": f"Foliage coverage ratio ({foliage_ratio:.3f}) below threshold ({self.foliage_green_threshold:.3f}). Image does not appear to contain a recognized plant.",
                "foliage_ratio": foliage_ratio,
                "image": img
            }

        return {
            "is_valid": True,
            "status": PredictionStatus.SUPPORTED.value,
            "reason": "Image passed all plant validation checks.",
            "foliage_ratio": foliage_ratio,
            "image": img
        }
