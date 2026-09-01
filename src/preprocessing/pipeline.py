"""
High-Level Preprocessing Pipeline Manager.

Integrates ImageValidator, ImageTransformer, and VideoFrameExtractor into a unified 
interface for processing images and video streams prior to CNN vision inference.
"""

from typing import Union, List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

from .image_transforms import ImageTransformer
from .image_validator import ImageValidator
from .video_extractor import VideoFrameExtractor
from src.contracts import PredictionStatus


class PreprocessingPipeline:
    """
    Unified end-to-end preprocessing pipeline.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        min_foliage_ratio: float = 0.05
    ):
        self.transformer = ImageTransformer(target_size=target_size, augment=augment)
        self.validator = ImageValidator(foliage_green_threshold=min_foliage_ratio)
        self.video_extractor = VideoFrameExtractor()

    def process_image(
        self,
        input_source: Union[str, Image.Image, np.ndarray],
        return_tensor: bool = True
    ) -> Tuple[bool, Optional[Any], Dict[str, Any]]:
        """
        Validates and transforms a single input image.
        
        Returns:
            Tuple of (is_valid: bool, transformed_data: Tensor/Array or None, validation_meta: Dict).
        """
        val_result = self.validator.validate(input_source)
        if not val_result["is_valid"]:
            return False, None, val_result

        img = val_result["image"]
        transformed = self.transformer.transform(img, return_tensor=return_tensor)
        return True, transformed, val_result

    def process_batch(
        self,
        input_sources: List[Union[str, Image.Image, np.ndarray]],
        return_tensor: bool = True
    ) -> List[Tuple[bool, Optional[Any], Dict[str, Any]]]:
        """Processes a batch of input images."""
        return [self.process_image(src, return_tensor=return_tensor) for src in input_sources]

    def process_video(
        self,
        video_path: str,
        return_tensor: bool = True
    ) -> Tuple[bool, List[Any], List[Dict[str, Any]]]:
        """
        Extracts, validates, and transforms frames from a video stream.
        """
        frames_meta = self.video_extractor.extract_frames_from_video(video_path)
        if not frames_meta:
            return False, [], [{"reason": "No valid frames extracted from video."}]

        valid_tensors = []
        valid_metas = []

        for pil_img, meta in frames_meta:
            ok, data, val_meta = self.process_image(pil_img, return_tensor=return_tensor)
            if ok:
                valid_tensors.append(data)
                meta.update(val_meta)
                valid_metas.append(meta)

        if not valid_tensors:
            return False, [], [{"reason": "All video frames failed plant validation checks."}]

        return True, valid_tensors, valid_metas
