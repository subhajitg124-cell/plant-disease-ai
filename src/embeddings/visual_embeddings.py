"""
Deep Visual Embedding Extractor.
"""

import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Union, List, Optional, Any
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.vision.model import PlantDiseaseCNN
from src.preprocessing.pipeline import PreprocessingPipeline


class VisualEmbeddingExtractor:
    """
    Feature Extractor for Deep Visual Embeddings.
    """
    def __init__(
        self,
        model_path: Optional[str] = "models/plant_disease_cnn.pth",
        embedding_dim: int = 128,
        device: Optional[str] = None
    ):
        self.embedding_dim = embedding_dim
        self.pipeline = PreprocessingPipeline(target_size=(224, 224))

        if HAS_TORCH:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.model = PlantDiseaseCNN(num_classes=38, embedding_dim=embedding_dim).to(self.device)
            if model_path and os.path.exists(model_path):
                try:
                    state = torch.load(model_path, map_location=self.device)
                    if isinstance(state, dict) and "state_dict" in state:
                        self.model.load_state_dict(state["state_dict"])
                    else:
                        self.model.load_state_dict(state)
                except Exception as e:
                    print(f"Warning: Could not load embedding model weights: {e}")
            self.model.eval()
        else:
            self.device = "cpu"
            self.model = None

    def extract(self, input_source: Union[str, Image.Image, np.ndarray, Any]) -> List[float]:
        """
        Extracts L2-normalized float embedding list for a single image input.
        """
        is_valid, data, _ = self.pipeline.process_image(input_source, return_tensor=True)
        if not is_valid or data is None:
            return [0.0] * self.embedding_dim

        if HAS_TORCH and isinstance(data, torch.Tensor):
            if data.ndim == 3:
                data = data.unsqueeze(0)
            data = data.to(self.device)

            with torch.no_grad():  # type: ignore[attr-defined]
                assert self.model is not None
                emb_tensor = self.model.extract_features(data)
                emb_np = emb_tensor.cpu().numpy()[0]
                return emb_np.tolist()
        else:
            arr = np.array(data).flatten()
            sub = arr[:self.embedding_dim]
            if len(sub) < self.embedding_dim:
                sub = np.pad(sub, (0, self.embedding_dim - len(sub)))
            norm = np.linalg.norm(sub)
            if norm > 0:
                sub = sub / norm
            return sub.tolist()

    def extract_batch(self, input_sources: List[Union[str, Image.Image, np.ndarray, Any]]) -> List[List[float]]:
        """Extracts embeddings for a batch of images."""
        return [self.extract(src) for src in input_sources]
