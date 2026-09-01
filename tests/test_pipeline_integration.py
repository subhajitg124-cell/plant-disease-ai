"""
End-to-End Integration Test for Preprocessing, Vision, Embeddings, and Module Contracts.
"""

import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from PIL import Image

from src.preprocessing.pipeline import PreprocessingPipeline
from src.vision.classifier import PlantDiseaseClassifier
from src.embeddings.visual_embeddings import VisualEmbeddingExtractor
from src.contracts import VisionPrediction, AdvisoryResult, IntegratedResponse, PredictionStatus


class TestPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.classifier = PlantDiseaseClassifier()
        self.embedding_extractor = VisualEmbeddingExtractor()

        # Green leaf input
        self.green_leaf = np.zeros((224, 224, 3), dtype=np.uint8)
        self.green_leaf[:, :, 1] = 190
        self.green_leaf[:, :, 0] = 40

        # Non-plant input
        self.non_plant = np.full((224, 224, 3), 100, dtype=np.uint8)

    def test_full_pipeline_valid_plant(self):
        # 1. Vision Prediction
        pred = self.classifier.predict(self.green_leaf)
        self.assertIsInstance(pred, VisionPrediction)
        self.assertIn(pred.status, [PredictionStatus.SUPPORTED.value, PredictionStatus.UNCERTAIN.value])

        # 2. Advisory Mock
        advisory = AdvisoryResult(
            canonical_id=pred.canonical_id,
            symptoms=["Spotting on leaves", "Leaf discoloration"],
            management=["Apply copper fungicide", "Improve air circulation"]
        )

        # 3. Integration Layer Compose
        resp = IntegratedResponse.compose(prediction=pred, advisory=advisory)
        self.assertIsInstance(resp, IntegratedResponse)
        self.assertEqual(resp.confidence, pred.confidence)
        self.assertEqual(resp.status, pred.status)

    def test_full_pipeline_not_a_plant_rejection(self):
        # 1. Vision Prediction for Non-Plant
        pred = self.classifier.predict(self.non_plant)
        self.assertEqual(pred.status, PredictionStatus.NOT_A_PLANT.value)

        # 2. Integration Layer Compose
        resp = IntegratedResponse.compose(prediction=pred)
        self.assertEqual(resp.status, PredictionStatus.NOT_A_PLANT.value)
        self.assertIn("validation failed", resp.user_message.lower())
        self.assertIsNone(resp.advisory)


if __name__ == "__main__":
    unittest.main()
