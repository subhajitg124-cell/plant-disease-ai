"""
Unit Tests for Vision Module and PlantDiseaseClassifier.
"""

import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import torch
import numpy as np
from PIL import Image

from src.vision.model import PlantDiseaseCNN
from src.vision.classifier import PlantDiseaseClassifier
from src.vision.train import train_model
from src.contracts import VisionPrediction, PredictionStatus


class TestVisionModule(unittest.TestCase):

    def test_cnn_model_forward_pass(self):
        model = PlantDiseaseCNN(num_classes=38, embedding_dim=128)
        dummy_input = torch.randn(2, 3, 224, 224)
        logits, embedding = model(dummy_input)

        self.assertEqual(logits.shape, (2, 38))
        self.assertEqual(embedding.shape, (2, 128))
        
        # Check L2 normalization of embedding
        norm = torch.norm(embedding[0], p=2).item()
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_classifier_prediction_supported(self):
        classifier = PlantDiseaseClassifier()
        img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        img_arr[:, :, 1] = 200  # Green leaf
        img_arr[:, :, 0] = 30

        pred = classifier.predict(img_arr)
        self.assertIsInstance(pred, VisionPrediction)
        self.assertIn(pred.status, [PredictionStatus.SUPPORTED.value, PredictionStatus.UNCERTAIN.value])
        self.assertIsNotNone(pred.embedding)
        assert pred.embedding is not None
        self.assertEqual(len(pred.embedding), 128)

    def test_classifier_prediction_not_a_plant(self):
        classifier = PlantDiseaseClassifier()
        img_arr = np.full((224, 224, 3), 120, dtype=np.uint8)

        pred = classifier.predict(img_arr)
        self.assertEqual(pred.status, PredictionStatus.NOT_A_PLANT.value)
        self.assertEqual(pred.canonical_id, "not_a_plant")
        self.assertEqual(pred.confidence, 0.0)

    def test_training_step(self):
        result = train_model(epochs=1, batch_size=8)
        self.assertEqual(result["status"], "success")
        self.assertTrue(os.path.exists("models/plant_disease_cnn.pth"))


if __name__ == "__main__":
    unittest.main()
