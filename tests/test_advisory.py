"""
Unit Tests for AdvisoryGenerator Module.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.contracts import VisionPrediction, PredictionStatus, IntegratedResponse
except ImportError:
    from contracts import VisionPrediction, PredictionStatus, IntegratedResponse

try:
    from src.advisory.advisory_generator import AdvisoryGenerator
except ImportError:
    try:
        from advisory.advisory_generator import AdvisoryGenerator
    except ImportError:
        from advisory_generator import AdvisoryGenerator


class TestAdvisoryGenerator(unittest.TestCase):
    """Test suite for AdvisoryGenerator safety and RAG integration."""

    def setUp(self):
        self.generator = AdvisoryGenerator()

    def test_supported_prediction_advisory(self):
        pred = VisionPrediction(
            plant="Tomato",
            disease="Early blight",
            canonical_id="tomato_early_blight",
            confidence=0.88,
            status=PredictionStatus.SUPPORTED.value,
            raw_label="Tomato___Early_blight",
            model_version="vision_v1"
        )
        res = self.generator.generate_advisory(pred)
        self.assertIsInstance(res, IntegratedResponse)
        self.assertIsNotNone(res.advisory)
        self.assertEqual(res.status, PredictionStatus.SUPPORTED.value)
        self.assertGreater(len(res.sources), 0)

    def test_uncertain_prediction_suppression(self):
        pred = VisionPrediction(
            plant="Tomato",
            disease="Early blight",
            canonical_id="tomato_early_blight",
            confidence=0.35,
            status=PredictionStatus.UNCERTAIN.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertIsNone(res.advisory)
        self.assertIn("confidence", res.user_message)
        self.assertGreater(len(res.warnings), 0)

    def test_not_a_plant_prediction_suppression(self):
        pred = VisionPrediction(
            plant="Non-Plant",
            disease="Invalid Image",
            canonical_id="not_a_plant",
            confidence=0.0,
            status=PredictionStatus.NOT_A_PLANT.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertIsNone(res.advisory)
        self.assertIn("failed plant foliage validation", res.user_message)


if __name__ == "__main__":
    unittest.main()
