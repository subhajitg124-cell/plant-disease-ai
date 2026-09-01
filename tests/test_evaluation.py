"""
Unit Tests for Evaluation Metrics and Report Generation.
"""

import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import tempfile
import numpy as np

from src.evaluation.metrics import calculate_metrics, ConfusionMatrix, calculate_topk_accuracy
from src.evaluation.evaluator import ModelEvaluator


class TestEvaluationModule(unittest.TestCase):

    def setUp(self):
        self.y_true = [0, 1, 2, 0, 1, 2]
        self.y_pred = [0, 1, 2, 0, 2, 2]
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_metrics_calculation(self):
        metrics = calculate_metrics(self.y_true, self.y_pred, num_classes=3)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertGreater(metrics["accuracy"], 0.5)

    def test_topk_accuracy(self):
        y_probs = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        y_true = [0, 1, 2]
        top1 = calculate_topk_accuracy(y_true, y_probs, k=1)
        top3 = calculate_topk_accuracy(y_true, y_probs, k=3)
        self.assertEqual(top1, 1.0)
        self.assertEqual(top3, 1.0)

    def test_evaluator_report_saving(self):
        evaluator = ModelEvaluator(reports_dir=self.temp_dir.name)
        metrics = calculate_metrics(self.y_true, self.y_pred, num_classes=3)
        json_p, md_p = evaluator.generate_and_save_reports(metrics)

        self.assertTrue(os.path.exists(json_p))
        self.assertTrue(os.path.exists(md_p))


if __name__ == "__main__":
    unittest.main()
