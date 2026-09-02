"""
Unit Tests for AdvisoryGenerator Module.

Tests:
  - Full advisory generation for supported predictions (grounding verification)
  - Advisory message synthesis includes all key fields
  - Evidence chunks populated from retrieved KB documents
  - Sources populated from retrieved KB documents
  - Low-confidence / uncertain prediction suppression
  - Non-plant input rejection
  - Unknown disease out-of-distribution suppression
  - Batch advisory generation
  - Advisory report formatting
  - Grounding quality check integration
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.contracts import VisionPrediction, PredictionStatus, IntegratedResponse, AdvisoryResult
from src.advisory.advisory_generator import AdvisoryGenerator


class TestAdvisoryGeneratorGrounding(unittest.TestCase):
    """Core grounding and advisory generation tests."""

    def setUp(self):
        self.generator = AdvisoryGenerator()

    # ── Supported predictions ──────────────────────────────────────────────

    def test_supported_prediction_returns_integrated_response(self):
        """Supported prediction should return a fully-populated IntegratedResponse."""
        pred = VisionPrediction(
            plant="Tomato", disease="Early blight",
            canonical_id="tomato_early_blight", confidence=0.88,
            status=PredictionStatus.SUPPORTED.value,
            raw_label="Tomato___Early_blight", model_version="vision_v1"
        )
        res = self.generator.generate_advisory(pred)
        self.assertIsInstance(res, IntegratedResponse)
        self.assertEqual(res.status, PredictionStatus.SUPPORTED.value)

    def test_supported_prediction_has_advisory(self):
        """Advisory object should be present for supported predictions."""
        pred = VisionPrediction(
            plant="Apple", disease="Apple scab",
            canonical_id="apple_apple_scab", confidence=0.92,
            status=PredictionStatus.SUPPORTED.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertIsNotNone(res.advisory)
        self.assertIsInstance(res.advisory, AdvisoryResult)

    def test_supported_prediction_evidence_populated(self):
        """Evidence chunks should be populated from retrieved KB content."""
        pred = VisionPrediction(
            plant="Apple", disease="Apple scab",
            canonical_id="apple_apple_scab", confidence=0.91,
            status=PredictionStatus.SUPPORTED.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertGreater(len(res.evidence), 0, "Evidence chunks must be non-empty")
        for chunk in res.evidence:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk), 0)

    def test_supported_prediction_sources_populated(self):
        """Sources list should be populated from retrieved KB document."""
        pred = VisionPrediction(
            plant="Tomato", disease="Early blight",
            canonical_id="tomato_early_blight", confidence=0.85,
            status=PredictionStatus.SUPPORTED.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertGreater(len(res.sources), 0, "Sources must be non-empty for grounded advisory")

    def test_supported_prediction_user_message_contains_diagnosis(self):
        """User message should mention the plant and disease names."""
        pred = VisionPrediction(
            plant="Tomato", disease="Early blight",
            canonical_id="tomato_early_blight", confidence=0.88,
            status=PredictionStatus.SUPPORTED.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertIn("Tomato", res.user_message)
        self.assertIn("Early blight", res.user_message)

    def test_supported_prediction_user_message_contains_confidence(self):
        """User message should include the confidence percentage."""
        pred = VisionPrediction(
            plant="Apple", disease="Apple scab",
            canonical_id="apple_apple_scab", confidence=0.76,
            status=PredictionStatus.SUPPORTED.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertIn("76.0%", res.user_message)

    def test_supported_prediction_message_includes_symptoms(self):
        """User message should reference symptoms from the KB document."""
        pred = VisionPrediction(
            plant="Apple", disease="Apple scab",
            canonical_id="apple_apple_scab", confidence=0.90,
            status=PredictionStatus.SUPPORTED.value
        )
        res = self.generator.generate_advisory(pred)
        # Message should include a symptom or management section
        lower_msg = res.user_message.lower()
        has_content = any(kw in lower_msg for kw in [
            "symptom", "prevent", "treatment", "management", "cause", "risk", "spots", "blight"
        ])
        self.assertTrue(has_content, f"Expected advisory content in message: {res.user_message[:200]}")

    def test_supported_prediction_no_safety_warnings_when_grounded(self):
        """A grounded, supported prediction should have no safety warnings."""
        pred = VisionPrediction(
            plant="Apple", disease="Apple scab",
            canonical_id="apple_apple_scab", confidence=0.91,
            status=PredictionStatus.SUPPORTED.value
        )
        res = self.generator.generate_advisory(pred)
        # If grounded, there should be no suppression warnings
        suppression_warnings = [
            w for w in res.warnings
            if "suppressed" in w.lower() or "validation" in w.lower()
        ]
        self.assertEqual(len(suppression_warnings), 0)

    # ── Confidence grounding check ─────────────────────────────────────────

    def test_confidence_in_response_matches_prediction(self):
        """Response confidence should match the prediction confidence."""
        pred = VisionPrediction(
            plant="Tomato", disease="Late blight",
            canonical_id="tomato_late_blight", confidence=0.78,
            status=PredictionStatus.SUPPORTED.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertAlmostEqual(res.confidence, 0.78, places=5)

    # ── Safety guardrails ──────────────────────────────────────────────────

    def test_uncertain_prediction_advisory_suppressed(self):
        """Low-confidence predictions should suppress the advisory."""
        pred = VisionPrediction(
            plant="Tomato", disease="Early blight",
            canonical_id="tomato_early_blight", confidence=0.38,
            status=PredictionStatus.UNCERTAIN.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertIsNone(res.advisory)
        self.assertEqual(res.status, PredictionStatus.UNCERTAIN.value)

    def test_uncertain_prediction_has_warnings(self):
        """Uncertain predictions should have at least one warning."""
        pred = VisionPrediction(
            plant="Tomato", disease="Early blight",
            canonical_id="tomato_early_blight", confidence=0.35,
            status=PredictionStatus.UNCERTAIN.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertGreater(len(res.warnings), 0)

    def test_uncertain_prediction_message_mentions_confidence(self):
        """Uncertain message should explain the confidence issue."""
        pred = VisionPrediction(
            plant="Tomato", disease="Early blight",
            canonical_id="tomato_early_blight", confidence=0.35,
            status=PredictionStatus.UNCERTAIN.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertIn("confidence", res.user_message.lower())

    def test_uncertain_prediction_evidence_empty(self):
        """No evidence chunks should be generated for uncertain predictions."""
        pred = VisionPrediction(
            plant="Corn", disease="Common rust",
            canonical_id="corn_common_rust", confidence=0.22,
            status=PredictionStatus.UNCERTAIN.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertEqual(len(res.evidence), 0)

    def test_not_a_plant_advisory_suppressed(self):
        """Non-plant inputs should suppress the advisory."""
        pred = VisionPrediction(
            plant="Non-Plant / Corrupted",
            disease="Invalid Image",
            canonical_id="not_a_plant",
            confidence=0.0,
            status=PredictionStatus.NOT_A_PLANT.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertIsNone(res.advisory)
        self.assertEqual(res.status, PredictionStatus.NOT_A_PLANT.value)

    def test_not_a_plant_message_mentions_validation(self):
        """Non-plant message should mention the validation failure."""
        pred = VisionPrediction(
            plant="Non-Plant / Corrupted",
            disease="Invalid Image",
            canonical_id="not_a_plant",
            confidence=0.0,
            status=PredictionStatus.NOT_A_PLANT.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertIn("failed plant foliage validation", res.user_message.lower())

    def test_unknown_prediction_advisory_suppressed(self):
        """Out-of-distribution disease should suppress the advisory."""
        pred = VisionPrediction(
            plant="SomePlant", disease="Unknown Pathology",
            canonical_id="unknown_disease_xyz",
            confidence=0.65,
            status=PredictionStatus.UNKNOWN.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertIsNone(res.advisory)
        self.assertEqual(res.status, PredictionStatus.UNKNOWN.value)

    def test_unknown_prediction_has_warnings(self):
        """Unknown predictions should produce at least one warning."""
        pred = VisionPrediction(
            plant="SomePlant", disease="Unknown",
            canonical_id="unknown_xyz",
            confidence=0.65,
            status=PredictionStatus.UNKNOWN.value
        )
        res = self.generator.generate_advisory(pred)
        self.assertGreater(len(res.warnings), 0)

    # ── Batch generation ───────────────────────────────────────────────────

    def test_batch_generation(self):
        """Batch generation should return one response per input prediction."""
        preds = [
            VisionPrediction(plant="Apple", disease="Apple scab",
                             canonical_id="apple_apple_scab", confidence=0.90,
                             status=PredictionStatus.SUPPORTED.value),
            VisionPrediction(plant="Tomato", disease="Early blight",
                             canonical_id="tomato_early_blight", confidence=0.82,
                             status=PredictionStatus.SUPPORTED.value),
            VisionPrediction(plant="Non-Plant", disease="Invalid",
                             canonical_id="not_a_plant", confidence=0.0,
                             status=PredictionStatus.NOT_A_PLANT.value),
        ]
        responses = self.generator.generate_batch(preds)
        self.assertEqual(len(responses), 3)
        self.assertIsNotNone(responses[0].advisory)   # supported
        self.assertIsNotNone(responses[1].advisory)   # supported
        self.assertIsNone(responses[2].advisory)      # not_a_plant

    # ── Report formatting ──────────────────────────────────────────────────

    def test_format_advisory_report_structure(self):
        """Formatted report should contain key section headers."""
        pred = VisionPrediction(
            plant="Apple", disease="Apple scab",
            canonical_id="apple_apple_scab", confidence=0.91,
            status=PredictionStatus.SUPPORTED.value
        )
        res = self.generator.generate_advisory(pred)
        report = self.generator.format_advisory_report(res)
        self.assertIn("PLANT DISEASE ADVISORY REPORT", report)
        self.assertIn("ADVISORY:", report)

    def test_format_advisory_report_contains_plant_name(self):
        """Formatted report should include the plant name."""
        pred = VisionPrediction(
            plant="Apple", disease="Apple scab",
            canonical_id="apple_apple_scab", confidence=0.91,
            status=PredictionStatus.SUPPORTED.value
        )
        res = self.generator.generate_advisory(pred)
        report = self.generator.format_advisory_report(res)
        self.assertIn("Apple", report)


class TestAdvisoryGroundingIntegration(unittest.TestCase):
    """Integration-level grounding verification tests."""

    def setUp(self):
        self.generator = AdvisoryGenerator()

    def test_multiple_diseases_all_have_evidence(self):
        """All supported disease predictions should produce non-empty evidence."""
        test_cases = [
            ("Apple", "Apple scab", "apple_apple_scab"),
            ("Tomato", "Early blight", "tomato_early_blight"),
            ("Tomato", "Late blight", "tomato_late_blight"),
            ("Corn", "Common rust", "corn_common_rust"),
        ]
        for plant, disease, cid in test_cases:
            with self.subTest(canonical_id=cid):
                pred = VisionPrediction(
                    plant=plant, disease=disease,
                    canonical_id=cid, confidence=0.80,
                    status=PredictionStatus.SUPPORTED.value
                )
                res = self.generator.generate_advisory(pred)
                self.assertGreater(
                    len(res.evidence), 0,
                    f"No evidence for {cid}"
                )
                self.assertGreater(
                    len(res.sources), 0,
                    f"No sources for {cid}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
