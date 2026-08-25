import unittest
from src.contracts import (
    PredictionStatus,
    VisionPrediction,
    RAGQueryInput,
    AdvisoryResult,
    IntegratedResponse
)

class TestModuleInterfaces(unittest.TestCase):
    """
    Test suite for Vision ↔ RAG ↔ Integration module interface contracts.
    """

    def test_valid_supported_prediction(self):
        pred = VisionPrediction(
            plant="Tomato",
            disease="Early Blight",
            canonical_id="tomato_early_blight",
            confidence=0.94,
            status=PredictionStatus.SUPPORTED.value,
            raw_label="Tomato___Early_blight",
            model_version="vision_v1"
        )
        self.assertEqual(pred.plant, "Tomato")
        self.assertEqual(pred.disease, "Early Blight")
        self.assertEqual(pred.canonical_id, "tomato_early_blight")
        self.assertEqual(pred.confidence, 0.94)
        self.assertEqual(pred.status, "supported")
        self.assertEqual(pred.model_version, "vision_v1")
        self.assertIsNone(pred.embedding)  # Optional embedding

        # Test dictionary conversion round-trip
        pred_dict = pred.to_dict()
        pred_reconstructed = VisionPrediction.from_dict(pred_dict)
        self.assertEqual(pred, pred_reconstructed)

    def test_model_version_optional(self):
        pred = VisionPrediction(
            plant="Tomato",
            disease="Early Blight",
            canonical_id="tomato_early_blight",
            confidence=0.94,
            status=PredictionStatus.SUPPORTED.value
        )
        self.assertIsNone(pred.model_version)
        self.assertIsNone(pred.embedding)

    def test_uncertain_prediction(self):
        pred = VisionPrediction(
            plant="Tomato",
            disease="Early Blight",
            canonical_id="tomato_early_blight",
            confidence=0.42,
            status=PredictionStatus.UNCERTAIN.value
        )
        self.assertEqual(pred.status, "uncertain")
        self.assertEqual(pred.confidence, 0.42)

    def test_unknown_prediction(self):
        pred = VisionPrediction(
            plant="Corn (maize)",
            disease="Unknown Pathology",
            canonical_id="corn_unknown_pathology",
            confidence=0.88,
            status=PredictionStatus.UNKNOWN.value
        )
        self.assertEqual(pred.status, "unknown")

    def test_not_a_plant_prediction(self):
        pred = VisionPrediction(
            plant="Non-Plant Object",
            disease="None",
            canonical_id="not_a_plant",
            confidence=0.12,
            status=PredictionStatus.NOT_A_PLANT.value
        )
        self.assertEqual(pred.status, "not_a_plant")

    def test_valid_rag_response(self):
        rag_input = RAGQueryInput(
            plant="Tomato",
            disease="Early Blight",
            canonical_id="tomato_early_blight"
        )
        self.assertEqual(rag_input.canonical_id, "tomato_early_blight")

        advisory = AdvisoryResult(
            canonical_id="tomato_early_blight",
            symptoms=["[EXAMPLE] Dark brown spots with concentric rings"],
            causes=["[EXAMPLE] Fungal pathogen Alternaria solani"],
            risk_factors=["[EXAMPLE] High humidity (24-29 C)"],
            prevention=["[EXAMPLE] Rotate crops every 2-3 years"],
            management=["[EXAMPLE] Apply copper fungicide sprays"],
            sources=["[EXAMPLE] USDA Agricultural Extension Bulletin No. 402"]
        )
        self.assertEqual(len(advisory.symptoms), 1)
        self.assertEqual(len(advisory.sources), 1)

        # Test dictionary round-trip
        adv_dict = advisory.to_dict()
        adv_reconstructed = AdvisoryResult.from_dict(adv_dict)
        self.assertEqual(advisory, adv_reconstructed)

    def test_integration_of_vision_and_rag_outputs(self):
        pred = VisionPrediction(
            plant="Tomato",
            disease="Early Blight",
            canonical_id="tomato_early_blight",
            confidence=0.94,
            status=PredictionStatus.SUPPORTED.value,
            model_version="vision_v1"
        )
        advisory = AdvisoryResult(
            canonical_id="tomato_early_blight",
            symptoms=["[EXAMPLE] Dark brown spots with concentric rings"],
            causes=["[EXAMPLE] Fungal pathogen Alternaria solani"],
            prevention=["[EXAMPLE] Rotate crops every 2-3 years"],
            management=["[EXAMPLE] Apply copper fungicide sprays"],
            sources=["[EXAMPLE] USDA Extension Bulletin 402"]
        )

        integrated = IntegratedResponse.compose(prediction=pred, advisory=advisory)
        self.assertEqual(integrated.status, "supported")
        self.assertIn("Tomato - Early Blight", integrated.user_message)
        self.assertIsNotNone(integrated.advisory)
        self.assertEqual(integrated.evidence, advisory.symptoms)
        self.assertEqual(integrated.sources, advisory.sources)
        self.assertEqual(len(integrated.warnings), 0)

    def test_integration_non_supported_status_suppresses_advisory(self):
        # Test uncertain prediction integration
        uncertain_pred = VisionPrediction(
            plant="Tomato",
            disease="Early Blight",
            canonical_id="tomato_early_blight",
            confidence=0.45,
            status=PredictionStatus.UNCERTAIN.value
        )
        integrated = IntegratedResponse.compose(prediction=uncertain_pred)
        self.assertIsNone(integrated.advisory)
        self.assertEqual(integrated.status, "uncertain")
        self.assertTrue(any("operational threshold" in w for w in integrated.warnings))

        # Test not_a_plant integration
        not_plant_pred = VisionPrediction(
            plant="Object",
            disease="None",
            canonical_id="not_a_plant",
            confidence=0.05,
            status=PredictionStatus.NOT_A_PLANT.value
        )
        integrated_not_plant = IntegratedResponse.compose(prediction=not_plant_pred)
        self.assertIsNone(integrated_not_plant.advisory)
        self.assertEqual(integrated_not_plant.status, "not_a_plant")
        self.assertTrue(any("failed plant validation" in w for w in integrated_not_plant.warnings))

    def test_missing_required_fields_rejected(self):
        # Invalid status
        with self.assertRaises(ValueError):
            VisionPrediction(
                plant="Tomato",
                disease="Early Blight",
                canonical_id="tomato_early_blight",
                confidence=0.9,
                status="invalid_status_value"
            )

        # Out-of-bounds confidence score
        with self.assertRaises(ValueError):
            VisionPrediction(
                plant="Tomato",
                disease="Early Blight",
                canonical_id="tomato_early_blight",
                confidence=1.5,  # Invalid confidence > 1.0
                status=PredictionStatus.SUPPORTED.value
            )

        # Empty required plant field
        with self.assertRaises(ValueError):
            VisionPrediction(
                plant="",
                disease="Early Blight",
                canonical_id="tomato_early_blight",
                confidence=0.9,
                status=PredictionStatus.SUPPORTED.value
            )

        # Missing field in from_dict
        with self.assertRaises(ValueError):
            VisionPrediction.from_dict({
                "plant": "Tomato",
                "disease": "Early Blight",
                # "canonical_id" missing
                "confidence": 0.9,
                "status": "supported"
            })


if __name__ == "__main__":
    unittest.main()
