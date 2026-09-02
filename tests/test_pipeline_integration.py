"""
End-to-End Pipeline Integration Tests.

Tests the complete Image → Vision → RAG → Advisory pipeline via PlantDiseasePipeline
and verifies each integration touchpoint:

  1. Valid plant image → VisionPrediction → IntegratedResponse
  2. Non-plant image → NOT_A_PLANT rejection → no advisory
  3. Direct KB advisory by canonical ID
  4. Natural language KB query
  5. Vision-only classification (no RAG)
  6. Batch pipeline execution
  7. VisionPrediction → AdvisoryResult grounding check
  8. Pipeline diagnostics / component info
  9. IntegratedResponse contract validation
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import PlantDiseasePipeline
from src.contracts import (
    VisionPrediction, AdvisoryResult, IntegratedResponse, PredictionStatus
)
from src.advisory.advisory_generator import AdvisoryGenerator
from src.retrieval.rag_retriever import RAGRetriever


# ---------------------------------------------------------------------------
# Synthetic image fixtures
# ---------------------------------------------------------------------------

def make_green_leaf(h: int = 224, w: int = 224) -> np.ndarray:
    """Simulates a healthy-looking green leaf input."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = 30   # R
    arr[:, :, 1] = 170  # G
    arr[:, :, 2] = 25   # B
    return arr


def make_brown_spotted_leaf(h: int = 224, w: int = 224) -> np.ndarray:
    """Simulates a brown-spotted diseased leaf."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = 100  # R (brownish)
    arr[:, :, 1] = 80   # G
    arr[:, :, 2] = 20   # B
    # Add dark spots in the centre
    arr[80:140, 80:140, :] = [40, 25, 10]
    return arr


def make_non_plant(h: int = 224, w: int = 224) -> np.ndarray:
    """Simulates a uniform grey non-plant image."""
    return np.full((h, w, 3), 100, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestPipelineInitialisation(unittest.TestCase):
    """Tests that the pipeline initialises correctly with all components loaded."""

    def setUp(self):
        self.pipeline = PlantDiseasePipeline()

    def test_pipeline_info_structure(self):
        """Pipeline info should expose classifier and KB statistics."""
        info = self.pipeline.get_pipeline_info()
        self.assertIn("classifier", info)
        self.assertIn("knowledge_base", info)
        self.assertIn("pipeline_version", info)

    def test_classifier_has_classes(self):
        """Classifier should have at least 38 known classes."""
        info = self.pipeline.get_pipeline_info()
        self.assertGreaterEqual(info["classifier"]["num_classes"], 38)

    def test_knowledge_base_has_documents(self):
        """Knowledge base should have at least 38 indexed documents."""
        info = self.pipeline.get_pipeline_info()
        self.assertGreaterEqual(info["knowledge_base"]["total_documents"], 38)


class TestVisionOnlyPipeline(unittest.TestCase):
    """Tests for vision-only classification path (no RAG)."""

    def setUp(self):
        self.pipeline = PlantDiseasePipeline()

    def test_classify_only_green_leaf(self):
        """classify_only should return a VisionPrediction for a valid leaf."""
        pred = self.pipeline.classify_only(make_green_leaf())
        self.assertIsInstance(pred, VisionPrediction)
        self.assertIn(pred.status, [
            PredictionStatus.SUPPORTED.value,
            PredictionStatus.UNCERTAIN.value
        ])

    def test_classify_only_non_plant(self):
        """classify_only should return NOT_A_PLANT for a non-plant image."""
        pred = self.pipeline.classify_only(make_non_plant())
        self.assertEqual(pred.status, PredictionStatus.NOT_A_PLANT.value)

    def test_classify_only_returns_canonical_id(self):
        """VisionPrediction should have a non-empty canonical_id."""
        pred = self.pipeline.classify_only(make_green_leaf())
        self.assertIsInstance(pred.canonical_id, str)
        self.assertGreater(len(pred.canonical_id), 0)

    def test_classify_only_confidence_in_range(self):
        """Confidence should be in [0.0, 1.0]."""
        pred = self.pipeline.classify_only(make_green_leaf())
        self.assertGreaterEqual(pred.confidence, 0.0)
        self.assertLessEqual(pred.confidence, 1.0)


class TestFullPipelineIntegration(unittest.TestCase):
    """End-to-end pipeline integration tests."""

    def setUp(self):
        self.pipeline = PlantDiseasePipeline()

    def test_valid_plant_returns_integrated_response(self):
        """Valid plant image should produce an IntegratedResponse."""
        result = self.pipeline.predict_and_advise(make_green_leaf())
        self.assertIsInstance(result, IntegratedResponse)

    def test_valid_plant_has_prediction(self):
        """Valid plant result should have a VisionPrediction."""
        result = self.pipeline.predict_and_advise(make_green_leaf())
        self.assertIsInstance(result.prediction, VisionPrediction)

    def test_valid_plant_status_is_supported_or_uncertain(self):
        """Valid plant result should be 'supported' or 'uncertain' status."""
        result = self.pipeline.predict_and_advise(make_green_leaf())
        self.assertIn(result.status, [
            PredictionStatus.SUPPORTED.value,
            PredictionStatus.UNCERTAIN.value
        ])

    def test_non_plant_image_rejected(self):
        """Non-plant image should return NOT_A_PLANT status."""
        result = self.pipeline.predict_and_advise(make_non_plant())
        self.assertEqual(result.status, PredictionStatus.NOT_A_PLANT.value)

    def test_non_plant_advisory_is_none(self):
        """Non-plant result should not produce an advisory."""
        result = self.pipeline.predict_and_advise(make_non_plant())
        self.assertIsNone(result.advisory)

    def test_non_plant_user_message_mentions_validation(self):
        """Non-plant result message should mention validation failure."""
        result = self.pipeline.predict_and_advise(make_non_plant())
        self.assertIn("validation", result.user_message.lower())

    def test_supported_prediction_has_advisory(self):
        """If status=supported, advisory should be populated."""
        result = self.pipeline.predict_and_advise(make_green_leaf())
        if result.status == PredictionStatus.SUPPORTED.value:
            self.assertIsNotNone(result.advisory)
            self.assertIsInstance(result.advisory, AdvisoryResult)

    def test_supported_prediction_evidence_populated(self):
        """Supported prediction should have non-empty evidence list."""
        result = self.pipeline.predict_and_advise(make_green_leaf())
        if result.status == PredictionStatus.SUPPORTED.value:
            self.assertGreater(len(result.evidence), 0)

    def test_supported_prediction_sources_populated(self):
        """Supported prediction should have source citations."""
        result = self.pipeline.predict_and_advise(make_green_leaf())
        if result.status == PredictionStatus.SUPPORTED.value:
            self.assertGreater(len(result.sources), 0)

    def test_confidence_passthrough(self):
        """IntegratedResponse.confidence should match VisionPrediction.confidence."""
        result = self.pipeline.predict_and_advise(make_green_leaf())
        self.assertAlmostEqual(result.confidence, result.prediction.confidence, places=5)

    def test_status_passthrough(self):
        """IntegratedResponse.status should match VisionPrediction.status."""
        result = self.pipeline.predict_and_advise(make_green_leaf())
        self.assertEqual(result.status, result.prediction.status)

    def test_user_message_non_empty(self):
        """User message should always be a non-empty string."""
        for img in [make_green_leaf(), make_non_plant(), make_brown_spotted_leaf()]:
            result = self.pipeline.predict_and_advise(img)
            self.assertIsInstance(result.user_message, str)
            self.assertGreater(len(result.user_message), 0)

    def test_pil_image_input(self):
        """Pipeline should accept PIL Image objects."""
        pil_img = Image.fromarray(make_green_leaf())
        result = self.pipeline.predict_and_advise(pil_img)
        self.assertIsInstance(result, IntegratedResponse)

    def test_file_path_input(self):
        """Pipeline should accept image file paths."""
        import tempfile
        arr = make_green_leaf()
        # Use delete=False and explicit close to avoid Windows file-lock
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_name = tmp.name
            Image.fromarray(arr).save(tmp_name)
        try:
            result = self.pipeline.predict_and_advise(tmp_name)
            self.assertIsInstance(result, IntegratedResponse)
        finally:
            os.unlink(tmp_name)


class TestKnowledgeBaseQueryPipeline(unittest.TestCase):
    """Tests for direct KB query and advisory shortcut paths."""

    def setUp(self):
        self.pipeline = PlantDiseasePipeline()

    def test_get_advisory_by_id_apple_scab(self):
        """Direct KB advisory should return structured AdvisoryResult."""
        adv = self.pipeline.get_advisory_by_id("apple_apple_scab")
        self.assertIsInstance(adv, AdvisoryResult)
        self.assertEqual(adv.canonical_id, "apple_apple_scab")
        self.assertGreater(len(adv.symptoms), 0)
        self.assertGreater(len(adv.management), 0)
        self.assertGreater(len(adv.sources), 0)

    def test_get_advisory_by_id_tomato_early_blight(self):
        """Direct KB advisory for tomato early blight."""
        adv = self.pipeline.get_advisory_by_id("tomato_early_blight")
        self.assertIsInstance(adv, AdvisoryResult)
        self.assertGreater(len(adv.prevention), 0)

    def test_kb_query_returns_results(self):
        """NL query should return relevant KB documents."""
        results = self.pipeline.query_knowledge_base(
            "apple scab olive green spots fungal venturia treatment", top_k=5
        )
        self.assertGreater(len(results), 0)

    def test_tomato_early_blight_query(self):
        """Tomato concentric rings query should match early blight in top results."""
        results = self.pipeline.query_knowledge_base(
            "tomato alternaria concentric brown rings early blight management chlorothalonil",
            top_k=10
        )
        ids = [doc["canonical_id"] for doc in results]
        self.assertIn("tomato_early_blight", ids,
                      f"Expected tomato_early_blight in top-10 results, got: {ids}")

    def test_kb_query_result_structure(self):
        """Each KB query result should have required fields."""
        results = self.pipeline.query_knowledge_base("tomato blight fungal", top_k=3)
        for hit in results:
            self.assertIn("canonical_id", hit)
            self.assertIn("plant", hit)
            self.assertIn("disease", hit)
            self.assertIn("similarity_score", hit)
            self.assertIn("symptoms", hit)
            self.assertIn("management", hit)
            self.assertIn("sources", hit)
            self.assertIsInstance(hit["similarity_score"], float)

    def test_kb_query_apple_scab_ranks_first(self):
        """Apple scab query should return apple_apple_scab as top hit."""
        results = self.pipeline.query_knowledge_base(
            "apple scab olive velvety spots copper sulfur prevention", top_k=5
        )
        self.assertGreater(len(results), 0)
        top_id = results[0]["canonical_id"]
        self.assertEqual(top_id, "apple_apple_scab")

    def test_kb_query_min_score_positive(self):
        """Query similarity scores should be positive."""
        results = self.pipeline.query_knowledge_base("tomato early blight", top_k=3)
        for hit in results:
            self.assertGreater(hit["similarity_score"], 0.0)


class TestBatchPipeline(unittest.TestCase):
    """Tests for batch prediction pipeline."""

    def setUp(self):
        self.pipeline = PlantDiseasePipeline()

    def test_batch_returns_correct_count(self):
        """Batch prediction should return one response per input."""
        images = [make_green_leaf(), make_non_plant(), make_brown_spotted_leaf()]
        results = self.pipeline.predict_batch(images)
        self.assertEqual(len(results), 3)

    def test_batch_non_plant_rejected(self):
        """Non-plant in batch should be rejected without crashing."""
        images = [make_non_plant(), make_green_leaf()]
        results = self.pipeline.predict_batch(images)
        self.assertEqual(results[0].status, PredictionStatus.NOT_A_PLANT.value)
        self.assertIn(results[1].status, [
            PredictionStatus.SUPPORTED.value,
            PredictionStatus.UNCERTAIN.value
        ])


class TestContractValidation(unittest.TestCase):
    """Validates that pipeline outputs conform to data contracts."""

    def setUp(self):
        self.pipeline = PlantDiseasePipeline()

    def test_integrated_response_to_dict_round_trip(self):
        """IntegratedResponse.to_dict() should serialise cleanly."""
        result = self.pipeline.predict_and_advise(make_green_leaf())
        d = result.to_dict()
        self.assertIn("prediction", d)
        self.assertIn("user_message", d)
        self.assertIn("confidence", d)
        self.assertIn("status", d)
        self.assertIn("evidence", d)
        self.assertIn("sources", d)
        self.assertIn("warnings", d)
        self.assertIsInstance(d["evidence"], list)
        self.assertIsInstance(d["sources"], list)

    def test_vision_prediction_contract_fields(self):
        """VisionPrediction in response should have all required contract fields."""
        result = self.pipeline.predict_and_advise(make_green_leaf())
        pred = result.prediction
        self.assertIsInstance(pred.plant, str)
        self.assertIsInstance(pred.disease, str)
        self.assertIsInstance(pred.canonical_id, str)
        self.assertIsInstance(pred.confidence, float)
        self.assertIsInstance(pred.status, str)
        self.assertIn(pred.status, [s.value for s in PredictionStatus])

    def test_advisory_result_contract_fields(self):
        """If advisory is present, it should conform to AdvisoryResult contract."""
        result = self.pipeline.predict_and_advise(make_green_leaf())
        if result.advisory is not None:
            adv = result.advisory
            self.assertIsInstance(adv.canonical_id, str)
            self.assertIsInstance(adv.symptoms, list)
            self.assertIsInstance(adv.causes, list)
            self.assertIsInstance(adv.risk_factors, list)
            self.assertIsInstance(adv.prevention, list)
            self.assertIsInstance(adv.management, list)
            self.assertIsInstance(adv.sources, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
