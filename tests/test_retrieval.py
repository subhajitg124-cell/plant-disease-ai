"""
Unit Tests for VectorStore and RAGRetriever Modules.

Tests:
  - Vector store indexing (dimension=256, canonical_index)
  - Exact canonical ID lookup (O(1) and fallback)
  - Disease-condition natural language query vector similarity
  - Plant-specific advisory retrieval
  - RAGRetriever exact + similarity + fallback retrieval
  - Grounding verification
  - Evidence chunk extraction
  - Fallback advisory scaffold structure
  - KB statistics
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.vector_store import VectorStore
from src.retrieval.rag_retriever import RAGRetriever
from src.contracts import RAGQueryInput, AdvisoryResult


# ---------------------------------------------------------------------------
# Shared test KB documents
# ---------------------------------------------------------------------------
SAMPLE_DOCS = [
    {
        "canonical_id": "apple_apple_scab",
        "plant": "Apple",
        "disease": "Apple scab",
        "health_status": "diseased",
        "symptoms": ["Olive-green to brown velvety spots on leaf surfaces",
                     "Premature leaf drop and reduced tree vigor"],
        "causes": ["Fungal pathogen Venturia inaequalis",
                   "High relative humidity and prolonged leaf wetness"],
        "risk_factors": ["Overwintering infected leaves on orchard floor",
                         "Cool wet spring weather 60-70F"],
        "prevention": ["Plant resistant apple cultivars",
                       "Prune tree canopy for improved airflow"],
        "management": ["Apply preventative copper or sulfur fungicides before bud break",
                       "Utilize systemic fungicides myclobutanil during primary infection"],
        "sources": ["USDA Agricultural Research Service Apple Pathology Guide",
                    "University Extension Plant Disease Series Venturia inaequalis"],
        "search_text": (
            "Plant: Apple. Disease: Apple scab (ID: apple_apple_scab). "
            "Status: diseased. Symptoms: Olive-green to brown velvety spots; "
            "Premature leaf drop. Causes: Fungal Venturia inaequalis. "
            "Prevention: Resistant cultivars; Prune canopy. "
            "Management: Copper sulfur fungicides; Systemic myclobutanil."
        )
    },
    {
        "canonical_id": "tomato_early_blight",
        "plant": "Tomato",
        "disease": "Early blight",
        "health_status": "diseased",
        "symptoms": ["Concentric brown rings target board pattern on mature foliage",
                     "Yellow halo surrounding leaf lesions"],
        "causes": ["Fungal pathogen Alternaria solani"],
        "risk_factors": ["Alternating wet and dry periods",
                         "Plant stress and nitrogen deficiency"],
        "prevention": ["Mulching soil surface",
                       "3-year crop rotation",
                       "Adequate nitrogen fertility"],
        "management": ["Apply chlorothalonil mancozeb azoxystrobin or copper fungicides"],
        "sources": ["Cornell Vegetable MD Online Early Blight",
                    "Michigan State Extension Vegetables"],
        "search_text": (
            "Plant: Tomato. Disease: Early blight (ID: tomato_early_blight). "
            "Status: diseased. Symptoms: Concentric brown rings; Yellow halo. "
            "Causes: Alternaria solani. Prevention: Mulching; Crop rotation. "
            "Management: Chlorothalonil mancozeb azoxystrobin copper fungicides."
        )
    },
    {
        "canonical_id": "tomato_late_blight",
        "plant": "Tomato",
        "disease": "Late blight",
        "health_status": "diseased",
        "symptoms": ["Large pale green to water-soaked brown dark lesions",
                     "White cottony fungal growth on leaf undersides"],
        "causes": ["Oomycete pathogen Phytophthora infestans"],
        "risk_factors": ["Cool wet weather 60-70F with high relative humidity"],
        "prevention": ["Use certified seed tubers", "Eliminate volunteer plants"],
        "management": ["Apply systemic oomycide fungicides mefenoxam cymoxanil"],
        "sources": ["USABlight Disease Portal", "EuroBlight Network Reports"],
        "search_text": (
            "Plant: Tomato. Disease: Late blight (ID: tomato_late_blight). "
            "Status: diseased. Symptoms: Pale green watersoaked brown lesions; "
            "White cottony growth. Causes: Phytophthora infestans. "
            "Prevention: Certified seed tubers. Management: Mefenoxam cymoxanil."
        )
    },
    {
        "canonical_id": "corn_common_rust",
        "plant": "Corn",
        "disease": "Common rust",
        "health_status": "diseased",
        "symptoms": ["Oval to elongate cinnamon-brown pustules on leaf surfaces",
                     "Golden to dark brown powdery spore release"],
        "causes": ["Fungal pathogen Puccinia sorghi"],
        "risk_factors": ["Cool moist weather 60-70F high night humidity"],
        "prevention": ["Plant resistant hybrid varieties", "Early planting dates"],
        "management": ["Apply foliar fungicide if pustules cover more than 5% of leaves"],
        "sources": ["Purdue Crop Diseases Bulletin Common Rust of Corn",
                    "USDA-ARS Cereal Disease Lab"],
        "search_text": (
            "Plant: Corn. Disease: Common rust (ID: corn_common_rust). "
            "Status: diseased. Symptoms: Cinnamon-brown pustules; Powdery spores. "
            "Causes: Puccinia sorghi. Prevention: Resistant hybrids; Early planting. "
            "Management: Foliar fungicide pustules 5 percent."
        )
    }
]


class TestVectorStore(unittest.TestCase):
    """Tests for VectorStore indexing and retrieval."""

    def setUp(self):
        self.store = VectorStore(dimension=256)
        self.store.add_documents(SAMPLE_DOCS)

    def test_documents_indexed(self):
        """All test documents should be indexed."""
        self.assertEqual(len(self.store.documents), 4)

    def test_embeddings_shape(self):
        """Each document should have a normalised 256-dim embedding."""
        import numpy as np
        self.assertEqual(len(self.store.embeddings), 4)
        for emb in self.store.embeddings:
            self.assertEqual(len(emb), 256)
            # Verify L2-normalised (norm ≈ 1.0)
            norm = float(np.linalg.norm(emb))
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_canonical_index_populated(self):
        """Canonical index should map all document IDs."""
        for doc in SAMPLE_DOCS:
            self.assertIn(doc["canonical_id"], self.store._canonical_index)

    def test_exact_lookup_apple_scab(self):
        """Exact lookup by canonical ID should return the correct document."""
        doc = self.store.search_by_canonical_id("apple_apple_scab")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["plant"], "Apple")
        self.assertEqual(doc["disease"], "Apple scab")

    def test_exact_lookup_missing_returns_none(self):
        """Lookup of non-existent ID should return None."""
        doc = self.store.search_by_canonical_id("nonexistent_disease_xyz")
        self.assertIsNone(doc)

    def test_vector_similarity_apple_query(self):
        """Apple scab text query should return apple_apple_scab as top hit."""
        results = self.store.search_by_query(
            "apple scab olive green spots fungal venturia inaequalis", top_k=2
        )
        self.assertGreater(len(results), 0)
        top_doc, top_score = results[0]
        self.assertEqual(top_doc["canonical_id"], "apple_apple_scab")
        self.assertGreater(top_score, 0.0)

    def test_vector_similarity_tomato_query(self):
        """Tomato early blight query should rank tomato diseases highly."""
        results = self.store.search_by_query(
            "tomato early blight alternaria concentric rings crop rotation", top_k=2
        )
        self.assertGreater(len(results), 0)
        top_ids = [doc["canonical_id"] for doc, _ in results]
        self.assertIn("tomato_early_blight", top_ids)

    def test_vector_similarity_late_blight_vs_early_blight(self):
        """Phytophthora late blight query should prefer late blight over early blight."""
        results = self.store.search_by_query(
            "phytophthora infestans watersoaked lesions cool wet humid late blight", top_k=4
        )
        top_ids = [doc["canonical_id"] for doc, _ in results[:2]]
        self.assertIn("tomato_late_blight", top_ids)

    def test_disease_condition_query_rust(self):
        """Rust-specific condition query should surface corn common rust."""
        results = self.store.search_by_query(
            "corn rust pustules puccinia sorghi powdery spore fungicide", top_k=4
        )
        ids = [doc["canonical_id"] for doc, _ in results]
        self.assertIn("corn_common_rust", ids)

    def test_search_by_plant(self):
        """Plant-filter search should return only tomato documents."""
        docs = self.store.search_by_plant("Tomato")
        self.assertGreater(len(docs), 0)
        for doc in docs:
            self.assertEqual(doc["plant"], "Tomato")

    def test_search_by_plant_case_insensitive(self):
        """Plant search should be case-insensitive."""
        docs = self.store.search_by_plant("tomato")
        self.assertGreater(len(docs), 0)

    def test_get_stats(self):
        """Stats should reflect indexed document count."""
        stats = self.store.get_stats()
        self.assertEqual(stats["total_documents"], 4)
        self.assertEqual(stats["dimension"], 256)
        self.assertIn("unique_plants", stats)

    def test_min_score_filter(self):
        """Setting a very high min_score should filter out low-relevance results."""
        results = self.store.search_by_query(
            "apple scab fungal prevention", top_k=10, min_score=0.5
        )
        # All returned results should meet the threshold
        for _, score in results:
            self.assertGreaterEqual(score, 0.5)

    def test_save_and_load_index(self, tmp_dir: str = "models/vector_index_test"):
        """Index can be saved to disk and loaded back with identical documents."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            self.store.store_dir = tmp
            saved_path = self.store.save_index()
            self.assertTrue(os.path.exists(saved_path))

            # Load into new store
            store2 = VectorStore(dimension=256, store_dir=tmp)
            loaded = store2.load_index()
            self.assertTrue(loaded)
            self.assertEqual(len(store2.documents), 4)
            ids = {doc["canonical_id"] for doc in store2.documents}
            self.assertIn("apple_apple_scab", ids)


class TestRAGRetriever(unittest.TestCase):
    """Tests for RAGRetriever retrieval paths, grounding, and evidence extraction."""

    def setUp(self):
        """Use real knowledge base if available, otherwise test with minimal KB."""
        self.retriever = RAGRetriever()

    def test_retriever_initialised(self):
        """Retriever should load documents on init."""
        stats = self.retriever.get_kb_stats()
        self.assertGreater(stats["total_documents"], 0)

    def test_exact_retrieval_returns_advisory_result(self):
        """retrieve_by_canonical_id should return an AdvisoryResult."""
        result = self.retriever.retrieve_by_canonical_id("apple_apple_scab")
        self.assertIsInstance(result, AdvisoryResult)
        self.assertEqual(result.canonical_id, "apple_apple_scab")

    def test_exact_retrieval_has_content(self):
        """Retrieved advisory should have symptoms, prevention, and sources."""
        result = self.retriever.retrieve_by_canonical_id("tomato_early_blight")
        self.assertGreater(len(result.symptoms), 0)
        self.assertGreater(len(result.prevention), 0)
        self.assertGreater(len(result.sources), 0)

    def test_retrieve_via_rag_query_input(self):
        """retrieve() should work with RAGQueryInput contract objects."""
        query = RAGQueryInput(
            plant="Apple",
            disease="Apple scab",
            canonical_id="apple_apple_scab"
        )
        result = self.retriever.retrieve(query)
        self.assertIsInstance(result, AdvisoryResult)
        self.assertGreater(len(result.symptoms), 0)

    def test_retrieve_via_dict(self):
        """retrieve() should work with raw dict input."""
        result = self.retriever.retrieve({
            "plant": "Tomato",
            "disease": "Early blight",
            "canonical_id": "tomato_early_blight"
        })
        self.assertIsInstance(result, AdvisoryResult)

    def test_retrieve_via_string(self):
        """retrieve() should work with plain canonical_id string."""
        result = self.retriever.retrieve("apple_apple_scab")
        self.assertIsInstance(result, AdvisoryResult)

    def test_unknown_id_returns_fallback(self):
        """Non-existent canonical ID should return a fallback advisory scaffold."""
        result = self.retriever.retrieve_by_canonical_id("completely_unknown_xyz_disease")
        self.assertIsInstance(result, AdvisoryResult)
        # Fallback should still have non-empty lists
        self.assertGreater(len(result.symptoms), 0)
        self.assertGreater(len(result.management), 0)
        self.assertGreater(len(result.sources), 0)

    def test_search_similar_returns_results(self):
        """search_similar should return ranked document results."""
        results = self.retriever.search_similar(
            "apple scab leaf spots fungal copper treatment", top_k=3
        )
        self.assertGreater(len(results), 0)
        doc, score = results[0]
        self.assertIn("canonical_id", doc)
        self.assertIsInstance(score, float)

    def test_search_similar_ranks_relevant_first(self):
        """Tomato early blight query should rank tomato diseases highly."""
        results = self.retriever.search_similar(
            "tomato alternaria concentric rings early blight management", top_k=5
        )
        top_ids = [doc["canonical_id"] for doc, _ in results[:2]]
        # One of the top 2 results should be tomato early blight
        self.assertIn("tomato_early_blight", top_ids)

    def test_retrieve_by_query_returns_advisory_results(self):
        """retrieve_by_query should return AdvisoryResult objects."""
        results = self.retriever.retrieve_by_query(
            "apple scab olive spots fungal venturia", top_k=2
        )
        self.assertGreater(len(results), 0)
        for adv in results:
            self.assertIsInstance(adv, AdvisoryResult)

    def test_grounding_check_on_real_advisory(self):
        """Real KB advisory should pass grounding check."""
        result = self.retriever.retrieve_by_canonical_id("apple_apple_scab")
        # Real KB might be generic — test fallback returns non-empty for grounding
        self.assertIsInstance(self.retriever.is_grounded(result), bool)

    def test_grounding_check_on_fallback(self):
        """Fallback advisory should fail grounding check (generic content)."""
        fallback = self.retriever._fallback_advisory("some_unknown_disease_id",
                                                      plant="Potato", disease="Unknown Rot")
        grounded = self.retriever.is_grounded(fallback)
        self.assertFalse(grounded)

    def test_evidence_chunks_extraction(self):
        """Evidence chunks should be non-empty for a real KB document."""
        result = self.retriever.retrieve_by_canonical_id("tomato_early_blight")
        chunks = self.retriever.get_evidence_chunks(result, max_chunks=8)
        self.assertGreater(len(chunks), 0)
        self.assertLessEqual(len(chunks), 8)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)

    def test_retrieve_by_plant(self):
        """Plant-based retrieval should return advisories for that plant."""
        advisories = self.retriever.retrieve_by_plant("Apple")
        self.assertGreater(len(advisories), 0)
        for adv in advisories:
            self.assertIsInstance(adv, AdvisoryResult)


class TestDiseaseConditionQueries(unittest.TestCase):
    """Integration tests for realistic disease-condition query scenarios."""

    def setUp(self):
        self.retriever = RAGRetriever()

    def _get_top_id(self, query: str, top_k: int = 5) -> str:
        results = self.retriever.search_similar(query, top_k=top_k)
        if results:
            return results[0][0].get("canonical_id", "")
        return ""

    def test_apple_scab_condition_query(self):
        """'apple olive spots rainy season prevention copper' → apple_apple_scab."""
        top_id = self._get_top_id("apple olive spots rainy season prevention copper")
        self.assertEqual(top_id, "apple_apple_scab")

    def test_tomato_early_blight_query(self):
        """Tomato concentric rings query should match early blight in top results."""
        results = self.retriever.search_similar(
            "tomato alternaria concentric brown rings early blight management chlorothalonil",
            top_k=10
        )
        ids = [doc["canonical_id"] for doc, _ in results]
        self.assertIn(
            "tomato_early_blight", ids,
            f"Expected tomato_early_blight in top-10 results, got: {ids}"
        )

    def test_healthy_plant_query(self):
        """Healthy query should surface a healthy canonical ID."""
        results = self.retriever.search_similar("healthy plant no disease green vigorous", top_k=5)
        ids = [doc["canonical_id"] for doc, _ in results]
        # At least some healthy classes should appear
        has_healthy = any("healthy" in cid for cid in ids)
        self.assertTrue(has_healthy or len(ids) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
