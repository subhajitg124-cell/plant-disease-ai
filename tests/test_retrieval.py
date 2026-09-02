"""
Unit Tests for VectorStore and RAGRetriever Modules.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.retrieval.vector_store import VectorStore
except ImportError:
    try:
        from retrieval.vector_store import VectorStore
    except ImportError:
        from vector_store import VectorStore

try:
    from src.retrieval.rag_retriever import RAGRetriever
except ImportError:
    try:
        from retrieval.rag_retriever import RAGRetriever
    except ImportError:
        from rag_retriever import RAGRetriever

try:
    from src.contracts import RAGQueryInput, AdvisoryResult
except ImportError:
    from contracts import RAGQueryInput, AdvisoryResult


class TestVectorStoreAndRetriever(unittest.TestCase):
    """Test suite verifying vector indexing and RAG retrieval functionality."""
    
    def setUp(self):
        self.test_docs = [
            {
                "canonical_id": "apple_apple_scab",
                "plant": "Apple",
                "disease": "Apple scab",
                "symptoms": ["Olive green leaf spots"],
                "causes": ["Venturia inaequalis"],
                "risk_factors": ["High humidity"],
                "prevention": ["Pruning canopy"],
                "management": ["Copper fungicide"],
                "sources": ["USDA Apple Guide"],
                "search_text": "Plant: Apple. Disease: Apple scab. Symptoms: Olive green leaf spots."
            },
            {
                "canonical_id": "tomato_early_blight",
                "plant": "Tomato",
                "disease": "Early blight",
                "symptoms": ["Concentric brown rings"],
                "causes": ["Alternaria solani"],
                "risk_factors": ["Plant stress"],
                "prevention": ["Crop rotation"],
                "management": ["Chlorothalonil"],
                "sources": ["Cornell Extension"],
                "search_text": "Plant: Tomato. Disease: Early blight. Symptoms: Concentric brown rings."
            }
        ]

    def test_vector_store_indexing_and_search(self):
        store = VectorStore(dimension=128)
        store.add_documents(self.test_docs)
        
        self.assertEqual(len(store.documents), 2)
        
        # Exact lookup test
        doc = store.search_by_canonical_id("apple_apple_scab")
        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc["plant"], "Apple")
        
        # Vector similarity search test
        q_vec = store._text_to_vector("Apple scab leaf spots")
        results = store.search_by_vector(q_vec, top_k=1)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0]["canonical_id"], "apple_apple_scab")

    def test_rag_retriever(self):
        retriever = RAGRetriever()
        
        # Retrieve by RAGQueryInput contract
        query = RAGQueryInput(plant="Apple", disease="Apple scab", canonical_id="apple_apple_scab")
        res = retriever.retrieve(query)
        
        self.assertIsInstance(res, AdvisoryResult)
        self.assertGreater(len(res.symptoms), 0)
        self.assertGreater(len(res.prevention), 0)
        self.assertGreater(len(res.sources), 0)


if __name__ == "__main__":
    unittest.main()
