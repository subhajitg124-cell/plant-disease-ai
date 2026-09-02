"""
RAG Retriever Module for Agricultural Knowledge Base.

Retrieves verified agricultural care, prevention, and treatment guides by canonical
disease ID or vector similarity search, returning typed AdvisoryResult objects.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, Union

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from src.contracts import RAGQueryInput, AdvisoryResult
except ImportError:
    from contracts import RAGQueryInput, AdvisoryResult

try:
    from src.retrieval.vector_store import VectorStore
except ImportError:
    try:
        from retrieval.vector_store import VectorStore
    except ImportError:
        from vector_store import VectorStore


class RAGRetriever:
    """
    RAG Retrieval Engine for Plant Disease Advisory System.
    """
    def __init__(
        self,
        kb_path: str = "data/knowledge_base/agricultural_documents.json",
        store_dir: str = "models/vector_index"
    ):
        self.kb_path = kb_path
        self.vector_store = VectorStore(store_dir=store_dir)
        self.kb_documents: Dict[str, Dict[str, Any]] = {}
        self._init_knowledge_base()

    def _init_knowledge_base(self):
        """Loads knowledge base documents and populates vector index."""
        if os.path.exists(self.kb_path):
            with open(self.kb_path, mode="r", encoding="utf-8") as f:
                docs = json.load(f)
                for doc in docs:
                    cid = doc.get("canonical_id")
                    if cid:
                        self.kb_documents[cid] = doc
                
                # Load into vector store
                self.vector_store.add_documents(docs)
                self.vector_store.save_index()
        else:
            print(f"Warning: Knowledge base JSON not found at '{self.kb_path}'.")

    def retrieve_by_canonical_id(self, canonical_id: str) -> AdvisoryResult:
        """
        Retrieves advisory result by exact canonical disease ID.
        """
        doc = self.kb_documents.get(canonical_id)
        if not doc:
            doc = self.vector_store.search_by_canonical_id(canonical_id)

        if doc:
            return AdvisoryResult(
                canonical_id=canonical_id,
                symptoms=doc.get("symptoms", []),
                causes=doc.get("causes", []),
                risk_factors=doc.get("risk_factors", []),
                prevention=doc.get("prevention", []),
                management=doc.get("management", []),
                sources=doc.get("sources", [])
            )

        # Fallback for unknown / generic canonical IDs
        return AdvisoryResult(
            canonical_id=canonical_id,
            symptoms=[f"No specific symptoms recorded for canonical ID '{canonical_id}'."],
            causes=["Unknown or unsupported disease agent."],
            risk_factors=["Unmanaged environmental foliage wetness."],
            prevention=["Maintain general plant nutrition and field sanitation."],
            management=["Consult a local agricultural extension specialist."],
            sources=["General Plant Pathology Knowledge System"]
        )

    def retrieve(self, query_input: Union[RAGQueryInput, str, Dict[str, Any]]) -> AdvisoryResult:
        """
        Retrieves advisory guidance given RAGQueryInput object, canonical ID string, or query dict.
        """
        if isinstance(query_input, RAGQueryInput):
            canonical_id = query_input.canonical_id
        elif isinstance(query_input, dict):
            canonical_id = query_input.get("canonical_id", "")
        else:
            canonical_id = str(query_input)

        return self.retrieve_by_canonical_id(canonical_id)

    def search_similar(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches knowledge base by natural language query vector similarity.
        """
        q_vec = self.vector_store._text_to_vector(query_text)
        results = self.vector_store.search_by_vector(q_vec, top_k=top_k)
        return [doc for doc, score in results]


if __name__ == "__main__":
    retriever = RAGRetriever()
    result = retriever.retrieve_by_canonical_id("apple_apple_scab")
    print("Retrieved Apple Scab Advisory Result:")
    print("Symptoms:", result.symptoms)
    print("Prevention:", result.prevention)
    print("Sources:", result.sources)
