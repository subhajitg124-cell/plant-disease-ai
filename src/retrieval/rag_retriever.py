"""
RAG Retriever Module for Agricultural Knowledge Base.

Retrieves verified agricultural care, prevention, and treatment guides by:
  - Exact canonical disease ID lookup (O(1))
  - Natural language / disease-condition vector similarity query
  - Fallback to graceful advisory scaffold when no match is found

Returns fully typed AdvisoryResult objects grounded in retrieved KB documents.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, Tuple, Union

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.contracts import RAGQueryInput, AdvisoryResult
from src.retrieval.vector_store import VectorStore


# Similarity threshold below which a match is deemed too low-confidence
_MIN_SIMILARITY_SCORE = 0.05


class RAGRetriever:
    """
    RAG Retrieval Engine for Plant Disease Advisory System.

    Provides grounded retrieval over the agricultural knowledge base via:
    1. Exact canonical ID lookup (primary path)
    2. Natural language disease/condition query (secondary path)
    3. Graceful fallback advisory scaffold (tertiary path)
    """

    def __init__(
        self,
        kb_path: str = "data/knowledge_base/agricultural_documents.json",
        store_dir: str = "models/vector_index"
    ):
        self.kb_path = kb_path
        self.vector_store = VectorStore(dimension=256, store_dir=store_dir)
        self.kb_documents: Dict[str, Dict[str, Any]] = {}
        self._index_loaded = False
        self._init_knowledge_base()

    def _init_knowledge_base(self):
        """Loads knowledge base documents and populates vector index."""
        # Try to load pre-built index first (faster startup)
        if self.vector_store.load_index():
            for doc in self.vector_store.documents:
                cid = doc.get("canonical_id")
                if cid:
                    self.kb_documents[cid] = doc
            self._index_loaded = True
            return

        # Fall back to building from JSON source
        if os.path.exists(self.kb_path):
            with open(self.kb_path, mode="r", encoding="utf-8") as f:
                docs = json.load(f)

            for doc in docs:
                cid = doc.get("canonical_id")
                if cid:
                    self.kb_documents[cid] = doc

            # Build and persist vector index
            self.vector_store.add_documents(docs)
            self.vector_store.save_index()
            self._index_loaded = True
        else:
            print(
                f"Warning: Knowledge base JSON not found at '{self.kb_path}'. "
                "RAG retrieval will use fallback advisories only."
            )

    # -----------------------------------------------------------------------
    # Primary retrieval: canonical ID
    # -----------------------------------------------------------------------

    def retrieve_by_canonical_id(self, canonical_id: str) -> AdvisoryResult:
        """
        Retrieves advisory by exact canonical disease ID.

        Search order:
          1. In-memory dict (O(1))
          2. Vector store exact lookup
          3. Fallback advisory scaffold
        """
        doc = self.kb_documents.get(canonical_id)
        if not doc:
            doc = self.vector_store.search_by_canonical_id(canonical_id)

        if doc:
            return self._doc_to_advisory(doc)

        # Fallback scaffold
        return self._fallback_advisory(canonical_id)

    # -----------------------------------------------------------------------
    # Secondary retrieval: natural language query
    # -----------------------------------------------------------------------

    def search_similar(
        self, query_text: str, top_k: int = 5, min_score: float = _MIN_SIMILARITY_SCORE
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches knowledge base by natural language disease/condition query.

        Supports queries such as:
          - "tomato early blight concentric rings prevention"
          - "apple scab olive spots fungicide management"
          - "humid conditions fungal leaf spot spray"

        Returns: list of (document, similarity_score) tuples, sorted desc.
        """
        return self.vector_store.search_by_query(query_text, top_k=top_k, min_score=min_score)

    def retrieve_by_query(
        self, query_text: str, top_k: int = 5
    ) -> List[AdvisoryResult]:
        """
        Returns advisory results for the top-k documents matching a natural language query.
        """
        results = self.search_similar(query_text, top_k=top_k)
        return [self._doc_to_advisory(doc) for doc, _ in results]

    def retrieve_by_plant(self, plant_name: str) -> List[AdvisoryResult]:
        """Returns all advisory results for a given plant species."""
        docs = self.vector_store.search_by_plant(plant_name)
        return [self._doc_to_advisory(doc) for doc in docs]

    # -----------------------------------------------------------------------
    # Unified entry point
    # -----------------------------------------------------------------------

    def retrieve(
        self, query_input: Union[RAGQueryInput, str, Dict[str, Any]]
    ) -> AdvisoryResult:
        """
        Retrieves agricultural advisory given RAGQueryInput, canonical ID string, or dict.

        Pipeline:
          1. Resolve canonical_id from input
          2. Try exact canonical ID lookup
          3. If not found, try similarity search on combined plant+disease text
          4. If still not found, return fallback advisory scaffold
        """
        if isinstance(query_input, RAGQueryInput):
            canonical_id = query_input.canonical_id
            plant = query_input.plant
            disease = query_input.disease
        elif isinstance(query_input, dict):
            canonical_id = query_input.get("canonical_id", "")
            plant = query_input.get("plant", "")
            disease = query_input.get("disease", "")
        else:
            canonical_id = str(query_input)
            plant = ""
            disease = ""

        # 1. Exact canonical ID lookup
        doc = self.kb_documents.get(canonical_id)
        if not doc:
            doc = self.vector_store.search_by_canonical_id(canonical_id)

        if doc:
            return self._doc_to_advisory(doc)

        # 2. Similarity fallback using plant+disease text
        if plant or disease:
            fallback_query = f"{plant} {disease} {canonical_id}".strip()
            results = self.search_similar(fallback_query, top_k=1)
            if results:
                best_doc, score = results[0]
                if score >= _MIN_SIMILARITY_SCORE:
                    advisory = self._doc_to_advisory(best_doc)
                    # Annotate that this was retrieved via similarity, not exact match
                    advisory.sources.append(
                        f"[Retrieved via similarity search — query: '{fallback_query}', score: {score:.3f}]"
                    )
                    return advisory

        # 3. Fallback advisory scaffold
        return self._fallback_advisory(canonical_id, plant=plant, disease=disease)

    # -----------------------------------------------------------------------
    # Grounding check
    # -----------------------------------------------------------------------

    def is_grounded(self, advisory: AdvisoryResult) -> bool:
        """
        Returns True if the advisory has substantive, non-fallback evidence.

        Checks:
          - At least one source that is NOT a fallback/generic citation
          - At least one management action that isn't just 'consult a specialist'
          - Symptoms are present
        """
        # A fallback advisory always has at least one source containing 'Fallback'
        all_sources_generic = bool(advisory.sources) and all(
            any(marker in s for marker in (
                "Fallback", "General Plant Pathology", "generic", "General Agricultural"
            ))
            for s in advisory.sources
        )
        if all_sources_generic:
            return False

        has_symptoms = bool(advisory.symptoms)
        has_sources = bool(advisory.sources)
        has_management = bool(advisory.management) and not all(
            m.strip().lower().startswith("consult")
            for m in advisory.management
        )
        return has_symptoms and has_sources and has_management

    def get_evidence_chunks(
        self, advisory: AdvisoryResult, max_chunks: int = 8
    ) -> List[str]:
        """
        Returns a flat list of grounding evidence text chunks from an AdvisoryResult.

        Used by AdvisoryGenerator to populate the `evidence` field.
        """
        chunks: List[str] = []
        chunks.extend(advisory.symptoms)
        chunks.extend(advisory.causes)
        chunks.extend(advisory.risk_factors)
        chunks.extend(advisory.prevention)
        chunks.extend(advisory.management)
        return chunks[:max_chunks]

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _doc_to_advisory(self, doc: Dict[str, Any]) -> AdvisoryResult:
        """Maps a knowledge base document dict to an AdvisoryResult contract."""
        return AdvisoryResult(
            canonical_id=str(doc.get("canonical_id", "unknown")),
            symptoms=list(doc.get("symptoms", [])),
            causes=list(doc.get("causes", [])),
            risk_factors=list(doc.get("risk_factors", [])),
            prevention=list(doc.get("prevention", [])),
            management=list(doc.get("management", [])),
            sources=list(doc.get("sources", []))
        )

    def _fallback_advisory(
        self,
        canonical_id: str,
        plant: str = "",
        disease: str = ""
    ) -> AdvisoryResult:
        """
        Returns a graceful fallback advisory scaffold when no KB document is found.
        """
        label = disease or canonical_id.replace("_", " ").title()
        plant_label = plant or "the affected plant"
        return AdvisoryResult(
            canonical_id=canonical_id,
            symptoms=[
                f"Visible foliage changes or lesions consistent with {label}.",
                "Generalised growth suppression or discoloration."
            ],
            causes=[
                f"Likely pathogen, pest, or environmental stressor associated with {label}."
            ],
            risk_factors=[
                "Unmanaged foliage wetness, warm temperatures, and poor air circulation."
            ],
            prevention=[
                f"Maintain balanced nutrition and good sanitation practices for {plant_label}.",
                "Avoid overhead irrigation; provide adequate plant spacing."
            ],
            management=[
                "Consult a local agricultural extension specialist for a confirmed diagnosis.",
                "Apply broad-spectrum protective fungicide or bactericide as a precaution."
            ],
            sources=[
                "Fallback: General Agricultural Extension Advisory",
                "Plant Disease Pathology Index (generic)"
            ]
        )

    def get_kb_stats(self) -> Dict[str, Any]:
        """Returns knowledge base statistics."""
        stats = self.vector_store.get_stats()
        stats["kb_path"] = self.kb_path
        stats["index_loaded"] = self._index_loaded
        return stats


if __name__ == "__main__":
    retriever = RAGRetriever()
    stats = retriever.get_kb_stats()
    print(f"KB Stats: {stats}")

    # Test exact retrieval
    result = retriever.retrieve_by_canonical_id("apple_apple_scab")
    print("\n[Exact Retrieval] apple_apple_scab")
    print("  Symptoms:", result.symptoms)
    print("  Management:", result.management)
    print("  Sources:", result.sources)
    print("  Grounded:", retriever.is_grounded(result))

    # Test natural language query retrieval
    print("\n[Query Search] 'tomato concentric rings early blight management'")
    similar = retriever.search_similar("tomato concentric rings early blight management", top_k=3)
    for doc, score in similar:
        print(f"  [{score:.4f}] {doc['canonical_id']} — {doc['plant']} / {doc['disease']}")

    # Test via RAGQueryInput
    query = RAGQueryInput(plant="Tomato", disease="Early blight", canonical_id="tomato_early_blight")
    adv = retriever.retrieve(query)
    print("\n[RAGQueryInput] tomato_early_blight")
    print("  Prevention:", adv.prevention)
