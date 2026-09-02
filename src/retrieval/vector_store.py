"""
Vector Store and Similarity Index Module.

Supports FAISS, ChromaDB, and NumPy cosine similarity indexing
for fast agricultural document retrieval via canonical ID or
natural language disease-condition queries.
"""

import os
import sys
import json
import math
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

HAS_CHROMADB = False
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


# ---------------------------------------------------------------------------
# Domain vocabulary for enriched agricultural query embeddings
# ---------------------------------------------------------------------------
_PLANT_TERMS = {
    "apple", "tomato", "potato", "grape", "corn", "maize", "strawberry",
    "cherry", "peach", "pepper", "squash", "soybean", "citrus", "orange",
    "lemon", "blueberry", "raspberry", "wheat", "rice", "cotton", "coffee",
    "banana", "mango", "cucumber", "bean", "pea", "sunflower"
}

_DISEASE_TERMS = {
    "scab", "blight", "rot", "rust", "mildew", "mold", "spot", "mosaic",
    "virus", "wilt", "canker", "lesion", "chlorosis", "necrosis", "streak",
    "yellowing", "browning", "curl", "burn", "greening", "dieback",
    "anthracnose", "cercospora", "septoria", "alternaria", "powdery",
    "downy", "bacterial", "fungal", "viral", "nematode", "oomycete",
    "early_blight", "late_blight", "black_rot", "leaf_mold", "leaf_scorch",
    "target_spot", "haunglongbing"
}

_SYMPTOM_TERMS = {
    "spots", "rings", "pustules", "lesion", "discoloration", "defoliation",
    "yellowing", "wilting", "stunting", "streaking", "mottling", "curling",
    "scorching", "webbing", "stippling", "laceration", "cracks", "cankers",
    "mummies", "galls", "spores", "coating", "watersoaked", "brown", "black",
    "olive", "grey", "tan", "orange", "yellow"
}

_CONDITION_TERMS = {
    "humidity", "moisture", "wet", "dry", "rain", "temperature", "warm",
    "cool", "hot", "wind", "soil", "residue", "stress", "rotation",
    "overhead", "irrigation", "canopy", "spacing", "pruning", "sanitation"
}

_SEVERITY_TERMS = {
    "severe", "mild", "moderate", "heavy", "light", "acute", "chronic"
}


class VectorStore:
    """
    Unified Vector Store and Similarity Indexing Engine.

    Supports:
    - Exact canonical ID lookup via hash map
    - Enriched TF-IDF-style text-to-vector encoding with agricultural domain boosting
    - FAISS (if available) or NumPy cosine fallback for top-k similarity search
    - ChromaDB persistence (if available)
    - Disease-condition natural language query support
    """

    def __init__(self, dimension: int = 256, store_dir: str = "models/vector_index"):
        self.dimension = dimension
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)

        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []

        # Fast canonical_id → document index lookup
        self._canonical_index: Dict[str, int] = {}

        # FAISS index initialization
        self.faiss_index = None
        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatIP(dimension)

        # ChromaDB client initialization
        self.chroma_client = None
        self.chroma_collection = None
        if HAS_CHROMADB:
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=os.path.join(self.store_dir, "chroma")
                )
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name="agricultural_kb"
                )
            except Exception as e:
                print(f"Warning: ChromaDB initialization fallback: {e}")
                self.chroma_client = None
                self.chroma_collection = None

    # -----------------------------------------------------------------------
    # Text Vectorisation
    # -----------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Tokenises text into lowercase alpha-numeric tokens."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _domain_boost(self, token: str) -> float:
        """
        Returns a domain importance multiplier for agricultural tokens.
        Plant names and disease names get the highest boost; symptom and
        condition terms get moderate boosts.
        """
        if token in _PLANT_TERMS:
            return 3.5
        if token in _DISEASE_TERMS:
            return 3.0
        if token in _SYMPTOM_TERMS:
            return 2.0
        if token in _CONDITION_TERMS:
            return 1.5
        if token in _SEVERITY_TERMS:
            return 1.2
        return 1.0

    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        Generates a domain-boosted, L2-normalised pseudo-embedding.

        Strategy: deterministic hash-based TF-style accumulation with:
        - Multiple overlapping hash planes for better collision spread
        - Agricultural domain vocabulary boosting
        - Token position decay (earlier tokens weighted slightly higher)
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(text)
        token_freq: Dict[str, int] = {}
        for tok in tokens:
            token_freq[tok] = token_freq.get(tok, 0) + 1

        # Compute IDF-like discount for very frequent generic tokens
        total = max(len(tokens), 1)

        for idx, token in enumerate(tokens):
            tf = token_freq[token] / total
            boost = self._domain_boost(token)
            pos_weight = 1.0 / (1.0 + 0.05 * idx)  # slight position decay

            # Use three independent hash planes to spread the signal
            char_sum = sum(ord(c) for c in token)
            for plane_offset in (0, self.dimension // 3, 2 * self.dimension // 3):
                slot = (char_sum + plane_offset * 7) % self.dimension
                vec[slot] += tf * boost * pos_weight

            # Add bigram signal for consecutive tokens
            if idx > 0:
                prev = tokens[idx - 1]
                bigram_val = sum(ord(c) for c in (prev + "_" + token))
                bigram_slot = bigram_val % self.dimension
                vec[bigram_slot] += 0.3 * boost * pos_weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    # -----------------------------------------------------------------------
    # Indexing
    # -----------------------------------------------------------------------

    def add_documents(
        self,
        docs: List[Dict[str, Any]],
        vectors: Optional[List[List[float]]] = None
    ):
        """
        Indexes a list of agricultural knowledge base documents.

        For each document, builds a rich search_text from all available fields
        if not already present, then generates and stores its embedding.
        """
        for i, doc in enumerate(docs):
            if vectors is not None and i < len(vectors):
                vec = np.array(vectors[i], dtype=np.float32)
            else:
                # Build or use existing search_text
                search_text = doc.get("search_text") or self._build_search_text(doc)
                vec = self._text_to_vector(search_text)

            # L2 normalise
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            doc_idx = len(self.documents)
            canonical_id = str(doc.get("canonical_id", f"doc_{doc_idx}"))

            self.documents.append(doc)
            self.embeddings.append(vec)
            self._canonical_index[canonical_id] = doc_idx

            # Index into FAISS
            if HAS_FAISS and self.faiss_index is not None:
                self.faiss_index.add(np.array([vec], dtype=np.float32))

            # Index into ChromaDB
            if self.chroma_collection is not None:
                try:
                    self.chroma_collection.upsert(
                        ids=[canonical_id],
                        embeddings=[vec.tolist()],
                        metadatas=[{
                            "canonical_id": canonical_id,
                            "plant": str(doc.get("plant", "")),
                            "disease": str(doc.get("disease", ""))
                        }],
                        documents=[doc.get("search_text", "")]
                    )
                except Exception:
                    pass

    def _build_search_text(self, doc: Dict[str, Any]) -> str:
        """Constructs a rich search text string from document fields."""
        plant = doc.get("plant", "")
        disease = doc.get("disease", "")
        cid = doc.get("canonical_id", "")
        status = doc.get("health_status", "diseased")

        parts = [f"Plant: {plant}. Disease: {disease} (ID: {cid}). Status: {status}."]

        for field in ("symptoms", "causes", "risk_factors", "prevention", "management"):
            items = doc.get(field, [])
            if items:
                label = field.replace("_", " ").title()
                parts.append(f"{label}: {'; '.join(items)}.")

        return " ".join(parts)

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    def search_by_vector(
        self, query_vector: Union[List[float], np.ndarray], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches index by vector embedding.

        Returns: list of (document, similarity_score) tuples sorted desc by score.
        """
        if not self.embeddings:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # FAISS path
        if HAS_FAISS and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            k = min(top_k, self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(
                np.array([q_vec], dtype=np.float32), k
            )
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
            return results

        # NumPy cosine fallback
        emb_matrix = np.array(self.embeddings, dtype=np.float32)
        sims = np.dot(emb_matrix, q_vec)
        k = min(top_k, len(self.documents))
        top_indices = np.argsort(sims)[::-1][:k]

        return [(self.documents[int(idx)], float(sims[idx])) for idx in top_indices]

    def search_by_query(
        self, query_text: str, top_k: int = 5, min_score: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches index by natural language disease/condition query string.

        Supports queries like:
        - "apple scab leaf spots"
        - "tomato early blight concentric rings management"
        - "humid conditions fungal spray prevention"

        Returns: list of (document, score) tuples filtered by min_score.
        """
        q_vec = self._text_to_vector(query_text)
        results = self.search_by_vector(q_vec, top_k=top_k)
        if min_score > 0.0:
            results = [(doc, score) for doc, score in results if score >= min_score]
        return results

    def search_by_canonical_id(self, canonical_id: str) -> Optional[Dict[str, Any]]:
        """
        O(1) exact-match lookup for document by canonical disease ID.
        """
        idx = self._canonical_index.get(canonical_id)
        if idx is not None:
            return self.documents[idx]
        # Linear fallback for safety
        for doc in self.documents:
            if doc.get("canonical_id") == canonical_id:
                return doc
        return None

    def search_by_plant(
        self, plant_name: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Returns all documents for a given plant species (case-insensitive).
        """
        name = plant_name.strip().lower()
        return [
            doc for doc in self.documents
            if doc.get("plant", "").lower() == name
        ][:top_k]

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save_index(self, path: Optional[str] = None) -> str:
        """Saves documents and embeddings to disk."""
        target_path = path or os.path.join(self.store_dir, "vector_store.npz")
        docs_path = os.path.join(self.store_dir, "vector_documents.json")
        index_path = os.path.join(self.store_dir, "canonical_index.json")

        emb_matrix = (
            np.array(self.embeddings, dtype=np.float32)
            if self.embeddings
            else np.zeros((0, self.dimension))
        )
        np.savez_compressed(target_path, embeddings=emb_matrix)

        with open(docs_path, mode="w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

        with open(index_path, mode="w", encoding="utf-8") as f:
            json.dump(self._canonical_index, f, indent=2)

        return target_path

    def load_index(self, path: Optional[str] = None) -> bool:
        """Loads saved index from disk.

        Returns False (without crashing) if the stored embeddings have a
        different dimension than the current VectorStore, so the caller
        can rebuild the index from source data.
        """
        target_path = path or os.path.join(self.store_dir, "vector_store.npz")
        docs_path = os.path.join(self.store_dir, "vector_documents.json")
        index_path = os.path.join(self.store_dir, "canonical_index.json")

        if not os.path.exists(target_path) or not os.path.exists(docs_path):
            return False

        data = np.load(target_path)
        loaded_embeddings = data["embeddings"]

        # ── Dimension mismatch guard ──────────────────────────────────────
        if loaded_embeddings.ndim == 2 and loaded_embeddings.shape[1] != self.dimension:
            print(
                f"Warning: Stored index dimension ({loaded_embeddings.shape[1]}) "
                f"does not match VectorStore dimension ({self.dimension}). "
                "Rebuilding index from source data."
            )
            return False

        with open(docs_path, mode="r", encoding="utf-8") as f:
            self.documents = json.load(f)

        if os.path.exists(index_path):
            with open(index_path, mode="r", encoding="utf-8") as f:
                raw = json.load(f)
                self._canonical_index = {k: int(v) for k, v in raw.items()}
        else:
            # Rebuild canonical index from documents
            self._canonical_index = {
                str(doc.get("canonical_id", f"doc_{i}")): i
                for i, doc in enumerate(self.documents)
            }

        self.embeddings = [row for row in loaded_embeddings]

        if HAS_FAISS and self.faiss_index is not None and len(self.embeddings) > 0:
            self.faiss_index.reset()
            self.faiss_index.add(np.array(self.embeddings, dtype=np.float32))

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the current index."""
        return {
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "has_faiss": HAS_FAISS,
            "has_chromadb": HAS_CHROMADB and self.chroma_collection is not None,
            "unique_plants": len({doc.get("plant") for doc in self.documents}),
            "unique_diseases": len({doc.get("canonical_id") for doc in self.documents}),
        }


if __name__ == "__main__":
    kb_path = "data/knowledge_base/agricultural_documents.json"
    if os.path.exists(kb_path):
        with open(kb_path, mode="r", encoding="utf-8") as f:
            docs = json.load(f)
        store = VectorStore(dimension=256)
        store.add_documents(docs)
        store.save_index()
        stats = store.get_stats()
        print(f"Indexed {stats['total_documents']} documents | {stats['unique_plants']} plants | dim={stats['dimension']}")

        # Quick similarity query test
        results = store.search_by_query("apple scab olive spots fungal prevention", top_k=3)
        for doc, score in results:
            print(f"  [{score:.4f}] {doc['canonical_id']} — {doc['plant']} / {doc['disease']}")
