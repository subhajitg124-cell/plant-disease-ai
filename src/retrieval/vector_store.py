"""
Vector Store and Similarity Index Module.

Supports FAISS, ChromaDB, and NumPy/SciPy cosine similarity indexing
for fast agricultural document retrieval.
"""

import os
import sys
import json
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


class VectorStore:
    """
    Unified Vector Store and Similarity Indexing Engine.
    """
    def __init__(self, dimension: int = 128, store_dir: str = "models/vector_index"):
        self.dimension = dimension
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)

        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []

        # FAISS index initialization
        self.faiss_index = None
        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatIP(dimension)

        # ChromaDB client initialization
        self.chroma_client = None
        self.chroma_collection = None
        if HAS_CHROMADB:
            try:
                self.chroma_client = chromadb.PersistentClient(path=os.path.join(self.store_dir, "chroma"))
                self.chroma_collection = self.chroma_client.get_or_create_collection(name="agricultural_kb")
            except Exception as e:
                print(f"Warning: ChromaDB initialization fallback: {e}")
                self.chroma_client = None
                self.chroma_collection = None

    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        Generates deterministic L2-normalized pseudo-embedding for text string.
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        for idx, word in enumerate(words):
            val = sum(ord(c) for c in word)
            slot = val % self.dimension
            vec[slot] += (idx + 1) * 0.1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def add_documents(
        self,
        docs: List[Dict[str, Any]],
        vectors: Optional[List[List[float]]] = None
    ):
        """
        Indexes a list of documents and their corresponding vector embeddings.
        """
        for i, doc in enumerate(docs):
            if vectors is not None and i < len(vectors):
                vec = np.array(vectors[i], dtype=np.float32)
            else:
                text = doc.get("search_text", "") or json.dumps(doc)
                vec = self._text_to_vector(text)

            # L2 normalize vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            doc_id = str(doc.get("canonical_id", f"doc_{len(self.documents)}"))

            self.documents.append(doc)
            self.embeddings.append(vec)

            # Index into FAISS
            if HAS_FAISS and self.faiss_index is not None:
                self.faiss_index.add(np.array([vec], dtype=np.float32))

            # Index into ChromaDB
            if self.chroma_collection is not None:
                try:
                    self.chroma_collection.add(
                        ids=[f"{doc_id}_{len(self.documents)}"],
                        embeddings=[vec.tolist()],
                        metadatas=[{
                            "canonical_id": str(doc.get("canonical_id", "")),
                            "plant": str(doc.get("plant", "")),
                            "disease": str(doc.get("disease", ""))
                        }],
                        documents=[doc.get("search_text", "")]
                    )
                except Exception as e:
                    pass

    def search_by_vector(self, query_vector: List[float], top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches index by vector embedding, returning list of (document, similarity_score).
        """
        if not self.embeddings:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # FAISS search path
        if HAS_FAISS and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            scores, indices = self.faiss_index.search(np.array([q_vec], dtype=np.float32), top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
            return results

        # NumPy cosine similarity fallback path
        emb_matrix = np.array(self.embeddings)
        sims = np.dot(emb_matrix, q_vec)
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(sims[idx])))

        return results

    def search_by_canonical_id(self, canonical_id: str) -> Optional[Dict[str, Any]]:
        """
        Direct exact-match lookup for document by canonical disease ID.
        """
        for doc in self.documents:
            if doc.get("canonical_id") == canonical_id:
                return doc
        return None

    def save_index(self, path: Optional[str] = None) -> str:
        """Saves documents and embeddings matrix to disk."""
        target_path = path or os.path.join(self.store_dir, "vector_store.npz")
        docs_json_path = os.path.join(self.store_dir, "vector_documents.json")

        emb_matrix = np.array(self.embeddings, dtype=np.float32) if self.embeddings else np.zeros((0, self.dimension))
        np.savez_compressed(target_path, embeddings=emb_matrix)

        with open(docs_json_path, mode="w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

        return target_path

    def load_index(self, path: Optional[str] = None) -> bool:
        """Loads saved index from disk."""
        target_path = path or os.path.join(self.store_dir, "vector_store.npz")
        docs_json_path = os.path.join(self.store_dir, "vector_documents.json")

        if not os.path.exists(target_path) or not os.path.exists(docs_json_path):
            return False

        with open(docs_json_path, mode="r", encoding="utf-8") as f:
            self.documents = json.load(f)

        data = np.load(target_path)
        self.embeddings = [row for row in data["embeddings"]]

        if HAS_FAISS and self.faiss_index is not None and len(self.embeddings) > 0:
            self.faiss_index.reset()
            self.faiss_index.add(np.array(self.embeddings, dtype=np.float32))

        return True


if __name__ == "__main__":
    kb_path = "data/knowledge_base/agricultural_documents.json"
    if os.path.exists(kb_path):
        with open(kb_path, mode="r", encoding="utf-8") as f:
            docs = json.load(f)
        store = VectorStore()
        store.add_documents(docs)
        store.save_index()
        print(f"Indexed {len(docs)} documents into VectorStore successfully!")
