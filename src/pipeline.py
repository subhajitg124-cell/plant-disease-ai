"""
Plant Disease Pipeline — End-to-End Integration Module.

Connects the CNN Vision Classifier, RAG Knowledge Base Retriever, and Advisory
Generator into a single, unified prediction pipeline.

Full pipeline flow:
  Image → Preprocessing → CNN Classification → VisionPrediction
       → RAGQueryInput → KB Retrieval → AdvisoryResult
       → Advisory Synthesis → IntegratedResponse

Usage:
    from src.pipeline import PlantDiseasePipeline

    pipeline = PlantDiseasePipeline()
    result = pipeline.predict_and_advise("path/to/leaf.jpg")
    print(result.user_message)
"""

import os
from typing import Optional, List, Dict, Any, Union

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

from src.contracts import VisionPrediction, AdvisoryResult, IntegratedResponse, PredictionStatus
from src.vision.classifier import PlantDiseaseClassifier
from src.retrieval.rag_retriever import RAGRetriever
from src.advisory.advisory_generator import AdvisoryGenerator


class PlantDiseasePipeline:
    """
    Unified end-to-end Plant Disease Detection and Advisory Pipeline.

    Integrates:
      - PlantDiseaseClassifier (Vision/CNN module)
      - RAGRetriever           (Knowledge Base Retrieval module)
      - AdvisoryGenerator      (Advisory Synthesis module)

    The pipeline can be run on a single image or in batch mode.

    Args:
        model_path:            Path to CNN checkpoint (.pth file).
        class_mapping_path:    Path to PlantVillage class mapping CSV.
        kb_path:               Path to agricultural knowledge base JSON.
        store_dir:             Directory for vector index persistence.
        confidence_threshold:  Minimum confidence for 'supported' prediction.
        device:                PyTorch device string ('cpu', 'cuda', or None for auto).
    """

    def __init__(
        self,
        model_path: Optional[str] = "models/plant_disease_cnn.pth",
        class_mapping_path: str = "data/metadata/plantvillage_class_mapping.csv",
        kb_path: str = "data/knowledge_base/agricultural_documents.json",
        store_dir: str = "models/vector_index",
        confidence_threshold: float = 0.60,
        device: Optional[str] = None
    ):
        # Vision module
        self.classifier = PlantDiseaseClassifier(
            model_path=model_path,
            class_mapping_path=class_mapping_path,
            confidence_threshold=confidence_threshold,
            device=device
        )

        # RAG retrieval module
        self.retriever = RAGRetriever(kb_path=kb_path, store_dir=store_dir)

        # Advisory generation module (shares retriever instance)
        self.generator = AdvisoryGenerator(retriever=self.retriever)

    # -----------------------------------------------------------------------
    # Single image prediction
    # -----------------------------------------------------------------------

    def predict_and_advise(
        self,
        image_input: Union[str, "Image.Image", np.ndarray],  # type: ignore[name-defined]
        extract_embedding: bool = False
    ) -> IntegratedResponse:
        """
        Runs the full end-to-end pipeline on a single image input.

        Args:
            image_input:       File path (str), PIL Image, or NumPy array.
            extract_embedding: Whether to extract and attach a visual embedding vector.

        Returns:
            IntegratedResponse with diagnosis, advisory, evidence, sources, and warnings.

        Pipeline steps:
          1. CNN Vision classification → VisionPrediction
          2. RAG retrieval → AdvisoryResult (gated by prediction status)
          3. Advisory synthesis → IntegratedResponse
        """
        # Step 1: Vision classification
        prediction = self.classifier.predict(
            image_input, extract_embedding=extract_embedding
        )

        # Step 2 + 3: Advisory generation (internally calls RAG retrieval)
        response = self.generator.generate_advisory(prediction)

        return response

    # -----------------------------------------------------------------------
    # Batch prediction
    # -----------------------------------------------------------------------

    def predict_batch(
        self,
        image_inputs: List[Union[str, "Image.Image", np.ndarray]],  # type: ignore[name-defined]
    ) -> List[IntegratedResponse]:
        """
        Runs the pipeline on a list of image inputs and returns a list of responses.
        """
        results: List[IntegratedResponse] = []
        for img in image_inputs:
            try:
                res = self.predict_and_advise(img)
            except Exception as exc:
                # Construct an error response rather than crashing the batch
                err_prediction = VisionPrediction(
                    plant="Error",
                    disease="Processing Error",
                    canonical_id="pipeline_error",
                    confidence=0.0,
                    status=PredictionStatus.NOT_A_PLANT.value
                )
                res = IntegratedResponse(
                    prediction=err_prediction,
                    advisory=None,
                    user_message=f"Pipeline error during processing: {exc}",
                    confidence=0.0,
                    status=PredictionStatus.NOT_A_PLANT.value,
                    evidence=[],
                    sources=[],
                    warnings=[f"Exception: {exc}"]
                )
            results.append(res)
        return results

    # -----------------------------------------------------------------------
    # Vision-only (no advisory) quick path
    # -----------------------------------------------------------------------

    def classify_only(
        self,
        image_input: Union[str, "Image.Image", np.ndarray],  # type: ignore[name-defined]
        extract_embedding: bool = False
    ) -> VisionPrediction:
        """
        Runs only the Vision CNN classification step without RAG retrieval.

        Useful when only the disease label and confidence score are needed.
        """
        return self.classifier.predict(image_input, extract_embedding=extract_embedding)

    # -----------------------------------------------------------------------
    # RAG query-only path
    # -----------------------------------------------------------------------

    def query_knowledge_base(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Direct natural language query against the agricultural knowledge base.

        Returns a list of (document, similarity_score) tuples sorted by relevance.

        Example queries:
          - "apple scab olive spots fungal prevention copper"
          - "tomato early blight alternaria management chlorothalonil"
          - "humid conditions bacterial spot warm weather overhead irrigation"
        """
        results = self.retriever.search_similar(query_text, top_k=top_k)
        return [
            {
                "canonical_id": doc.get("canonical_id"),
                "plant": doc.get("plant"),
                "disease": doc.get("disease"),
                "similarity_score": round(score, 4),
                "symptoms": doc.get("symptoms", []),
                "management": doc.get("management", []),
                "sources": doc.get("sources", [])
            }
            for doc, score in results
        ]

    # -----------------------------------------------------------------------
    # Canonical ID advisory shortcut
    # -----------------------------------------------------------------------

    def get_advisory_by_id(self, canonical_id: str) -> AdvisoryResult:
        """
        Retrieves an agricultural advisory directly by canonical disease ID.

        Bypasses the CNN classifier — useful for integration testing.
        """
        advisory = self.retriever.retrieve_by_canonical_id(canonical_id)
        return advisory

    # -----------------------------------------------------------------------
    # Pipeline diagnostics
    # -----------------------------------------------------------------------

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Returns diagnostic info about the loaded pipeline components."""
        kb_stats = self.retriever.get_kb_stats()
        return {
            "classifier": {
                "model_path": self.classifier.model_path,
                "num_classes": self.classifier.num_classes,
                "confidence_threshold": self.classifier.confidence_threshold,
                "device": str(self.classifier.device)
            },
            "knowledge_base": kb_stats,
            "pipeline_version": "1.0.0"
        }


if __name__ == "__main__":
    print("Initialising Plant Disease Pipeline...")
    pipeline = PlantDiseasePipeline()

    info = pipeline.get_pipeline_info()
    print(f"\nPipeline Info:")
    print(f"  Classifier: {info['classifier']['num_classes']} classes, device={info['classifier']['device']}")
    print(f"  Knowledge Base: {info['knowledge_base']['total_documents']} documents indexed")

    # Test 1: Synthetic green leaf (valid plant)
    print("\n[Test 1: Synthetic Green Leaf]")
    green_leaf = np.zeros((224, 224, 3), dtype=np.uint8)
    green_leaf[:, :, 1] = 190
    green_leaf[:, :, 0] = 30
    result = pipeline.predict_and_advise(green_leaf)
    print(f"  Status: {result.status}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Plant/Disease: {result.prediction.plant} / {result.prediction.disease}")
    print(f"  Evidence chunks: {len(result.evidence)}")
    print(f"  Sources: {result.sources}")
    print(f"  Message: {result.user_message[:150]}...")

    # Test 2: Direct KB advisory
    print("\n[Test 2: Direct KB Advisory — tomato_early_blight]")
    adv = pipeline.get_advisory_by_id("tomato_early_blight")
    print(f"  Symptoms: {adv.symptoms[:2]}")
    print(f"  Management: {adv.management[:2]}")
    print(f"  Sources: {adv.sources}")

    # Test 3: Natural language KB query
    print("\n[Test 3: NL Query — 'apple scab olive fungal treatment']")
    hits = pipeline.query_knowledge_base("apple scab olive green spots fungal treatment", top_k=3)
    for hit in hits:
        print(f"  [{hit['similarity_score']}] {hit['canonical_id']} — {hit['plant']}/{hit['disease']}")

    print("\nPipeline end-to-end test complete.")
