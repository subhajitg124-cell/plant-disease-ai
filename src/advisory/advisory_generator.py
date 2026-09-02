"""
Advisory Generator Module.

Integrates Vision predictions and RAG Knowledge Base retrieval to construct
grounded IntegratedResponse objects matching inter-module contracts.
"""

import os
import sys
from typing import Optional, Dict, Any, Union

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from src.contracts import VisionPrediction, RAGQueryInput, AdvisoryResult, IntegratedResponse, PredictionStatus
except ImportError:
    from contracts import VisionPrediction, RAGQueryInput, AdvisoryResult, IntegratedResponse, PredictionStatus

try:
    from src.retrieval.rag_retriever import RAGRetriever
except ImportError:
    try:
        from retrieval.rag_retriever import RAGRetriever
    except ImportError:
        from rag_retriever import RAGRetriever


class AdvisoryGenerator:
    """
    Advisory Generation Engine for Plant Disease Diagnosis.
    """
    def __init__(self, retriever: Optional[RAGRetriever] = None):
        self.retriever = retriever if retriever is not None else RAGRetriever()

    def generate_advisory(self, prediction: VisionPrediction) -> IntegratedResponse:
        """
        Generates structured IntegratedResponse combining Vision prediction and RAG advisory.
        """
        status = prediction.status
        confidence = prediction.confidence
        canonical_id = prediction.canonical_id

        # 1. Non-plant status
        if status == PredictionStatus.NOT_A_PLANT.value:
            return IntegratedResponse(
                prediction=prediction,
                advisory=None,
                user_message="Input image failed plant foliage validation. Please upload a clear image of a plant leaf.",
                confidence=confidence,
                status=status,
                evidence=[],
                sources=[],
                warnings=["Validation failure: Non-plant image detected."]
            )

        # 2. Uncertain status
        if status == PredictionStatus.UNCERTAIN.value:
            return IntegratedResponse(
                prediction=prediction,
                advisory=None,
                user_message="Prediction confidence is below operational threshold. Advisory suppressed for safety. Please provide a clearer leaf image.",
                confidence=confidence,
                status=status,
                evidence=[],
                sources=[],
                warnings=["Low confidence prediction suppressed."]
            )

        # 3. Unknown / unsupported status
        if status == PredictionStatus.UNKNOWN.value:
            return IntegratedResponse(
                prediction=prediction,
                advisory=None,
                user_message="Input plant foliage disease is unsupported by current baseline taxonomy.",
                confidence=confidence,
                status=status,
                evidence=[],
                sources=[],
                warnings=["Out-of-distribution plant disease detected."]
            )

        # 4. Supported status: Retrieve agricultural evidence via RAG
        rag_query = RAGQueryInput(
            plant=prediction.plant,
            disease=prediction.disease,
            canonical_id=canonical_id
        )
        advisory = self.retriever.retrieve(rag_query)

        evidence_chunks = advisory.symptoms + advisory.prevention + advisory.management

        return IntegratedResponse(
            prediction=prediction,
            advisory=advisory,
            user_message=f"Identified {prediction.plant} with {prediction.disease} ({confidence*100:.1f}% confidence). Grounded advisory retrieved.",
            confidence=confidence,
            status=status,
            evidence=evidence_chunks,
            sources=advisory.sources,
            warnings=[]
        )


if __name__ == "__main__":
    generator = AdvisoryGenerator()
    pred = VisionPrediction(
        plant="Apple",
        disease="Apple scab",
        canonical_id="apple_apple_scab",
        confidence=0.92,
        status="supported",
        raw_label="Apple___Apple_scab",
        model_version="vision_v1"
    )
    res = generator.generate_advisory(pred)
    print("Generated IntegratedResponse:")
    print("User Message:", res.user_message)
    print("Sources:", res.sources)
