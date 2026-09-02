"""
Advisory Generator Module.

Integrates Vision CNN predictions with RAG Knowledge Base retrieval to construct
grounded IntegratedResponse objects.

Advisory Generation Flow:
  1. Validate prediction status (gate out non-plant, uncertain, unknown)
  2. Build RAG retrieval query from Vision prediction fields
  3. Retrieve grounded advisory from knowledge base
  4. Verify evidence grounding before final response assembly
  5. Synthesise user-facing advisory message from retrieved evidence
  6. Return fully-populated IntegratedResponse with evidence + sources

All advisory text is grounded in retrieved KB documents.
"""

import os
import sys
from typing import Optional, List, Dict, Any, Union

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.contracts import (
    VisionPrediction, RAGQueryInput, AdvisoryResult,
    IntegratedResponse, PredictionStatus
)
from src.retrieval.rag_retriever import RAGRetriever


class AdvisoryGenerator:
    """
    Advisory Generation Engine for Plant Disease Diagnosis.

    Orchestrates the full advisory prompt flow:
      VisionPrediction → RAGQueryInput → AdvisoryResult → IntegratedResponse

    Evidence grounding is verified before the response is assembled.
    Ungrounded or low-confidence predictions are handled with appropriate
    safety guardrails and clear user warnings.
    """

    def __init__(self, retriever: Optional[RAGRetriever] = None):
        self.retriever = retriever if retriever is not None else RAGRetriever()

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def generate_advisory(self, prediction: VisionPrediction) -> IntegratedResponse:
        """
        Full advisory generation pipeline for a single VisionPrediction.

        Returns a fully-populated IntegratedResponse with:
          - user_message: human-readable diagnosis summary
          - advisory: structured AdvisoryResult (or None for gated statuses)
          - evidence: flat list of grounding evidence chunks
          - sources: KB source citations
          - warnings: any safety or grounding warnings
        """
        status = prediction.status

        # ── Gate 1: Non-plant rejection ──────────────────────────────────────
        if status == PredictionStatus.NOT_A_PLANT.value:
            return IntegratedResponse(
                prediction=prediction,
                advisory=None,
                user_message=(
                    "Input image failed plant foliage validation. "
                    "Please upload a clear, well-lit photograph of a plant leaf."
                ),
                confidence=prediction.confidence,
                status=status,
                evidence=[],
                sources=[],
                warnings=["Validation failure: Non-plant or corrupted image detected."]
            )

        # ── Gate 2: Low-confidence prediction ───────────────────────────────
        if status == PredictionStatus.UNCERTAIN.value:
            return IntegratedResponse(
                prediction=prediction,
                advisory=None,
                user_message=(
                    f"Prediction confidence is too low ({prediction.confidence * 100:.1f}%) "
                    "to generate a reliable advisory. "
                    "Please provide a clearer, close-up image of the affected leaf."
                ),
                confidence=prediction.confidence,
                status=status,
                evidence=[],
                sources=[],
                warnings=[
                    "Low-confidence prediction suppressed for safety.",
                    f"Confidence {prediction.confidence:.3f} is below the operational threshold."
                ]
            )

        # ── Gate 3: Out-of-distribution / unknown disease ───────────────────
        if status == PredictionStatus.UNKNOWN.value:
            return IntegratedResponse(
                prediction=prediction,
                advisory=None,
                user_message=(
                    f"The detected pathology on {prediction.plant} does not match "
                    "any supported disease in the current knowledge base taxonomy. "
                    "Please consult a local agricultural extension specialist."
                ),
                confidence=prediction.confidence,
                status=status,
                evidence=[],
                sources=[],
                warnings=["Out-of-distribution plant disease detected — advisory unavailable."]
            )

        # ── Supported: Full RAG retrieval + grounding flow ───────────────────
        return self._generate_grounded_advisory(prediction)

    # -----------------------------------------------------------------------
    # Grounded advisory pipeline
    # -----------------------------------------------------------------------

    def _generate_grounded_advisory(
        self, prediction: VisionPrediction
    ) -> IntegratedResponse:
        """
        Full grounded advisory generation for a 'supported' prediction.

        Steps:
          1. Build RAGQueryInput from prediction
          2. Retrieve AdvisoryResult from knowledge base
          3. Verify grounding quality
          4. Extract evidence chunks
          5. Synthesise advisory message
          6. Assemble IntegratedResponse
        """
        warnings: List[str] = []

        # Step 1: Build retrieval query
        rag_query = RAGQueryInput(
            plant=prediction.plant,
            disease=prediction.disease,
            canonical_id=prediction.canonical_id
        )

        # Step 2: Retrieve advisory from KB
        advisory = self.retriever.retrieve(rag_query)

        # Step 3: Grounding verification
        is_grounded = self.retriever.is_grounded(advisory)
        if not is_grounded:
            warnings.append(
                "Advisory grounding quality is low — retrieved content may be generic. "
                "Verify with a certified agronomist before taking action."
            )

        # Step 4: Extract evidence chunks
        evidence_chunks = self.retriever.get_evidence_chunks(advisory, max_chunks=8)

        # Step 5: Synthesise user-facing advisory message
        user_message = self._synthesise_message(prediction, advisory, is_grounded)

        # Step 6: Assemble IntegratedResponse
        return IntegratedResponse(
            prediction=prediction,
            advisory=advisory,
            user_message=user_message,
            confidence=prediction.confidence,
            status=prediction.status,
            evidence=evidence_chunks,
            sources=advisory.sources,
            warnings=warnings
        )

    # -----------------------------------------------------------------------
    # Prompt / message synthesis
    # -----------------------------------------------------------------------

    def _synthesise_message(
        self,
        prediction: VisionPrediction,
        advisory: AdvisoryResult,
        is_grounded: bool
    ) -> str:
        """
        Synthesises a structured, human-readable advisory message from retrieved evidence.

        The message is grounded in KB-retrieved content: symptoms, top prevention
        steps, and top management actions are drawn directly from the AdvisoryResult.
        """
        lines: List[str] = []

        # Header
        conf_pct = prediction.confidence * 100
        lines.append(
            f"Diagnosis: {prediction.plant} — {prediction.disease} "
            f"({conf_pct:.1f}% confidence)."
        )

        # Symptoms section (grounded)
        if advisory.symptoms:
            top_symptoms = advisory.symptoms[:3]
            lines.append(
                "Observed symptoms: " + "; ".join(top_symptoms) + "."
            )

        # Causes section (grounded)
        if advisory.causes:
            lines.append(
                "Likely cause: " + advisory.causes[0] + "."
            )

        # Risk factors
        if advisory.risk_factors:
            lines.append(
                "Key risk factor: " + advisory.risk_factors[0] + "."
            )

        # Prevention (grounded)
        if advisory.prevention:
            top_prevention = advisory.prevention[:2]
            lines.append(
                "Recommended prevention: " + "; ".join(top_prevention) + "."
            )

        # Management (grounded)
        if advisory.management:
            top_management = advisory.management[:2]
            lines.append(
                "Treatment actions: " + "; ".join(top_management) + "."
            )

        # Grounding quality notice
        if not is_grounded:
            lines.append(
                "Note: Advisory is based on general guidelines. "
                "Consult an agronomist for site-specific recommendations."
            )

        return " | ".join(lines)

    # -----------------------------------------------------------------------
    # Batch generation
    # -----------------------------------------------------------------------

    def generate_batch(
        self, predictions: List[VisionPrediction]
    ) -> List[IntegratedResponse]:
        """
        Generates advisory responses for a list of VisionPrediction objects.
        """
        return [self.generate_advisory(pred) for pred in predictions]

    # -----------------------------------------------------------------------
    # Image + advisory end-to-end (convenience wrapper)
    # -----------------------------------------------------------------------

    def format_advisory_report(self, response: IntegratedResponse) -> str:
        """
        Formats an IntegratedResponse into a multi-section advisory report string.
        """
        pred = response.prediction
        lines = [
            "=" * 60,
            "PLANT DISEASE ADVISORY REPORT",
            "=" * 60,
            f"Plant:    {pred.plant}",
            f"Disease:  {pred.disease}",
            f"Confidence: {pred.confidence * 100:.1f}%",
            f"Status:   {response.status}",
            "",
            "ADVISORY:",
            response.user_message,
        ]

        if response.advisory:
            adv = response.advisory
            if adv.symptoms:
                lines += ["", "SYMPTOMS:"]
                lines += [f"  • {s}" for s in adv.symptoms]
            if adv.causes:
                lines += ["", "CAUSES:"]
                lines += [f"  • {c}" for c in adv.causes]
            if adv.risk_factors:
                lines += ["", "RISK FACTORS:"]
                lines += [f"  • {r}" for r in adv.risk_factors]
            if adv.prevention:
                lines += ["", "PREVENTION:"]
                lines += [f"  • {p}" for p in adv.prevention]
            if adv.management:
                lines += ["", "MANAGEMENT:"]
                lines += [f"  • {m}" for m in adv.management]
            if adv.sources:
                lines += ["", "SOURCES:"]
                lines += [f"  [{i+1}] {s}" for i, s in enumerate(adv.sources)]

        if response.warnings:
            lines += ["", "WARNINGS:"]
            lines += [f"  ⚠ {w}" for w in response.warnings]

        lines.append("=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    generator = AdvisoryGenerator()

    # Test 1: Supported prediction — should produce full grounded advisory
    pred_supported = VisionPrediction(
        plant="Apple",
        disease="Apple scab",
        canonical_id="apple_apple_scab",
        confidence=0.92,
        status="supported",
        raw_label="Apple___Apple_scab",
        model_version="vision_v1"
    )
    res = generator.generate_advisory(pred_supported)
    print(generator.format_advisory_report(res))
    print(f"\nGrounding verified: {bool(res.evidence)}")
    print(f"Sources: {res.sources}")

    # Test 2: Uncertain prediction — should suppress advisory
    print("\n" + "=" * 60)
    pred_uncertain = VisionPrediction(
        plant="Tomato",
        disease="Early blight",
        canonical_id="tomato_early_blight",
        confidence=0.38,
        status="uncertain"
    )
    res2 = generator.generate_advisory(pred_uncertain)
    print(f"Uncertain → Advisory: {res2.advisory}")
    print(f"Warning: {res2.warnings}")

    # Test 3: Query-based retrieval
    print("\n[Query-based Retrieval]")
    similar = generator.retriever.search_similar(
        "tomato late blight phytophthora infestans cool wet weather management", top_k=3
    )
    for doc, score in similar:
        print(f"  [{score:.4f}] {doc['canonical_id']} — {doc['plant']}/{doc['disease']}")
