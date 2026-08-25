"""
Module Interface Contracts for Plant Disease AI.

Defines typed dataclass schemas and data contracts governing inter-module 
communication across Vision, RAG Knowledge Base, and Integration layers.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any


class PredictionStatus(str, Enum):
    """Supported prediction status categories."""
    SUPPORTED = "supported"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"
    NOT_A_PLANT = "not_a_plant"


@dataclass
class VisionPrediction:
    """
    Data contract for Vision/CNN module output (Tohidur).
    
    Required Core Fields:
    - plant: Crop species name (e.g., 'Tomato')
    - disease: Disease name or 'Healthy' (e.g., 'Early Blight')
    - canonical_id: Machine-friendly taxonomy ID (e.g., 'tomato_early_blight')
    - confidence: Prediction confidence score [0.0, 1.0]
    - status: Prediction status category ('supported', 'uncertain', 'unknown', 'not_a_plant')
    
    Optional / Future Extension Fields:
    - raw_label: Original dataset directory label if applicable
    - embedding: Deep visual feature vector (Optional future extension for open-set matching)
    - model_version: Identifier for the CNN checkpoint/architecture version (e.g., 'vision_v1')
    """
    plant: str
    disease: str
    canonical_id: str
    confidence: float
    status: str
    raw_label: Optional[str] = None
    embedding: Optional[List[float]] = None
    model_version: Optional[str] = None

    def __post_init__(self):
        valid_statuses = {s.value for s in PredictionStatus}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of {valid_statuses}")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence score {self.confidence} must be between 0.0 and 1.0")

        if not isinstance(self.plant, str) or not self.plant.strip():
            raise ValueError("Field 'plant' must be a non-empty string.")
        if not isinstance(self.disease, str) or not self.disease.strip():
            raise ValueError("Field 'disease' must be a non-empty string.")
        if not isinstance(self.canonical_id, str) or not self.canonical_id.strip():
            raise ValueError("Field 'canonical_id' must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionPrediction":
        required = ["plant", "disease", "canonical_id", "confidence", "status"]
        for req in required:
            if req not in data or data[req] is None:
                raise ValueError(f"Missing required field: '{req}'")
        return cls(
            plant=str(data["plant"]),
            disease=str(data["disease"]),
            canonical_id=str(data["canonical_id"]),
            confidence=float(data["confidence"]),
            status=str(data["status"]),
            raw_label=data.get("raw_label"),
            embedding=data.get("embedding"),
            model_version=data.get("model_version")
        )


@dataclass
class RAGQueryInput:
    """
    Data contract for Knowledge Base / RAG module input query (Saiyab).
    """
    plant: str
    disease: str
    canonical_id: str

    def __post_init__(self):
        if not isinstance(self.plant, str) or not self.plant.strip():
            raise ValueError("Field 'plant' must be a non-empty string.")
        if not isinstance(self.disease, str) or not self.disease.strip():
            raise ValueError("Field 'disease' must be a non-empty string.")
        if not isinstance(self.canonical_id, str) or not self.canonical_id.strip():
            raise ValueError("Field 'canonical_id' must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGQueryInput":
        for req in ["plant", "disease", "canonical_id"]:
            if req not in data or not data[req]:
                raise ValueError(f"Missing required field: '{req}'")
        return cls(
            plant=str(data["plant"]),
            disease=str(data["disease"]),
            canonical_id=str(data["canonical_id"])
        )


@dataclass
class AdvisoryResult:
    """
    Data contract for Knowledge Base / RAG module output advisory (Saiyab).
    
    NOTE: All text content held in instances of this class during Phase 2 are 
    FOR DEMONSTRATION & TESTING SCHEMA STRUCTURE ONLY. The production RAG system 
    will populate these fields from verified, authoritative agricultural extension sources in Phase 3.
    """
    canonical_id: str
    symptoms: List[str] = field(default_factory=list)
    causes: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    prevention: List[str] = field(default_factory=list)
    management: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.canonical_id, str) or not self.canonical_id.strip():
            raise ValueError("Field 'canonical_id' must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvisoryResult":
        if "canonical_id" not in data or not data["canonical_id"]:
            raise ValueError("Missing required field: 'canonical_id'")
        return cls(
            canonical_id=str(data["canonical_id"]),
            symptoms=list(data.get("symptoms", [])),
            causes=list(data.get("causes", [])),
            risk_factors=list(data.get("risk_factors", [])),
            prevention=list(data.get("prevention", [])),
            management=list(data.get("management", [])),
            sources=list(data.get("sources", []))
        )


@dataclass
class IntegratedResponse:
    """
    Data contract for Integration Layer output (Asikul).
    Combines VisionPrediction + AdvisoryResult into a final structured user response.
    """
    prediction: VisionPrediction
    advisory: Optional[AdvisoryResult]
    user_message: str
    confidence: float
    status: str
    evidence: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["prediction"] = self.prediction.to_dict()
        res["advisory"] = self.advisory.to_dict() if self.advisory else None
        return res

    @classmethod
    def compose(cls, prediction: VisionPrediction, advisory: Optional[AdvisoryResult] = None) -> "IntegratedResponse":
        """
        Combines VisionPrediction and AdvisoryResult into an IntegratedResponse object.
        """
        warnings: List[str] = []
        evidence: List[str] = []
        sources: List[str] = []

        if prediction.status == PredictionStatus.NOT_A_PLANT.value:
            user_msg = "Image validation failed: The uploaded image does not appear to contain a recognized plant."
            warnings.append("Rejection: Input image failed plant validation checks.")
            final_advisory = None

        elif prediction.status == PredictionStatus.UNCERTAIN.value:
            user_msg = f"Uncertain prediction ({prediction.confidence * 100:.1f}% confidence): The plant image resembles {prediction.plant} - {prediction.disease}, but confidence is low. Please upload a clearer close-up image."
            warnings.append("Low confidence score below operational threshold. Advisory suppressed.")
            final_advisory = None

        elif prediction.status == PredictionStatus.UNKNOWN.value:
            user_msg = "Unknown disease detected: The image contains a valid plant, but the pathology does not match supported dataset classes."
            warnings.append("Out-of-distribution plant disease sample.")
            final_advisory = None

        elif prediction.status == PredictionStatus.SUPPORTED.value:
            user_msg = f"Detected {prediction.plant} - {prediction.disease} ({prediction.confidence * 100:.1f}% confidence)."
            if advisory:
                evidence.extend(advisory.symptoms)
                sources.extend(advisory.sources)
                final_advisory = advisory
            else:
                warnings.append("Pathology advisory context unavailable for this class.")
                final_advisory = None
        else:
            user_msg = "System error processing image."
            warnings.append(f"Unrecognized prediction status '{prediction.status}'.")
            final_advisory = None

        return cls(
            prediction=prediction,
            advisory=final_advisory,
            user_message=user_msg,
            confidence=prediction.confidence,
            status=prediction.status,
            evidence=evidence,
            sources=sources,
            warnings=warnings
        )
