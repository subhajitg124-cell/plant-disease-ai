# Module Interface Specifications

> **Status**: **Phase 2 Specification** (`[IMPLEMENTED SCHEMAS]`)

This document defines the exact data contracts, inputs, outputs, Python dataclasses (`src/contracts.py`), and canonical JSON schemas governing inter-module communication across the **Plant Disease AI** platform.

---

## 1. Module Pipeline Dataflow

```text
User Image Input
      ↓
Vision / CNN Module (Tohidur) ───[VisionPrediction Object]───┐
      ↓                                                     │
Canonical Disease ID ("tomato_early_blight")               │
      ↓                                                     │
RAG Knowledge Module (Saiyab) ───[AdvisoryResult Object]───┤
                                                            ↓
Integration Pipeline (Asikul) ───────────────────[IntegratedResponse Object]
```

---

## 2. Status Categories & Prediction Classification

The system categorizes predictions into four distinct status categories (`PredictionStatus` enum in `src/contracts.py`):

| Status | Category Enum | Description | Action / Flow |
| :--- | :--- | :--- | :--- |
| **Valid Supported Disease** | `"supported"` | Image passes plant validation and model confidence \(\ge \tau\) for a recognized taxonomy class. | Passed to RAG module for full advisory retrieval. |
| **Uncertain Prediction** | `"uncertain"` | Plant is recognized, but model confidence is below operational threshold \(\tau\). | Advisory suppressed; user prompted for a clearer image. |
| **Unknown Disease** | `"unknown"` | Valid plant image detected, but pathology is out-of-distribution / unsupported class. | Returns unknown pathology notification; advisory suppressed. |
| **Not A Plant** | `"not_a_plant"` | Input image rejected by the plant validation filter. | Rejection notice returned immediately. |

*Note: Confidence threshold \(\tau\) is NOT hardcoded in schemas and will be calibrated experimentally during evaluation.*

---

## 3. Vision Module Contract (`src/vision/` / `src/contracts.py`)

### Responsibility (Tohidur)
Ingests preprocessed image tensors, runs CNN model inference, and outputs a `VisionPrediction` object.

### Core Required Fields vs. Optional Extensions
- **Required Core Fields**: `plant`, `disease`, `canonical_id`, `confidence`, `status`.
- **Optional Extensions**:
  - `model_version`: Identifier for the CNN checkpoint/architecture version (e.g., `'vision_v1'`).
  - `embedding`: Deep visual feature vector (Optional future extension for open-set matching/similarity search; vector dimensions and backbone architecture are TBD).
  - `raw_label`: Original dataset directory label if applicable.

### Python Dataclass Schema (`VisionPrediction`)
```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class VisionPrediction:
    plant: str
    disease: str
    canonical_id: str
    confidence: float
    status: str  # "supported" | "uncertain" | "unknown" | "not_a_plant"
    raw_label: Optional[str] = None
    embedding: Optional[List[float]] = None  # Optional future extension
    model_version: Optional[str] = None      # Optional version tag e.g. "vision_v1"
```

### Example JSON Payload (Supported Disease)
```json
{
  "plant": "Tomato",
  "disease": "Early Blight",
  "canonical_id": "tomato_early_blight",
  "confidence": 0.94,
  "status": "supported",
  "model_version": "vision_v1"
}
```

---

## 4. RAG / Knowledge Base Contract (`src/retrieval/` / `src/advisory/` / `src/contracts.py`)

> [!IMPORTANT]
> **Demonstration Data Notice**: All advisory text, symptoms, and sources shown below are **EXAMPLE / DEMONSTRATION DATA ONLY** for schema validation purposes. The production RAG knowledge base will retrieve verified, authoritative agricultural extension facts during Phase 3.

### Responsibility (Saiyab)
Receives the canonical disease metadata, executes semantic vector lookup against the agricultural knowledge base, and returns a structured `AdvisoryResult` object.

### Input Schema (`RAGQueryInput`)
```python
@dataclass
class RAGQueryInput:
    plant: str
    disease: str
    canonical_id: str
```

```json
{
  "plant": "Tomato",
  "disease": "Early Blight",
  "canonical_id": "tomato_early_blight"
}
```

### Output Schema (`AdvisoryResult`)
```python
@dataclass
class AdvisoryResult:
    canonical_id: str
    symptoms: List[str]
    causes: List[str]
    risk_factors: List[str]
    prevention: List[str]
    management: List[str]
    sources: List[str]
```

```json
{
  "canonical_id": "tomato_early_blight",
  "symptoms": [
    "[EXAMPLE] Dark brown spots with concentric rings on lower leaves"
  ],
  "causes": [
    "[EXAMPLE] Fungal pathogen Alternaria solani"
  ],
  "risk_factors": [
    "[EXAMPLE] High humidity and warm temperatures (24-29°C)"
  ],
  "prevention": [
    "[EXAMPLE] Rotate crops every 2-3 years with non-solanaceous crops"
  ],
  "management": [
    "[EXAMPLE] Apply copper-based or chlorothalonil fungicides at first onset"
  ],
  "sources": [
    "[EXAMPLE] USDA Agricultural Extension Publication No. 402"
  ]
}
```

---

## 5. Integration Contract (`src/integration/` / `src/contracts.py`)

### Responsibility (Asikul)
Combines `VisionPrediction` and `AdvisoryResult` into a final `IntegratedResponse` object providing user messaging, evidence snippets, sources, and warnings.

### Python Dataclass Schema (`IntegratedResponse`)
```python
@dataclass
class IntegratedResponse:
    prediction: VisionPrediction
    advisory: Optional[AdvisoryResult]
    user_message: str
    confidence: float
    status: str
    evidence: List[str]
    sources: List[str]
    warnings: List[str]
```

### Example Integrated Payload (Supported Disease)
```json
{
  "prediction": {
    "plant": "Tomato",
    "disease": "Early Blight",
    "canonical_id": "tomato_early_blight",
    "confidence": 0.94,
    "status": "supported",
    "model_version": "vision_v1"
  },
  "advisory": {
    "canonical_id": "tomato_early_blight",
    "symptoms": ["[EXAMPLE] Dark brown spots with concentric rings on lower leaves"],
    "causes": ["[EXAMPLE] Fungal pathogen Alternaria solani"],
    "risk_factors": ["[EXAMPLE] High humidity (24-29°C)"],
    "prevention": ["[EXAMPLE] Rotate crops every 2-3 years"],
    "management": ["[EXAMPLE] Apply copper-based fungicide sprays"],
    "sources": ["[EXAMPLE] USDA Agricultural Extension Publication No. 402"]
  },
  "user_message": "Detected Tomato - Early Blight (94.0% confidence).",
  "confidence": 0.94,
  "status": "supported",
  "evidence": ["[EXAMPLE] Dark brown spots with concentric rings on lower leaves"],
  "sources": ["[EXAMPLE] USDA Agricultural Extension Publication No. 402"],
  "warnings": []
}
```

---

## 6. Implementation References

- Python Schemas: [`src/contracts.py`](file:///c:/Users/HP/Downloads/Plant%20Disease/plant-disease-ai/src/contracts.py)
- Unit Test Suite: [`tests/test_module_interfaces.py`](file:///c:/Users/HP/Downloads/Plant%20Disease/plant-disease-ai/tests/test_module_interfaces.py)
