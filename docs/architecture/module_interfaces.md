# Module Interface Specifications

> **Status**: **Phase 2 Specification** (`[IN PROGRESS]`)

This document defines the exact data contracts, inputs, outputs, and canonical data schemas governing inter-module communication across the **Plant Disease AI** platform.

---

## 1. Vision Module Contract (`src/vision/`)

### Responsibility
Ingests preprocessed image tensors, runs CNN inference/embedding extraction, evaluates prediction confidence, maps raw predictions to canonical disease IDs, and determines classification status.

### Function Signature Conceptual Specification
```python
def predict_disease(
    processed_image: torch.Tensor,
    return_embedding: bool = True
) -> Dict[str, Any]:
    ...
```

### Output Schema
```json
{
  "plant": "Tomato",
  "disease": "Early Blight",
  "canonical_id": "tomato_early_blight",
  "confidence": 0.9421,
  "status": "supported",
  "raw_label": "Tomato___Early_blight",
  "embedding": [0.024, -0.115, 0.842]
}
```

### Supported Status Values
- `"supported"`: Confidence exceeds calibrated threshold \(\tau\); prediction belongs to a trained dataset class.
- `"uncertain"`: Confidence falls below threshold \(\tau\); model is hesitant.
- `"unknown"`: Sample detected as an unknown/out-of-distribution plant disease.
- `"not_a_plant"`: Input image rejected by the plant validation filter.

*Note: Confidence thresholds will be determined experimentally during evaluation.*

---

## 2. RAG / Knowledge Base Contract (`src/retrieval/`)

### Responsibility
Receives prediction metadata containing the canonical disease ID, performs semantic and key-based vector lookup against the agricultural knowledge store, and returns structured pathology context.

### Input Schema
```json
{
  "plant": "Tomato",
  "disease": "Early Blight",
  "canonical_id": "tomato_early_blight"
}
```

### Output Schema
```json
{
  "canonical_id": "tomato_early_blight",
  "symptoms": [
    "Dark brown spots with concentric rings on lower leaves",
    "Yellowing of leaf tissue surrounding spots",
    "Premature leaf drop leading to sunscald on fruit"
  ],
  "causes": [
    "Fungal pathogen Alternaria solani",
    "Spores spread by rain splash, wind, and contaminated tools"
  ],
  "risk_factors": [
    "High humidity and warm temperatures (24-29°C)",
    "Extended leaf wetness periods"
  ],
  "prevention": [
    "Rotate crops with non-solanaceous crops every 2-3 years",
    "Mulch beneath plants to reduce rain splash from soil"
  ],
  "management": [
    "Apply copper-based or chlorothalonil fungicides at early onset",
    "Prune lower infected leaves to improve air circulation"
  ],
  "sources": [
    "USDA Agricultural Extension Publication No. 402",
    "Global Plant Pathology Database - Alternaria Solani Guide"
  ]
}
```

---

## 3. Advisory Generator Contract (`src/advisory/`)

### Responsibility
Synthesizes the Vision prediction object and retrieved structured knowledge snippets into user-ready agricultural advice.

### Input Parameters
- `prediction_object`: Output dictionary from Vision Module.
- `knowledge_object`: Output dictionary from RAG Module.

### Output Schema
```json
{
  "headline": "Tomato Early Blight Detected (94.2% Confidence)",
  "canonical_id": "tomato_early_blight",
  "status": "supported",
  "symptoms_and_causes": "The image shows characteristic symptoms of Early Blight caused by the fungus Alternaria solani...",
  "prevention_guidance": "Implement crop rotation with non-nightshade crops and mulch soil surfaces...",
  "management_actions": "Remove lower infected foliage immediately and apply approved copper-based spray...",
  "reference_sources": [
    "USDA Agricultural Extension Publication No. 402"
  ]
}
```

---

## 4. System Integration Pipeline Contract (`src/integration/`)

### High-Level Dataflow Chain
```text
Image Input
  ↓
Vision Module (Output: Prediction Object)
  ↓
Canonical Disease ID ("tomato_early_blight")
  ↓
RAG Knowledge Module (Output: Structured Pathology Snippets)
  ↓
Advisory Module (Output: Structured Advisory Object)
  ↓
Integration Response Output
```

---

## 5. Preprocessing & Evaluation Contracts

### 5.1 Preprocessing Module (`src/preprocessing/`)
- **Input**: `image` (`PIL.Image` or `np.ndarray`), `target_size: Tuple[int, int] = (224, 224)`
- **Output**: `processed_image: torch.Tensor` (Shape: `[3, 224, 224]`, normalized via standard ImageNet mean/std)

### 5.2 Evaluation Module (`src/evaluation/`)
- **Input**: `predictions: List[str]`, `ground_truth: List[str]`, `latencies: List[float]`
- **Output**: `metrics: Dict[str, float]` (`accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`, `mean_latency_ms`)
