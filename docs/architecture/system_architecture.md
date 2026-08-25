# System Architecture Specification

> **Status**: **Phase 2 — Preprocessing & Multi-Dataset Taxonomy Engine** (`[IN PROGRESS]`)

---

## 1. Executive Summary & Objective

The **Plant Disease AI** system is an extensible computer vision and knowledge-retrieval platform. Given an input plant image, the system validates plant presence, identifies the crop species and disease status, computes a prediction confidence score, retrieves grounded pathology insights from a verified agricultural knowledge base, and outputs structured symptom analysis, cause explanations, and management/prevention guidance.

---

## 2. Multi-Dataset Architecture Design

To support heterogeneous datasets (PlantVillage, PlantDoc, Plant Pathology 2021, and unseen evaluation sets) without re-engineering core downstream components, label translation relies on a decoupled, canonical taxonomy pipeline:

```text
Dataset-Specific Label (e.g., PlantVillage "Tomato___Early_blight" / PlantDoc "Tomato_Early_Blight")
        ↓
Dataset-Specific Mapping (data/metadata/plantvillage_class_mapping.csv)
        ↓
Canonical Disease ID (e.g., "tomato_early_blight")
        ↓
Unified Project Taxonomy (data/metadata/class_mapping.csv)
        ↓
CNN Training Classes / Embeddings / RAG Knowledge Store
```

### Taxonomy Files Distinction
- **`data/metadata/plantvillage_class_mapping.csv`**: Holds dataset-specific mappings preserving original dataset directory labels (`original_label`), class indices (`class_id`), and image counts.
- **`data/metadata/class_mapping.csv`**: Serves as the single source of truth for the project's unified canonical disease taxonomy across all integrated datasets.

---

## 3. End-to-End System Pipelines

### 3.1 Primary Inference & Advisory Pipeline

```text
User Image Input
       ↓
Plant Validation (Is it a plant image?)
       ↓ [Yes]
Image Preprocessing (Resizing, Normalization, Tensor Transform)
       ↓
CNN / Vision Model Inference
       ↓
Prediction Object (Plant, Disease, Confidence, Status)
       ↓
Canonical Disease ID Mapping ("tomato_early_blight")
       ↓
Knowledge Retrieval / Vector RAG Query
       ↓
Retrieved Grounded Pathology Documents (Symptoms, Causes, Risk Factors)
       ↓
Advisory Generator & LLM Reasoning
       ↓
Final Structured Advisory Response (Symptoms, Causes, Prevention, Management, Sources)
```

### 3.2 Unknown & Low-Confidence Handling Logic

The system must not generate confident predictions for non-plant images or out-of-distribution diseased samples:

```text
Input Image
       ↓
Plant Validation Filter
       ↓ 
[Not A Plant] → Return Status: "not_a_plant" (Rejection)
       ↓ [Valid Plant]
Vision Disease Classifier
       ↓
Confidence Evaluation against Calibrated Threshold (τ)
       ├── Confidence >= τ → Status: "supported" (Pass to RAG & Advisory)
       └── Confidence < τ  → Status: "uncertain" / "unknown" (Fallback guidance & re-sampling prompt)
```

*Note: Confidence threshold \(\tau\) will be empirically calibrated during validation and evaluation phases.*

### 3.3 Video Input Pipeline (FUTURE EXTENSION)

```text
Video File / Stream [FUTURE EXTENSION]
       ↓
Frame Extraction & Keyframe Selection
       ↓
Image Preprocessing
       ↓
Vision Classifier & Embedding Extraction
       ↓
Temporal Aggregation & Disease Detection Output
```

*Note: Video processing, frame extraction, text-only datasets, and multimodal video understanding are explicitly designated as future extensions.*

---

## 4. Dataset Imbalance & Mitigation Strategy

Inspection of the baseline PlantVillage dataset (54,305 images across 38 classes) reveals significant class imbalance:

- **Dominant / Large Classes**:
  - `Orange___Haunglongbing_(Citrus_greening)`: 5,507 images
  - `Tomato___Tomato_Yellow_Leaf_Curl_Virus`: 5,357 images
  - `Soybean___healthy`: 5,090 images
- **Underrepresented / Small Classes**:
  - `Potato___healthy`: 152 images
  - `Apple___Cedar_apple_rust`: 275 images
  - `Peach___healthy`: 360 images

### Mitigation Protocol for Phase 2 & 3:
- Classes will **NOT** be blindly deleted or manually pruned.
- The team will evaluate the following strategies during data preprocessing and training:
  1. Focal Loss and Weighted Cross-Entropy loss functions.
  2. Stratified data splitting for train/validation/test sets.
  3. Class-aware data augmentation (mixup, cutmix, rotation, color jittering).
  4. Balanced mini-batch sampling and oversampling of minority classes.

---

## 5. Agricultural Advisory Safety & Knowledge Grounding Rules

> [!IMPORTANT]
> **Safety Directive**: Large Language Models (LLMs) must **NEVER** hallucinate or invent unverified chemical treatments or agricultural care steps. All advisory outputs must be strictly grounded in retrieved evidence from validated agricultural extension sources, pathology databases, and peer-reviewed guides retrieved by the RAG module.

---

## 6. Team Ownership & Responsibilities

- **Subhajit**: AI Architecture, Technical Direction, Interface Definition, Module Review.
- **Tohidur**: Computer Vision, Preprocessing Pipelines, Baseline CNN Classifier Training.
- **Saiyab**: Knowledge Base Indexing, RAG Retrieval, Advisory Generation.
- **Asikul**: System Integration, Benchmark Metrics, Latency & Evaluation Testing.
- **Nazid**: Multi-Dataset Metadata Management, Few-Shot Generalization, Documentation.
