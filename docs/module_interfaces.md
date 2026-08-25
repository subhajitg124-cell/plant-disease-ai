# Module Interface Specifications

> [!NOTE]
> **Specification Notice**: Phase 2 defines exact data contracts, inputs, outputs, and canonical JSON schemas governing inter-module communication. Comprehensive specs are maintained in [`docs/architecture/module_interfaces.md`](architecture/module_interfaces.md).

---

## 1. Overview

To ensure smooth system integration across vision, embedding, retrieval, advisory, evaluation, and adaptation workflows, all modules must strictly adhere to the input/output signatures defined below.

---

## 2. Interface Definitions

### 2.1 Preprocessing Module (`src/preprocessing/`)

**Responsibility**: Converts raw images or video frame inputs into normalized tensor/image representations suitable for model inference.

- **Input**:
  - `image`: `PIL.Image`, `numpy.ndarray`, or file path string.
  - `target_size`: `Tuple[int, int]` (Default: `(224, 224)`).
- **Output**:
  - `processed_image`: `torch.Tensor` or `numpy.ndarray` (normalized, standard shape `[C, H, W]`).

---

### 2.2 Vision Module (`src/vision/`)

**Responsibility**: Performs convolutional neural network classification and joint embedding feature extraction.

- **Input**:
  - `image`: Processed image tensor/frame from Preprocessing module.
  - `return_embedding`: `bool` (Default: `True`).
- **Output**:
  ```json
  {
      "disease": "Tomato___Bacterial_spot",
      "confidence": 0.964,
      "embedding": [0.024, -0.115, 0.842, "... (vector of size D)"]
  }
  ```

---

### 2.3 Embedding Module (`src/embeddings/`)

**Responsibility**: Extracts dense visual embeddings for similarity search, clustering, and open-set recognition.

- **Input**:
  - `processed_image`: Processed image tensor.
- **Output**:
  - `embedding`: `List[float]` or `numpy.ndarray` (numerical embedding vector, e.g., dimension 512 or 768).

---

### 2.4 Retrieval Module (`src/retrieval/`)

**Responsibility**: Queries the agricultural knowledge vector store for domain-specific context.

- **Input**:
  - `query`: String (disease name e.g., `"Tomato___Bacterial_spot"`) **OR** `query_embedding` vector.
  - `top_k`: `int` (Default: `3`).
- **Output**:
  - `retrieved_documents`: `List[Dict]` containing top-k relevant knowledge documents:
    ```json
    [
        {
            "doc_id": "doc_104",
            "disease": "Tomato___Bacterial_spot",
            "content": "Copper-based fungicides applied at first sign of disease help control bacterial spot...",
            "score": 0.892
        }
    ]
    ```

---

### 2.5 Advisory Module (`src/advisory/`)

**Responsibility**: Generates actionable, grounded agricultural recommendations using retrieved knowledge context and disease predictions.

- **Input**:
  - `disease`: String (predicted disease name).
  - `retrieved_context`: `List[Dict]` (documents from Retrieval module).
- **Output**:
  ```json
  {
      "treatment": "Apply copper sprays every 7-10 days during warm, moist weather.",
      "prevention": "Rotate crops every 2-3 years, avoid overhead irrigation, and use certified disease-free seeds.",
      "care": "Monitor bottom leaves daily for small dark spots; remove infected plant debris promptly."
  }
  ```

---

### 2.6 Evaluation Module (`src/evaluation/`)

**Responsibility**: Computes model performance metrics and benchmark statistics.

- **Input**:
  - `predictions`: `List[str]` (predicted disease labels).
  - `ground_truth`: `List[str]` (true target labels).
  - `inference_times`: `List[float]` (per-sample inference latencies in seconds).
- **Output**:
  - `metrics`: Dictionary containing:
    - `accuracy`: `float`
    - `precision`: `float`
    - `recall`: `float`
    - `f1_score`: `float`
    - `confusion_matrix`: `List[List[int]]`
    - `mean_inference_time_ms`: `float`

---

### 2.7 Adaptation Module (`src/embeddings/` / `src/vision/`)

**Responsibility**: Adapts the classification index using few-shot samples or embedding similarity for unseen plant datasets.

- **Input**:
  - `new_dataset_samples`: Few-shot image samples from new/unseen dataset.
  - `sample_labels`: Class labels corresponding to few-shot samples.
  - `embeddings`: Visual embeddings extracted from few-shot samples.
- **Output**:
  - `adapted_index`: Updated class index map and similarity baseline matching structure.
