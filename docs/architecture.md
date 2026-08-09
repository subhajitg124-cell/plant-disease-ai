# Architecture & Design Specification

> [!NOTE]
> **Implementation Status Notice**: This document outlines the Phase-1 architectural foundation for `plant-disease-ai`. All model architectures, vector databases, RAG pipelines, and LLM integrations are **[PLANNED]** and explicitly marked as such. The current repository state is in **Phase 1: Project Foundation & Architecture**.

---

## 1. System Overview

The **Plant Disease AI** system is designed as a generalized, embedding-driven plant disease detection and advisory system. Beyond static closed-set classification, it incorporates visual embeddings, few-shot adaptation for unseen datasets, retrieval-augmented generation (RAG), and LLM reasoning to deliver actionable treatment, prevention, and care guidance to farmers and agricultural experts.

---

## 2. System Dataflow Pipelines

### 2.1 Primary Disease Detection & Advisory Pipeline [PLANNED]

```
IMAGE / VIDEO
      ↓
Data Ingestion
      ↓
Image / Video Preprocessing
      ↓
Vision CNN
      ↓
Disease Classification
      +
Visual Embedding Extraction
      ↓
Disease Identification / Similarity Analysis
      ↓
Knowledge Retrieval / Vector Database
      ↓
Retrieved Agricultural Knowledge
      ↓
LLM / AI Reasoning
      ↓
Treatment + Prevention + Care Advisory
```

### 2.2 Unseen-Dataset Adaptation Pipeline [PLANNED]

```
NEW / UNSEEN DATASET
      ↓
Dataset Analysis
      ↓
Preprocessing
      ↓
Visual Embedding Extraction
      ↓
Few-Shot / Embedding-Based Adaptation
      ↓
Class Index / Similarity Matching
      ↓
Disease Identification
      ↓
RAG Retrieval
      ↓
Advisory Generation
```

---

## 3. Core Module Breakdown

| Module | Status | Description & Responsibilities |
| :--- | :--- | :--- |
| **1. Data Ingestion & Preprocessing** | `[PLANNED]` | Ingests single images or video streams, performs resizing, normalization, color space alignment, and frame extraction for video input. |
| **2. Vision Engine / CNN** | `[PLANNED]` | Executes convolutional backbone inference to extract spatial features and compute closed-set disease class probabilities. |
| **3. Visual Embeddings** | `[PLANNED]` | Extracts high-dimensional deep feature representations (embeddings) from penultimate CNN layers for similarity search and open-set recognition. |
| **4. Knowledge Base / Vector Database** | `[PLANNED]` | Stores structured agricultural treatment guides, pathology facts, and preventative measures indexed by vector embeddings. |
| **5. Retrieval / RAG** | `[PLANNED]` | Performs semantic search against the knowledge vector database given predicted disease labels or query embeddings to retrieve top-k context snippets. |
| **6. Advisory Generator** | `[PLANNED]` | Combines retrieved knowledge snippets with vision predictions in a prompt template for an LLM to generate clear treatment, prevention, and care advice. |
| **7. Adaptation Pipeline** | `[PLANNED]` | Facilitates few-shot or embedding-based adaptation on unseen plant species/diseases without full model retraining. |
| **8. Evaluation** | `[PLANNED]` | Measures system performance across classification metrics (accuracy, precision, recall, F1, confusion matrix) and system benchmarks (inference latency). |

---

## 4. Module Ownership

### Subhajit
- AI Architecture
- Technical Direction
- Module Interface Definition
- Final Technical Review

### Tohidur
- Computer Vision
- Image/Video Processing
- CNN Training

### Saiyab
- Knowledge Base
- RAG Retrieval
- Advisory Generation

### Asikul
- System Integration
- Evaluation
- Performance Testing

### Nazid
- Dataset Management
- Few-Shot Adaptation
- Documentation and Testing

---

## 5. Technology Stack Specification

| Component Layer | Technology | Implementation Status |
| :--- | :--- | :--- |
| Core Language | **Python 3.10+** | `[CURRENTLY IMPLEMENTED]` |
| Deep Learning Framework | **PyTorch (torch)** | `[CURRENTLY INSTALLED]` |
| Vision Library | **TorchVision** | `[CURRENTLY INSTALLED]` |
| Image/Video Processing | **OpenCV (cv2) & Pillow (PIL)** | `[CURRENTLY INSTALLED]` |
| Data & Metrics Utilities | **NumPy, pandas, scikit-learn, matplotlib** | `[CURRENTLY INSTALLED]` |
| Model Architectures / Backbones | **HuggingFace Transformers** | `[PLANNED]` |
| Vector Database | **FAISS and/or ChromaDB** | `[PLANNED]` |
| RAG Orchestration | **LangChain and/or LlamaIndex** | `[PLANNED]` |
