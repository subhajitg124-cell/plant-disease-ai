# Plant Disease AI

> **Status**: **Phase 1 — Project Foundation & Architecture** (`[IN PROGRESS]`)

---

## Project Overview

**Plant Disease AI** is a generalized AI-driven plant disease detection and agricultural advisory system. By combining computer vision, visual embeddings, few-shot adaptation, vector-based retrieval augmented generation (RAG), and large language model (LLM) reasoning, the system aims to provide accurate disease identification alongside actionable, grounded treatment and care recommendations for farmers and agricultural domain experts.

---

## Problem Statement

Conventional deep learning plant disease classifiers rely heavily on static, closed-set datasets. When deployed in real-world scenarios or introduced to unseen crop varieties and changing environmental conditions, fixed classifiers struggle to generalize without extensive model retraining. Furthermore, standard classifiers only output disease labels without providing contextual guidance on treatment, preventative care, or risk mitigation.

---

## Objectives

- **Image and Video Processing**: Support multi-modal inputs, including static images and continuous video frame streams.
- **CNN-Based Disease Classification**: Train robust convolutional models for baseline disease recognition.
- **Visual Embedding Extraction**: Extract high-dimensional visual feature vectors to enable similarity matching and open-set recognition.
- **Few-Shot / Embedding-Based Adaptation**: Enable rapid adaptation to unseen plant species and new disease datasets without full model retraining.
- **RAG-Based Knowledge Retrieval**: Retrieve verified, domain-specific agricultural literature and care guides using vector similarity.
- **Grounded Advisory Generation**: Leverage AI reasoning to produce clear, structured recommendations covering **Treatment**, **Prevention**, and **Care**.
- **Comprehensive System Evaluation**: Evaluate model accuracy, generalization on unseen test sets, and real-time inference latency.

---

## Architecture

The system operates across two key execution paths:
1. **Primary Inference & Advisory Path**: Ingestion → Preprocessing → CNN Classification & Embedding Extraction → Vector Retrieval → LLM Advisory Generation.
2. **Unseen-Dataset Adaptation Path**: Unseen Dataset Ingestion → Feature Embedding Extraction → Few-Shot Similarity Adaptation → Class Index Updating → Retrieval & Advisory.

For detailed architecture diagrams, pipeline breakdowns, and component specifications, refer to [docs/architecture.md](docs/architecture.md).

For inter-module data contracts and interface definitions, refer to [docs/module_interfaces.md](docs/module_interfaces.md).

---

## Project Structure

```
plant-disease-ai/
├── data/
│   ├── external/       # External datasets and reference benchmarks
│   ├── processed/      # Normalized, transformed, and augmented data
│   └── raw/            # Raw un-processed image datasets
├── docs/               # Architecture design & interface specifications
│   ├── architecture.md
│   └── module_interfaces.md
├── models/             # Saved model checkpoints, weights, and indexes
├── notebooks/          # Exploratory data analysis & experiment notebooks
├── reports/            # Performance reports, logs, and benchmark outputs
├── scripts/            # Utility and diagnostic scripts
│   └── verify_env.py   # Environment & GPU verification script
├── src/                # Core application source code
│   ├── advisory/       # LLM prompt templates and advisory generation
│   ├── embeddings/     # Deep feature extraction & similarity indexes
│   ├── evaluation/     # Metrics calculation & benchmark suites
│   ├── preprocessing/  # Image/video frame preprocessing & transforms
│   ├── retrieval/      # Vector database indexing & RAG search
│   └── vision/         # CNN backbones & disease classification heads
├── tests/              # Unit and integration test suites
├── .gitignore          # Repository ignore rules
├── README.md           # Project documentation root
└── requirements.txt    # Python dependencies manifest
```

---

## Technology Stack

### Currently Installed & Active
- **Python 3.10+** (Core Runtime)
- **PyTorch & TorchVision** (Deep Learning & Computer Vision)
- **OpenCV & Pillow** (Image & Video Processing)
- **NumPy, pandas, scikit-learn, matplotlib** (Data Processing & Evaluation Metrics)

### Planned (Future Phases)
- **HuggingFace Transformers** (Vision & Language Models)
- **FAISS / ChromaDB** (Vector Similarity Search)
- **LangChain / LlamaIndex** (RAG Orchestration & Knowledge Retrieval)

---

## Team & Roles

- **Subhajit** — AI Architecture, Technical Direction, Interface Definition & Final Review
- **Tohidur** — Computer Vision, Image/Video Processing & CNN Training
- **Saiyab** — Knowledge Base, RAG Retrieval & Advisory Generation
- **Asikul** — System Integration, Evaluation & Performance Testing
- **Nazid** — Dataset Management, Few-Shot Adaptation & Documentation/Testing

---

## Dataset Policy

- Public agricultural datasets (e.g., PlantVillage) will be used for initial experimentation and baseline training during development.
- Final evaluation will be conducted on unseen datasets provided by the project evaluation committee.
- **Note**: Raw datasets are excluded from Git tracking via `.gitignore` and are not committed directly to the repository.

---

## Environment Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

- **Windows (PowerShell)**:
  ```powershell
  \.venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Verification

Verify that all required ML/CV packages and GPU/CUDA acceleration are properly configured by running:

```bash
python scripts/verify_env.py
```

Expected output:

```text
============================================================
Package / Component       | Status     | Version
------------------------------------------------------------
Python                    | OK         | 3.14.0a4
PyTorch (torch)           | OK         | 2.13.0+cu126
torchvision               | OK         | 0.28.0+cu126
OpenCV (cv2)              | OK         | 5.0.0.93
Pillow (PIL)              | OK         | 12.3.0
NumPy (numpy)             | OK         | 2.5.1
pandas                    | OK         | 3.0.5
matplotlib                | OK         | 3.11.1
scikit-learn (sklearn)    | OK         | 1.9.0
============================================================
CUDA Available for PyTorch: True
CUDA Device Name: NVIDIA GeForce RTX 3050 Laptop GPU
============================================================
```

---

## Development Roadmap

- [x] **Phase 1: Architecture & Data Sourcing Foundation** *(Current)*
  - Repository structure setup & tracking controls
  - System architecture & interface contract specification
  - Environment verification & setup documentation
- [ ] **Phase 2: Preprocessing & Vision Baseline Engine**
  - Image/video processing pipelines
  - Baseline CNN classifier training & embedding extraction
- [ ] **Phase 3: RAG Knowledge Base & Advisory Pipeline**
  - Vector database indexing for agricultural guides
  - Semantic retrieval & LLM advisory prompt engineering
- [ ] **Phase 4: Few-Shot Adaptation & Open-Set Generalization**
  - Embedding adaptation on unseen plant species/datasets
- [ ] **Phase 5: Integration, System Evaluation & Testing**
  - End-to-end integration and metric benchmarking on unseen test sets

---

## Current Status

**Current Phase**: `Phase 1 — Project Foundation & Architecture`.
All pipeline designs and module interfaces are defined. Model implementations, training pipelines, vector databases, and UI interfaces will be developed sequentially in subsequent phases.
