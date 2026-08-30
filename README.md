# Plant Disease Detection

> **Status**: **Phase 2 — Multi-Dataset Taxonomy & Architecture Specifications** (`[IN PROGRESS]`)

---

## Project Overview

**Plant Disease Detection** is a generalized AI-driven plant disease detection and agricultural advisory system. By combining computer vision, visual embeddings, few-shot adaptation, vector-based retrieval augmented generation (RAG), and large language model (LLM) reasoning, the system aims to provide accurate disease identification alongside actionable, grounded treatment and care recommendations for farmers and agricultural domain experts.

---

## Problem Statement

Conventional deep learning plant disease classifiers rely heavily on static, closed-set datasets. When deployed in real-world scenarios or introduced to unseen crop varieties and changing environm[...]

---

## Objectives

- **Image and Video Processing**: Support multi-modal inputs, including static images and continuous video frame streams (future extension).
- **CNN-Based Disease Classification**: Train robust convolutional models for baseline disease recognition.
- **Canonical Multi-Dataset Taxonomy**: Translate dataset-specific class labels (PlantVillage, PlantDoc, Plant Pathology) into unified, machine-friendly canonical disease IDs.
- **Visual Embedding Extraction**: Extract high-dimensional visual feature vectors to enable similarity matching and open-set recognition.
- **Few-Shot / Embedding-Based Adaptation**: Enable rapid adaptation to unseen plant species and new disease datasets without full model retraining.
- **RAG-Based Knowledge Retrieval**: Retrieve verified, domain-specific agricultural literature and care guides using vector similarity.
- **Grounded Advisory Generation**: Leverage AI reasoning to produce clear, structured recommendations covering **Treatment**, **Prevention**, and **Care**.
- **Comprehensive System Evaluation**: Evaluate model accuracy, generalization on unseen test sets, and real-time inference latency.

---

## Architecture & Dataflow

The system operates across key execution paths:
1. **Primary Inference & Advisory Path**: User Image Ingestion → Plant Validation → Image Preprocessing → CNN Classification & Embedding Extraction → Canonical Disease ID Translation → V[...]
2. **Unseen-Dataset Adaptation Path**: Unseen Dataset Ingestion → Feature Embedding Extraction → Few-Shot Similarity Adaptation → Class Index Updating → Retrieval & Advisory.

For detailed architecture diagrams, pipeline breakdowns, and component specifications, refer to [docs/architecture/system_architecture.md](docs/architecture/system_architecture.md).

For inter-module data contracts and interface definitions, refer to [docs/architecture/module_interfaces.md](docs/architecture/module_interfaces.md).

---

## Project Structure

```
plant-disease-ai/
├── data/
│   ├── external/       # External datasets and reference benchmarks
│   ├── metadata/       # Canonical class taxonomies & dataset mappings
│   │   ├── class_mapping.csv
│   │   └── plantvillage_class_mapping.csv
│   ├── processed/      # Normalized, transformed, and augmented data
│   └── raw/            # Raw un-processed image datasets
├── docs/               # Architecture design & interface specifications
│   ├── architecture/
│   │   ├── module_interfaces.md
│   │   └── system_architecture.md
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
│   └── test_taxonomy_mapping.py  # Taxonomy verification tests
├── .gitignore          # Repository ignore rules
├── README.md           # Project documentation root
└── requirements.txt    # Python dependencies manifest
```

---

## Dataset Status (PlantVillage Baseline)

- **Dataset**: PlantVillage
- **Total Images**: 54,305
- **Total Classes**: 38 (0..37)
- **Status**: Inspected & Mapped (`data/metadata/plantvillage_class_mapping.csv`)
- **Imbalance Notes**: Class frequencies range from 5,507 images (`Orange___Haunglongbing_(Citrus_greening)`) to 152 images (`Potato___healthy`). Class-weighted loss and balanced sampling strategi[...]

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

## Dataset Policy & Storage Architecture

- **GitHub Repository**: Source code, metadata CSVs, tests, and documentation.
- **Google Drive**: Large raw datasets, processed feature caches, and model weight checkpoints.
- **Google Colab**: High-performance GPU compute workspace for dataset exploration and training.
- **Note**: Raw dataset files are excluded from Git tracking via `.gitignore`. Dataset storage paths are configurable via environment variables and must not be hardcoded to machine-specific local[...]

---

## Environment Setup & Verification

### 1. Activate Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  \.venv\Scripts\Activate.ps1
  ```

### 2. Run Diagnostics
```bash
python scripts/verify_env.py
```

### 3. Run Taxonomy Verification Tests
```bash
python -m unittest tests/test_taxonomy_mapping.py
```

---

## Development Roadmap

- [x] **Phase 1: Architecture & Data Sourcing Foundation**
  - Repository structure setup & tracking controls
  - Baseline architecture & interface specification
- [/] **Phase 2: Multi-Dataset Taxonomy & Vision Baseline Engine** *(Current)*
  - [x] Multi-dataset canonical class taxonomy design & CSV mapping (`class_mapping.csv`)
  - [x] Detailed architecture & interface specifications (`docs/architecture/`)
  - [x] Automated taxonomy verification test suite (`tests/test_taxonomy_mapping.py`)
  - [ ] Image preprocessing & augmentation pipelines
  - [ ] Baseline CNN classifier training & embedding extraction
- [ ] **Phase 3: RAG Knowledge Base & Advisory Pipeline**
  - Vector database indexing for agricultural guides
  - Semantic retrieval & LLM advisory prompt engineering
- [ ] **Phase 4: Few-Shot Adaptation & Open-Set Generalization**
  - Embedding adaptation on unseen plant species/datasets
- [ ] **Phase 5: Integration, System Evaluation & Testing**
  - End-to-end integration and metric benchmarking on unseen test sets
