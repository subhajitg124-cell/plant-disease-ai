# Plant Disease AI — Project Progress Report

**Project:** Plant Disease AI

**Report Type:** Project Progress Report — Phase 2

**Reporting Period:** Phase 2 — Multi-Dataset Taxonomy & Architecture Foundation

**Repository:** `subhajitg124-cell/plant-disease-ai`

**Status:** Phase 2 foundation completed and verified; AI model training, production RAG, and end-to-end evaluation remain in subsequent phases.

---

## 1. Abstract

Plant Disease AI is being developed as a modular plant disease detection and agricultural advisory system. The proposed system combines computer vision, a canonical multi-dataset disease taxonomy, retrieval-augmented generation (RAG), and AI-assisted reasoning to move beyond simple disease classification toward structured, evidence-grounded agricultural guidance.

During Phase 2, the team concentrated on establishing a reliable and extensible software and data foundation. A canonical taxonomy was created for the PlantVillage baseline, inter-module data contracts were implemented, system and interface specifications were documented, and automated validation tests were developed. The current architecture is designed so that the Vision, Knowledge/RAG, and Integration modules can evolve independently while communicating through stable typed contracts.

The verified PlantVillage metadata contains 38 classes and 54,305 images. The taxonomy and interface test suites provide automated checks for class counts, canonical identifiers, required metadata, confidence validation, prediction statuses, serialization, and integration behavior. The Phase 2 test suite currently passes all 16 tests.

> **Scope note:** Phase 2 established the architecture, taxonomy, contracts, documentation, and validation framework. Production CNN training/inference, authoritative RAG indexing, and final end-to-end performance evaluation are planned for the following implementation phases and are not claimed as completed in this report.

---

## 2. Problem Statement

Conventional plant disease classifiers generally operate as closed-set image classification systems. They may return a disease label but do not inherently provide standardized cross-dataset disease identifiers, evidence-grounded explanations, or safe handling for non-plant, uncertain, and previously unsupported inputs.

The project addresses these limitations through a modular architecture in which:

1. Input images are validated before disease inference.
2. Dataset-specific labels are translated into standardized canonical disease IDs.
3. A Vision module produces a structured prediction with confidence and operational status.
4. A future RAG module retrieves relevant agricultural evidence using the canonical disease ID.
5. An Integration module combines prediction and retrieved evidence into a structured advisory response.
6. Low-confidence, unknown, and non-plant inputs are handled explicitly rather than being treated as normal disease predictions.

---

## 3. Project Objectives

The overall project aims to:

- Develop a CNN-based plant disease recognition baseline.
- Establish a canonical taxonomy across heterogeneous agricultural datasets.
- Preserve original dataset labels while providing machine-friendly canonical identifiers.
- Support confidence-aware and open-set-aware prediction handling.
- Extract visual embeddings as a future extension for similarity matching and few-shot adaptation.
- Build a verified agricultural knowledge base and RAG retrieval pipeline.
- Generate structured guidance covering symptoms, causes, risk factors, prevention, and management.
- Integrate the Vision and RAG components through stable software contracts.
- Evaluate classification accuracy, generalization, and inference latency.

---

## 4. Team Members and Responsibilities

| Team Member | Module / Area | Responsibilities |
|---|---|---|
| **Subhajit — Team Lead** | AI Architecture & System Design | System dataflow, canonical taxonomy definition, inter-module contracts, technical review and coordination. |
| **Tohidur** | Vision Engine | Image preprocessing, CNN classifier development/training, and future visual feature extraction. |
| **Saiyab** | Knowledge & RAG | Agricultural knowledge-base preparation, vector indexing, semantic retrieval, and structured advisory generation. |
| **Asikul** | Integration & Evaluation | End-to-end pipeline assembly, benchmark design, accuracy/precision/recall measurement, and latency evaluation. |
| **Nazid** | Metadata & Generalization | Dataset mappings, multi-dataset metadata, few-shot/generalization support, documentation and testing assistance. |

The ownership boundaries are intentionally modular so that changes to one component do not require redesigning the entire system.

---

## 5. Work Completed During Phase 2

### 5.1 Canonical Multi-Dataset Taxonomy

Different datasets may represent the same disease using different label formats. For example, a dataset-specific label such as `Tomato___Early_blight` is translated into the canonical identifier `tomato_early_blight`.

The taxonomy pipeline is:

```text
Dataset-Specific Label
        ↓
Dataset-Specific Mapping
        ↓
Canonical Disease ID
        ↓
Unified Project Taxonomy
        ↓
Vision Classes / Future Embeddings / RAG Knowledge Store
```

Two metadata files were established:

- `data/metadata/plantvillage_class_mapping.csv` — PlantVillage-specific mapping that preserves the original label, class ID, disease, health status, source dataset, and image count.
- `data/metadata/class_mapping.csv` — unified project taxonomy intended to serve as the single source of truth as additional datasets are integrated.

The original dataset labels and raw images remain unchanged.

### 5.2 Verified PlantVillage Baseline

| Property | Verified Value |
|---|---:|
| Dataset | PlantVillage |
| Total images | **54,305** |
| Total classes | **38** |
| Class IDs | **0–37** |
| Duplicate canonical IDs | **0** |
| Missing canonical IDs | **0** |
| Source-dataset consistency | **PlantVillage** |

The mapping also records class imbalance. Examples include Orange Huanglongbing with 5,507 images and Potato Healthy with 152 images. This imbalance has been documented as a consideration for subsequent preprocessing and model-training strategies.

### 5.3 Inter-Module Data Contracts

The team implemented `src/contracts.py` containing typed dataclass contracts for communication among the Vision, RAG, and Integration layers.

#### VisionPrediction

The Vision module produces:

- `plant`
- `disease`
- `canonical_id`
- `confidence`
- `status`
- optional `raw_label`
- optional `model_version`
- optional `embedding`

The `embedding` field is explicitly an optional future extension and is not required by the Phase 2 core Vision contract.

#### RAGQueryInput

The RAG module receives:

- `plant`
- `disease`
- `canonical_id`

#### AdvisoryResult

The planned RAG output is structured into:

- symptoms
- causes
- risk factors
- prevention
- management
- sources

During Phase 2, advisory text in tests and interface examples is demonstration/schema data only. Production authoritative agricultural evidence will be indexed during the RAG implementation phase.

#### IntegratedResponse

The Integration layer combines the Vision prediction and optional advisory into a final structured response containing:

- prediction
- advisory
- user message
- confidence
- status
- evidence
- sources
- warnings

---

## 6. Confidence and Safety-Oriented Status Handling

Four operational prediction states have been defined:

| Status | Meaning | System Behavior |
|---|---|---|
| `supported` | Recognized class with confidence meeting the operational threshold | Continue to RAG/advisory retrieval. |
| `uncertain` | Prediction confidence is below the operational threshold | Suppress advisory and request a clearer image. |
| `unknown` | Valid plant image but disease is unsupported/out-of-distribution | Do not generate a normal disease advisory. |
| `not_a_plant` | Input fails plant validation | Reject the input and return a validation message. |

The confidence threshold `τ` is intentionally not hardcoded in the contract. It will be calibrated experimentally during model validation and evaluation.

This design prevents the downstream advisory system from treating every Vision output as a reliable diagnosis.

---

## 7. System Architecture

The planned primary inference and advisory flow is:

```text
User Image Input
        ↓
Plant Validation Filter
        ↓
Image Preprocessing
        ↓
CNN / Vision Inference
        ↓
VisionPrediction
        ↓
Canonical Disease ID
        ↓
RAG / Vector Knowledge Retrieval
        ↓
Retrieved Agricultural Evidence
        ↓
Advisory Generation / LLM Reasoning
        ↓
IntegratedResponse
        ↓
Structured User Guidance
```

The architecture also defines explicit fallback behavior for non-plant, uncertain, and unsupported inputs.

### Architectural Safety Rule

Advisory generation is intended to be grounded in retrieved agricultural evidence. The LLM must not invent unverified chemical treatments or agricultural care instructions. Production advisory content will therefore be tied to validated agricultural extension, pathology, and other authoritative sources during the RAG implementation phase.

---

## 8. Documentation Completed

The following project documentation has been established:

| File | Purpose |
|---|---|
| `docs/architecture/system_architecture.md` | System architecture, dataflow, fallback handling, dataset imbalance considerations, and safety rules. |
| `docs/architecture/module_interfaces.md` | Formal module contracts, status definitions, Python schemas, and JSON examples. |
| `src/contracts.py` | Executable typed dataclass contracts and validation logic. |
| `README.md` | Project overview, objectives, structure, technology stack, dataset status, team roles, and roadmap. |

These documents provide a common technical reference for all team members.

---

## 9. Automated Testing and Verification

Two dedicated test files were implemented:

### `tests/test_taxonomy_mapping.py`

The suite validates:

- mapping files exist;
- exactly 38 PlantVillage classes are present;
- canonical IDs are non-empty and unique;
- class IDs are unique and cover 0–37;
- required metadata fields are populated;
- the source dataset is PlantVillage;
- the total mapped image count is 54,305.

### `tests/test_module_interfaces.py`

The suite validates:

- supported predictions;
- optional model-version and embedding behavior;
- uncertain predictions;
- unknown predictions;
- not-a-plant predictions;
- RAG input/output structures;
- dictionary serialization and reconstruction;
- Vision + RAG integration;
- advisory suppression for non-supported statuses;
- invalid confidence and missing-field rejection.

### Verification Command

```bash
python -m unittest discover tests
```

### Verification Result

**16/16 tests passed — 100% pass rate.**

This provides automated evidence that the Phase 2 taxonomy and inter-module contract foundation is internally consistent.

---

## 10. Development Infrastructure

The project uses the following development/storage arrangement:

- **GitHub:** source code, metadata, tests, and documentation.
- **Google Drive:** large raw datasets and model-related files that should not be stored in Git.
- **Google Colab:** GPU-enabled workspace for dataset exploration and model-development activities.

The repository also contains an environment verification script and a structured project layout separating data, models, source modules, tests, documentation, notebooks, and reports.

---

## 11. Technology Stack

### Active Foundation

- Python 3.10+
- PyTorch
- TorchVision
- OpenCV
- Pillow
- NumPy
- pandas
- scikit-learn
- matplotlib

### Planned AI/RAG Components

- CNN/transfer-learning based Vision model
- Visual embeddings
- FAISS and/or ChromaDB for vector similarity search
- LangChain and/or LlamaIndex for RAG orchestration where appropriate
- Transformer/LLM components for grounded advisory generation

The planned components are distinguished from the currently verified Phase 2 foundation.

---

## 12. Current Progress Status

| Component | Phase 2 Status | Progress State |
|---|---|---|
| Repository & project structure | Established | **Completed** |
| PlantVillage metadata inspection | Verified | **Completed** |
| Canonical taxonomy | Implemented | **Completed** |
| Unified class mapping | Established | **Completed** |
| Vision/RAG/Integration contracts | Implemented | **Completed** |
| Status and validation rules | Implemented | **Completed** |
| Architecture documentation | Implemented | **Completed** |
| Interface documentation | Implemented | **Completed** |
| Automated unit tests | Implemented and passing | **Completed** |
| GPU/Colab development infrastructure | Established | **Completed** |
| Image preprocessing pipeline | Planned/under implementation | **Next phase** |
| CNN baseline training | Planned/under implementation | **Next phase** |
| Production visual embeddings | Future extension | **Pending** |
| Production RAG knowledge base | Planned | **Next phase** |
| Vector retrieval | Planned | **Next phase** |
| LLM advisory generation | Planned | **Next phase** |
| End-to-end evaluation | Planned | **Later phase** |

---

## 13. Challenges Identified

### 13.1 Heterogeneous Dataset Labels

Different datasets use different naming conventions. The canonical taxonomy solves this by separating original labels from standardized project identifiers.

### 13.2 Class Imbalance

The PlantVillage baseline contains substantial variation in class frequency. This can bias a classifier toward dominant classes. The project therefore records class counts and will evaluate class-aware strategies during model training.

### 13.3 Open-Set and Low-Confidence Inputs

A model trained on a fixed set of diseases can encounter unsupported diseases or poor-quality images. The explicit `unknown` and `uncertain` states provide a defined fallback path.

### 13.4 Safe Agricultural Advice

A disease label alone is insufficient for reliable agricultural guidance. The RAG design therefore requires evidence retrieval and source attribution before production advisory generation.

### 13.5 Team Integration

Independent module development can cause compatibility problems. The typed contracts and interface documentation were introduced specifically to reduce integration risk.

---

## 14. Next Phase Plan

The next implementation phase will focus on converting the established architecture into working AI components.

### Vision Work

1. Implement image preprocessing and augmentation.
2. Establish train/validation/test splits.
3. Train and validate a CNN baseline.
4. Measure accuracy, precision, recall, F1-score and confusion matrix.
5. Calibrate the confidence threshold `τ`.
6. Investigate class imbalance mitigation.

### RAG Work

1. Collect and verify authoritative agricultural documents.
2. Prepare and chunk source documents.
3. Generate embeddings.
4. Build the vector index.
5. Implement canonical-ID-based retrieval.
6. Return structured advisory evidence with sources.

### Integration Work

1. Connect Vision output to canonical taxonomy mapping.
2. Route supported predictions to RAG.
3. Suppress advisory generation for uncertain/unknown/not-a-plant states.
4. Produce the final `IntegratedResponse`.
5. Perform end-to-end tests and latency measurements.

### Generalization Work

1. Prepare additional datasets such as PlantDoc/Plant Pathology where applicable.
2. Extend the canonical mapping without modifying the core interfaces.
3. Evaluate unseen-class and few-shot adaptation strategies.

---

## 15. Evidence / Screenshots to Attach

**Screenshots are required for the submitted progress report.** The following screenshots should be captured from the team's actual development environment and inserted at the indicated locations. Do not use fabricated screenshots.

### Screenshot 1 — GitHub Repository

**What to capture:** Repository homepage showing the project name, directory structure, and recent project files.

**Suggested caption:** *Figure 1. GitHub repository containing the Plant Disease AI source code, documentation, metadata, and test suites.*

**Insert here:**

`[SCREENSHOT 1: GitHub repository overview]`

### Screenshot 2 — Canonical Taxonomy Mapping

**What to capture:** `data/metadata/plantvillage_class_mapping.csv` opened in GitHub or a local editor, showing columns such as `class_id`, `canonical_id`, `plant`, `disease`, `original_label`, and `image_count`.

**Suggested caption:** *Figure 2. PlantVillage class mapping showing dataset-specific labels translated into canonical disease identifiers.*

**Insert here:**

`[SCREENSHOT 2: PlantVillage class mapping CSV]`

### Screenshot 3 — Inter-Module Contracts

**What to capture:** `src/contracts.py` showing `PredictionStatus` and the `VisionPrediction`, `RAGQueryInput`, `AdvisoryResult`, or `IntegratedResponse` definitions.

**Suggested caption:** *Figure 3. Typed data contracts defining communication between Vision, RAG, and Integration modules.*

**Insert here:**

`[SCREENSHOT 3: contracts.py]`

### Screenshot 4 — Automated Test Result

**What to capture:** Terminal/PowerShell output after running:

```bash
python -m unittest discover tests
```

The screenshot should visibly show the successful test result.

**Suggested caption:** *Figure 4. Automated verification of the Phase 2 taxonomy and module-interface test suites.*

**Insert here:**

`[SCREENSHOT 4: terminal showing 16/16 tests passing]`

### Screenshot 5 — System Architecture

**What to capture:** `docs/architecture/system_architecture.md` showing the architecture/dataflow section, or a clean diagram prepared specifically for the report.

**Suggested caption:** *Figure 5. Proposed end-to-end Plant Disease AI architecture from image input to grounded agricultural advisory.*

**Insert here:**

`[SCREENSHOT 5: system architecture/dataflow]`

### Screenshot 6 — Google Colab / GPU Environment

**What to capture:** The team's actual Colab notebook showing the GitHub connection, dataset/class loading, or GPU environment verification.

**Suggested caption:** *Figure 6. Google Colab GPU development environment connected to the project repository.*

**Insert here:**

`[SCREENSHOT 6: Colab environment / class loading / GPU verification]`

> **Important:** Only include screenshots that actually exist in your team's environment. If a screenshot cannot be produced from the current implementation, omit it rather than presenting it as completed work.

---

## 16. Conclusion

Phase 2 successfully established the technical foundation of Plant Disease AI. The project now has a standardized PlantVillage taxonomy containing 38 classes and 54,305 images, formal inter-module contracts, explicit prediction-status handling, architecture and interface documentation, and automated validation with all 16 tests passing.

The major outcome of this phase is a decoupled architecture that allows the Vision, RAG, and Integration components to be developed and improved independently while maintaining a stable communication interface. The next stage will focus on implementing and evaluating the actual CNN and RAG components and then connecting them into a complete end-to-end advisory pipeline.

The current project status should therefore be described as **a verified Phase 2 architectural and data foundation, progressing toward the complete AI implementation**, rather than as a finished production disease-detection system.

---

## 17. Repository References

- Project repository: `https://github.com/subhajitg124-cell/plant-disease-ai`
- System architecture: `docs/architecture/system_architecture.md`
- Module interfaces: `docs/architecture/module_interfaces.md`
- Contracts: `src/contracts.py`
- PlantVillage mapping: `data/metadata/plantvillage_class_mapping.csv`
- Taxonomy tests: `tests/test_taxonomy_mapping.py`
- Interface tests: `tests/test_module_interfaces.py`
