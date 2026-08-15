# Mermaid Charts

This page provides section-by-section Mermaid diagrams for the AutoQC system.

## 1) System Overview

```mermaid
flowchart LR
    A[Config] --> B[Dataset Parser]
    B --> C[VLM Inference]
    C --> D[Scoring and QC Decision]
    D --> E[Results Persistence]
    E --> F[Run Artifacts]
    E --> G[MLflow Logging Optional]
```

## 2) Config Loading and Validation

```mermaid
flowchart TD
    A[load_run_config] --> B[load .env]
    B --> C[read YAML or JSON]
    C --> D[expand env placeholders]
    D --> E[coerce to RunConfig]
    E --> F[validate required paths]
    F --> G{mlflow enabled}
    G -- no --> I[return config]
    G -- yes --> H[validate tracking uri not file:]
    H --> I[return config]
```

## 3) Dataset Parsing Flow

```mermaid
flowchart TD
    A[labels_path rglob *.json] --> B[images_path rglob *.jpg index by stem]
    B --> C[for each label file]
    C --> D{image exists}
    D -- no --> D1[skip missing image]
    D -- yes --> E[read JSON frames and objects]
    E --> F{category in class_names}
    F -- no --> F1[skip unknown category]
    F -- yes --> G{box2d exists}
    G -- no --> G1[skip missing box]
    G -- yes --> H[normalize box]
    H --> I{valid and min size}
    I -- no --> I1[skip invalid or small box]
    I -- yes --> J[emit DetectionSample]
    J --> K[collect results]
    K --> L{results empty}
    L -- yes --> M[raise runtime error with diagnostics]
    L -- no --> N[return samples]
```

## 4) Inference Mode Selection

```mermaid
flowchart TD
    A[inference_mode] --> B{mode}
    B -- without_red_rectangle --> C[full image + grounding box tokens]
    B -- with_red_rectangle --> D[draw red rectangle + full image]
    B -- coordinates_text --> E[full image + x,y,w,h text]
    B -- crop_only --> F[crop object box only]
    C --> G[processor and model generate]
    D --> G
    E --> G
    F --> G
    G --> H[parse response + first-token class logits]
    H --> I[InferenceResult]
```

## 5) Scoring and Decision Logic

```mermaid
flowchart TD
    A[pred_probs and given_label] --> B[compute self_confidence]
    B --> C[compute normalized margin]
    C --> D[fetch class threshold]
    D --> E{predicted label equals given label}
    E -- yes --> F[accepted_match]
    E -- no --> G{self_confidence <= threshold}
    G -- yes --> H[flag mismatch_low_self_confidence]
    G -- no --> I{high margin and high alt confidence}
    I -- yes --> J[flag mismatch_high_margin]
    I -- no --> K[accepted_uncertain_mismatch]
```

## 6) Threshold Derivation

```mermaid
flowchart TD
    A[samples + results] --> B[group self-confidence by class]
    B --> C[sort each class list]
    C --> D[pick ~20th percentile index]
    D --> E[set threshold per class]
```

## 7) Pipeline Orchestration

```mermaid
sequenceDiagram
    participant CLI
    participant Pipe as AutoQCPipeline
    participant Parser as BDDDatasetParser
    participant VLM as QwenGroundingInference
    participant Score as scoring.py
    participant Res as results.py
    participant MLF as mlflow_tracking.py

    CLI->>Pipe: run()
    Pipe->>Res: create_run_dir()
    Pipe->>Parser: iter_samples()
    loop each sample
        Pipe->>VLM: predict(sample)
        VLM-->>Pipe: InferenceResult
        Pipe->>Pipe: log object_inference
        Pipe->>Pipe: checkpoint every N
    end
    Pipe->>Score: derive_thresholds()
    Pipe->>Score: make_decisions()
    Pipe->>Pipe: log object_decision
    Pipe->>Res: persist_run()
    Res-->>Pipe: RunSummary
    Pipe->>MLF: log_run_to_mlflow()
    Pipe-->>CLI: summary + run_dir
```

## 8) Run Artifacts and Tracking

```mermaid
flowchart LR
    A[persist_run] --> B[run_config.json]
    A --> C[run_summary.json]
    A --> D[metrics.json]
    A --> E[run_manifest.json]
    A --> F[all_samples parquet or csv]
    A --> G[flagged_samples parquet or csv]
    A --> H[class_summary.csv]
    A --> I[confusion_matrix.csv]
    E --> J[runs/run_index.json update]
```

## 9) Azure DAG Pipeline

```mermaid
flowchart LR
    Z[env placeholders resolved] --> A[prepare_data step]
    A --> B[prepared_inputs.json]
    B --> C[run_inference step]
    C --> D[inference_output]
    D --> E[evaluate step]
    E --> F[evaluation_summary]
    F --> G{total_samples > 0}
    G -- yes --> H[quality gate pass]
    G -- no --> I[quality gate fail]
```

## 10) Experiment Matrix

```mermaid
flowchart TD
    A[experiment config] --> B[mode list]
    B --> C1[without_red_rectangle run]
    B --> C2[with_red_rectangle run]
    B --> C3[coordinates_text run]
    B --> C4[crop_only run]
    C1 --> D[collect metrics]
    C2 --> D
    C3 --> D
    C4 --> D
    D --> E[experiment_summary.json]
    E --> F[pick best mode]
```
