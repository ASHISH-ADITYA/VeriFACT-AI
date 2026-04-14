# Architecture

## System Overview

VeriFACT AI is a 5-stage verification pipeline with three user interfaces sharing a single core engine.

```
┌──────────────────────────────────────────────────┐
│             User Interfaces                       │
│                                                   │
│  Chrome Extension   Streamlit Dashboard   API     │
│  (beacon overlay)   (deep analysis)     (REST)    │
└────────────┬──────────────┬───────────────┬──────┘
             │              │               │
             └──────────────┼───────────────┘
                            │
             ┌──────────────▼──────────────┐
             │       VeriFactPipeline       │
             │                              │
             │  1. Claim Decomposition      │
             │  2. Evidence Retrieval       │
             │  3. NLI Verdict + Gate       │
             │  4. Confidence Scoring       │
             │  5. Annotated Output         │
             └──────────────────────────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
         ┌───▼───┐   ┌─────▼─────┐  ┌────▼────┐
         │Ollama │   │   FAISS   │  │DeBERTa  │
         │ (LLM) │   │  (index)  │  │  (NLI)  │
         └───────┘   └───────────┘  └─────────┘
```

## Pipeline Stages

### Stage 1: Claim Decomposition
- **Input**: Raw text from LLM response
- **Output**: List of atomic factual claims with type labels
- **Primary**: Ollama LLM with structured JSON prompt
- **Fallback**: spaCy sentence segmentation

### Stage 2: Evidence Retrieval
- **Input**: Claim text
- **Output**: Top-k evidence passages with similarity scores
- **Method**: sentence-transformers encoding → FAISS cosine search
- **Corpus**: 369K Wikipedia Simple English passages

### Stage 3: NLI Verdict with Specificity Gate
- **Input**: (evidence, claim) pairs
- **Output**: SUPPORTED / CONTRADICTED / UNVERIFIABLE verdict
- **Model**: DeBERTa-v3-base (92% MultiNLI accuracy)
- **Gate**: `specificity = entailment × similarity` — prevents topical false entailment

### Stage 4: Bayesian Confidence Scoring
- **Input**: NLI scores, retrieval scores, source metadata
- **Output**: Calibrated confidence (0–1)
- **Weights**: NLI 40%, retrieval 25%, source 15%, cross-ref 20%

### Stage 5: Annotated Output
- **Input**: Claims with verdicts
- **Output**: Color-coded HTML, structured JSON, corrections

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| NLI over LLM-as-judge | DeBERTa cross-encoder | No circular reasoning; deterministic; runs locally |
| Specificity gate | entailment × similarity | Prevents false entailment from topical overlap |
| Local-first | Ollama + FAISS on device | Zero cost; zero data leakage; works offline |
| Fallback chain | Ollama → API → spaCy | Graceful degradation; never crashes |
