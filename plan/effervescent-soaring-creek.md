# VeriFactAI — Complete Technical Specification & Implementation Plan
### LLM Hallucination Detection & Factual Grounding System
#### Final Year CSE Capstone Project

---

## 1. EXECUTIVE SUMMARY

**VeriFactAI** is an AI safety system that automatically detects factual hallucinations in Large Language Model (LLM) outputs, verifies claims against trusted knowledge sources, and presents users with annotated, citation-backed results. 

**The Problem**: LLMs hallucinate 3-27% of the time depending on domain (Ji et al., 2023, "Survey of Hallucination in NLG"). In high-stakes domains (medicine, law, finance), a single hallucinated fact can cause real harm. No robust, open-source, end-to-end hallucination detection system exists today.

**The Solution**: A 5-stage pipeline that decomposes LLM text into atomic claims, retrieves evidence from trusted corpora, classifies each claim as Supported/Contradicted/Unverifiable using Natural Language Inference, computes calibrated confidence scores, and returns annotated output with source citations.

**Novel Contributions**:
1. Multi-stage claim verification pipeline combining claim decomposition + retrieval + NLI + Bayesian scoring (no existing open-source system integrates all four)
2. Calibrated confidence scoring that fuses retrieval similarity, NLI probability, source reliability, and cross-reference agreement
3. Domain-adaptive verification (general knowledge, medical, scientific) with domain-specific knowledge corpora
4. Quantitative evaluation on established benchmarks (TruthfulQA, HaluEval, FEVER) with published baselines for comparison

---

## 2. RESEARCH FOUNDATION & LITERATURE REVIEW

### 2.1 The Hallucination Problem
- **Prevalence**: GPT-4 hallucinates on ~3% of factual questions (OpenAI, 2023); GPT-3.5 on ~15-20%; open-source models on 20-30%+ (Lin et al., TruthfulQA, 2022)
- **Types of hallucination**:
  - **Intrinsic**: contradicts the source/prompt
  - **Extrinsic**: cannot be verified from any source (fabricated facts)
  - **Entity hallucination**: real entities combined in false relationships
  - **Numerical hallucination**: fabricated statistics, dates, quantities

### 2.2 Existing Approaches & Their Limitations
| Approach | Method | Limitation |
|---|---|---|
| SelfCheckGPT (Manakul et al., 2023) | Sample multiple LLM outputs, check consistency | Doesn't use external evidence; consistent hallucinations pass |
| RARR (Gao et al., 2023) | Retrieve-and-revise | Focused on editing, not detection/annotation |
| FActScore (Min et al., 2023) | Decompose into atomic facts + verify | Single-domain (biographies); not real-time; no UI |
| FactCheck-GPT (Wang et al., 2023) | LLM-based verification | Uses LLM to verify LLM — circular; prone to same hallucinations |
| SAFE (Wei et al., 2024, Google) | Search + LLM judge | Expensive (multiple LLM calls per claim); no confidence calibration |

**Our advantage over each**:
- Unlike SelfCheckGPT: we use external evidence, not just self-consistency
- Unlike RARR: we detect AND annotate, not just edit
- Unlike FActScore: we're multi-domain, real-time, with a usable UI
- Unlike FactCheck-GPT: we use NLI models (not LLMs) for verification — no circular reasoning
- Unlike SAFE: we're cost-efficient (NLI model is local, not API-based) with calibrated confidence

### 2.3 Key Technical Foundations
- **Claim Decomposition**: Based on FActScore methodology (Min et al., 2023) — LLM breaks text into atomic facts
- **Dense Retrieval**: DPR (Karpukhin et al., 2020) and sentence-transformers (Reimers & Gurevych, 2019) for semantic search
- **Natural Language Inference**: DeBERTa-v3 (He et al., 2023) achieves 92.0% on MultiNLI — state-of-art for entailment/contradiction detection
- **Bayesian Confidence**: Inspired by calibration techniques from Guo et al. (2017), "On Calibration of Modern Neural Networks"

---

## 3. SYSTEM ARCHITECTURE

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VERIFACTAI SYSTEM                      │
│                                                           │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │  INPUT    │───▶│  LLM ENGINE  │───▶│    CLAIM       │  │
│  │  (Query)  │    │  (Claude/    │    │  DECOMPOSER    │  │
│  │           │    │   GPT-4)     │    │  (Atomic facts)│  │
│  └──────────┘    └──────────────┘    └───────┬────────┘  │
│                                              │            │
│                                              ▼            │
│  ┌──────────────────────────────────────────────────┐    │
│  │              EVIDENCE RETRIEVAL ENGINE             │    │
│  │                                                    │    │
│  │  ┌────────────┐  ┌──────────┐  ┌──────────────┐  │    │
│  │  │ Sentence   │  │  FAISS   │  │  Knowledge   │  │    │
│  │  │ Transformer│─▶│  Index   │◀─│  Corpus      │  │    │
│  │  │ Encoder    │  │ (Vector  │  │  (Wikipedia, │  │    │
│  │  │            │  │  Search) │  │   PubMed,    │  │    │
│  │  │            │  │          │  │   etc.)      │  │    │
│  │  └────────────┘  └──────────┘  └──────────────┘  │    │
│  └──────────────────────┬───────────────────────────┘    │
│                         │ Top-k evidence per claim        │
│                         ▼                                 │
│  ┌──────────────────────────────────────────────────┐    │
│  │               VERDICT ENGINE                       │    │
│  │                                                    │    │
│  │  ┌────────────┐  ┌──────────────┐  ┌──────────┐  │    │
│  │  │ NLI Model  │  │  Bayesian    │  │ Cross-   │  │    │
│  │  │ (DeBERTa)  │─▶│  Confidence  │◀─│ Reference│  │    │
│  │  │ Entail/    │  │  Scorer      │  │ Checker  │  │    │
│  │  │ Contradict │  │              │  │          │  │    │
│  │  └────────────┘  └──────────────┘  └──────────┘  │    │
│  └──────────────────────┬───────────────────────────┘    │
│                         │ Verdict + Confidence per claim  │
│                         ▼                                 │
│  ┌──────────────────────────────────────────────────┐    │
│  │            ANNOTATED OUTPUT GENERATOR              │    │
│  │                                                    │    │
│  │  • Color-coded text (green/yellow/red)             │    │
│  │  • Inline citations with source links              │    │
│  │  • Confidence scores per claim                     │    │
│  │  • Suggested corrections for contradicted claims   │    │
│  │  • Overall Factuality Score (0-100)                │    │
│  └──────────────────────┬───────────────────────────┘    │
│                         │                                 │
│                         ▼                                 │
│  ┌──────────────────────────────────────────────────┐    │
│  │             STREAMLIT DASHBOARD                    │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow (Detailed)

```
Input: "Albert Einstein invented the telephone in 1876 and won the Nobel Prize in Physics."

Step 1 — Claim Decomposition:
  Claim 1: "Albert Einstein invented the telephone"
  Claim 2: "The telephone was invented in 1876"
  Claim 3: "Albert Einstein won the Nobel Prize in Physics"

Step 2 — Evidence Retrieval (per claim):
  Claim 1 Evidence: ["Alexander Graham Bell is credited with inventing the telephone...", ...]
  Claim 2 Evidence: ["Alexander Graham Bell patented the telephone in 1876...", ...]
  Claim 3 Evidence: ["Einstein received the Nobel Prize in Physics in 1921...", ...]

Step 3 — NLI Verdict:
  Claim 1: CONTRADICTED (evidence says Bell, not Einstein) — NLI contradiction score: 0.94
  Claim 2: SUPPORTED (1876 is correct for telephone patent) — NLI entailment score: 0.89
  Claim 3: SUPPORTED (Einstein did win Nobel in Physics) — NLI entailment score: 0.92

Step 4 — Confidence Scoring:
  Claim 1: 8% confidence (high contradiction + high retrieval relevance)
  Claim 2: 87% confidence (strong entailment + high retrieval relevance)
  Claim 3: 91% confidence (strong entailment + multiple source agreement)

Step 5 — Annotated Output:
  "[❌ Albert Einstein invented the telephone] [✅ in 1876] and [✅ won the Nobel Prize in Physics]."
  
  ❌ Claim 1 — CONTRADICTED (8% confidence)
     Evidence: "Alexander Graham Bell is credited with inventing the telephone"
     Source: Wikipedia - Telephone
     Correction: The telephone was invented by Alexander Graham Bell, not Einstein.
  
  ✅ Claim 2 — VERIFIED (87% confidence)
     Evidence: "Bell patented the telephone in 1876"
     Source: Wikipedia - Alexander Graham Bell
  
  ✅ Claim 3 — VERIFIED (91% confidence)
     Evidence: "Einstein received the Nobel Prize in Physics in 1921"
     Source: Wikipedia - Albert Einstein

  Overall Factuality Score: 62/100 (2 of 3 claims verified)
```

---

## 4. DETAILED COMPONENT SPECIFICATIONS

### 4.1 Claim Decomposer (`core/claim_decomposer.py`)

**Purpose**: Break any LLM-generated text into a list of atomic, verifiable factual claims.

**Algorithm**:
1. Send text to Claude/GPT-4 with structured prompt
2. Parse JSON response into list of claims
3. Filter out non-factual statements (opinions, hedges, questions)
4. Map each claim back to its source sentence in the original text (for annotation)

**Prompt Engineering** (critical for quality):
```
System: You are a factual claim extractor. Your job is to decompose text into 
atomic factual claims that can be independently verified.

Rules:
1. Each claim must be a single factual statement
2. Each claim must be self-contained (understandable without original context)
3. Resolve pronouns and references (e.g., "He" → "Albert Einstein")
4. Split compound claims ("X did A and B" → two claims)
5. Preserve numerical claims exactly (dates, statistics, quantities)
6. SKIP: opinions, subjective statements, hedged language ("might", "possibly"), 
   questions, instructions, greetings
7. SKIP: trivially true statements ("The sky is above us")

Input text: {text}

Output format (strict JSON):
[
  {
    "claim": "self-contained factual statement",
    "source_sentence": "original sentence this came from",
    "claim_type": "entity_fact|numerical|temporal|causal|relational"
  }
]
```

**Fallback Strategy**: If LLM API fails or rate-limited:
- Use spaCy for sentence segmentation
- Use simple heuristics to filter non-factual sentences (remove questions, sentences with hedging words)
- Lower quality but ensures system doesn't break

**Expected Output Quality**: Based on FActScore paper, LLM-based decomposition achieves ~95% recall of factual claims with ~90% precision (10% non-factual claims slip through).

---

### 4.2 Evidence Retriever (`core/evidence_retriever.py`)

**Purpose**: For each claim, retrieve the most relevant evidence passages from trusted knowledge sources.

**Knowledge Corpus Construction**:

| Source | Size | Domain | How to Acquire |
|---|---|---|---|
| Wikipedia (Simple English) | ~200K articles | General knowledge | HuggingFace `datasets` library: `load_dataset("wikipedia", "20220301.simple")` |
| Wikipedia (English, subset) | ~1M articles (top articles by pageviews) | General knowledge | HuggingFace `datasets` or direct dump |
| PubMed Abstracts | ~500K abstracts | Medical/biomedical | HuggingFace `pubmed_qa` dataset |
| Textbook passages (optional) | Variable | Academic/scientific | OpenStax textbooks (freely available) |

**Chunking Strategy**:
- Split articles into passages of **200 tokens** with **50-token overlap**
- Preserve paragraph boundaries where possible
- Store metadata per chunk: `{source, title, url, section, chunk_id}`
- Expected total chunks: ~2-5M (depending on corpus size)

**Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional embeddings
- Trained on 1B+ sentence pairs
- Inference speed: ~14,000 sentences/second on GPU, ~2,000/sec on CPU
- Quality: 68.1 on STS benchmark (good balance of speed vs. quality)

**Alternative (higher quality, slower)**: `sentence-transformers/all-mpnet-base-v2`
- 768-dimensional
- 69.6 on STS benchmark
- ~3x slower but more accurate retrieval

**FAISS Index Configuration**:
```python
# For corpus up to 5M chunks on CPU:
index = faiss.IndexFlatIP(384)  # Inner product (cosine sim with normalized vectors)

# For larger corpus (>5M), use IVF for speed:
quantizer = faiss.IndexFlatIP(384)
index = faiss.IndexIVFFlat(quantizer, 384, nlist=1024, faiss.METRIC_INNER_PRODUCT)
index.nprobe = 10  # Search 10 clusters (balance speed/recall)
```

**Retrieval Pipeline**:
1. Encode claim using same sentence-transformer
2. Search FAISS for top-k (k=5) most similar passages
3. Apply relevance threshold: discard results with similarity < 0.30
4. If no results pass threshold → mark claim as "No Evidence Found"
5. Return ranked evidence list with metadata

**Performance Targets**:
- Retrieval latency: <100ms per claim (FAISS in-memory)
- Recall@5: target >80% (relevant evidence is in top 5 results)

---

### 4.3 Verdict Engine (`core/verdict_engine.py`)

**Purpose**: Determine whether retrieved evidence supports, contradicts, or is neutral to each claim.

#### 4.3.1 NLI Classification

**Model**: `cross-encoder/nli-deberta-v3-base` (HuggingFace)
- Trained on MultiNLI + SNLI + FEVER
- Accuracy: 92.0% on MultiNLI
- Output: 3-class probability distribution [entailment, neutral, contradiction]
- Inference: ~50ms per pair on CPU

**NLI Inference**:
```python
# For each claim, run against each evidence passage:
for evidence in retrieved_evidence:
    input = f"{evidence} [SEP] {claim}"
    scores = nli_model.predict(input)  # [entailment, neutral, contradiction]
```

**Aggregation Strategy** (across top-5 evidence passages per claim):
```
max_entailment = max(entailment scores across all evidence)
max_contradiction = max(contradiction scores across all evidence)
avg_neutral = mean(neutral scores across all evidence)

if max_contradiction > 0.75 AND max_contradiction > max_entailment:
    verdict = "CONTRADICTED"
    contradiction_evidence = evidence with highest contradiction score
elif max_entailment > 0.65:
    verdict = "SUPPORTED"
    supporting_evidence = evidence with highest entailment score
elif no evidence retrieved (all below relevance threshold):
    verdict = "NO_EVIDENCE"
else:
    verdict = "UNVERIFIABLE"
```

#### 4.3.2 Bayesian Confidence Scoring

**Purpose**: Produce a single calibrated confidence score (0-100%) for each claim.

**Formula**:
```
confidence = (
    w1 * nli_support_score +
    w2 * retrieval_relevance +
    w3 * source_reliability +
    w4 * cross_reference_agreement
)

Where:
  w1 = 0.40  (NLI is most discriminative signal)
  w2 = 0.25  (how relevant was the best evidence?)
  w3 = 0.15  (Wikipedia/PubMed = 1.0; unknown = 0.5)
  w4 = 0.20  (do multiple independent sources agree?)

  nli_support_score = max_entailment - max_contradiction  (range: -1 to 1, normalized to 0-1)
  retrieval_relevance = max cosine similarity from FAISS (range: 0-1)
  source_reliability = pre-assigned per source (Wikipedia=1.0, PubMed=1.0, etc.)
  cross_reference_agreement = fraction of top-5 evidences that entail the claim (range: 0-1)
```

**Confidence Thresholds → Verdict Mapping**:
| Confidence | Verdict | Color | Meaning |
|---|---|---|---|
| ≥ 75% | ✅ VERIFIED | Green | Claim is well-supported by evidence |
| 40-74% | ⚠️ UNCERTAIN | Yellow | Some evidence exists but inconclusive |
| < 40%, no contradiction | ❓ UNVERIFIABLE | Orange | Cannot find sufficient evidence |
| < 40%, contradiction detected | ❌ CONTRADICTED | Red | Evidence directly contradicts this claim |

**Calibration**: After initial implementation, we'll check if confidence scores are well-calibrated (i.e., claims with 80% confidence are correct ~80% of the time) on the evaluation set. If not, apply temperature scaling (Guo et al., 2017).

---

### 4.4 Annotated Output Generator (`core/annotator.py`)

**Purpose**: Map verdicts back to original text and generate rich, annotated output.

**Algorithm**:
1. For each claim, find its source sentence in original text
2. Highlight the relevant span with appropriate color
3. Attach tooltip data: verdict, confidence, evidence snippet, source citation
4. For CONTRADICTED claims: generate a correction using the contradicting evidence
5. Compute overall Factuality Score: `(verified_claims / total_claims) * 100`

**Output Formats**:
- **HTML**: For Streamlit rendering (color-coded spans with tooltips)
- **JSON**: For programmatic access
  ```json
  {
    "original_text": "...",
    "overall_factuality_score": 73.5,
    "total_claims": 8,
    "verified": 5,
    "contradicted": 1,
    "unverifiable": 2,
    "claims": [
      {
        "claim": "...",
        "verdict": "VERIFIED",
        "confidence": 0.89,
        "evidence": "...",
        "source": {"title": "...", "url": "..."},
        "original_span": {"start": 0, "end": 45}
      }
    ]
  }
  ```
- **Markdown**: For report generation

**Correction Generation** (for contradicted claims):
- Use the contradicting evidence + LLM to generate a corrected version:
  ```
  Claim: "Einstein invented the telephone"
  Evidence: "Alexander Graham Bell invented the telephone"
  Correction: "The telephone was invented by Alexander Graham Bell, not Albert Einstein."
  ```

---

### 4.5 Pipeline Orchestrator (`core/pipeline.py`)

**Purpose**: Chain all components into a single, clean API.

**Class Design**:
```python
class VeriFactPipeline:
    def __init__(self, config: Config):
        self.llm = LLMClient(config.llm_provider, config.api_key)
        self.decomposer = ClaimDecomposer(self.llm)
        self.retriever = EvidenceRetriever(config.faiss_index_path, config.embedding_model)
        self.verdict_engine = VerdictEngine(config.nli_model)
        self.annotator = AnnotatedOutputGenerator()
        self.cache = {}  # Simple in-memory cache
    
    def verify(self, text: str, domain: str = "general") -> VerificationResult:
        """Full pipeline: text → annotated, verified output"""
        # Step 1: Decompose into claims
        claims = self.decomposer.decompose(text)
        
        # Step 2: Retrieve evidence for each claim (parallelizable)
        for claim in claims:
            claim.evidence = self.retriever.retrieve(claim.text, domain=domain, top_k=5)
        
        # Step 3: Run NLI verdict on each claim
        for claim in claims:
            claim.verdict, claim.confidence = self.verdict_engine.judge(claim)
        
        # Step 4: Generate annotated output
        result = self.annotator.annotate(text, claims)
        return result
    
    def verify_query(self, query: str, domain: str = "general") -> VerificationResult:
        """Full pipeline: query → LLM response → verified output"""
        llm_response = self.llm.generate(query)
        return self.verify(llm_response, domain)
```

**Performance Targets** (end-to-end):
| Metric | Target |
|---|---|
| Total latency (short text, ~5 claims) | < 10 seconds |
| Total latency (long text, ~20 claims) | < 30 seconds |
| Memory usage | < 4GB (FAISS index + models) |
| Concurrent users (Streamlit) | 1-5 (demo scope) |

---

## 5. KNOWLEDGE CORPUS STRATEGY

### 5.1 Corpus Selection & Justification

**Primary corpus: Wikipedia Simple English**
- Size: ~200K articles, ~70M words
- After chunking: ~350K passages
- Covers: general knowledge, history, science, geography, people
- Quality: curated, neutral, regularly updated
- Acquisition: `datasets.load_dataset("wikipedia", "20220301.simple")`

**Secondary corpus: PubMed QA abstracts**
- Size: ~500K abstracts
- After chunking: ~1M passages
- Covers: medical, biomedical, clinical research
- Quality: peer-reviewed scientific literature
- Acquisition: `datasets.load_dataset("pubmed_qa", "pqa_labeled")`

**Index build time estimates**:
- Embedding 1.35M passages with all-MiniLM-L6-v2:
  - On GPU (A100): ~15 minutes
  - On MacBook M2: ~2-3 hours
  - On CPU (cloud): ~4-5 hours
- FAISS index construction: < 5 minutes after embeddings are computed

**Storage requirements**:
- Embeddings (384-dim float32, 1.35M vectors): ~2GB
- FAISS index: ~2GB
- Metadata (JSON): ~500MB
- Total: ~4.5GB

### 5.2 Corpus Update Strategy (for future scalability)
- Design the indexing pipeline to be incremental (add new documents without rebuilding)
- Use FAISS `index.add()` for incremental additions
- Future: could integrate real-time web search as fallback when corpus retrieval confidence is low

---

## 6. FRONTEND SPECIFICATION (Streamlit Dashboard)

### 6.1 Page Layout

```
┌─────────────────────────────────────────────────────────────┐
│  🔍 VeriFactAI — LLM Hallucination Detector                 │
│  ─────────────────────────────────────────────────────────── │
│                                                               │
│  [Domain: General ▼]  [Model: Claude ▼]                      │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Enter your question or paste LLM-generated text:        │ │
│  │  ┌───────────────────────────────────────────────────┐   │ │
│  │  │                                                    │   │ │
│  │  │  [Text input area]                                 │   │ │
│  │  │                                                    │   │ │
│  │  └───────────────────────────────────────────────────┘   │ │
│  │  [🔍 Verify]  [📋 Paste LLM Text]  [🎲 Try Example]    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │  FACTUALITY SCORE    │  │  CLAIM BREAKDOWN              │ │
│  │                      │  │                                │ │
│  │     ┌────┐           │  │  ✅ Verified:     5 (62.5%)   │ │
│  │     │ 73 │           │  │  ⚠️ Uncertain:    1 (12.5%)   │ │
│  │     │/100│           │  │  ❌ Contradicted: 1 (12.5%)   │ │
│  │     └────┘           │  │  ❓ No Evidence:  1 (12.5%)   │ │
│  │  [circular gauge]    │  │  [pie chart]                   │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  ANNOTATED OUTPUT                                        │ │
│  │                                                           │ │
│  │  [Color-coded text with hover tooltips showing           │ │
│  │   evidence, confidence, and source for each claim.       │ │
│  │   Green = verified, Yellow = uncertain, Red = false]     │ │
│  │                                                           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  DETAILED CLAIM ANALYSIS                                  │ │
│  │                                                           │ │
│  │  ▶ Claim 1: "Einstein won Nobel Prize..." ✅ 91%         │ │
│  │    Evidence: "Einstein received the Nobel..."             │ │
│  │    Source: Wikipedia — Albert Einstein                     │ │
│  │                                                           │ │
│  │  ▶ Claim 2: "Einstein invented telephone" ❌ 8%          │ │
│  │    Evidence: "Bell invented the telephone..."             │ │
│  │    Source: Wikipedia — Telephone                           │ │
│  │    Correction: "The telephone was invented by A.G. Bell"  │ │
│  │                                                           │ │
│  │  [expandable cards for each claim]                        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  CONFIDENCE DISTRIBUTION                                  │ │
│  │  [histogram of confidence scores across all claims]       │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Interactive Features
1. **Hover Tooltips**: Hover over any highlighted text to see evidence + source
2. **Expandable Claim Cards**: Click to see full evidence, NLI scores, all retrieved passages
3. **Example Queries**: Pre-loaded examples demonstrating each verdict type
4. **Domain Selector**: Switch between General / Medical / Scientific corpora
5. **Comparison Mode**: Toggle to see raw LLM output vs. verified output side-by-side
6. **Export**: Download full verification report as JSON or PDF

### 6.3 Example Queries (Pre-loaded for demo)
1. "Tell me about the history of penicillin" (mostly correct, tests verification)
2. "What are the side effects of aspirin?" (medical domain, tests PubMed retrieval)
3. "Explain quantum computing" (technical, tests scientific accuracy)
4. A deliberately hallucinated paragraph (tests contradiction detection)

---

## 7. EVALUATION METHODOLOGY

### 7.1 Benchmarks

#### TruthfulQA (Lin et al., 2022)
- **What**: 817 questions across 38 categories designed to elicit LLM hallucinations
- **Why**: Gold standard for measuring LLM truthfulness
- **How we use it**:
  1. Run each question through an LLM to get a response
  2. Run response through VeriFactAI
  3. Compare VeriFactAI's verdicts against ground-truth labels
  4. Measure: Does VeriFactAI correctly flag known false answers?
- **Metrics**: Precision, Recall, F1 for hallucination detection
- **Baseline comparisons**: Raw LLM truthfulness vs. VeriFactAI-assisted

#### HaluEval (Li et al., 2023)
- **What**: 35,000 samples with hallucinated vs. correct LLM outputs across QA, dialogue, and summarization
- **Why**: Largest hallucination evaluation dataset
- **How we use it**: Run 500-1000 sample subset, measure binary classification accuracy (hallucinated vs. not)
- **Metrics**: Accuracy, Precision, Recall, F1, AUROC

#### FEVER (Thorne et al., 2018) — if time permits
- **What**: 185K claims labeled as SUPPORTED, REFUTED, or NOT ENOUGH INFO against Wikipedia
- **Why**: Direct benchmark for claim verification
- **How we use it**: Run claims through our retrieval + NLI pipeline, compare verdicts
- **Metrics**: Label accuracy (3-class), evidence retrieval accuracy

### 7.2 Evaluation Metrics

| Metric | What It Measures | Target |
|---|---|---|
| Hallucination Detection Precision | Of claims flagged as false, how many truly are? | ≥ 80% |
| Hallucination Detection Recall | Of truly false claims, how many did we catch? | ≥ 70% |
| Hallucination Detection F1 | Harmonic mean of precision/recall | ≥ 75% |
| Verification Accuracy | Overall verdict correctness (3-class) | ≥ 75% |
| Evidence Retrieval Recall@5 | Is relevant evidence in top 5 results? | ≥ 80% |
| Confidence Calibration Error | Expected Calibration Error (ECE) | ≤ 0.15 |
| End-to-End Latency | Time from input to annotated output | ≤ 15s (5 claims) |

### 7.3 Ablation Studies
To demonstrate each component's contribution, we'll measure performance when removing:
1. **No retrieval**: Just NLI on claim vs. random Wikipedia passage → shows retrieval value
2. **No NLI**: Just retrieval similarity threshold → shows NLI model value
3. **No confidence scoring**: Binary verdict only → shows calibration value
4. **No claim decomposition**: Verify full sentences instead of atomic claims → shows decomposition value

### 7.4 Comparative Analysis
- **Raw LLM** (no verification) vs. **VeriFactAI** on same questions
- Show: percentage of hallucinations caught, false positive rate
- Generate table + bar chart for the report

---

## 8. IMPLEMENTATION TIMELINE (3 DAYS)

### Day 1: Foundation (8-10 hours)

| Time | Task | Deliverable |
|---|---|---|
| Hour 1 | Project setup, git init, install dependencies, configure API keys | Working dev environment |
| Hour 2 | Download Wikipedia Simple English + PubMed QA datasets | Raw data available |
| Hour 3-4 | Chunk documents, generate embeddings (start on GPU/M-series) | Embeddings computed (can run in background) |
| Hour 5 | Build FAISS index from embeddings | Searchable vector index |
| Hour 6-7 | Implement `ClaimDecomposer` with LLM prompt | Claim extraction working |
| Hour 7-8 | Implement `EvidenceRetriever` with FAISS search | Retrieval pipeline working |
| Hour 8-9 | Integration test: text → claims → evidence | End-of-day smoke test |
| Hour 9-10 | Buffer for debugging, optimization | Clean Day 1 codebase |

**Day 1 Exit Criteria**: Given any text, system extracts claims and retrieves relevant evidence for each. Manually verify on 5 test cases.

### Day 2: Intelligence (8-10 hours)

| Time | Task | Deliverable |
|---|---|---|
| Hour 1-2 | Implement `VerdictEngine` with DeBERTa NLI model | NLI classification working |
| Hour 3 | Implement NLI aggregation logic (across top-5 evidence) | Multi-evidence verdicts |
| Hour 4 | Implement Bayesian confidence scoring formula | Calibrated scores |
| Hour 5-6 | Implement `AnnotatedOutputGenerator` (HTML + JSON) | Rich annotated output |
| Hour 7-8 | Build `VeriFactPipeline` orchestrator, connect all components | Full pipeline working |
| Hour 8-9 | End-to-end testing on 10 diverse queries | Pipeline validated |
| Hour 9-10 | Add caching, error handling, logging | Production-ready pipeline |

**Day 2 Exit Criteria**: Complete working pipeline. Input any question → get LLM response → see annotated output with verdicts, confidence, and citations.

### Day 3: Polish & Prove (8-10 hours)

| Time | Task | Deliverable |
|---|---|---|
| Hour 1-3 | Build Streamlit dashboard (all panels, visualizations) | Working UI |
| Hour 3-4 | Add interactive features (tooltips, expandable cards, examples) | Polished UI |
| Hour 4-6 | Run TruthfulQA evaluation (~800 questions) | Benchmark results |
| Hour 6-7 | Run HaluEval evaluation (~500 samples) | Second benchmark |
| Hour 7-8 | Generate evaluation plots (bar charts, confusion matrix, ROC) | Visual results |
| Hour 8-9 | Run ablation study (at least 2 of 4 ablations) | Component contribution analysis |
| Hour 9-10 | Write report, create architecture diagram, document results | Final report |

**Day 3 Exit Criteria**: Polished demo + quantified evaluation results + comprehensive report.

---

## 9. COMPLETE FILE STRUCTURE

```
verifactai/
├── app.py                          # Streamlit main application
├── config.py                       # Configuration (API keys, model paths, thresholds)
├── requirements.txt                # All Python dependencies
├── README.md                       # Project overview and setup instructions
│
├── core/
│   ├── __init__.py
│   ├── claim_decomposer.py         # LLM-based atomic claim extraction
│   ├── evidence_retriever.py       # FAISS vector search over knowledge corpus
│   ├── verdict_engine.py           # NLI classification + Bayesian confidence
│   ├── annotator.py                # Annotated output generation (HTML/JSON/MD)
│   └── pipeline.py                 # VeriFactPipeline orchestrator
│
├── data/
│   ├── download_data.py            # Script to download Wikipedia + PubMed
│   ├── build_index.py              # Chunk, embed, build FAISS index
│   ├── index/                      # FAISS index files (generated)
│   └── metadata/                   # Chunk metadata JSON files (generated)
│
├── evaluation/
│   ├── evaluate.py                 # Main evaluation runner
│   ├── benchmarks.py               # TruthfulQA + HaluEval loaders
│   ├── metrics.py                  # Precision, recall, F1, ECE calculations
│   ├── ablation.py                 # Ablation study runner
│   └── plots.py                    # Generate evaluation visualizations
│
├── utils/
│   ├── __init__.py
│   └── helpers.py                  # Shared utilities (logging, timing, etc.)
│
├── tests/
│   ├── test_decomposer.py          # Unit tests for claim decomposition
│   ├── test_retriever.py           # Unit tests for evidence retrieval
│   ├── test_verdict.py             # Unit tests for NLI verdict engine
│   └── test_pipeline.py            # Integration tests for full pipeline
│
└── assets/
    ├── architecture_diagram.png    # System architecture (for report)
    └── examples/                   # Pre-loaded example queries
```

---

## 10. DEPENDENCIES

```
# requirements.txt
# Core ML/NLP
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
spacy>=3.7.0
datasets>=2.15.0

# LLM APIs
anthropic>=0.39.0
openai>=1.6.0

# Web UI
streamlit>=1.29.0
plotly>=5.18.0
streamlit-extras>=0.3.0

# Data processing
pandas>=2.1.0
numpy>=1.24.0
tqdm>=4.66.0

# Evaluation
scikit-learn>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0

# Utilities
python-dotenv>=1.0.0
loguru>=0.7.0
```

---

## 11. RISK ASSESSMENT & MITIGATION

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| LLM API rate limits | Medium | High | Implement retry with exponential backoff + response caching |
| FAISS index too large for local machine | Low | High | Use Wikipedia Simple English (smaller); or use cloud VM |
| Embedding computation takes too long | Medium | Medium | Start Day 1 Hour 2; use GPU if available; pre-compute overnight |
| NLI model accuracy insufficient | Low | High | DeBERTa-v3-base is 92% on MultiNLI — well-validated |
| Claim decomposition quality poor | Medium | Medium | Extensive prompt engineering + fallback to sentence-level |
| Time overrun on Day 1 | Medium | High | Prioritize: if indexing is slow, use smaller corpus (50K articles) |
| Streamlit UI takes too long | Low | Medium | Use pre-built Streamlit components; skip comparison mode if needed |

---

## 12. SCALABILITY & FUTURE WORK

### Immediate Extensions (post-submission)
- Real-time web search fallback (when corpus doesn't have evidence)
- Multi-language support (Hindi, Spanish, etc.)
- Fine-tuned claim decomposer (instead of LLM API calls)
- Browser extension (verify any webpage text)

### Research Extensions (publishable)
- Temporal hallucination detection (outdated facts vs. current truth)
- Multi-hop claim verification (claims requiring multiple evidence pieces)
- Adversarial robustness (can the system be fooled by cleverly worded hallucinations?)
- Human-in-the-loop: user feedback improves confidence calibration over time

### Production Scale
- Replace FAISS with Pinecone/Weaviate for distributed vector search
- Add streaming API for real-time verification
- Kubernetes deployment for horizontal scaling
- Continuous corpus update pipeline

---

## 13. WHAT MAKES THIS STANFORD/MIT GRADE

1. **Addresses the #1 problem in applied AI** — hallucination detection is the most active research area at every top AI lab
2. **Novel architecture** — no existing open-source system combines claim decomposition + dense retrieval + NLI + Bayesian confidence in a single pipeline
3. **Rigorous evaluation** — benchmarked on TruthfulQA, HaluEval with published baselines; ablation studies demonstrate component contributions
4. **Practical utility** — directly usable by anyone deploying LLMs; every company needs this
5. **Research-grade methodology** — grounded in FActScore, DPR, DeBERTa literature; not just tool assembly
6. **Measurable impact** — quantifiable reduction in hallucination exposure
7. **Publishable** — architecture + evaluation methodology suitable for ACL/EMNLP workshop paper
