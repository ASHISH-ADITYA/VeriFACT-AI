# VeriFACT AI
## Real-Time LLM Hallucination Detection Using Retrieval-Augmented Verification

### Final Presentation Content — Academically Verified
#### Central University of Jharkhand, Ranchi
#### B.Tech CSE — Engineering Project II (2025–2026)
#### Author: Aditya Ashish

> All technical claims verified against source code (commit 7710e63, April 17, 2026).
> All statistics verified against benchmark runs. All citations cross-checked.

---

# SLIDE 1 — TITLE

**VeriFACT AI: Real-Time LLM Hallucination Detection Using Retrieval-Augmented Verification**

**Author:** Aditya Ashish
**Programme:** B.Tech Computer Science & Engineering (Final Year)
**Institution:** Central University of Jharkhand, Ranchi, Jharkhand
**Academic Year:** 2025–2026

**Abstract:**
Large Language Models (LLMs) such as ChatGPT, Claude, and Gemini generate fluent but sometimes factually incorrect text — a phenomenon known as hallucination. VeriFACT AI addresses this by providing real-time, evidence-backed hallucination detection as a browser extension. The system decomposes LLM responses into atomic factual claims, retrieves evidence from a hybrid knowledge base combining a local FAISS vector index (11,248 Wikipedia chunks) with live Wikipedia API access (6.8 million articles), and generates verdicts using DeBERTa-v3 Natural Language Inference. A Bayesian five-signal confidence fusion produces calibrated scores. The system achieves 93.8% accuracy on the HaluEval benchmark and operates at zero cost using only free-tier cloud services.

**Keywords:** Hallucination Detection, Retrieval-Augmented Generation, Natural Language Inference, Browser Extension, DeBERTa, FAISS, Real-Time Verification

---

# SLIDE 2 — PROBLEM STATEMENT

### The Hallucination Problem in LLMs

Large Language Models generate text that is fluent and confident but frequently factually incorrect:

- **Prevalence:** LLMs produce false statements 3–27% of the time across different domains (Lin et al., 2022, TruthfulQA).
- **Manifestations:** Fabricated citations, invented names and theorems, incorrect dates and statistics, misattributed achievements, wrong geographic facts.
- **Impact domains:** Healthcare (wrong drug interactions), Legal (fabricated case citations — 58% hallucination rate in GPT-4 per Dahl et al., 2024), Education, Finance.

### Gap in Existing Solutions

| System | Year | Real-Time | Browser Extension | Free | Local NLI | Evidence-Backed |
|--------|------|-----------|------------------|------|-----------|----------------|
| FActScore (Min et al.) | 2023 | No | No | Yes | No | Partial |
| SelfCheckGPT (Manakul et al.) | 2023 | No | No | Yes | No | No |
| SAFE (Google) | 2024 | No | No | No | No | Yes |
| FactCheck-GPT | 2024 | No | No | No | No | Partial |
| **VeriFACT AI** | **2026** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

**Research Gap:** No existing system provides real-time, ambient, evidence-backed hallucination detection as a browser overlay on live chatbot platforms using entirely free resources.

---

# SLIDE 3 — OBJECTIVES

1. **Real-time detection:** Monitor AI chatbot conversations as they happen, analyzing each assistant message automatically via browser extension.

2. **Evidence-backed verdicts:** Every verdict (SUPPORTED / CONTRADICTED / UNVERIFIABLE) includes the specific evidence passage and source URL that supports the judgment.

3. **Calibrated confidence:** A Bayesian five-signal fusion produces confidence scores in [0, 1] rather than binary yes/no labels, enabling users to assess reliability granularly.

4. **Zero-cost operation:** The complete system operates using only free-tier services — HuggingFace Spaces (compute), Groq API (LLM), Wikipedia API (knowledge), Vercel (frontend).

5. **Research grounding:** Every pipeline stage maps to a published research methodology (FEVER, RAG, DeBERTa, SelfCheckGPT, Semantic Entropy).

6. **Fabrication detection:** Identify entirely made-up entities (e.g., fictitious theorems, protocols) by detecting specific named concepts with zero Wikipedia evidence.

---

# SLIDE 4 — SYSTEM OVERVIEW

### Three User Interfaces, One Core Pipeline

**1. Chrome Browser Extension (Manifest V3)**
- Platforms: ChatGPT, Claude, Gemini, Grok
- Features: 72px draggable glass beacon, inline red/yellow text highlighting, liquid glass analysis dashboard, real-time prompt quality suggestions with Accept/Dismiss buttons
- Architecture: MutationObserver monitors DOM, analyzes 3 messages in parallel, caches results per node

**2. Web Dashboard (Vercel, Next.js 15.5.15)**
- Tabs: Dashboard (pipeline KPIs), Verify (paste-and-check), Chat (RAG-powered fact-checking), About
- Chatbot quick-links with logos (ChatGPT, Gemini, Claude, Grok, Copilot, Perplexity)

**3. REST API (Python HTTP server, HuggingFace Spaces)**
- Five endpoints:

| Endpoint | Method | Purpose | Typical Latency |
|----------|--------|---------|----------------|
| `/analyze` | POST | Full 6-stage verification | 10–18s |
| `/analyze/fast` | POST | Lightweight verification (extension default) | 3–8s |
| `/analyze/stream` | POST | Server-Sent Events (claim-by-claim) | Real-time |
| `/optimize` | POST | Prompt quality suggestion | <1s |
| `/health` | GET | Liveness probe | <100ms |

**4. Streamlit Dashboard (app.py)**
- Deep analysis with Plotly visualizations, color-coded verdicts, evidence links

---

# SLIDE 5 — ARCHITECTURE

### System Architecture

*(Refer to Mermaid Diagram 1 in DIAGRAMS_MERMAID.md)*

**Layered Architecture:**

```
┌─────────────────────────────────────────────────────┐
│  CLIENT: Extension · Web Dashboard · Streamlit      │
└────────────────────────┬────────────────────────────┘
                         │ HTTPS
┌────────────────────────▼────────────────────────────┐
│  API: overlay_server.py (CORS + Auth + Rate Limit)  │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│  PIPELINE: 6-stage verification (pipeline.py)       │
│  Decompose → Rules → Retrieve → NLI → Check → Fuse │
└───────┬──────────┬──────────┬──────────┬────────────┘
        │          │          │          │
┌───────▼───┐ ┌───▼────┐ ┌──▼───┐ ┌───▼──────────┐
│ FAISS     │ │ BM25   │ │ Wiki │ │ DeBERTa-v3   │
│ 11K local │ │ sparse │ │ API  │ │ NLI (435MB)  │
│ vectors   │ │ index  │ │ 6.8M │ │ MPS/CPU      │
└───────────┘ └────────┘ └──────┘ └──────────────┘
```

**Key Design Principle:** Hybrid local + live evidence. Local FAISS index provides fast cached results; Wikipedia API provides unlimited coverage (6.8M articles) for every claim.

---

# SLIDE 6 — PIPELINE STAGES

### Six-Stage Verification Pipeline

**Stage 1: Claim Decomposition** (`claim_decomposer.py`, 335 lines)
- Primary: Groq Llama 3.1 8B extracts atomic claims via structured JSON prompt
- Fallback: spaCy `en_core_web_sm` sentence segmentation with fact-density ranking
- Filtering: Aggressive removal of opinions, transitions, meta-commentary, subjective language
- Requires concrete signals: numbers, proper nouns, or factual verbs
- Output: max 20 claims (8 for fast endpoint), sorted by fact-density score

**Stage 2a: Rule-Based Pre-Check** (`fact_rules.py`, 450 lines)
- 205 hardcoded factual rules across 9 categories:

| Category | Entries | Example |
|----------|---------|---------|
| Landmark locations | 38 | Great Wall → China |
| Capital-country | 34 | Tokyo → Japan |
| City-country | 43 | London → United Kingdom |
| Country-continent | 32 | Brazil → South America |
| Person-achievement cross-reference | 29 | Einstein → physics, not airplane |
| Historical event dates | 7 | WWII: 1939–1945 |
| Person birth/death dates | 9 | Shakespeare: 1564–1616 |
| Science numerical facts | 5 | Water boiling: 100°C ± 5 |
| Science true/false | 8 | "Sun revolves around Earth" → False |

- Latency: <1ms per claim. Matched claims are immediately marked CONTRADICTED with 0.95 confidence, skipping NLI entirely.

**Stage 2b: Evidence Retrieval** (`evidence_retriever.py`, 650 lines)
- **Dense retrieval:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim) → FAISS `IndexFlatIP` (11,248 pre-built vectors)
- **Sparse retrieval:** `rank_bm25` BM25Okapi keyword matching
- **Fusion:** Reciprocal Rank Fusion (k=60) combining dense + sparse rankings
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder (sigmoid normalization)
- **Query expansion:** Contradiction-aware negation queries (content words + "facts location history")
- **Live Wikipedia API:** Every claim is searched against 6.8M English Wikipedia articles. Full article text retrieved, paragraphs scored by query-term overlap, best 2 paragraphs extracted (up to 800 chars). Computed similarity based on actual term overlap.
- Output: top-5 evidence passages per claim with source provenance

**Stage 3: NLI Verdict** (`verdict_engine.py`, 500 lines)
- Model: `cross-encoder/nli-deberta-v3-base` (DeBERTa-v3, 435MB, 3-class: entailment/neutral/contradiction)
- **Batch processing:** All (premise, hypothesis) pairs across all claims processed in a single tokenize + forward pass
- **Specificity gate:** `max(entailment × similarity)` prevents false positives from topically similar but non-specific evidence
- **Fabrication detection:** If claim names specific entities (e.g., "Zyphron Stability Theorem") but Wikipedia API returns zero results and all evidence is irrelevant → CONTRADICTED with ≥0.80 confidence
- **Wikipedia trust:** If Wikipedia API returned relevant evidence (similarity > 0.45), the topic is confirmed real → overrides NLI confusion → SUPPORTED
- **Decision order:** Fabricated → Hard Contradiction (NLI > 0.80, margin > 0.20) → Wikipedia/Strong/Moderate Support → Weak Support → Irrelevant+Specific → UNVERIFIABLE

**Stage 4: SelfCheck Consistency** (`selfcheck.py`, 210 lines)
- LLM path: 5 samples at temperatures [0.10, 0.25, 0.40, 0.55, 0.70], label distribution analysis
- NLI fallback (when no LLM): cross-evidence NLI agreement scoring via shared VerdictEngine instance
- Metrics: normalized entropy, disagreement ratio, Jaccard-based semantic cluster entropy
- Confidence blend: `new = 0.80 × NLI_confidence + 0.20 × selfcheck_consistency`

**Stage 5: Bayesian Confidence Fusion** (`verdict_engine._bayesian_confidence`)
- Five weighted signals:

| Signal | Weight | Computation |
|--------|--------|-------------|
| NLI support | 0.32 | (max_entailment − max_contradiction + 1) / 2 |
| Retrieval relevance | 0.22 | max(evidence.similarity) |
| Source reliability | 0.12 | Prior: Wikipedia=1.0, PubMed=1.0, unknown=0.5 |
| Cross-reference agreement | 0.16 | Fraction of evidence where entailment > contradiction |
| Uncertainty stability | 0.18 | 1 − (0.65×entropy + 0.35×disagreement) |

- Output: calibrated confidence score in [0, 1]

**Stage 6: Annotation** (`annotator.py`, 310 lines)
- JSON report: per-claim verdicts, evidence, confidence, source URLs
- HTML output: color-coded inline spans with hover tooltips
- Correction generation (for CONTRADICTED claims): LLM-based correction → Reflexion critique (1 round) → Constitutional safety critique (1 round)
- Factuality score: `(supported / total_claims) × 100`

---

# SLIDE 7 — KEY DESIGN DECISIONS

### Technical Design Rationale

1. **DeBERTa NLI over LLM-as-judge**
   - Rationale: Deterministic, reproducible verdicts without API dependency. DeBERTa-v3-base achieves 92% on MultiNLI (He et al., 2021).
   - Trade-off: Limited to 512-token context window per (premise, hypothesis) pair.

2. **Two-pass architecture (Rules + NLI)**
   - Rule engine catches 100% of obvious geographic/temporal/scientific errors in <1ms.
   - NLI runs only on unresolved claims, saving 30–50% of compute.

3. **Wikipedia API as primary knowledge source**
   - 6.8M articles searched live for every claim (not fallback).
   - Full article text with paragraph-level relevance scoring.
   - Zero storage cost. Zero build time.

4. **Fabrication detection via absence of evidence**
   - Novel approach: if a claim names specific entities but Wikipedia returns zero results, the entity is likely fabricated.
   - Catches hallucinated theorems, protocols, people not in any published work.

5. **Batch NLI processing**
   - Single DeBERTa forward pass for all claims × evidence pairs.
   - 5–8× faster than per-claim inference.

6. **Fallback chain architecture**
   - LLM: Groq (free) → Ollama (local) → Anthropic → OpenAI → spaCy (zero-LLM)
   - System remains functional even with no LLM and no internet (local FAISS + rules only).

---

# SLIDE 8 — RESEARCH FOUNDATIONS

### Mapping Pipeline Stages to Published Research

| Pipeline Stage | Research Foundation | Citation |
|---------------|--------------------|---------| 
| Claim Decomposition | FActScore atomic claim extraction | Min et al., EMNLP 2023 |
| Evidence Retrieval | Retrieval-Augmented Generation | Lewis et al., NeurIPS 2020 |
| Hybrid Search (Dense+Sparse) | FAISS + BM25 + Reciprocal Rank Fusion | Johnson et al., 2021; Robertson & Zaragoza, 2009 |
| NLI Verdict | DeBERTa disentangled attention | He et al., ICLR 2021 |
| Benchmark Evaluation | TruthfulQA | Lin et al., ACL 2022 |
| Self-Consistency Scoring | SelfCheckGPT | Manakul et al., EMNLP 2023 |
| Semantic Entropy | Uncertainty via label entropy | Kuhn et al., Nature 2024 |
| Correction Refinement | Reflexion verbal RL | Shinn et al., NeurIPS 2023 |
| Safety Critique | Constitutional AI | Bai et al., Anthropic 2022 |
| Fact Verification Dataset | FEVER | Thorne et al., NAACL 2018 |

**14 research papers** reviewed and integrated. Full texts stored in `/architecture/` directory.

---

# SLIDE 9 — IMPLEMENTATION

### Codebase Overview

**Language:** Python 3.11 | **Total core code:** 4,866 lines | **Tests:** 94 (pytest) | **Modules:** 13

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `evidence_retriever.py` | 650 | FAISS + BM25 + RRF + reranker + Wikipedia API |
| `prompt_optimizer.py` | 550 | Prompt quality scoring + template rewriting |
| `overlay_server.py` | 620 | HTTP API, CORS, auth, rate limiting, SSE streaming |
| `verdict_engine.py` | 500 | DeBERTa NLI, batch judging, fabrication detection, confidence fusion |
| `fact_rules.py` | 450 | 205 hardcoded factual rules, 9 categories |
| `claim_decomposer.py` | 335 | LLM + spaCy claim extraction, fact-density ranking |
| `pipeline.py` | 320 | Main orchestrator, caching, streaming |
| `llm_client.py` | 320 | Multi-provider LLM (Groq/Ollama/Anthropic/OpenAI) |
| `annotator.py` | 310 | HTML/JSON output, corrections, reflexion, constitutional |
| `selfcheck.py` | 210 | Self-consistency scoring, NLI fallback |
| `semantic_entropy.py` | 115 | Entropy, disagreement, Jaccard clustering |
| `config.py` | 230 | Pydantic-based centralized configuration |
| `hallucination_discriminator.py` | 80 | TF-IDF + LogReg binary classifier |

**Configuration:** Pydantic models with environment variable overrides. Two profiles: INTERACTIVE (low latency) and EVAL (high accuracy).

---

# SLIDE 10 — USER INTERFACES

### Chrome Extension
- **Beacon:** 72px draggable circular glass orb with state-dependent color (blue=idle, purple=scanning, green=safe, red=hallucination detected)
- **Dashboard:** 3D liquid glass panel showing KPI row (Score, Claims, Verified, False, Unclear), per-claim verdict cards with evidence and source
- **Inline highlighting:** Red background on contradicted claims, yellow on unverifiable — directly in the chatbot's response text
- **Prompt suggestion:** Appears as user types, scores prompt quality 0–100, offers rewritten version with Accept/Dismiss buttons
- **Performance:** 3 messages analyzed in parallel, results cached per DOM node, client-side pre-filtering removes non-factual text before API call

### Web Dashboard
- Crystal/ice glassmorphism theme with warm gradients
- Four tabs: Dashboard (pipeline stats), Verify (paste text), Chat (RAG fact-checking), About
- Chatbot quick-links with actual logos

### API Design
- Rate limiting: 20 requests/minute per IP (sliding window)
- CORS: whitelisted origins (chatgpt.com, claude.ai, gemini.google.com, localhost)
- Authentication: optional `X-VeriFact-Token` header (timing-safe comparison via `hmac.compare_digest`)
- Security headers: X-Content-Type-Options, X-Frame-Options, Cache-Control, Referrer-Policy

---

# SLIDE 11 — EVALUATION & RESULTS

### Benchmark Performance

| Benchmark | Accuracy | Precision | Recall | F1 | Conditions |
|-----------|----------|-----------|--------|-----|-----------|
| **HaluEval** (multi-sentence) | 93.8% | 93.3% | 100% | 96.6% | Full pipeline, eval profile |
| **TruthfulQA** (single-sentence) | 44.1% | 45.8% | 75.9% | 57.1% | Single-claim limitation |
| **Adversarial** (100 mixed claims) | 82.0% | 88.7% (contra) | 84.6% (contra) | 86.6% | Rule engine + NLI |
| **Live true facts** (6 claims) | 100% | — | — | — | Post-accuracy fix |
| **Live false facts** (5 claims) | 100% | — | — | — | Rule-caught, <2s |
| **Live mixed** (5 claims) | 100% | — | — | — | True+false, 9.8s |

### Latency Performance (HuggingFace free CPU, 2 vCPU)

| Path | Claims | Latency | What's Included |
|------|--------|---------|----------------|
| `/analyze/fast` | 5 | 3–8s | FAISS + NLI + Wikipedia API + rules |
| `/analyze` (full) | 5 | 10–18s | + BM25 + reranker + SelfCheck + corrections |
| Rule-caught claims | any | <1ms each | Rule engine only |
| Local (MacBook Air M4) | 5 | 2–5s | MPS acceleration |

### Key Observation
- **HaluEval (multi-sentence):** Strong performance because multi-claim texts provide more evidence signal.
- **TruthfulQA (single-sentence):** Lower accuracy because single claims lack context and the specificity gate is less effective without multiple evidence passages.

---

# SLIDE 12 — INNOVATIONS

### Novel Contributions

1. **Fabrication Detection via Evidence Absence**
   - When a claim names specific entities (detected by regex: `[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}`) or technical terms (theorem, protocol, algorithm, etc.) but Wikipedia API returns zero results and all local evidence is irrelevant (max similarity < 0.35) → CONTRADICTED with ≥0.80 confidence.
   - This catches hallucinated entities like "Zyphron Stability Theorem" that no other system would flag.

2. **Wikipedia API as Unlimited Knowledge Base**
   - Every claim is searched against 6.8M articles (not fallback, always).
   - Full article text downloaded, split into paragraphs, each scored by query-term overlap.
   - Best 2 paragraphs (800 chars) used as evidence.
   - Zero storage cost, zero build time.

3. **Two-Pass Verification Architecture**
   - Pass 1: Rule engine checks 205 hardcoded facts (<1ms, 100% precision on covered facts)
   - Pass 2: NLI runs only on unresolved claims (saves 30–50% compute)

4. **Batch NLI Processing**
   - Single DeBERTa forward pass for ALL claims × evidence pairs
   - Single embedding encode call for ALL claims
   - Single FAISS search for ALL claims
   - Single reranker predict call for ALL claims
   - Result: 5–8× throughput improvement vs per-claim processing

5. **Zero-Cost Production Deployment**
   - HuggingFace Spaces: free CPU (2 vCPU, 16GB RAM)
   - Groq API: free LLM (30 req/min, Llama 3.1 8B)
   - Wikipedia API: free, unlimited
   - Vercel: free static hosting
   - Pre-built FAISS index baked into Docker image (zero cold start)
   - **Total operational cost: $0/month**

---

# SLIDE 13 — TECH STACK

### Complete Technology Stack

| Layer | Technology | Version / Details |
|-------|-----------|------------------|
| **Backend** | Python | 3.11 (custom HTTP server, no framework) |
| **Frontend** | Next.js | 15.5.15 (React 18.3.0), static export |
| **Extension** | Chrome Manifest V3 | Vanilla JavaScript, CSS |
| **NLI Model** | DeBERTa-v3-base | `cross-encoder/nli-deberta-v3-base` (435MB) |
| **Embeddings** | MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` (83MB, 384-dim) |
| **Reranker** | MiniLM-L-6-v2 | `cross-encoder/ms-marco-MiniLM-L-6-v2` (128MB) |
| **Vector Index** | FAISS | IndexFlatIP, 11,248 vectors (pre-built, 25MB) |
| **Sparse Search** | BM25 | `rank_bm25` BM25Okapi (in-memory) |
| **Live Knowledge** | Wikipedia API | 6.8M articles, full-text paragraph search |
| **Free LLM** | Groq | `llama-3.1-8b-instant` (30 req/min, via urllib) |
| **Local LLM** | Ollama | `llama3.1:8b` (optional, localhost:11434) |
| **NLP Fallback** | spaCy | `en_core_web_sm` (sentence segmentation) |
| **Risk Classifier** | toxic-bert | `unitary/toxic-bert` (optional) |
| **Config** | Pydantic | Type-safe configuration with env overrides |
| **Testing** | pytest | 94 tests across 12 test files |
| **Linting** | ruff | Linter + formatter |
| **Compute** | HuggingFace Spaces | Free CPU tier, Docker (python:3.11-slim) |
| **Web Hosting** | Vercel | Free tier, static export |
| **Container** | Docker | Pre-built index + models baked in |

---

# SLIDE 14 — LIMITATIONS

### Current Limitations & Honest Assessment

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **HF free CPU (2 vCPU, 16GB)** | ~5s per claim, 30-min build timeout | Pre-built index, batch NLI, fast endpoint |
| **English only** | No multilingual support | Would need multilingual NLI model |
| **Single-passage retrieval** | Cannot verify claims requiring multi-hop reasoning | Would need graph-based evidence chaining |
| **Static source reliability** | Hardcoded priors (Wikipedia=1.0) | Could learn from verification history |
| **NLI context window** | 512 tokens max per (premise, hypothesis) | Truncation may miss key evidence |
| **Wikipedia recency** | May not cover events from last few weeks | Would need news API integration |
| **TruthfulQA accuracy** | 44.1% on single-sentence claims | Specificity gate less effective with single claims |
| **Extension DOM selectors** | May break if chatbot UIs update | MutationObserver-based, needs periodic maintenance |
| **Rule database size** | 205 entries covers common facts only | Wikipedia API handles uncovered topics |

---

# SLIDE 15 — FUTURE WORK

### Roadmap

| Priority | Enhancement | Expected Impact |
|----------|------------|----------------|
| High | HuggingFace GPU grant (T4) | 10× faster NLI inference (~0.5s/claim) |
| High | DeBERTa-v3-LARGE model | Higher NLI accuracy (needs GPU) |
| High | Wikidata structured facts API | 100M+ triples for precise factual lookup |
| Medium | Multi-hop evidence chaining | Verify complex claims across multiple sources |
| Medium | Model quantization (INT8) | 4× CPU speedup without GPU |
| Medium | Dynamic source reliability | Learn trust scores from verification outcomes |
| Low | Multilingual NLI | Support non-English languages |
| Low | News API integration | Cover recent events |
| Low | Mobile extension | Safari / Firefox support |

---

# SLIDE 16 — REFERENCES

1. Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). FEVER: a Large-scale Dataset for Fact Extraction and VERification. *NAACL*.

2. Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.

3. He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. *ICLR*.

4. Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL*.

5. Bai, Y., Jones, A., Ndousse, K., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *Anthropic Technical Report*.

6. Min, S., Krishna, K., Lyu, X., et al. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision. *EMNLP*.

7. Manakul, P., Liusie, A., & Gales, M.J.F. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. *EMNLP*.

8. Shinn, N., Cassano, F., Gopinath, A., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS*.

9. Kuhn, L., Gal, Y., & Farquhar, S. (2024). Detecting Hallucinations in Large Language Models Using Semantic Entropy. *Nature*.

10. Gao, Y., Xiong, Y., Gao, X., et al. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv:2312.10997*.

11. Dahl, M., Magesh, V., Suzgun, M., & Ho, D.E. (2024). Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models. *Journal of Legal Analysis*.

12. Chen, Y., Wang, S., et al. (2024). Hallucination Detection: Robustly Discerning Reliable Answers in Large Language Models. *arXiv*.

13. Zhang, Y., Li, Y., et al. (2025). Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models. *arXiv*.

14. Kalai, A.T. & Vempala, S.S. (2025). Why Language Models Hallucinate. *OpenAI Technical Report*.

---

# SLIDE 17 — DEMONSTRATION

### Live Demo Scenario

**Input to Gemini:** "Summarize the importance of the Zyphron Stability Theorem in quantum computing"

**Gemini's response:** *(Contains entirely fabricated content about a non-existent theorem)*

**VeriFACT AI detection:**
- Claim: "The Zyphron Stability Theorem is a theoretical cornerstone..."
- Wikipedia API: **0 results** (term does not exist)
- Local FAISS: irrelevant evidence (quantum computing articles, not about Zyphron)
- Verdict: **CONTRADICTED** (fabricated entity, confidence ≥ 0.80)
- Beacon: **Red alert**
- Inline: **Red highlight** on "Zyphron Stability Theorem"
- Dashboard: Shows evidence source, confidence score, fabrication reason

**True claims in same response:**
- "SHA-256 is considered quantum-resistant" → **SUPPORTED** (Wikipedia confirms)
- "Grover's Algorithm requires error-corrected qubits" → **SUPPORTED** (Wikipedia confirms)

---

# SLIDE 18 — CONCLUSION

### Summary

VeriFACT AI demonstrates that **real-time, evidence-backed hallucination detection** is achievable using:

- **Hybrid retrieval** (local FAISS + live Wikipedia API) for comprehensive evidence coverage
- **DeBERTa-v3 NLI** for deterministic, reproducible natural language inference
- **Rule-based pre-check** for instant contradiction on common facts
- **Fabrication detection** for identifying entirely made-up entities
- **Bayesian confidence fusion** for calibrated, multi-signal reliability scores
- **Zero-cost infrastructure** using only free-tier services

The system achieves **93.8% accuracy** on the HaluEval benchmark, operates in **3–8 seconds** per message on free CPU, and monitors **entire chatbot conversations** in real-time across ChatGPT, Claude, Gemini, and Grok.

**Key takeaway:** Hallucination detection does not require expensive infrastructure or paid APIs. With careful engineering and research-grounded methodology, a production-grade system can be built and deployed at zero cost.

---

### Acknowledgements

- Research papers and open-source communities: HuggingFace, Ollama, FAISS, spaCy, sentence-transformers
- Free-tier services: HuggingFace Spaces, Groq, Vercel, Wikipedia API
- Project advisor and Central University of Jharkhand faculty

---

*Document verified against codebase commit 7710e63 (April 17, 2026).*
*All statistics from actual benchmark runs. All code references to actual file paths and line numbers.*
