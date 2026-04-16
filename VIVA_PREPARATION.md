# VeriFACT AI — Complete Viva Preparation Guide

> Ye file padh lo puri, presentation aur viva dono ke liye ready ho jaoge.
> Har cheez technical + simple dono mein explained hai.

---

## 1. PROJECT KA EK-LINE SUMMARY

**VeriFACT AI ek Chrome browser extension hai jo ChatGPT, Claude, Gemini jaise AI chatbots ki responses ko real-time mein fact-check karta hai — galat claims ko red mein highlight karta hai aur evidence ke saath batata hai ki kya sahi hai kya galat.**

English: "VeriFACT AI is a real-time hallucination detection system that monitors AI chatbot conversations, identifies factually incorrect claims, and highlights them with evidence-backed verdicts."

---

## 2. PROBLEM KYA HAI? (Why This Project?)

### Technical:
LLMs (Large Language Models) like ChatGPT **hallucinate** — they generate text that sounds confident but is factually wrong. Research shows this happens **3-27% of the time** (Source: TruthfulQA paper, Lin et al., ACL 2022).

### Simple:
ChatGPT kabhi kabhi jo bolta hai wo sach nahi hota. Wo aise bolega jaise use full confidence hai, lekin actually wo cheez exist hi nahi karti. Jaise agar tum pucho "Zyphron Stability Theorem kya hai?" to ChatGPT ek pura essay likh dega — lekin Zyphron Theorem exist hi nahi karta. Ye **hallucination** hai.

### Examples of Hallucination:
- "The Haldiram Novel Theory revolutionized computer science" → **Fabricated** (ye exist nahi karta)
- "Einstein invented the telephone" → **Wrong** (Alexander Graham Bell ne invent kiya)
- "The Great Wall of China is in South America" → **Wrong** (China mein hai)

### Why Existing Tools Don't Work:
| Tool | Problem |
|------|---------|
| SelfCheckGPT | Offline, no real-time UI |
| FActScore | Offline, research-only |
| SAFE (Google) | Not free, not public |
| **VeriFACT AI** | **Real-time, free, browser extension** |

---

## 3. SYSTEM KAISE KAAM KARTA HAI? (Complete Flow)

### Step-by-Step (Jab User Gemini Pe Chat Karta Hai):

```
Step 1: User Gemini pe question puchta hai
Step 2: Gemini reply deta hai
Step 3: VeriFACT extension automatically detect karta hai ki naya message aaya
Step 4: Extension message ka text nikal ke backend ko bhejta hai
Step 5: Backend 6 stages se text ko check karta hai
Step 6: Result aata hai — kaunsa claim sahi hai, kaunsa galat
Step 7: Extension galat claims ko RED mein highlight kar deta hai
Step 8: User beacon click kare to dashboard dikhta hai with details
```

---

## 4. THE 6-STAGE PIPELINE (Ye Sab Backend Mein Hota Hai)

### Stage 1: CLAIM DECOMPOSITION (Claim Todna)

**Full Form:** Claim = factual statement, Decomposition = break into pieces

**Kya karta hai:** Chatbot ka pura paragraph leke usme se individual facts nikalta hai.

**Example:**
```
Input: "Napoleon lost Waterloo in 1815. The Earth has 3 moons."

Output:
  Claim 1: "Napoleon lost Waterloo in 1815"
  Claim 2: "The Earth has 3 moons"
```

**Technology:**
- **Groq API** → Free cloud LLM service
  - Model: **Llama 3.1 70B** (Meta ka open-source LLM, 70 billion parameters)
  - 30 requests/minute free mein milta hai
  - 500 tokens/second speed — bahut fast
- **spaCy** → Backup (agar Groq down ho)
  - Full form: Not an acronym, it's a library name
  - Python NLP library — sentence splitting karta hai
  - Model: `en_core_web_sm` (English language small model, 43MB)

**File:** `claim_decomposer.py` (335 lines)

**Fact-Density Score:** Har sentence ko score deta hai — jisme numbers, names, dates hain unko priority milti hai:
```
score = 1.0 + 2.0×(has_number) + 0.5×(proper_nouns) + 0.5×(factual_verbs)
```

---

### Stage 2a: RULE ENGINE (Instant Check)

**Kya karta hai:** 205 hardcoded facts ke against check karta hai. Agar match mil gaya to turant CONTRADICTED bol deta hai — AI model ko call karne ki zaroorat nahi.

**Speed:** <1 millisecond (almost instant)

**9 Categories:**
| Category | Count | Example |
|----------|-------|---------|
| Landmarks | 38 | Great Wall → China |
| Capitals | 34 | Tokyo → Japan |
| Cities | 43 | London → UK |
| Countries | 32 | Brazil → South America |
| Person achievements | 29 | Einstein → physics, not telephone |
| Historical dates | 7 | WWII: 1939-1945 |
| Person dates | 9 | Shakespeare: 1564-1616 |
| Science numbers | 5 | Water boils at 100°C |
| Science T/F | 8 | "Sun revolves around Earth" → False |

**Simple mein:** Ye ek dictionary hai — agar galat fact dictionary mein mil gaya to turant bol deta hai "GALAT HAI" bina kisi AI ke.

**File:** `fact_rules.py` (464 lines)

---

### Stage 2b: EVIDENCE RETRIEVAL (Saboot Dhundhna)

**Kya karta hai:** Har claim ke liye Wikipedia se evidence dhundhta hai — kya sahi hai kya galat, uska proof.

**5 Layers of Search:**

#### Layer 1: FAISS Dense Search (Vector Search)
- **FAISS** = Facebook AI Similarity Search
  - Facebook ne banaya hai, open-source
  - Vectors mein search karta hai (numbers ki list mein similar numbers dhundhta hai)
- **Index:** IndexFlatIP (Flat Inner Product — exact cosine similarity search)
  - 11,248 Wikipedia chunks stored as 384-dimensional vectors
  - 25 MB size
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
  - MiniLM = Mini Language Model (Microsoft ka compact model)
  - L6 = 6 layers, v2 = version 2
  - 83 MB size, 384-dimensional output
  - Kya karta hai: Text ko numbers mein convert karta hai (embedding)

**Simple mein:** Claim ko numbers mein badalta hai, phir Wikipedia ke numbers mein se sabse milte-julte dhundhta hai.

#### Layer 2: BM25 Sparse Search (Keyword Search)
- **BM25** = Best Match 25 (classic information retrieval algorithm)
  - Okapi BM25 variant
  - Keywords match karta hai — agar claim mein "Waterloo" hai to "Waterloo" wale documents dhundhta hai
- Library: `rank_bm25`

**Simple mein:** Google search jaisa — keywords match karta hai.

#### Layer 3: RRF (Reciprocal Rank Fusion)
- **RRF** = Reciprocal Rank Fusion
  - Dense (FAISS) aur Sparse (BM25) ke results ko combine karta hai
  - Formula: `RRF(d) = Σ 1/(60 + rank_i(d))`
  - k=60 (constant, tuned for best results)

**Simple mein:** Dono search results ko mila ke ek final ranking banata hai.

#### Layer 4: Cross-Encoder Reranking
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - MS MARCO = Microsoft Machine Reading Comprehension
  - Ye evidence ko accurately rank karta hai
  - 128 MB size
- Sigmoid normalization: score ko 0-1 range mein lata hai

**Simple mein:** Top results ko dubara check karta hai ki kaunsa sabse relevant hai.

#### Layer 5: Wikipedia API (6.8 Million Articles!)
- **Har claim ke liye** Wikipedia search API call hota hai
- 6.8 million English articles mein se search
- Full article text download hota hai (not just introduction)
- Paragraphs mein split karke, query terms se match karta hai
- Best 2 paragraphs (800 chars) nikalta hai

**Entity Verification (NEW — our innovation):**
- Check karta hai: "Haldiram Novel Theory" ka Wikipedia article exist karta hai?
- API call: `en.wikipedia.org/w/api.php?action=query&titles=Haldiram+Novel+Theory`
- Agar article MISSING hai → ye fabricated/made-up hai

**Simple mein:** Wikipedia se puchta hai "ye cheez real hai?" — agar Wikipedia ko pata nahi to probably fake hai.

**File:** `evidence_retriever.py` (872 lines — sabse bada module)

---

### Stage 3: NLI VERDICT (AI Judge)

**NLI** = Natural Language Inference
- Input: (Evidence, Claim) pair
- Output: Kya evidence claim ko support karta hai ya contradict?

**Model:** `cross-encoder/nli-deberta-v3-base`
- **DeBERTa** = Decoding-Enhanced BERT with Disentangled Attention
  - BERT = Bidirectional Encoder Representations from Transformers (Google, 2019)
  - DeBERTa = BERT ka improved version (Microsoft, 2021)
  - v3 = version 3
  - base = medium size (435 MB)
  - 92% accuracy on MultiNLI benchmark
- **cross-encoder** = dono texts (evidence + claim) ek saath model mein jaate hain

**3 Labels:**
```
Entailment (support)    → evidence claim ko support karta hai
Neutral                 → evidence claim se related nahi hai
Contradiction           → evidence claim ko galat kehta hai
```

**Output:** Softmax probabilities (teen numbers jo 1.0 mein add hote hain):
```
P(entailment) = 0.75  → 75% chance sahi hai
P(neutral)    = 0.15  → 15% chance related nahi
P(contradiction) = 0.10 → 10% chance galat hai
```

**Specificity Gate:**
```
s = entailment_score × evidence_similarity
```
Ye prevent karta hai ki topically similar but actually irrelevant evidence se false positive na aaye.

**Batch Processing:**
- Saare claims × evidence pairs ek saath process hote hain
- Single tokenize + single forward pass
- 5-8x faster than one-by-one

**Fabrication Detection:**
```
IF specific_entity_in_claim AND no_wikipedia_article_exists
   AND all_evidence_irrelevant
THEN → CONTRADICTED (90% confidence) — ye fake hai
```

**LLM-as-Judge (Groq 70B):**
- Jab NLI confused ho (entailment 0.35-0.65 ke beech)
- LLM se puchta hai: "Does this evidence SPECIFICALLY verify this EXACT claim?"
- Agar LLM bole "NO" → entailment score 0.1 kar deta hai

**File:** `verdict_engine.py` (586 lines)

---

### Stage 4: SELFCHECK (Double-Check)

**Based on:** SelfCheckGPT paper (Manakul et al., EMNLP 2023)

**Kya karta hai:** Ek aur baar check karta hai ki verdict sahi hai — multiple angles se.

**Two Paths:**
1. **LLM Path:** LLM ko 5 baar puchta hai same question, different temperatures
   - Temperatures: 0.10, 0.25, 0.40, 0.55, 0.70
   - Agar 5 mein se 4 baar same answer aaye → consistent → more confident
2. **NLI Fallback:** (jab LLM nahi hai) Different evidence passages se NLI run karta hai

**Metrics:**
- **Entropy** = kitna uncertain hai (0 = certain, 1 = confused)
  - `H = -Σ p_i × log(p_i) / log(3)`
- **Disagreement** = kitne samples disagree karte hain
  - `D = 1 - (max_count / total)`

**Uncertainty Formula:**
```
U = 0.65 × entropy + 0.35 × disagreement
```

**File:** `selfcheck.py` (212 lines), `semantic_entropy.py` (112 lines)

---

### Stage 5: CONFIDENCE FUSION (Final Score)

**Kya karta hai:** 5 different signals ko combine karke ek final confidence score nikalta hai.

**5 Signals with Weights:**
```
Signal 1: NLI Score         (weight: 0.32) → NLI ne kya kaha
Signal 2: Retrieval Sim     (weight: 0.22) → evidence kitna relevant tha
Signal 3: Source Trust       (weight: 0.12) → Wikipedia = 1.0, unknown = 0.5
Signal 4: Cross-Reference   (weight: 0.16) → kitne evidence agree karte hain
Signal 5: Stability          (weight: 0.18) → 1 - uncertainty
```

**Formula:**
```
C = σ(0.32×s₁ + 0.22×s₂ + 0.12×s₃ + 0.16×s₄ + 0.18×s₅)
```
where σ = sigmoid function (output 0 to 1)

**Simple mein:** 5 judges ka weighted average — sabko ek vote milta hai, kisi ka vote zyada important hai.

---

### Stage 6: ANNOTATOR (Output Generation)

**Kya karta hai:** Final result banata hai — JSON report, HTML with colored highlights, corrections for wrong claims.

**Correction Loops:**
1. **Reflexion** (Shinn et al., NeurIPS 2023) → LLM apna correction critique karta hai
2. **Constitutional AI** (Bai et al., Anthropic 2022) → Safety check on corrections

**File:** `annotator.py` (307 lines)

---

## 5. CHROME EXTENSION (User Ko Kya Dikhta Hai)

### Beacon (Floating Glass Ball):
- **72 pixels** circular glass orb
- **Draggable** — kahin bhi move kar sakte ho
- **Colors:**
  - 🔵 Blue = idle (kuch check nahi ho raha)
  - 🟣 Purple (pulsing) = checking... (analysis ho rahi hai)
  - 🟢 Green = safe (sab sahi hai)
  - 🔴 Red = hallucination detected! (galat claim mila)

### Dashboard (Click Beacon):
- **KPI Row:** Score | Claims | Verified | False | Unclear
- **Claim Cards:** har flagged claim with evidence and source
- **VF Logo** → click karke website pe jao

### Inline Highlighting:
- **Red background** directly chatbot response mein on false claims
- Hover karo to tooltip dikhta hai with verdict + confidence %

### Parallel Analysis:
- **3 messages simultaneously** analyze hote hain
- Results cache hote hain per DOM node
- Re-check only when content changes

### Prompt Suggestion:
- Jab user type karta hai, 3 second baad suggestion aata hai
- **Accept** button → improved prompt insert ho jata hai
- **Dismiss** → original rakhta hai

**Files:** `content.js`, `styles.css`, `background.js`, `manifest.json`

---

## 6. TECH STACK (Sab Kuch Jo Use Hua)

| Layer | Tech | Kya Hai |
|-------|------|---------|
| **Backend** | Python 3.11 | Programming language |
| **Frontend** | Next.js 15.5.15 | React-based web framework (Vercel ne banaya) |
| **Extension** | Chrome Manifest V3 | Browser extension standard |
| **NLI** | DeBERTa-v3-base | Microsoft ka NLI model (435 MB) |
| **Embeddings** | all-MiniLM-L6-v2 | Text → numbers converter (83 MB, 384-dim) |
| **Reranker** | ms-marco-MiniLM-L-6-v2 | Evidence re-ranking model (128 MB) |
| **Vector DB** | FAISS IndexFlatIP | Facebook ka vector search (11K vectors) |
| **Keyword Search** | BM25 (rank_bm25) | Classic information retrieval |
| **Knowledge** | Wikipedia API | 6.8M articles, free, unlimited |
| **Free LLM** | Groq (Llama 3.1 70B) | Free cloud LLM, 30 req/min |
| **Local LLM** | Ollama (llama3.1:8b) | Local machine pe LLM |
| **NLP** | spaCy (en_core_web_sm) | Sentence splitting fallback |
| **Config** | Pydantic | Type-safe Python config |
| **Testing** | pytest (94 tests) | Automated testing |
| **Linting** | ruff | Code quality checker |
| **Hosting** | HuggingFace Spaces | Free Docker hosting |
| **Web** | Vercel | Free static site hosting |
| **Container** | Docker (python:3.11-slim) | Containerization |

---

## 7. DEPLOYMENT (Kahan Deploy Hai)

```
User's Browser
    ↓ (HTTPS)
HuggingFace Spaces (Free CPU, Docker container)
    ├── Pre-built FAISS index (25 MB, baked in Docker image)
    ├── Pre-downloaded ML models (DeBERTa + MiniLM + reranker)
    ├── overlay_server.py (port 7860)
    ↓ (HTTPS)
    ├── Groq API (free LLM, Llama 70B)
    └── Wikipedia API (6.8M articles)

Vercel (Web Dashboard)
    └── Next.js static site
```

**Total Cost: $0/month** — sab free hai.

---

## 8. API ENDPOINTS (5 URLs)

| URL | Method | Kya Karta Hai | Speed |
|-----|--------|--------------|-------|
| `/analyze` | POST | Full 6-stage check | 10-18s |
| `/analyze/fast` | POST | Quick check (extension default) | 3-8s |
| `/analyze/stream` | POST | SSE streaming (live updates) | Real-time |
| `/optimize` | POST | Prompt suggestion | <1s |
| `/health` | GET | "Server alive hai?" | <100ms |

---

## 9. BENCHMARK RESULTS (Numbers)

| Test | Accuracy | What It Tests |
|------|----------|--------------|
| **HaluEval** | **93.8%** | Multi-sentence hallucination detection |
| **Adversarial** | **82%** | 100 mixed true/false claims |
| **Live Mix** | **100%** | 5 true+false claims together |
| **Rules** | **100%** | 12 geographic/date/science facts |
| **TruthfulQA** | **44.1%** | Single-sentence nuanced claims (hard) |
| **Fabrication** | **90%** | Made-up terms (Haldiram Theory, etc.) |

### Why TruthfulQA is Low (44.1%)?
- Single sentences don't give enough context for specificity gate
- Questions like "Did Einstein win Nobel for relativity?" need deep reasoning
- (He won for photoelectric effect, not relativity — subtle distinction)

---

## 10. INNOVATIONS (What's New — Evaluators Will Ask)

### 1. Fabrication Detection (Novel Contribution)
- **No other system does this**
- Check: does Wikipedia have an article with this EXACT name?
- "Zyphron Stability Theorem" → Wikipedia says NO → FABRICATED
- Confidence: 90%

### 2. Two-Pass Architecture
- Pass 1: Rule engine (instant, <1ms)
- Pass 2: NLI (only for unresolved claims)
- Saves 30-50% compute

### 3. Batch NLI
- Single DeBERTa forward pass for ALL claims
- 5-8x faster than per-claim processing

### 4. Wikipedia API as Knowledge Base
- 6.8M articles searched for EVERY claim
- Full article text with paragraph-level scoring
- Not just intro — finds deep facts

### 5. Zero-Cost Deployment
- Total: $0/month
- HF Spaces + Groq + Wikipedia + Vercel = all free

---

## 11. LIMITATIONS (Be Honest About These)

1. **English only** — no Hindi, no other languages
2. **5-10% false positive rate** — sometimes marks true facts as wrong
3. **5s per claim on free CPU** — not instant
4. **Single-passage retrieval** — can't combine info from multiple articles
5. **Wikipedia recency** — last few weeks ke events missing ho sakte hain
6. **TruthfulQA 44.1%** — subtle nuances catch nahi kar pata
7. **Extension DOM selectors** — ChatGPT update kare to break ho sakta hai

---

## 12. RESEARCH PAPERS USED (14 Papers)

| # | Paper | Year | Where Used |
|---|-------|------|-----------|
| 1 | FEVER (Thorne et al.) | 2018, NAACL | Fact verification methodology |
| 2 | RAG (Lewis et al.) | 2020, NeurIPS | FAISS + BM25 retrieval architecture |
| 3 | DeBERTa (He et al.) | 2021, ICLR | NLI model for verdict engine |
| 4 | TruthfulQA (Lin et al.) | 2022, ACL | Benchmark evaluation |
| 5 | Constitutional AI (Bai et al.) | 2022, Anthropic | Safety critique in corrections |
| 6 | FActScore (Min et al.) | 2023, EMNLP | Atomic claim extraction |
| 7 | SelfCheckGPT (Manakul et al.) | 2023, EMNLP | Self-consistency scoring |
| 8 | Reflexion (Shinn et al.) | 2023, NeurIPS | Critique-and-revise loop |
| 9 | Semantic Entropy (Kuhn et al.) | 2024, Nature | Uncertainty estimation |
| 10 | RAG Survey (Gao et al.) | 2024 | Architecture reference |
| 11 | Legal Hallucinations (Dahl et al.) | 2024 | Motivation (58% legal hallucination) |
| 12 | Hallucination Detection (Chen et al.) | 2024 | Detection methods |
| 13 | Hallucination Survey (Zhang et al.) | 2025 | Taxonomy |
| 14 | Why LLMs Hallucinate (Kalai et al.) | 2025, OpenAI | Root causes |

---

## 13. CODEBASE STATS

- **Total Code:** 5,336 lines across 13 Python modules
- **Tests:** 94 automated tests, 12 test files
- **Extension:** 4 files (JS + CSS + manifest + logo)
- **Web:** Next.js app with 4 tabs
- **Research Papers:** 14 reviewed and integrated
- **Rule Database:** 205 facts, 9 categories
- **Knowledge Base:** 11,248 local chunks + 6.8M Wikipedia articles (live API)

---

## 14. COMMON VIVA QUESTIONS & ANSWERS

**Q: What is hallucination in LLMs?**
A: When an LLM generates text that sounds confident and fluent but is factually incorrect. It's not lying — the model doesn't know it's wrong. It happens because LLMs predict the most likely next token, not the most true one.

**Q: How is this different from just Googling?**
A: Google requires you to manually search each claim. VeriFACT automatically decomposes text into claims, retrieves evidence, runs NLI inference, and produces calibrated confidence scores — all in 3-8 seconds per message, running continuously as you chat.

**Q: Why not just use GPT-4 to check GPT-4?**
A: That's circular — using one LLM to verify another LLM means you're trusting a potentially hallucinating system to catch hallucinations. We use DeBERTa NLI (a specialized classification model) + Wikipedia evidence (factual ground truth) instead.

**Q: What's your novel contribution?**
A: Fabrication detection via entity verification — checking if Wikipedia has an article with the exact name mentioned in the claim. If not, the entity is likely fabricated. No existing system does this.

**Q: Why is TruthfulQA accuracy so low?**
A: TruthfulQA tests single-sentence nuanced claims like "Did Einstein win the Nobel Prize for relativity?" — the answer is no (it was for photoelectric effect). Our specificity gate needs multi-passage context to work well, and single sentences don't provide that.

**Q: Is it truly real-time?**
A: On free CPU, each claim takes ~5 seconds. For a typical chatbot message with 3-5 claims, total time is 7-15 seconds. It's not instant, but it runs automatically in the background while you continue chatting. With GPU, it would be under 1 second per claim.

**Q: What happens if Wikipedia doesn't have info?**
A: If neither local FAISS nor Wikipedia API finds relevant evidence, AND the claim names a specific entity that doesn't have a Wikipedia article, we flag it as potentially fabricated (CONTRADICTED at 90% confidence). If the claim is generic without specific entities, we mark it UNVERIFIABLE.

**Q: How do you handle the cold start on HuggingFace?**
A: We pre-build the FAISS index and bake it into the Docker image along with all ML models. So startup only needs to load models into memory (~30 seconds) instead of rebuilding the index (which used to take 15 minutes).

**Q: What's RAG and how do you use it?**
A: RAG = Retrieval-Augmented Generation. Instead of relying only on what the AI model "knows," we first RETRIEVE evidence from a knowledge base (Wikipedia), then AUGMENT the model's input with this evidence, and then GENERATE a verdict. Our RAG uses FAISS + BM25 + Wikipedia API for retrieval.

**Q: Cost kitna hai?**
A: Zero. HuggingFace Spaces free hai, Groq API free hai (30 req/min), Wikipedia API free hai, Vercel free hai. Total monthly cost: $0.

---

*Last verified: April 17, 2026. All numbers from actual codebase and live tests.*
