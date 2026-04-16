# VeriFACT AI — Architecture Diagrams (Verified Against Source Code)

> Every arrow and connection verified against actual Python source.
> Optimized for Claude Opus rendering. Paste into claude.ai for diagrams.

---

## 1. System Architecture

```mermaid
graph TB
    subgraph CLIENTS["Client Layer"]
        direction LR
        EXT["Chrome Extension\n(content.js)\nChatGPT · Claude · Gemini · Grok"]
        WEB["Web Dashboard\n(Next.js 15.5.15)\nVercel"]
        STREAM["Streamlit\n(app.py)"]
    end

    subgraph SERVER["API Server (overlay_server.py, port 7860)"]
        direction LR
        FAST["POST /analyze/fast"]
        FULL["POST /analyze"]
        SSE["POST /analyze/stream"]
        OPT["POST /optimize"]
        HEALTH["GET /health"]
    end

    subgraph PIPELINE["VeriFactPipeline (pipeline.py)"]
        direction TB
        S1["Stage 1: ClaimDecomposer\n(claim_decomposer.py)"]
        S2A["Stage 2a: RuleEngine\n(fact_rules.py)\n400+ facts, 9 rule types"]
        S2B["Stage 2b: EvidenceRetriever\n(evidence_retriever.py)"]
        S3["Stage 3: VerdictEngine\n(verdict_engine.py)\nDeBERTa-v3 NLI"]
        S4["Stage 4: SelfCheck\n(selfcheck.py)"]
        S5["Stage 5: ConfidenceFusion\n(verdict_engine._bayesian_confidence)"]
        S6["Stage 6: Annotator\n(annotator.py)\nReflexion + Constitutional"]
    end

    subgraph KNOWLEDGE["Knowledge Layer"]
        direction LR
        FAISS[("FAISS IndexFlatIP\n11K vectors, 384-dim\n(pre-built in Docker)")]
        BM25[("BM25 (rank_bm25)\nIn-memory sparse index")]
        WIKI_API[("Wikipedia API\n6.8M articles\nFull-text paragraph search")]
        RULES_DB[("Rule Database\n400+ hardcoded facts")]
        METADATA[("chunks.jsonl\ntext · source · title · url")]
    end

    subgraph MODELS["ML Models (pre-downloaded)"]
        direction LR
        DEBERTA["DeBERTa-v3-base\n(NLI, 435MB)"]
        MINILM["all-MiniLM-L6-v2\n(Embeddings, 83MB, 384-dim)"]
        RERANKER["ms-marco-MiniLM-L-6-v2\n(Reranker, 128MB)"]
        SPACY_M["spaCy en_core_web_sm\n(Fallback NLP, 43MB)"]
        TOXIC["unitary/toxic-bert\n(Risk classifier, optional)"]
    end

    subgraph EXTERNAL["External APIs (Free)"]
        direction LR
        GROQ["Groq API\nllama-3.1-8b-instant\n30 req/min free"]
        OLLAMA["Ollama\nllama3.1:8b\nlocalhost:11434"]
    end

    EXT -->|"POST /analyze/fast\n(default for extension)"| FAST
    WEB -->|"POST /analyze"| FULL
    STREAM -->|"Direct Python call"| S1

    FAST --> S1
    FULL --> S1
    SSE --> S1

    S1 -->|"claims[] (max 8 fast, 20 full)"| S2A
    S2A -->|"rule_resolved[] instantly"| S6
    S2A -->|"unresolved[] claims"| S2B
    S2B -->|"evidence[][] (top-5 per claim)"| S3
    S3 -->|"verdicts[]"| S4
    S4 -->|"uncertainty metrics"| S5
    S5 -->|"confidence scores"| S6
    S6 -->|"JSON + HTML response"| SERVER

    S1 -.->|"LLM call\n(claim extraction)"| GROQ
    S1 -.->|"LLM call\n(if Groq fails)"| OLLAMA
    S1 -.->|"Fallback\n(if no LLM)"| SPACY_M

    S2B -->|"Batch encode queries"| MINILM
    S2B -->|"Vector search"| FAISS
    S2B -->|"Keyword search"| BM25
    S2B -->|"Rerank candidates"| RERANKER
    S2B -->|"Live search\n(EVERY claim)"| WIKI_API
    S2B -->|"Load chunk text"| METADATA
    S2A -->|"Lookup"| RULES_DB

    S3 -->|"Batch NLI\n(single forward pass)"| DEBERTA

    style CLIENTS fill:#E3F2FD
    style SERVER fill:#F3E5F5
    style PIPELINE fill:#E8F5E9
    style KNOWLEDGE fill:#FFF3E0
    style MODELS fill:#FCE4EC
    style EXTERNAL fill:#F1F8E9
```

---

## 2. Complete Request-Response Flow

```mermaid
sequenceDiagram
    participant User as User / Extension
    participant API as overlay_server.py
    participant CD as ClaimDecomposer
    participant LLM as Groq / Ollama / spaCy
    participant Rules as fact_rules.py (400+)
    participant ER as EvidenceRetriever
    participant FAISS as FAISS (11K local)
    participant BM25 as BM25 (sparse)
    participant WIKI as Wikipedia API (6.8M)
    participant NLI as DeBERTa-v3 NLI
    participant SC as SelfCheck
    participant CF as Confidence Fusion
    participant AN as Annotator

    User->>API: POST /analyze/fast {text, top_claims:8}
    API->>API: Validate: CORS + rate limit (20/min) + auth + size (50K max)

    Note over API,CD: STAGE 1: Claim Decomposition
    API->>CD: decompose(text)
    CD->>LLM: Generate JSON claims (or spaCy fallback)
    LLM-->>CD: Raw JSON / sentences
    CD->>CD: Parse + filter non-factual + fact-density rank
    CD-->>API: claims[] (capped at 8 for fast)

    Note over API,Rules: STAGE 2a: Rule Check (instant)
    loop Each claim
        API->>Rules: check_rules(claim.text)
        alt Rule matches (landmark/capital/date/science/person)
            Rules-->>API: RuleViolation → CONTRADICTED (0.95 conf)
        else No rule
            Rules-->>API: None → needs NLI
        end
    end

    Note over API,WIKI: STAGE 2b: Evidence Retrieval (unresolved only)
    API->>ER: batch_retrieve(unresolved_claims)
    ER->>ER: _expand_queries() → original + negation query
    ER->>FAISS: Batch encode (MiniLM, single call) + search
    FAISS-->>ER: Dense results (top-15 per claim)
    ER->>BM25: Keyword search per claim
    BM25-->>ER: Sparse results (top-15 per claim)
    ER->>ER: RRF fusion (k=60) + deduplicate
    ER->>ER: Cross-encoder rerank (single predict call)

    loop Each claim (ALWAYS, not fallback)
        ER->>WIKI: Search 6.8M articles + get full text
        WIKI-->>ER: Best 2 paragraphs (800 chars) + computed similarity
    end
    ER->>ER: Merge local + wiki, sort by similarity, top-5
    ER-->>API: evidence[][] (top-5 per unresolved claim)

    Note over API,NLI: STAGE 3: NLI Verdict (batched)
    API->>NLI: batch_judge(claims + evidence)
    NLI->>NLI: Collect ALL (evidence, claim) pairs
    NLI->>NLI: Single tokenize + single forward pass
    NLI->>NLI: Check: has_wiki_evidence? fabrication? hard_contradiction?
    NLI->>NLI: Decision: FABRICATED → CONTRADICTED → SUPPORTED → UNVERIFIABLE
    NLI-->>API: verdicts[] with labels + confidence

    Note over API,SC: STAGE 4: SelfCheck (skipped in fast mode)
    opt Full mode only
        API->>SC: score_claim(claim, evidence)
        SC-->>API: consistency + entropy + uncertainty
    end

    Note over API,CF: STAGE 5: Confidence Fusion
    API->>CF: _bayesian_confidence(5 signals)
    Note over CF: NLI(0.32) + Retrieval(0.22) + Source(0.12) + CrossRef(0.16) + Uncertainty(0.18)
    CF-->>API: final confidence [0,1]

    Note over API,AN: STAGE 6: Annotation
    API->>AN: generate_json(text, claims)
    AN-->>API: {factuality_score, flags[], alerts[]}

    API-->>User: JSON response
    Note over User: Extension: red/yellow highlights + dashboard update
```

---

## 3. Evidence Retrieval Detail

```mermaid
graph TD
    CLAIM["Input: claim text"]

    EXPAND["_expand_queries()\n1. Original claim\n2. Content words + 'facts location history'"]

    subgraph LOCAL["Local Index Search"]
        ENCODE["Batch encode\n(MiniLM-L6-v2, 384-dim)"]
        FSEARCH["FAISS IndexFlatIP\nsearch(embeddings, k=15)"]
        BSEARCH["BM25Okapi\nget_top_n(tokenized, k=15)"]
        RRF["Reciprocal Rank Fusion\nscore = 1/(60+rank_dense)\n      + 1/(60+rank_sparse)"]
        DEDUP["Deduplicate by chunk_id"]
        RERANK["Cross-encoder rerank\n(ms-marco-MiniLM-L-6-v2)\nSigmoid normalize scores"]
        LOCAL_TOP["Local top-5"]
    end

    subgraph LIVE["Wikipedia API Search (EVERY claim)"]
        WSEARCH["en.wikipedia.org/w/api.php\n?action=query&list=search\n&srsearch={claim}"]
        WFULL["Get FULL article text\n(not just intro)"]
        WPARA["Split into paragraphs\nScore each by query-term overlap"]
        WBEST["Best 2 paragraphs\n(up to 800 chars)\nComputed similarity = term_overlap × 0.85"]
    end

    MERGE["Merge local + wiki results\nDeduplicate by title\nSort by similarity desc\nKeep top-5"]

    OUTPUT["Output: list[Evidence]\ntext · source · title · url · similarity"]

    CLAIM --> EXPAND
    EXPAND --> ENCODE
    ENCODE --> FSEARCH
    EXPAND --> BSEARCH
    FSEARCH --> RRF
    BSEARCH --> RRF
    RRF --> DEDUP
    DEDUP --> RERANK
    RERANK --> LOCAL_TOP

    CLAIM --> WSEARCH
    WSEARCH --> WFULL
    WFULL --> WPARA
    WPARA --> WBEST

    LOCAL_TOP --> MERGE
    WBEST --> MERGE
    MERGE --> OUTPUT

    style LOCAL fill:#BBDEFB
    style LIVE fill:#E1BEE7
```

---

## 4. Verdict Decision Tree (Exact Code Logic)

```mermaid
graph TD
    INPUT["claim + evidence\n+ nli_results"]

    RULE{"fact_rules.check_rules()\nLandmark? Capital? Date?\nScience? Person?"}
    RULE_HIT["CONTRADICTED\nconfidence=0.95\nsource=factual_rule\n(skip all NLI)"]

    NLI_CALC["Compute:\nmax_con, max_raw_ent\nmax_specific_ent\nmax_similarity\nbest_wiki_sim\nhas_wiki_evidence"]

    FAB{"FABRICATION?\nhas_specific_entity OR has_technical_term\nAND NOT has_wiki_evidence\nAND evidence_is_irrelevant\nAND max_raw_ent < 0.40"}
    FAB_YES["CONTRADICTED\nconfidence ≥ 0.80\nreason: fabricated"]

    HARD{"HARD CONTRADICTION?\nmax_con > 0.80\nAND max_con > max_raw_ent + 0.20\nAND best_contra.sim > 0.50"}

    WIKI_OVERRIDE{"wiki_confirms?\nbest_wiki_sim > 0.45"}
    WIKI_WIN["SUPPORTED\nreason: Wikipedia confirms"]
    HARD_WIN["CONTRADICTED\nNLI-based"]

    STRONG{"strong_support?\nmax_specific_ent > 0.65"}
    MOD{"moderate_support?\nmax_raw_ent > 0.35\nAND max_similarity > 0.30"}
    WIKI_OR_STRONG{"wiki_confirms OR\nstrong OR moderate?"}

    WEAK{"weak_support?\nmax_similarity > 0.25\nAND max_con < 0.60"}

    IRREL_ENT{"evidence_is_irrelevant\nAND has_specific_entity\nAND NOT has_wiki_evidence?"}

    IRREL{"evidence_is_irrelevant?\nmax_sim < 0.35\nAND avg_sim < 0.25"}

    SUPPORTED["SUPPORTED"]
    CONTRADICTED_IRREL["CONTRADICTED\n(specific + no evidence)"]
    UNVERIFIABLE["UNVERIFIABLE\n(generic + no evidence)"]
    SUPPORTED_DEFAULT["SUPPORTED\n(default)"]

    INPUT --> RULE
    RULE -->|"Match"| RULE_HIT
    RULE -->|"No match"| NLI_CALC
    NLI_CALC --> FAB
    FAB -->|"Yes"| FAB_YES
    FAB -->|"No"| HARD
    HARD -->|"Yes"| WIKI_OVERRIDE
    WIKI_OVERRIDE -->|"Yes"| WIKI_WIN
    WIKI_OVERRIDE -->|"No"| HARD_WIN
    HARD -->|"No"| WIKI_OR_STRONG
    WIKI_OR_STRONG -->|"Yes"| SUPPORTED
    WIKI_OR_STRONG -->|"No"| WEAK
    WEAK -->|"Yes"| SUPPORTED
    WEAK -->|"No"| IRREL_ENT
    IRREL_ENT -->|"Yes"| CONTRADICTED_IRREL
    IRREL_ENT -->|"No"| IRREL
    IRREL -->|"Yes"| UNVERIFIABLE
    IRREL -->|"No"| SUPPORTED_DEFAULT

    style RULE_HIT fill:#FFCDD2
    style FAB_YES fill:#FFCDD2
    style HARD_WIN fill:#FFCDD2
    style CONTRADICTED_IRREL fill:#FFCDD2
    style SUPPORTED fill:#C8E6C9
    style WIKI_WIN fill:#C8E6C9
    style SUPPORTED_DEFAULT fill:#C8E6C9
    style UNVERIFIABLE fill:#FFF9C4
```

---

## 5. Bayesian Confidence Fusion (5 Signals)

```mermaid
graph LR
    S1["Signal 1: NLI\n(max_ent - max_con + 1) / 2\nRange: [0, 1]\nWeight: 0.32"]
    S2["Signal 2: Retrieval\nmax(evidence.similarity)\nRange: [0, 1]\nWeight: 0.22"]
    S3["Signal 3: Source\nmean(source_reliability)\nwikipedia=1.0, pubmed=1.0\nunknown=0.5\nWeight: 0.12"]
    S4["Signal 4: Cross-Ref\nfraction where\nentailment > contradiction\nAND entailment > 0.5\nWeight: 0.16"]
    S5["Signal 5: Stability\n1 - uncertainty\nuncertainty = 0.65×entropy\n+ 0.35×disagreement\nWeight: 0.18"]

    FUSION["confidence = σ(\nΣ w_i × signal_i\n)"]

    S1 --> FUSION
    S2 --> FUSION
    S3 --> FUSION
    S4 --> FUSION
    S5 --> FUSION

    BOOST{"is_likely_fabricated?"}
    BOOST_YES["confidence = max(conf, 0.80)"]

    FUSION --> BOOST
    BOOST -->|"Yes"| BOOST_YES
    BOOST -->|"No"| FINAL

    FINAL["Final confidence\n[0.0 - 1.0]"]

    style FUSION fill:#B2DFDB
    style BOOST_YES fill:#FFCDD2
```

---

## 6. SelfCheck (Two Paths)

```mermaid
graph TD
    INPUT["Input: claim_text + evidence[]"]

    CHECK{"LLM available?\n(provider ≠ 'none')"}

    subgraph LLM_PATH["LLM Sampling Path"]
        SAMPLE["5 LLM calls at temps:\n0.10, 0.25, 0.40, 0.55, 0.70"]
        PARSE["Parse JSON response:\n{label, rationale}"]
        LABELS_LLM["Labels: supported | contradicted | uncertain"]
    end

    subgraph NLI_PATH["NLI Fallback Path (no LLM)"]
        NLI_RUN["Run _batch_nli(claim, evidence)\nvia shared VerdictEngine"]
        MAP["Map NLI scores:\nmax(ent) → supported\nmax(con) → contradicted\nelse → uncertain"]
        LABELS_NLI["Same label format"]
    end

    ENTROPY["normalized_entropy(labels, 3 classes)"]
    DISAGREE["disagreement = 1 - max_count/total"]
    CLUSTER["cluster_entropy(rationales, jaccard=0.5)\n(LLM path only)"]

    UNCERTAINTY["uncertainty =\n0.65 × entropy\n+ 0.35 × disagreement\n+ 0.30 × cluster_entropy"]

    BLEND["Blend with NLI confidence:\nnew_conf = 0.80 × nli_conf\n+ 0.20 × consistency"]

    INPUT --> CHECK
    CHECK -->|"Yes"| SAMPLE
    CHECK -->|"No"| NLI_RUN
    SAMPLE --> PARSE --> LABELS_LLM
    NLI_RUN --> MAP --> LABELS_NLI
    LABELS_LLM --> ENTROPY
    LABELS_LLM --> DISAGREE
    LABELS_LLM --> CLUSTER
    LABELS_NLI --> ENTROPY
    LABELS_NLI --> DISAGREE
    ENTROPY --> UNCERTAINTY
    DISAGREE --> UNCERTAINTY
    CLUSTER --> UNCERTAINTY
    UNCERTAINTY --> BLEND

    style LLM_PATH fill:#E1BEE7
    style NLI_PATH fill:#BBDEFB
```

---

## 7. Chrome Extension Architecture

```mermaid
graph TB
    subgraph CHATBOT["Chatbot Page (ChatGPT / Claude / Gemini)"]
        DOM["Assistant message DOM nodes"]
        INPUT_FIELD["User input textarea / contenteditable"]
    end

    subgraph EXT["Chrome Extension (content.js)"]
        OBSERVER["MutationObserver\n(watches DOM changes)"]
        SCANNER["scan() function\n3 messages in parallel\nCache per DOM node"]
        FILTER["extractFactualText()\nClient-side pre-filter\nRemove questions, meta, filler"]
        HIGHLIGHT["highlightNode()\nRed = CONTRADICTED\nYellow = UNVERIFIABLE"]

        BEACON["Beacon (72px glass orb)\nDraggable anywhere\nStates: idle/scanning/safe/alert"]
        DASHBOARD["Dashboard (glass panel)\nDraggable, KPI row\nClaim cards with evidence\nVF logo → website link"]
        PROMPT_TIP["Prompt Suggestion\nDraggable, X to close\nAccept / Use Original buttons"]
    end

    subgraph BACKEND["HuggingFace Space API"]
        ANALYZE["/analyze/fast"]
        OPTIMIZE["/optimize"]
    end

    DOM -->|"MutationObserver\ndetects new messages"| OBSERVER
    OBSERVER -->|"debounce 1.5s"| SCANNER
    SCANNER -->|"extract innerText"| FILTER
    FILTER -->|"POST (up to 3 parallel)"| ANALYZE
    ANALYZE -->|"JSON: flags[], score"| SCANNER
    SCANNER -->|"Apply highlights"| HIGHLIGHT
    HIGHLIGHT -->|"Modify DOM"| DOM
    SCANNER -->|"Update state"| BEACON
    SCANNER -->|"Render cards"| DASHBOARD

    INPUT_FIELD -->|"keyup event\n3s debounce"| PROMPT_TIP
    PROMPT_TIP -->|"POST"| OPTIMIZE
    OPTIMIZE -->|"suggested prompt + score"| PROMPT_TIP
    PROMPT_TIP -->|"Accept: insert text"| INPUT_FIELD

    BEACON -->|"Click"| DASHBOARD

    style CHATBOT fill:#E3F2FD
    style EXT fill:#F3E5F5
    style BACKEND fill:#E8F5E9
```

---

## 8. Deployment Architecture

```mermaid
graph TB
    subgraph USER["End User"]
        BROWSER["Chrome + Extension"]
    end

    subgraph PROD["Production (All Free)"]
        subgraph HF["HuggingFace Spaces (Free CPU)"]
            DOCKER["Docker: python:3.11-slim"]
            PREBUILD["Pre-built FAISS index\n(baked into image, 25MB)"]
            PREMODELS["Pre-downloaded models\n(baked into image, ~700MB)"]
            API_SERVER["overlay_server.py\nPort 7860"]
        end

        subgraph VERCEL["Vercel (Free)"]
            NEXTJS["Next.js 15.5.15\nStatic export"]
            LOGOS["AI chatbot logos\nVF logo"]
        end
    end

    subgraph FREE_APIS["Free External APIs"]
        GROQ_SVC["Groq Cloud\nLlama 3.1 8B\n30 req/min, 0 cost"]
        WIKI_SVC["Wikipedia API\n6.8M articles\nUnlimited, 0 cost"]
    end

    subgraph LOCAL["Local Development"]
        MAC["MacBook Air M4\n16GB RAM, MPS"]
        OLLAMA_SVC["Ollama\nllama3.1:8b"]
        BIG_INDEX["Full FAISS index\n369K vectors"]
    end

    BROWSER -->|"HTTPS"| API_SERVER
    BROWSER -->|"HTTPS"| NEXTJS
    API_SERVER -->|"HTTPS"| GROQ_SVC
    API_SERVER -->|"HTTPS"| WIKI_SVC

    DOCKER --- PREBUILD
    DOCKER --- PREMODELS
    DOCKER --- API_SERVER

    style HF fill:#E8F5E9
    style VERCEL fill:#E3F2FD
    style FREE_APIS fill:#F1F8E9
    style LOCAL fill:#FFF3E0
```

---

## 9. Rule Engine (9 Rule Types, Actual Execution Order)

```mermaid
graph TD
    CLAIM["claim_text.lower()"]

    R1{"Rule 1: LANDMARK_LOCATIONS\n68 entries\n'great wall' → China\n'eiffel tower' → France"}
    R2{"Rule 2: CAPITAL_COUNTRY\n36 entries\n'tokyo' → Japan (requires 'capital' in text)"}
    R3{"Rule 3: CITY_COUNTRY\n19 entries\n'london' → UK, 'sydney' → Australia"}
    R4{"Rule 4: COUNTRY_CONTINENT\n30 entries\n'brazil' → South America\n(requires location verb)"}
    R5{"Rule 5: PERSON_ACHIEVEMENT\n30+ entries\n'einstein' → physics, not airplane\n'newton' → gravity, not telephone"}
    R6{"Rule 6: HISTORICAL_EVENTS\n7 events with year ranges\nWWII: 1939-1945 (±10yr tolerance)\nMoon landing: 1969"}
    R7{"Rule 7: PERSON_DATES\n9 people with birth/death years\nEinstein: 1879-1955\nShakespeare: 1564-1616"}
    R8{"Rule 8: SCIENCE_NUMBERS\n5 facts with tolerances\nWater: 100°C (95-105)\nBones: 206 (200-210)"}
    R9{"Rule 9: SCIENCE_TRUTHS\n8 boolean facts\n'sun revolves around earth' → False\n'gold is a gas' → False"}

    HIT["Return RuleViolation(\nrule_name, claim,\nreason, correct_fact)\n→ CONTRADICTED, conf=0.95"]
    MISS["Return None\n→ Proceed to evidence retrieval + NLI"]

    CLAIM --> R1
    R1 -->|"landmark + wrong location"| HIT
    R1 -->|"no match"| R2
    R2 -->|"capital + wrong country"| HIT
    R2 -->|"no match"| R3
    R3 -->|"city + wrong country"| HIT
    R3 -->|"no match"| R4
    R4 -->|"country + wrong continent"| HIT
    R4 -->|"no match"| R5
    R5 -->|"person + other's achievement"| HIT
    R5 -->|"no match"| R6
    R6 -->|"event + impossible year"| HIT
    R6 -->|"no match"| R7
    R7 -->|"person + wrong birth/death"| HIT
    R7 -->|"no match"| R8
    R8 -->|"science + wrong number"| HIT
    R8 -->|"no match"| R9
    R9 -->|"science statement = False"| HIT
    R9 -->|"no match"| MISS

    style HIT fill:#FFCDD2
    style MISS fill:#C8E6C9
```

---

## 10. LLM Provider Fallback Chain (Actual Code)

```mermaid
graph TD
    REQUEST["LLM.generate()\ncalled from: ClaimDecomposer,\nPromptOptimizer, Annotator"]

    PROVIDER{"config.llm.provider"}

    NONE_CHECK["provider = 'none'\n→ Return None immediately\n→ spaCy fallback in caller"]

    GROQ_AUTO{"GROQ_API_KEY set\nAND provider = 'none'?"}
    AUTO_SWITCH["Auto-switch to 'groq'\n(config.py line 77)"]

    PRIMARY["Try primary provider"]

    OLLAMA_TRY["Ollama\nPOST localhost:11434/api/generate\nmodel: llama3.1:8b\n2 retries, 0.5s backoff"]

    GROQ_TRY["Groq\nPOST api.groq.com/openai/v1/chat/completions\nmodel: llama-3.1-8b-instant\nBearer token auth, urllib"]

    ANTHRO_TRY["Anthropic\nclient.messages.create()\nmodel: claude-sonnet-4-20250514\n(only if ANTHROPIC_API_KEY set)"]

    OPENAI_TRY["OpenAI\nclient.chat.completions.create()\nmodel: gpt-4o-mini\n(only if OPENAI_API_KEY set)"]

    SUCCESS["Return response text"]
    FAIL["Return None\n→ Caller uses spaCy fallback"]

    REQUEST --> PROVIDER
    PROVIDER -->|"'none'"| GROQ_AUTO
    GROQ_AUTO -->|"Yes"| AUTO_SWITCH
    AUTO_SWITCH --> GROQ_TRY
    GROQ_AUTO -->|"No"| NONE_CHECK

    PROVIDER -->|"'ollama'"| OLLAMA_TRY
    PROVIDER -->|"'groq'"| GROQ_TRY
    PROVIDER -->|"'anthropic'"| ANTHRO_TRY
    PROVIDER -->|"'openai'"| OPENAI_TRY

    OLLAMA_TRY -->|"success"| SUCCESS
    OLLAMA_TRY -->|"fail"| GROQ_TRY
    GROQ_TRY -->|"success"| SUCCESS
    GROQ_TRY -->|"fail"| ANTHRO_TRY
    ANTHRO_TRY -->|"success"| SUCCESS
    ANTHRO_TRY -->|"fail"| OPENAI_TRY
    OPENAI_TRY -->|"success"| SUCCESS
    OPENAI_TRY -->|"fail"| FAIL

    style SUCCESS fill:#C8E6C9
    style FAIL fill:#FFCDD2
    style NONE_CHECK fill:#E0E0E0
```

---

*All arrows verified against source code commit 3c282bd.*
*Every decision branch matches actual if/elif/else in Python.*
