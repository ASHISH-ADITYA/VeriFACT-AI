# VeriFACT AI — All Architecture & Flow Diagrams (Mermaid Code)

> Paste each code block into https://mermaid.live to render

---

## 1. System Architecture (High-Level)

```mermaid
graph TB
    subgraph CLIENT["Client Layer"]
        EXT["Chrome Extension<br/>(Manifest V3)<br/>ChatGPT | Claude | Gemini | Grok"]
        WEB["Web Dashboard<br/>(Next.js 15.5.15)<br/>Vercel"]
        STR["Streamlit Dashboard<br/>(app.py:8501)"]
    end

    subgraph API["API Layer (overlay_server.py)"]
        AF["/analyze/fast<br/>(3-8s)"]
        AN["/analyze<br/>(10-15s)"]
        AS["/analyze/stream<br/>(SSE)"]
        OP["/optimize<br/>(<1s)"]
        HE["/health"]
    end

    subgraph PIPE["Pipeline Layer (pipeline.py)"]
        CD["Stage 1<br/>Claim Decomposer"]
        RE["Stage 2a<br/>Rule Engine<br/>(400+ facts)"]
        ER["Stage 2b<br/>Evidence Retriever"]
        VE["Stage 3<br/>Verdict Engine<br/>(DeBERTa NLI)"]
        SC["Stage 4<br/>SelfCheck"]
        CF["Stage 5<br/>Confidence Fusion"]
        AO["Stage 6<br/>Annotator"]
    end

    subgraph DATA["Data & Knowledge Layer"]
        FAISS[("FAISS Index<br/>IndexFlatIP<br/>11K vectors<br/>384-dim")]
        BM25[("BM25 Index<br/>rank_bm25<br/>In-memory")]
        WIKI[("Wikipedia API<br/>6.8M articles<br/>Full-text search")]
        RULES[("Rule Database<br/>400+ facts<br/>9 rule types")]
        META[("Metadata JSONL<br/>text | source<br/>title | url")]
    end

    subgraph MODELS["ML Models"]
        DEBERTA["DeBERTa-v3-base<br/>NLI (435MB)"]
        MINILM["all-MiniLM-L6-v2<br/>Embeddings (83MB)"]
        RERANK["ms-marco-MiniLM<br/>Reranker (128MB)"]
        SPACY["spaCy en_core_web_sm<br/>Fallback (43MB)"]
    end

    subgraph EXTERNAL["External Services (Free)"]
        GROQ["Groq API<br/>Llama 3.1 8B<br/>30 req/min free"]
        OLLAMA["Ollama<br/>Local LLM<br/>(optional)"]
    end

    EXT -->|"POST /analyze/fast"| AF
    WEB -->|"POST /analyze"| AN
    STR -->|"Direct Python"| PIPE

    AF --> PIPE
    AN --> PIPE
    AS --> PIPE
    OP --> PIPE

    CD -->|"claims[]"| RE
    RE -->|"unresolved"| ER
    ER -->|"evidence[]"| VE
    VE -->|"verdicts[]"| SC
    SC -->|"uncertainty"| CF
    CF -->|"confidence"| AO

    ER --- FAISS
    ER --- BM25
    ER --- WIKI
    ER --- META
    RE --- RULES
    VE --- DEBERTA
    ER --- MINILM
    ER --- RERANK
    CD --- GROQ
    CD --- OLLAMA
    CD --- SPACY

    style CLIENT fill:#e3f2fd,stroke:#1565c0
    style API fill:#f3e5f5,stroke:#7b1fa2
    style PIPE fill:#e8f5e9,stroke:#2e7d32
    style DATA fill:#fff3e0,stroke:#e65100
    style MODELS fill:#fce4ec,stroke:#c62828
    style EXTERNAL fill:#f1f8e9,stroke:#558b2f
```

---

## 2. Pipeline Data Flow (Request → Response)

```mermaid
sequenceDiagram
    participant U as User/Extension
    participant S as API Server
    participant D as Claim Decomposer
    participant R as Rule Engine
    participant E as Evidence Retriever
    participant W as Wikipedia API
    participant N as DeBERTa NLI
    participant C as Confidence Fusion
    participant A as Annotator

    U->>S: POST /analyze/fast {text: "..."}
    S->>S: CORS check + Rate limit + Auth

    S->>D: decompose(text)
    alt LLM available (Groq/Ollama)
        D->>D: LLM JSON extraction
    else No LLM
        D->>D: spaCy sentence split + fact-density ranking
    end
    D-->>S: claims[] (max 8 for fast, 20 for full)

    loop For each claim
        S->>R: check_rules(claim)
        alt Rule matches (geography/date/science)
            R-->>S: CONTRADICTED (95% confidence, <1ms)
        else No rule match
            R-->>S: null (proceed to NLI)
        end
    end

    S->>E: batch_retrieve(unresolved_claims)
    E->>E: Batch encode (MiniLM-L6, single call)
    E->>E: FAISS search (11K vectors)
    E->>E: BM25 search (keyword matching)
    E->>E: RRF fusion (k=60)
    E->>E: Cross-encoder rerank

    loop For each claim
        E->>W: Wikipedia API search (6.8M articles)
        W-->>E: Full article text + paragraph scoring
    end
    E-->>S: evidence[][] (top-5 per claim)

    S->>N: batch_judge(claims + evidence)
    N->>N: Single tokenize + forward pass (all pairs)
    N->>N: Fabrication check (no wiki = fake?)
    N->>N: Specificity gate (entailment × similarity)
    N-->>S: verdicts[] (SUPPORTED/CONTRADICTED/UNVERIFIABLE)

    S->>C: bayesian_confidence()
    Note over C: 5 signals: NLI(0.32) + Retrieval(0.22)<br/>+ Source(0.12) + CrossRef(0.16) + Uncertainty(0.18)
    C-->>S: confidence scores[]

    S->>A: generate_json(text, claims)
    A-->>S: {factuality_score, flags[], alerts[]}

    S-->>U: JSON response
    Note over U: Extension: highlight red/yellow<br/>Dashboard: show verdict cards
```

---

## 3. Evidence Retrieval Pipeline (Hybrid Search)

```mermaid
graph LR
    CLAIM["Claim Text<br/>'Napoleon lost Waterloo in 1815'"]

    subgraph EXPAND["Query Expansion"]
        Q1["Original query"]
        Q2["Negation query<br/>(content words + 'facts location history')"]
    end

    subgraph DENSE["Dense Retrieval"]
        ENC["Encode<br/>(MiniLM-L6-v2<br/>384-dim)"]
        FAISS["FAISS IndexFlatIP<br/>search(query, k=15)"]
        DR["Dense Results<br/>(15 candidates)"]
    end

    subgraph SPARSE["Sparse Retrieval"]
        TOK["Tokenize<br/>(lowercase, split)"]
        BM25["BM25Okapi<br/>score(query)"]
        SR["Sparse Results<br/>(15 candidates)"]
    end

    subgraph FUSION["Reciprocal Rank Fusion"]
        RRF["RRF Score =<br/>1/(60+rank_dense) +<br/>1/(60+rank_sparse)"]
        DEDUP["Deduplicate<br/>by chunk_id"]
        FUSED["Fused Candidates<br/>(15 unique)"]
    end

    subgraph RERANK["Cross-Encoder Reranking"]
        CE["ms-marco-MiniLM-L-6<br/>predict(query, passage)"]
        SIG["Sigmoid normalization"]
        TOP["Top-5 reranked"]
    end

    subgraph WIKIAPI["Wikipedia API (6.8M articles)"]
        SEARCH["Search API<br/>srsearch=query"]
        FULL["Get full article text"]
        PARA["Score paragraphs by<br/>query term overlap"]
        BEST["Best 2 paragraphs<br/>(800 chars)"]
    end

    subgraph MERGE["Final Evidence"]
        MRG["Merge local + wiki<br/>Deduplicate by title<br/>Sort by similarity"]
        FINAL["Top-5 Evidence<br/>with source + URL"]
    end

    CLAIM --> EXPAND
    Q1 --> ENC
    Q2 --> ENC
    ENC --> FAISS
    FAISS --> DR

    Q1 --> TOK
    TOK --> BM25
    BM25 --> SR

    DR --> RRF
    SR --> RRF
    RRF --> DEDUP
    DEDUP --> FUSED
    FUSED --> CE
    CE --> SIG
    SIG --> TOP

    CLAIM --> SEARCH
    SEARCH --> FULL
    FULL --> PARA
    PARA --> BEST

    TOP --> MRG
    BEST --> MRG
    MRG --> FINAL

    style DENSE fill:#bbdefb,stroke:#1565c0
    style SPARSE fill:#ffe0b2,stroke:#e65100
    style FUSION fill:#c8e6c9,stroke:#2e7d32
    style RERANK fill:#f8bbd0,stroke:#c62828
    style WIKIAPI fill:#e1bee7,stroke:#7b1fa2
    style MERGE fill:#fff9c4,stroke:#f57f17
```

---

## 4. Verdict Engine Decision Tree

```mermaid
graph TD
    START["Claim + Evidence<br/>(from retriever)"]

    RULE{"Rule Engine<br/>check_rules()"}
    RULE_YES["CONTRADICTED<br/>confidence: 0.95<br/>source: factual_rule"]

    NLI["DeBERTa NLI<br/>batch_nli()"]
    SCORES["max_contradiction<br/>max_entailment<br/>max_similarity"]

    WIKI_CHECK{"Wikipedia API<br/>returned evidence?"}
    WIKI_SIM{"Wiki similarity<br/>> 0.45?"}

    FAB{"Fabrication Check<br/>Specific entity + No wiki?"}
    FAB_YES["CONTRADICTED<br/>confidence: 0.80+<br/>reason: fabricated"]

    HARD{"Hard Contradiction<br/>NLI contra > 0.80<br/>AND contra > ent + 0.20<br/>AND sim > 0.50?"}
    HARD_YES["CONTRADICTED<br/>confidence: NLI-based"]

    WIKI_TRUST{"Wikipedia confirms<br/>topic exists?"}
    WIKI_SUP["SUPPORTED<br/>reason: Wikipedia confirms"]

    STRONG{"Strong Support<br/>specific_ent > 0.65?"}
    MOD{"Moderate Support<br/>raw_ent > 0.35<br/>AND sim > 0.30?"}
    WEAK{"Weak Support<br/>sim > 0.25<br/>AND contra < 0.60?"}

    SUP["SUPPORTED"]
    UNVER["UNVERIFIABLE"]

    START --> RULE
    RULE -->|"Match"| RULE_YES
    RULE -->|"No match"| NLI
    NLI --> SCORES
    SCORES --> FAB
    FAB -->|"Yes"| FAB_YES
    FAB -->|"No"| HARD
    HARD -->|"Yes"| WIKI_TRUST
    WIKI_TRUST -->|"Yes"| WIKI_SUP
    WIKI_TRUST -->|"No"| HARD_YES
    HARD -->|"No"| STRONG
    STRONG -->|"Yes"| SUP
    STRONG -->|"No"| MOD
    MOD -->|"Yes"| SUP
    MOD -->|"No"| WEAK
    WEAK -->|"Yes"| SUP
    WEAK -->|"No"| UNVER

    style RULE_YES fill:#ffcdd2,stroke:#c62828
    style FAB_YES fill:#ffcdd2,stroke:#c62828
    style HARD_YES fill:#ffcdd2,stroke:#c62828
    style SUP fill:#c8e6c9,stroke:#2e7d32
    style WIKI_SUP fill:#c8e6c9,stroke:#2e7d32
    style UNVER fill:#fff9c4,stroke:#f57f17
```

---

## 5. Bayesian Confidence Fusion

```mermaid
graph LR
    subgraph SIGNALS["5 Input Signals"]
        S1["NLI Score<br/>(max_ent - max_con + 1) / 2<br/>weight: 0.32"]
        S2["Retrieval Similarity<br/>max(evidence.similarity)<br/>weight: 0.22"]
        S3["Source Reliability<br/>wikipedia: 1.0<br/>pubmed: 1.0<br/>unknown: 0.5<br/>weight: 0.12"]
        S4["Cross-Reference<br/>fraction(ent > con)<br/>weight: 0.16"]
        S5["Uncertainty Stability<br/>1 - entropy(NLI dist)<br/>weight: 0.18"]
    end

    FUSE["Weighted Fusion<br/>Σ(w_i × signal_i)<br/>σ(result) → [0, 1]"]

    THRESH{"Threshold"}
    HIGH["Verified<br/>confidence ≥ 0.75"]
    MED["Uncertain<br/>0.40 ≤ conf < 0.75"]
    LOW["Hallucination<br/>confidence < 0.50"]

    S1 --> FUSE
    S2 --> FUSE
    S3 --> FUSE
    S4 --> FUSE
    S5 --> FUSE

    FUSE --> THRESH
    THRESH -->|"≥ 0.75"| HIGH
    THRESH -->|"0.40-0.75"| MED
    THRESH -->|"< 0.50"| LOW

    style HIGH fill:#c8e6c9,stroke:#2e7d32
    style MED fill:#fff9c4,stroke:#f57f17
    style LOW fill:#ffcdd2,stroke:#c62828
```

---

## 6. SelfCheck Consistency Scoring

```mermaid
graph TD
    CLAIM["Claim + Evidence"]

    subgraph LLM_PATH["LLM Path (if available)"]
        S1["Sample 1<br/>temp=0.10"]
        S2["Sample 2<br/>temp=0.25"]
        S3["Sample 3<br/>temp=0.40"]
        S4["Sample 4<br/>temp=0.55"]
        S5["Sample 5<br/>temp=0.70"]
        LABELS["Labels:<br/>supported | contradicted | uncertain"]
    end

    subgraph NLI_PATH["NLI Fallback (no LLM)"]
        E1["Evidence 1 × Claim → NLI"]
        E2["Evidence 2 × Claim → NLI"]
        E3["Evidence 3 × Claim → NLI"]
        NLILABELS["Map: max(ent)→supported<br/>max(con)→contradicted<br/>else→uncertain"]
    end

    subgraph METRICS["Uncertainty Metrics"]
        ENT["Normalized Entropy<br/>H(p_sup, p_con, p_unc) / log(3)"]
        DIS["Disagreement Ratio<br/>1 - (max_count / total)"]
        SEM["Semantic Cluster Entropy<br/>Jaccard clustering (thresh=0.5)"]
    end

    BLEND["Confidence Blend<br/>new = (1-0.2) × NLI_conf + 0.2 × consistency"]

    CLAIM --> LLM_PATH
    CLAIM --> NLI_PATH

    S1 --> LABELS
    S2 --> LABELS
    S3 --> LABELS
    S4 --> LABELS
    S5 --> LABELS

    E1 --> NLILABELS
    E2 --> NLILABELS
    E3 --> NLILABELS

    LABELS --> ENT
    LABELS --> DIS
    LABELS --> SEM
    NLILABELS --> ENT
    NLILABELS --> DIS

    ENT --> BLEND
    DIS --> BLEND
    SEM --> BLEND

    style LLM_PATH fill:#e1bee7,stroke:#7b1fa2
    style NLI_PATH fill:#bbdefb,stroke:#1565c0
    style METRICS fill:#fff3e0,stroke:#e65100
```

---

## 7. Chrome Extension Architecture

```mermaid
graph TB
    subgraph BROWSER["Browser (ChatGPT / Claude / Gemini)"]
        DOM["Chatbot DOM<br/>(assistant messages)"]
        INPUT["User Input Field"]
    end

    subgraph EXTENSION["Chrome Extension (Manifest V3)"]
        BG["background.js<br/>(service worker)"]
        CS["content.js<br/>(injected script)"]
        CSS["styles.css<br/>(3D liquid glass)"]

        subgraph BEACON["Beacon (72px glass orb)"]
            B_IDLE["Idle (blue)"]
            B_SCAN["Scanning (purple pulse)"]
            B_SAFE["Safe (green)"]
            B_ALERT["Alert (red pulse)"]
        end

        subgraph DASH["Dashboard (glass panel)"]
            KPI["KPI Row<br/>Score | Claims | Verified | False | Unclear"]
            CARDS["Claim Cards<br/>verdict + confidence + evidence"]
            LOGO["VF Logo → Website"]
        end

        subgraph PROMPT["Prompt Suggestion"]
            TIP["Suggestion Tooltip<br/>Score + Improvements"]
            ACCEPT["Accept Suggested"]
            DISMISS["Use Original"]
            CLOSE["× Close"]
        end
    end

    subgraph BACKEND["HuggingFace Space"]
        FAST["/analyze/fast"]
        OPT["/optimize"]
    end

    DOM -->|"MutationObserver"| CS
    CS -->|"Extract text"| CS
    CS -->|"POST (3 parallel)"| FAST
    FAST -->|"JSON response"| CS
    CS -->|"Highlight red/yellow"| DOM
    CS -->|"Update"| BEACON
    CS -->|"Render"| DASH

    INPUT -->|"keyup (3s debounce)"| CS
    CS -->|"POST"| OPT
    OPT -->|"suggestion"| PROMPT

    ACCEPT -->|"Insert into"| INPUT
    DISMISS -->|"Hide"| PROMPT

    style BROWSER fill:#e3f2fd,stroke:#1565c0
    style EXTENSION fill:#f3e5f5,stroke:#7b1fa2
    style BACKEND fill:#e8f5e9,stroke:#2e7d32
```

---

## 8. Deployment Architecture

```mermaid
graph TB
    subgraph DEV["Development (Local)"]
        MAC["MacBook Air M4<br/>16GB RAM"]
        OLL["Ollama<br/>llama3.1:8b"]
        LOC_FAISS["Local FAISS<br/>(369K vectors)"]
    end

    subgraph PROD["Production (Free Tier)"]
        subgraph HF["HuggingFace Spaces"]
            DOCKER["Docker Container<br/>Python 3.11-slim"]
            PRE["Pre-built FAISS Index<br/>(11K vectors, baked in)"]
            MODELS["Pre-downloaded Models<br/>DeBERTa + MiniLM + Reranker"]
            SERVER["overlay_server.py<br/>Port 7860"]
        end

        subgraph VERCEL["Vercel"]
            NEXTJS["Next.js 15.5.15<br/>Static Export"]
            ASSETS["Logo + AI logos<br/>Crystal theme"]
        end

        subgraph FREE_APIS["Free External APIs"]
            GROQ_API["Groq API<br/>Llama 3.1 8B<br/>30 req/min"]
            WIKI_API["Wikipedia API<br/>6.8M articles<br/>Unlimited"]
        end
    end

    subgraph USER["End User"]
        CHROME["Chrome Browser<br/>+ Extension"]
    end

    CHROME -->|"HTTPS"| SERVER
    CHROME -->|"HTTPS"| NEXTJS
    SERVER -->|"HTTP"| GROQ_API
    SERVER -->|"HTTP"| WIKI_API
    DOCKER --- PRE
    DOCKER --- MODELS
    DOCKER --- SERVER

    style DEV fill:#fff3e0,stroke:#e65100
    style HF fill:#e8f5e9,stroke:#2e7d32
    style VERCEL fill:#e3f2fd,stroke:#1565c0
    style FREE_APIS fill:#f1f8e9,stroke:#558b2f
```

---

## 9. Rule Engine Coverage

```mermaid
graph TD
    INPUT["Claim Text"]

    R1{"Rule 1<br/>Landmark Location<br/>(68 entries)"}
    R2{"Rule 2<br/>Capital-Country<br/>(36 entries)"}
    R3{"Rule 3<br/>City-Country<br/>(19 entries)"}
    R4{"Rule 4<br/>Country-Continent<br/>(30 entries)"}
    R5{"Rule 5<br/>Person Achievement<br/>(30+ entries)"}
    R6{"Rule 6<br/>Historical Events<br/>(7 events)"}
    R7{"Rule 7<br/>Person Dates<br/>(9 people)"}
    R8{"Rule 8<br/>Science Numbers<br/>(5 facts)"}
    R9{"Rule 9<br/>Science T/F<br/>(8 statements)"}

    MATCH["CONTRADICTED<br/>confidence: 0.95<br/>skip NLI"]
    NONE["No match<br/>→ proceed to NLI"]

    INPUT --> R1
    R1 -->|"Great Wall → China"| MATCH
    R1 -->|"No"| R2
    R2 -->|"Tokyo → Japan"| MATCH
    R2 -->|"No"| R3
    R3 -->|"London → UK"| MATCH
    R3 -->|"No"| R4
    R4 -->|"Brazil → South America"| MATCH
    R4 -->|"No"| R5
    R5 -->|"Einstein → physics"| MATCH
    R5 -->|"No"| R6
    R6 -->|"WWII → 1939-1945"| MATCH
    R6 -->|"No"| R7
    R7 -->|"Shakespeare → 1564-1616"| MATCH
    R7 -->|"No"| R8
    R8 -->|"Water → 100°C"| MATCH
    R8 -->|"No"| R9
    R9 -->|"Sun revolves Earth → False"| MATCH
    R9 -->|"No"| NONE

    style MATCH fill:#ffcdd2,stroke:#c62828
    style NONE fill:#c8e6c9,stroke:#2e7d32
```

---

## 10. LLM Fallback Chain

```mermaid
graph LR
    REQ["LLM Request<br/>(claim decomposition,<br/>prompt optimization,<br/>corrections)"]

    P1{"Primary Provider<br/>(configured)"}
    OLLAMA["Ollama<br/>llama3.1:8b<br/>localhost:11434"]
    GROQ["Groq API<br/>llama-3.1-8b-instant<br/>Free, 500 tok/s"]
    ANTHRO["Anthropic<br/>Claude Sonnet<br/>(if API key set)"]
    OPENAI["OpenAI<br/>GPT-4o-mini<br/>(if API key set)"]
    SPACY["spaCy Fallback<br/>en_core_web_sm<br/>Sentence segmentation"]
    NONE["provider='none'<br/>Skip LLM entirely"]

    REQ --> P1
    P1 -->|"ollama"| OLLAMA
    P1 -->|"groq"| GROQ
    P1 -->|"none"| NONE

    OLLAMA -->|"fail"| GROQ
    GROQ -->|"fail"| ANTHRO
    ANTHRO -->|"fail"| OPENAI
    OPENAI -->|"fail"| SPACY
    NONE --> SPACY

    style OLLAMA fill:#c8e6c9,stroke:#2e7d32
    style GROQ fill:#c8e6c9,stroke:#2e7d32
    style ANTHRO fill:#fff9c4,stroke:#f57f17
    style OPENAI fill:#fff9c4,stroke:#f57f17
    style SPACY fill:#ffcdd2,stroke:#c62828
    style NONE fill:#e0e0e0,stroke:#616161
```

---

*All diagrams verified against codebase commit 3ec97e2 (April 17, 2026).*
*Paste each mermaid code block into https://mermaid.live to render.*
