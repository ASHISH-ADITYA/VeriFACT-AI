# VeriFact — Production Product Blueprint
### Principal Architecture & Engineering Plan

---

## Section A. Executive Summary

### A.1 Product Concept
VeriFact is an ambient AI truth layer that runs as a browser extension on ChatGPT, Claude, and other LLM web interfaces. A small floating beacon continuously monitors assistant responses, decomposes them into atomic factual claims, verifies each claim against trusted knowledge sources using dense retrieval + NLI, and surfaces a real-time factuality score with per-claim evidence and corrections — all without interrupting the user's workflow.

### A.2 Why This Wins
Existing fact-checkers (SelfCheckGPT, FActScore, SAFE) are research tools — batch-mode, CLI-only, no UI. VeriFact is the first **ambient, real-time verification product** embedded directly in the user's chat flow. The differentiators:
1. **Zero friction** — no copy-paste, no tab switching. Verification happens automatically.
2. **Claim-level granularity** — not just "this is wrong" but exactly which sentence, why, and what the correction is.
3. **Local-first privacy** — inference runs on the user's machine (Ollama + MPS). No data leaves the device unless the user opts in.
4. **Research-grade accuracy** — DeBERTa NLI + FAISS retrieval, benchmarked on TruthfulQA and HaluEval.

### A.3 Core User Journeys
1. **Passive monitoring**: User chats with ChatGPT. Beacon is green. ChatGPT says something false → beacon turns red → user clicks → sees exactly which claim is wrong + evidence.
2. **Deep analysis**: User pastes a long AI-generated report into the VeriFact dashboard → gets annotated output with per-claim verdicts, confidence histogram, downloadable report.
3. **Conversation audit**: User clicks "Analyze full thread" → gets a conversation-level risk summary with claim timeline.

### A.4 Success Metrics
| Timeframe | Metric | Target |
|---|---|---|
| 30 days | Working extension on Chrome Web Store (unlisted) | Installable, functional on ChatGPT + Claude |
| 30 days | Beacon-to-result latency | < 8s for 5-claim message |
| 60 days | Hallucination detection F1 (TruthfulQA) | ≥ 0.72 |
| 60 days | Active daily users (friends + classmates beta) | 20+ |
| 90 days | Public Chrome Web Store listing | Published |
| 90 days | Precision on flagged claims | ≥ 0.80 |

---

## Section B. Product Requirements Document

### B.1 Personas

**Persona 1: Student / Researcher (Primary)**
- Uses ChatGPT 5-10x daily for coursework, literature review, coding help
- Has been burned by hallucinated citations, incorrect dates, fabricated facts
- Needs: passive safety net, quick verification, citation links
- Technical skill: moderate (can install extension, run local server)

**Persona 2: Content Creator / Journalist**
- Uses LLMs to draft articles, social media, research summaries
- Professional reputation at stake if facts are wrong
- Needs: high-precision flagging, exportable reports, source verification
- Technical skill: low-moderate (needs polished UI, no CLI)

**Persona 3: AI-Heavy Professional**
- Uses Claude/ChatGPT for legal research, medical questions, financial analysis
- High-stakes domain where a single hallucination has real consequences
- Needs: domain-specific accuracy, confidence calibration, audit trail
- Technical skill: varies

### B.2 Jobs To Be Done
1. "When I'm chatting with an AI, I want to know immediately if something it says is factually wrong, so I don't unknowingly use false information."
2. "When I'm reviewing AI-generated content before publishing, I want every claim checked against sources, so I can confidently publish."
3. "When I'm making a decision based on AI advice, I want to see the confidence level per claim, so I know what to double-check."

### B.3 User Stories (Prioritized)

| Priority | Story |
|---|---|
| P0 | As a user, I see a beacon on ChatGPT/Claude that changes color based on the factuality of the latest response |
| P0 | As a user, I click the beacon and see a popup with factuality score, claim count, and flagged claims |
| P0 | As a user, each flagged claim shows the verdict (supported/contradicted/unverifiable), confidence %, and evidence snippet |
| P1 | As a user, I can paste any text into a web dashboard for deep analysis with full annotation |
| P1 | As a user, I can export a verification report as JSON |
| P1 | As a user, contradicted claims show a suggested correction with source citation |
| P2 | As a user, I can analyze an entire conversation thread and see a risk timeline |
| P2 | As a user, I can configure which domains (medical, legal, general) to prioritize |
| P3 | As a user, I can use a desktop menu-bar companion app |

### B.4 UX Outcomes & Usability Targets
- **Time to first insight**: < 10 seconds after assistant response appears
- **Clicks to evidence**: 1 click (beacon) → verdict visible. 1 more click → full evidence.
- **Cognitive load**: traffic-light metaphor (green/yellow/red) requires zero learning
- **Accessibility**: WCAG 2.1 AA contrast ratios. Keyboard-navigable popup.

### B.5 Functional Requirements

**Live Beacon Behavior**
- Floating circular button, bottom-right corner, 56px
- States: idle (blue), analyzing (yellow pulse), good (green), caution (yellow), high-risk (red), error (purple)
- Auto-detects new assistant messages via DOM polling (3.5s interval)
- Sends latest message to local analysis server
- Updates beacon color based on factuality score

**On-Demand Deep Analysis**
- Streamlit dashboard at localhost:8501
- Text input → full annotated output with color-coded claim spans
- Per-claim expandable cards with NLI scores, evidence, source links
- Confidence distribution histogram
- Factuality score gauge

**Claim-Level Explanation**
- Each claim shows: verdict, confidence %, best evidence passage, source title/URL
- Contradicted claims include LLM-generated correction
- NLI score breakdown (entailment/neutral/contradiction bar)

**Confidence Scoring**
- Bayesian fusion of: NLI probability (40%), retrieval relevance (25%), source reliability (15%), cross-reference agreement (20%)
- Calibrated against TruthfulQA ground truth
- Displayed as percentage per claim and as aggregate per message

**Source Linking**
- Every evidence passage links to: Wikipedia article URL or PubMed abstract URL
- Source reliability badge (Wikipedia = high, PubMed = high, unknown = low)

**Conversation Risk Summary** (P2)
- Aggregates all claims across conversation messages
- Timeline view: which messages had highest contradiction rates
- Overall conversation factuality trend

### B.6 Non-Functional Requirements

| Requirement | Target | Rationale |
|---|---|---|
| Beacon-to-result latency (5 claims) | < 8s (local), < 4s (cloud) | User attention span on chat is ~10s |
| Hallucination detection precision | ≥ 0.80 | False positives erode trust faster than false negatives |
| Hallucination detection recall | ≥ 0.65 | Missing some hallucinations is acceptable; flagging real ones is not |
| Uptime (local server) | 99% during active session | Extension must not crash mid-conversation |
| Memory footprint | < 3GB (local models + index) | Must coexist with browser on 16GB M4 |
| Data privacy | Zero data transmitted to external servers (local mode) | Trust requirement for professional users |
| Extension size | < 500KB (no bundled models) | Chrome Web Store limit and install speed |

### B.7 Non-Goals for MVP
- Real-time streaming analysis (analyzing as tokens arrive) — too complex, defer to v2
- Multi-language support — English only for MVP
- Mobile app — browser extension only
- Chrome Web Store monetization — free for MVP
- User accounts / cloud sync — local-only
- Image/video fact-checking — text only
- Fine-tuning custom models — use off-the-shelf DeBERTa + sentence-transformers

---

## Section C. End-to-End System Architecture

### C.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER'S BROWSER                            │
│                                                                   │
│  ┌────────────────────┐     ┌──────────────────────────────┐    │
│  │  ChatGPT / Claude  │     │  VeriFact Chrome Extension   │    │
│  │  (web chat UI)     │────▶│  - Content script (DOM poll) │    │
│  │                    │     │  - Beacon UI overlay          │    │
│  └────────────────────┘     │  - Popup panel renderer       │    │
│                              └──────────────┬───────────────┘    │
│                                             │ HTTP POST /analyze │
└─────────────────────────────────────────────┼───────────────────┘
                                              │
                         ┌────────────────────▼────────────────────┐
                         │         LOCAL ANALYSIS SERVER            │
                         │         (127.0.0.1:8765)                │
                         │                                         │
                         │  ┌─────────────────────────────────┐   │
                         │  │       API Gateway / Router       │   │
                         │  │   /health  /analyze  /analyze-   │   │
                         │  │                      conversation│   │
                         │  └──────────────┬──────────────────┘   │
                         │                 │                       │
                         │  ┌──────────────▼──────────────────┐   │
                         │  │      VeriFactPipeline            │   │
                         │  │                                  │   │
                         │  │  ┌──────────┐  ┌─────────────┐  │   │
                         │  │  │  Claim    │  │  Evidence    │  │   │
                         │  │  │  Decomp   │  │  Retriever   │  │   │
                         │  │  │  (LLM /   │  │  (FAISS +    │  │   │
                         │  │  │  spaCy)   │  │  sent-trans) │  │   │
                         │  │  └──────────┘  └─────────────┘  │   │
                         │  │                                  │   │
                         │  │  ┌──────────┐  ┌─────────────┐  │   │
                         │  │  │  Verdict  │  │  Annotator   │  │   │
                         │  │  │  Engine   │  │  (HTML/JSON) │  │   │
                         │  │  │  (NLI +   │  │              │  │   │
                         │  │  │  Bayes)   │  │              │  │   │
                         │  │  └──────────┘  └─────────────┘  │   │
                         │  └─────────────────────────────────┘   │
                         │                                         │
                         │  ┌─────────┐  ┌──────────┐  ┌───────┐ │
                         │  │  Ollama  │  │  FAISS   │  │ Cache │ │
                         │  │  (LLM)   │  │  Index   │  │(dict) │ │
                         │  └─────────┘  └──────────┘  └───────┘ │
                         └─────────────────────────────────────────┘

                         ┌─────────────────────────────────────────┐
                         │       STREAMLIT DASHBOARD (:8501)       │
                         │       (Deep analysis / evaluation)      │
                         └─────────────────────────────────────────┘
```

### C.2 Sequence Flow: Live Analysis (Beacon)

```
User sends message to ChatGPT
  │
  ▼
ChatGPT renders assistant response in DOM
  │
  ▼ (3.5s poll interval)
Content Script detects new message (fingerprint changed)
  │
  ▼
Beacon → yellow (analyzing)
  │
  ▼
Content Script → POST http://127.0.0.1:8765/analyze
  Body: { "text": "<assistant message>", "top_claims": 6 }
  │
  ▼
Overlay Server receives request
  │
  ▼ (cached? → return immediately)
Pipeline.verify_text(text)
  │
  ├── Stage 1: Claim Decomposition (Ollama or spaCy fallback) ........... ~4-6s
  ├── Stage 2: Evidence Retrieval (FAISS batch search) .................. ~0.3s
  ├── Stage 3: NLI Verdict (DeBERTa on MPS) ............................ ~0.5s
  ├── Stage 4: Confidence Scoring ...................................... ~0.01s
  └── Stage 5: Annotate + corrections .................................. ~2-4s
  │
  ▼
Response JSON → Content Script
  │
  ▼
Beacon → green/yellow/red based on factuality_score
Panel renders: score, confidence, flagged claims
```

### C.3 Sequence Flow: Deep Analysis (Dashboard)

```
User pastes text into Streamlit input
  │
  ▼
Pipeline.verify_text(text)
  │  (same stages as above)
  ▼
Streamlit renders:
  - Color-coded annotated text with hover tooltips
  - Expandable claim cards with NLI bars
  - Factuality score gauge + pie chart
  - Confidence distribution histogram
  - Downloadable JSON report
```

### C.4 Real-Time Event Model

| Event | Source | Destination | Trigger |
|---|---|---|---|
| `new_message_detected` | Content Script | Overlay Server | DOM mutation / poll |
| `analysis_started` | Overlay Server | Content Script | Request received |
| `analysis_complete` | Overlay Server | Content Script | Pipeline finished |
| `beacon_state_change` | Content Script | Beacon UI | Score threshold |
| `claim_flagged` | Pipeline | JSON response | Contradiction detected |
| `fallback_activated` | LLM Client | Server logs | Provider failure |

### C.5 Data Model

```
Session
  ├── id: UUID
  ├── platform: "chatgpt" | "claude"
  ├── started_at: ISO timestamp
  ├── messages: [Message]
  └── conversation_factuality: float

Message
  ├── id: UUID
  ├── session_id: FK
  ├── text: string
  ├── role: "assistant"
  ├── fingerprint: string (length:first80:last80)
  ├── analyzed_at: ISO timestamp
  ├── factuality_score: float (0-100)
  ├── processing_time_s: float
  └── claims: [Claim]

Claim
  ├── id: string (c-0, c-1, ...)
  ├── message_id: FK
  ├── text: string
  ├── claim_type: enum
  ├── char_start: int
  ├── char_end: int
  ├── verdict: "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE" | "NO_EVIDENCE"
  ├── confidence: float (0-1)
  ├── nli_scores: {entailment, neutral, contradiction}
  ├── correction: string | null
  └── evidence: [Evidence]

Evidence
  ├── text: string
  ├── source: "wikipedia" | "pubmed"
  ├── title: string
  ├── url: string
  ├── similarity: float
  └── chunk_id: int

Verdict (aggregated)
  ├── label: string
  ├── confidence: float
  ├── best_evidence: Evidence
  └── all_nli: [NLIResult]
```

### C.6 Storage Strategy
- **Transient** (in-memory dict cache): message analysis results keyed by content hash. Cleared on server restart. Size-limited to 500 entries (LRU).
- **Persistent** (optional, SQLite): conversation history for audit trail. Disabled by default. User opt-in via settings.
- **Vector store**: FAISS flat index, read-only after build. Stored on disk (~36MB for 5K articles, ~2GB for full Wikipedia).

### C.7 Security Architecture
- **CORS**: Overlay server allows only `127.0.0.1` origin. Extension communicates via localhost only.
- **No auth for local mode**: server binds to `127.0.0.1` only (not `0.0.0.0`), inaccessible from network.
- **Extension identity**: Manifest V3 with minimal permissions (`storage` only, no `tabs`, no `<all_urls>`). Host permissions scoped to chatgpt.com, claude.ai, and localhost.
- **Abuse controls**: Rate limit at 10 requests/minute per content script. Minimum text length (40 chars) before analysis.
- **No telemetry in local mode**: zero external network calls except Ollama (localhost) and HuggingFace model download (one-time).

### C.8 Multi-Tenant Readiness (Future)
Not needed for MVP. Path: if deploying as a cloud service, add JWT auth, tenant isolation in cache keys, and per-tenant rate limits. FAISS index can be shared (read-only). LLM inference would need per-tenant queuing.

**Acceptance Criteria for Section C:**
- Architecture diagram matches actual deployed components
- All data flows traced from user action to UI update
- No external network calls in local mode (verified by packet capture)
- Data model supports all P0 and P1 user stories

**Assumptions:** User has Ollama installed. 16GB RAM minimum. Chrome browser.
**Unknowns:** ChatGPT/Claude DOM structure may change without notice (addressed in Section D resilience).

---

## Section D. Extension Architecture & UX Blueprint

### D.1 Manifest & Permissions Strategy

```json
{
  "manifest_version": 3,
  "permissions": ["storage"],
  "host_permissions": [
    "https://chatgpt.com/*",
    "https://chat.openai.com/*", 
    "https://claude.ai/*",
    "http://127.0.0.1:8765/*"
  ],
  "content_scripts": [{
    "matches": ["https://chatgpt.com/*", "https://chat.openai.com/*", "https://claude.ai/*"],
    "js": ["content.js"],
    "css": ["styles.css"],
    "run_at": "document_idle"
  }]
}
```

**Rationale**: Minimal permissions. No `tabs`, no `activeTab`, no `<all_urls>`. `storage` for user preferences only. Host permissions scoped to exact domains.

**Alternative considered**: `activeTab` + popup page. Rejected because it requires user click to activate — violates "ambient, always-on" requirement.

### D.2 Site Adapters

```javascript
// adapter interface
class SiteAdapter {
  getPlatformName()         // "chatgpt" | "claude"
  getAssistantMessages()    // returns string[]
  isStreaming()             // true if response is still generating
}
```

**ChatGPT Adapter**
- Selector: `[data-message-author-role='assistant']`
- Streaming detection: check for presence of streaming cursor element
- Stability: HIGH (data attributes are stable API)

**Claude Adapter**
- Selector: `div[data-is-streaming], div.prose, div.font-claude-message`
- Streaming detection: `data-is-streaming` attribute
- Stability: MEDIUM (class-based selectors may change)

**Future Adapter Framework**
- Abstract `SiteAdapter` base class
- New adapters registered in a `ADAPTERS` map keyed by hostname pattern
- DOM selectors stored in a config object per adapter (hot-updatable via extension storage)

### D.3 Beacon States & Transitions

```
                    ┌───────────┐
                    │   IDLE    │ (blue, no activity)
                    └─────┬─────┘
                          │ new message detected
                          ▼
                    ┌───────────┐
             ┌──────│ ANALYZING │ (yellow, pulsing)
             │      └─────┬─────┘
             │            │ result received
     timeout │            ▼
     (15s)   │   ┌────────────────┐
             │   │ Score >= 80?   │
             │   └───┬────┬───┬──┘
             │    Y  │    │   │ N (< 55)
             │       ▼    │   ▼
             │  ┌──────┐  │ ┌──────────┐
             │  │ GOOD │  │ │HIGH RISK │
             │  │(green)│  │ │  (red)   │
             │  └──────┘  │ └──────────┘
             │            │
             │            ▼ (55-79)
             │       ┌─────────┐
             │       │ CAUTION │
             │       │ (yellow)│
             │       └─────────┘
             │
             ▼
        ┌─────────┐
        │  ERROR  │ (purple, server unreachable)
        └─────────┘
```

### D.4 Popup Information Architecture

```
┌──────────────────────────────┐
│ VeriFact Live (ChatGPT)      │
├──────────────────────────────┤
│ ┌────────┬────────┬────────┐ │
│ │  73    │  81%   │   5    │ │
│ │ Score  │ Conf.  │ Claims │ │
│ └────────┴────────┴────────┘ │
│                              │
│ 3 supported, 1 contradicted, │
│ 1 unverifiable               │
├──────────────────────────────┤
│ ❌ CONTRADICTED (55%)        │
│ "Einstein invented the       │
│  telephone"                  │
│ Source: Wikipedia — Telephone │
│                              │
│ ✅ SUPPORTED (94%)           │
│ "The Eiffel Tower is in      │
│  Paris, France"              │
│ Source: Wikipedia — Eiffel    │
├──────────────────────────────┤
│ ⚙ Settings  📋 Copy report  │
└──────────────────────────────┘
```

### D.5 Visual Design System

**Colors**
- Good: `#16a34a` (green-600)
- Caution: `#f59e0b` (amber-500)
- High risk: `#dc2626` (red-600)
- Idle: `#2563eb` (blue-600)
- Error: `#7c3aed` (violet-600)
- Panel background: `rgba(17, 24, 39, 0.92)` (dark glass)
- Panel text: `#f9fafb` (gray-50)

**Typography**: System UI stack (`-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`)

**Spacing**: 4px base grid. Padding: 14px panel, 8px claim cards.

**Motion**: Beacon pulse animation 1.4s infinite on analyzing state. Panel fade-in 200ms ease.

**Accessibility**: All foreground/background combinations meet WCAG 2.1 AA (4.5:1 contrast ratio minimum). Beacon is keyboard-focusable. Panel is scrollable with keyboard. Verdict icons have aria-labels.

### D.6 DOM Resilience Strategy
1. **Primary selectors**: Use `data-*` attributes where available (ChatGPT has stable data attributes)
2. **Fallback selectors**: CSS class chains as backup
3. **Graceful degradation**: If no messages found, beacon stays idle (doesn't crash)
4. **Hot-updatable selectors**: Store selector config in `chrome.storage.local`. Can push updates without extension version bump.
5. **MutationObserver** (future optimization): Replace polling with DOM observer for lower CPU usage

### D.7 Privacy UX
- **Local-only by default**: All analysis happens on localhost. Badge in panel: "🔒 Local mode — no data leaves your device"
- **User controls**: Toggle extension on/off per site via popup settings
- **Opt-in scopes**: User can disable analysis for specific conversation types
- **Redaction**: Extension never stores raw chat text persistently (only transient cache, cleared on restart)

### D.8 Abuse & False Positive Handling
- Claims with confidence < 40% show "⚠️ Low confidence — verify manually" instead of a verdict
- Users can dismiss/hide individual claim flags (stored in session, not persistent)
- Beacon tooltip shows "Last analyzed: [timestamp]" so user knows freshness

**Acceptance Criteria for Section D:**
- Extension loads cleanly on ChatGPT and Claude without console errors
- Beacon appears within 2s of page load
- Panel renders correctly with all claim types
- Extension passes Chrome Web Store review (no excessive permissions)

**Assumptions:** ChatGPT and Claude are accessed via their web UIs, not API.
**Unknowns:** Exact DOM mutation timing on fast-streaming responses.

---

## Section E. Accuracy Strategy

### E.1 Hallucination Taxonomy

| Type | Description | Detection Method |
|---|---|---|
| Entity hallucination | Real entities in false relationships | NLI contradiction on entity claims |
| Numerical hallucination | Fabricated statistics, dates | NLI + exact match against retrieved evidence |
| Citation hallucination | Non-existent papers, sources | Retrieval returns no matching evidence |
| Causal hallucination | False cause-effect claims | NLI on causal claim type |
| Temporal hallucination | Wrong dates, anachronisms | NLI + temporal evidence matching |

### E.2 Multi-Stage Verification Pipeline

```
Text Input
  │
  ▼
Stage 1: Claim Extraction
  Model: Ollama (llama3.1:8b) with structured JSON prompt
  Fallback: spaCy sentence segmentation
  Output: List[Claim] with types and source spans
  │
  ▼
Stage 2: Dense Retrieval
  Model: all-MiniLM-L6-v2 (384-dim)
  Index: FAISS IndexFlatIP (cosine similarity)
  Corpus: Wikipedia Simple English + PubMed QA
  Top-k: 3 (interactive) / 5 (eval)
  Threshold: 0.30 minimum similarity
  │
  ▼
Stage 3: NLI Verification
  Model: cross-encoder/nli-deberta-v3-base
  Input: (evidence_passage, claim) pairs
  Output: P(entailment), P(neutral), P(contradiction)
  Aggregation: max across top-k evidence per claim
  Hardware: Apple MPS (Metal) acceleration
  │
  ▼
Stage 4: Confidence Calibration
  Method: Bayesian weighted fusion
  Weights: NLI (0.40), retrieval (0.25), source (0.15), cross-ref (0.20)
  Thresholds: ≥ 0.75 verified, 0.40-0.74 uncertain, < 0.40 unverifiable
  │
  ▼
Stage 5: Output Assembly
  Annotated HTML + structured JSON
  Corrections for contradicted claims (LLM-generated)
```

**Future upgrade path (v2)**: Add cross-encoder reranker (ms-marco-MiniLM) between retrieval and NLI for better evidence selection.

### E.3 Confidence Design

| Level | Computation | Display |
|---|---|---|
| Claim confidence | Bayesian fusion of 4 signals | Percentage per claim card |
| Message confidence | Mean of claim confidences | Displayed in beacon popup |
| Conversation confidence | Weighted mean (recent messages weighted higher) | Conversation risk summary |

### E.4 Calibration Methodology
1. Run pipeline on TruthfulQA validation set (817 questions)
2. Bin predictions by confidence decile
3. Compute Expected Calibration Error (ECE)
4. If ECE > 0.15, apply temperature scaling (Guo et al., 2017)
5. Re-evaluate until ECE ≤ 0.15
6. Report reliability diagram in project report

### E.5 Benchmark Plan

| Dataset | Type | Samples | Metrics |
|---|---|---|---|
| TruthfulQA (fixed-response) | Core benchmark | 817 | Precision, Recall, F1, macro F1 |
| HaluEval (QA) | Core benchmark | 500 | Accuracy, AUROC, confusion matrix |
| TruthfulQA (live-gen) | Stress test | 100 | Factuality distribution, verdict breakdown |
| Hand-crafted test set | Sanity check | 10 | Manual verification |

**Human eval protocol** (if time permits):
- 3 evaluators rate 50 claim verdicts as correct/incorrect
- Compute inter-annotator agreement (Cohen's kappa)
- Compare against pipeline verdicts

**Precision-recall targets**: Precision ≥ 0.80, Recall ≥ 0.65, F1 ≥ 0.72

### E.6 Error Analysis Process
1. Collect all false positives and false negatives from benchmark runs
2. Categorize by: retrieval failure, NLI misclassification, decomposition error, corpus gap
3. Prioritize: corpus gaps → more data. NLI errors → threshold tuning. Decomposition errors → prompt improvement.

### E.7 Safety Guardrails
- Claims with confidence < 0.40: show "Insufficient evidence" instead of a verdict
- Never assert "this is true" — always say "supported by evidence" or "no contradicting evidence found"
- Medical/legal claims: add disclaimer "This is not professional advice. Verify with authoritative sources."

**Acceptance Criteria for Section E:**
- Pipeline achieves F1 ≥ 0.72 on TruthfulQA fixed-response
- ECE ≤ 0.15 on calibration test
- No claim is displayed as "verified" if confidence < 0.75

---

## Section F. Performance Strategy

### F.1 Latency Budget (5-Claim Message, Local M4)

| Stage | Budget | Actual (measured) |
|---|---|---|
| Claim decomposition (Ollama) | 5s | 4-6s |
| Evidence retrieval (FAISS) | 0.5s | 0.3s |
| NLI verdict (DeBERTa MPS) | 1s | 0.5-1.2s |
| Confidence scoring | 0.05s | 0.01s |
| Correction generation (Ollama) | 3s | 2-4s |
| **Total** | **< 10s** | **7-12s** |

### F.2 Fast Path vs Deep Path

| Mode | When | What's Different |
|---|---|---|
| **Fast path** (beacon) | Every new message | top_k=3, skip correction generation, max 6 claims reported |
| **Deep path** (dashboard) | User-initiated paste | top_k=5, full corrections, all claims, detailed NLI scores |

### F.3 Caching Layers

| Layer | Key | TTL | Purpose |
|---|---|---|---|
| Message hash cache | MD5(text) | Session | Avoid re-analyzing identical messages |
| Claim cache | MD5(claim_text) | Session | Skip re-retrieval for repeated claims |
| FAISS index | On-disk | Permanent | Pre-built, read-only |
| NLI model | In-memory | Process lifetime | DeBERTa stays loaded after first use |

### F.4 Parallelization Plan
- **Batch retrieval**: All claims embedded and searched in one FAISS call (already implemented)
- **Batch NLI**: All (evidence, claim) pairs processed in one DeBERTa forward pass (already implemented)
- **Claim decomposition**: Sequential (LLM call), but could be parallelized with spaCy pre-segmentation + LLM refinement

### F.5 Backpressure Handling
- Overlay server: `ThreadingHTTPServer` handles concurrent requests
- If analysis takes > 15s, content script times out and shows "Analysis timed out" in beacon
- Maximum queue depth: 3 pending requests (newer requests replace older ones)

### F.6 Cost Optimization
- **Local mode**: Zero marginal cost. Electricity only.
- **Cloud mode (future)**: Cache aggressively. Batch similar queries. Use smaller models for fast path.
- **Corpus**: Simple English Wikipedia (200K articles) is 10x smaller than full English but covers 90%+ of common knowledge claims.

### F.7 SLOs & Alerts

| SLO | Target | Alert Threshold |
|---|---|---|
| P50 latency | < 8s | > 12s |
| P95 latency | < 15s | > 20s |
| Error rate | < 5% | > 10% |
| Ollama availability | 99% during session | 3 consecutive failures |

**Acceptance Criteria for Section F:**
- P50 latency < 8s measured on 10 diverse messages
- Cache hit returns < 100ms
- No OOM on 16GB M4 with browser + Ollama + pipeline running

---

## Section G. Deployment Strategy

### G.1 Infrastructure by Phase

| Phase | Infra | Cost |
|---|---|---|
| **MVP** (now) | Local only: Ollama + Python server + Chrome extension | $0 |
| **Beta** (month 2) | Add Vercel landing page + docs. Analysis stays local. | $0-20/mo |
| **Production** (month 3+) | Optional cloud endpoint on Railway/Fly.io for users without local GPU | $20-50/mo |

### G.2 Frontend on Vercel, Inference Local
- **Vercel**: Landing page, documentation, demo video, download links. Static site, no backend.
- **Inference**: Always local by default. Cloud inference is opt-in for users who can't run Ollama.
- **Rationale**: ML inference on Vercel serverless would be slow (cold starts) and expensive. Keep inference on user's machine or a dedicated GPU server.

### G.3 Environments

| Environment | Purpose | Config |
|---|---|---|
| `local-dev` | Development and testing | Ollama, 5K article index, interactive profile |
| `local-full` | Full local deployment | Ollama, 200K article index, eval profile |
| `staging` | Pre-release testing | Cloud GPU, full index, staging API endpoint |
| `production` | Public release | Cloud GPU + CDN, full index, production API |

### G.4 CI/CD Flow
```
git push → GitHub Actions
  ├── Lint (ruff + mypy)
  ├── Unit tests (pytest)
  ├── Integration test (smoke_test.py with mock LLM)
  ├── Extension lint (web-ext lint)
  └── Build → artifacts
        ├── Extension .zip for Chrome Web Store
        └── Docker image for cloud deployment
```

### G.5 Secrets & Config
- `.env` for local secrets (never committed)
- GitHub Secrets for CI
- Vercel environment variables for landing page
- Extension uses `chrome.storage.local` for user preferences (no secrets needed — local mode)

### G.6 Rollback
- Extension: Chrome Web Store supports version rollback
- Server: Docker image tagged by version, rollback = redeploy previous tag
- FAISS index: versioned on disk, rollback = symlink to previous version

### G.7 Monitoring
- **Local**: `verifactai.log` with structured logging (loguru)
- **Cloud** (future): Prometheus metrics + Grafana dashboard
- Metrics: request count, latency percentiles, error rate, cache hit rate, verdict distribution

**Acceptance Criteria for Section G:**
- `streamlit run app.py` starts cleanly from fresh clone
- `python overlay_server.py` serves /health within 30s of boot
- Extension loads in Chrome without errors

---

## Section H. Desktop App & Mobile Path

### H.1 Desktop Companion App (Phase 2)
- **Architecture**: Electron or Tauri app wrapping the overlay server
- **UX**: System tray / menu bar icon. Click to see latest analysis. Always-on background process.
- **Advantage**: Works with any browser (not just Chrome). Works with desktop apps (Slack, email).
- **Timeline**: Month 3-4, after extension is stable.

### H.2 System Tray Architecture
```
┌─────────────────────┐
│  Menu Bar: "VF ✅"  │ ← shows beacon color
├─────────────────────┤
│  Latest: 87% score  │
│  3 claims verified  │
│  ──────────────     │
│  Open Dashboard     │
│  Preferences        │
│  Quit               │
└─────────────────────┘
```

### H.3 Mobile Constraints
- No browser extensions on iOS Safari or Android Chrome
- Possible paths: Share extension (analyze shared text), standalone app with paste input
- **Realistic timeline**: Month 6+, not MVP scope

### H.4 Phased Strategy
1. **Phase 1 (now)**: Chrome extension + local server
2. **Phase 2 (month 3)**: Desktop menu bar app (Tauri)
3. **Phase 3 (month 6)**: Mobile share extension (iOS/Android)
4. **Phase 4 (month 9)**: Cloud API for third-party integrations

**Acceptance Criteria:** Each phase is independently useful. No phase depends on a later phase.

---

## Section I. Legal, Privacy & Compliance

### I.1 Data Handling Policy
- **Local mode**: Zero data transmitted externally. All processing on `127.0.0.1`.
- **Cloud mode** (future): Text sent over HTTPS. Not stored beyond request lifetime. No training on user data.
- **Extension**: Reads DOM text only from matched sites. Does not inject into form fields. Does not modify page content (beacon is overlay only).

### I.2 Terms of Service Risk
- ChatGPT ToS: Does not prohibit browser extensions that read public page content. Extensions cannot automate actions (we don't — read-only).
- Claude ToS: Similar — read-only overlay is permissible.
- **Mitigation**: Extension clearly labeled as "analysis overlay, does not modify or automate chat."

### I.3 Consent & Transparency
- Extension description clearly states: "Reads assistant responses for fact-checking. No data leaves your device."
- First-run onboarding screen explains: what is analyzed, where analysis happens, what is stored.
- User can disable per-site.

### I.4 Regional Compliance
- GDPR: No personal data collected or transmitted in local mode. Cloud mode would need privacy policy + data processing agreement.
- CCPA: Same as GDPR.
- FERPA: Not applicable (no educational records).

### I.5 Security Checklist
- [ ] Extension passes Chrome Web Store security review
- [ ] Server binds to 127.0.0.1 only (no network exposure)
- [ ] No `eval()` or `innerHTML` with unsanitized user input in extension
- [ ] CORS restricted to localhost
- [ ] No hardcoded secrets in extension code

**Acceptance Criteria:** Extension passes Chrome Web Store review. No data leaves device in local mode (verified by network inspection).

---

## Section J. Testing & QA Plan

### J.1 Test Strategy

| Level | Framework | Scope |
|---|---|---|
| Unit | pytest | Claim decomposer, metrics, config, helpers |
| Integration | pytest + mock LLM | Full pipeline with mock LLM responses |
| E2E | Selenium + extension | Extension → beacon → panel on ChatGPT test page |
| Benchmark | Custom runner | TruthfulQA, HaluEval with metrics export |

### J.2 Extension Testing
- **Manual**: Load unpacked in Chrome, navigate to ChatGPT, verify beacon appears
- **Automated** (future): Puppeteer with extension loaded, mock chat page
- **Cross-browser**: Chrome primary, Edge secondary (same Manifest V3)

### J.3 Test Packs
- `tests/test_pipeline.py` — 11 unit tests (parsing, filtering, data models)
- `evaluation/evaluate.py --benchmark sanity` — 3 hand-crafted samples
- `smoke_test.py` — 8-point prerequisite and component check
- Future: `tests/test_extension.js` — extension-specific tests

### J.4 Regression Harness
- All tests run on every git push via GitHub Actions
- Benchmark metrics stored in `assets/evaluation/evaluation_results.json`
- Any regression > 5% in F1 blocks merge

### J.5 Load & Chaos Testing
- **Load**: Send 50 requests in 60s to overlay server, measure P95
- **Chaos**: Kill Ollama mid-request, verify fallback to spaCy works
- **Memory**: Monitor RSS while processing 20 consecutive messages, verify < 3GB

### J.6 Release Gates

| Gate | Criteria |
|---|---|
| Unit tests | 100% pass |
| Smoke test | 8/8 pass |
| Benchmark F1 | ≥ 0.70 |
| P95 latency | < 15s |
| Extension lint | 0 errors |
| Manual QA | 5 diverse queries verified |

**Acceptance Criteria:** All gates pass before any release.

---

## Section K. Roadmap & Delivery Plan

### K.1 12-Week Execution Roadmap

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Core pipeline hardened | All audit fixes, smoke test passing, FAISS index built |
| 2 | Extension MVP functional | Beacon + popup working on ChatGPT with live analysis |
| 3 | Dashboard polished | Streamlit UI with all panels, export, examples |
| 4 | Evaluation complete | TruthfulQA + HaluEval benchmarks run, metrics exported, plots generated |
| 5 | Report draft | 15-page report with architecture, results, plots |
| 6 | **Capstone submission** | Report + code + demo video submitted |
| 7 | Chrome Web Store prep | Privacy policy, screenshots, listing description |
| 8 | Beta launch | Extension published (unlisted), 10 beta testers |
| 9 | Feedback iteration | Fix top 5 issues from beta feedback |
| 10 | Landing page | Vercel site with demo, docs, install instructions |
| 11 | Public launch | Chrome Web Store public listing |
| 12 | Desktop app alpha | Tauri menu bar app prototype |

### K.2 Workstream Breakdown

| Workstream | Owner | Weeks |
|---|---|---|
| ML Pipeline & Accuracy | You | 1-4 |
| Extension & UX | You | 2-4 |
| Dashboard & Frontend | You | 3-4 |
| Evaluation & Report | You | 4-6 |
| Infra & DevOps | You | 1-2 |
| QA | Continuous | 1-12 |

### K.3 Detailed Task Board (Immediate)

| Task | Est. | Priority | Status |
|---|---|---|---|
| Fix .env + config + profiles | 0.5d | P0 | DONE |
| Fallback chain + strict mode | 0.5d | P0 | DONE |
| FAISS index build (5K articles) | 0.25d | P0 | DONE |
| Overlay server functional | 0.25d | P0 | DONE |
| Extension beacon + popup functional | 0.5d | P0 | DONE |
| Evaluation suite (fixed + live) | 0.5d | P0 | DONE |
| Smoke test + unit tests | 0.25d | P0 | DONE |
| Run TruthfulQA benchmark | 0.5d | P0 | TODO |
| Run HaluEval benchmark | 0.5d | P0 | TODO |
| Generate evaluation plots | 0.25d | P0 | TODO |
| Write project report | 2d | P0 | TODO |
| Prepare demo video | 0.5d | P0 | TODO |
| Polish Streamlit dashboard | 0.5d | P1 | TODO |
| Full Wikipedia index (200K articles) | 0.5d | P1 | TODO |
| Chrome Web Store listing | 0.5d | P2 | TODO |

### K.4 Critical Path
```
FAISS Index → Pipeline Working → Evaluation → Report → Submission
                    ↓
            Extension Working → Demo Video → Submission
```
Both paths must converge at submission. Index and extension are parallelizable.

### K.5 Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| ChatGPT DOM changes break extension | Medium | High | DOM resilience strategy + hot-updatable selectors |
| Ollama unavailable on demo machine | Low | Critical | Pre-install + verify in smoke test |
| TruthfulQA F1 below 0.70 | Medium | High | Tune thresholds, expand corpus |
| Chrome Web Store rejection | Low | Medium | Minimal permissions, clear privacy policy |
| Report deadline pressure | High | High | Start report draft in parallel with coding |

---

## Section L. Build Specification Artifacts

### L.1 API Contracts

**POST /analyze** (Overlay Server)
```
Request:
  POST http://127.0.0.1:8765/analyze
  Content-Type: application/json
  
  {
    "text": "Albert Einstein invented the telephone in 1876.",
    "top_claims": 6,
    "source": "chatgpt"
  }

Response (200):
  {
    "factuality_score": 33.3,
    "overall_confidence": 0.55,
    "total_claims": 3,
    "supported": 1,
    "contradicted": 1,
    "unverifiable": 1,
    "no_evidence": 0,
    "processing_time": 8.2,
    "summary": "1 supported, 1 contradicted, 1 unverifiable.",
    "flags": [
      {
        "claim": "Albert Einstein invented the telephone",
        "verdict": "CONTRADICTED",
        "confidence": 0.55,
        "evidence": "Alexander Graham Bell is credited with...",
        "source": "wikipedia",
        "url": "https://simple.wikipedia.org/wiki/Telephone"
      }
    ]
  }
```

**GET /health**
```
Response (200):
  {
    "ok": true,
    "service": "verifact-overlay",
    "model": "llama3.1:8b",
    "index_vectors": 24605,
    "profile": "interactive"
  }
```

### L.2 Extension → Backend Message Schema
```javascript
// Content script → Overlay server
{
  "text": string,        // assistant message text
  "source": string,      // "chatgpt" | "claude"
  "top_claims": number,  // max claims to return (default 6)
}

// Overlay server → Content script
{
  "factuality_score": number,    // 0-100
  "overall_confidence": number,  // 0-1
  "total_claims": number,
  "supported": number,
  "contradicted": number,
  "unverifiable": number,
  "no_evidence": number,
  "processing_time": number,     // seconds
  "summary": string,
  "flags": ClaimFlag[]
}
```

### L.3 Error Code Matrix

| HTTP Status | Error Code | Meaning | Client Action |
|---|---|---|---|
| 400 | `text_required` | Empty or missing text field | Show "No text to analyze" |
| 404 | `not_found` | Unknown endpoint | Ignore |
| 500 | `pipeline_error` | Internal analysis failure | Show "Analysis failed" in beacon |
| 503 | `ollama_unavailable` | Ollama not running | Show "Start Ollama" guidance |
| 504 | `timeout` | Analysis exceeded 30s | Show "Timed out" |

**Retry policy**: 1 retry after 2s for 500/503. No retry for 400/404.

### L.4 Telemetry Event Schema (Local Only)

```json
{
  "event": "analysis_complete",
  "timestamp": "2026-04-14T10:30:00Z",
  "platform": "chatgpt",
  "claims_total": 5,
  "claims_supported": 3,
  "claims_contradicted": 1,
  "latency_s": 8.2,
  "cache_hit": false,
  "provider": "ollama",
  "fallback_used": false
}
```
Logged to `verifactai.log` only. No external transmission.

**Acceptance Criteria for Section L:**
- API contracts match actual server implementation
- Extension JS matches message schema exactly
- Error codes handled in content.js

---

## Section M. Final Recommendation

### M.1 Recommended Architecture
**Local-first, extension-driven, ambient verification.**
- Chrome Extension (Manifest V3) for ChatGPT/Claude detection
- Python overlay server (127.0.0.1:8765) for analysis
- Ollama for LLM inference (free, local, M4-optimized)
- DeBERTa NLI + FAISS for verification (no API calls)
- Streamlit dashboard for deep analysis and evaluation

### M.2 Why This Is the Best Balance
- **Speed**: 7-12s latency is acceptable for ambient monitoring (user is still reading the response)
- **Accuracy**: DeBERTa achieves 92% on MultiNLI — no cloud API can match this at zero cost
- **Beauty**: Glass-morphism popup with traffic-light beacon is premium UX
- **Cost**: $0 operational cost. All local inference.
- **Privacy**: Zero data leaves the device. Strongest possible privacy story.

### M.3 Immediate Next 10 Actions

1. Run TruthfulQA benchmark: `cd verifactai && python evaluation/evaluate.py --benchmark truthfulqa-fixed --max-samples 50`
2. Run HaluEval benchmark: `python evaluation/evaluate.py --benchmark halueval --max-samples 50`
3. Generate evaluation plots from results JSON
4. Test extension end-to-end on ChatGPT (load unpacked → send a query → verify beacon works)
5. Capture 3 demo screenshots (annotated output, claim panel, factuality gauge)
6. Start writing project report (abstract + architecture + results)
7. Record 3-minute demo video showing beacon in action
8. Build full Wikipedia index (200K articles) overnight
9. Polish Streamlit dashboard (add comparison mode, improve styling)
10. Prepare Chrome Web Store listing assets (icon, screenshots, description)

### M.4 Definition of Done: MVP Launch
- [ ] Extension installs cleanly from unpacked folder
- [ ] Beacon appears on ChatGPT and Claude
- [ ] Analysis returns within 15s for any message
- [ ] Popup shows factuality score, confidence, flagged claims
- [ ] Dashboard provides deep analysis with export
- [ ] Smoke test: 8/8 pass
- [ ] Unit tests: 11/11 pass
- [ ] TruthfulQA F1 ≥ 0.70
- [ ] Project report submitted
- [ ] Demo video recorded

### M.5 Definition of Done: Production Launch
- [ ] All MVP criteria met
- [ ] Chrome Web Store published (public listing)
- [ ] Landing page live on Vercel
- [ ] Full Wikipedia + PubMed index deployed
- [ ] 20+ beta users tested
- [ ] Top 5 bugs from beta fixed
- [ ] Privacy policy published
- [ ] Desktop app alpha functional
