# VeriFact AI Architecture Audit

Date: 2026-04-14
Scope: Core verification architecture, extension integration, deployment readiness

## Executive Verdict

The current system is NOT Ollama-only.
It already uses a hybrid verification architecture:

1. Claim extraction: LLM-assisted with spaCy fallback
2. Retrieval: FAISS + sentence-transformer dense retrieval
3. Verification: DeBERTa NLI contradiction/entailment scoring
4. Confidence fusion: weighted multi-signal scoring
5. Correction path: contradiction-aware correction generation with evidence attribution
6. Extension signaling: live beacon, inline highlighting, alert payloads

## Confirmed Implementation Mapping

- Orchestration: `core/pipeline.py`
- Claim decomposition: `core/claim_decomposer.py`
- Retrieval: `core/evidence_retriever.py`
- NLI verification + confidence: `core/verdict_engine.py`
- Annotation + correction: `core/annotator.py`
- Local API server: `overlay_server.py`
- Extension client: `integrations/web_beacon_extension/content.js`

## Critical Fix Applied During Audit

Problem observed in extension behavior:
- "Analyzer offline: TypeError: Failed to fetch"

Root cause:
- CORS preflight did not allow custom request header `X-VeriFact-Token`

Fix:
- Updated `overlay_server.py` response headers to include:
  - `Access-Control-Allow-Headers: Content-Type, X-VeriFact-Token`

## What Is Strong Right Now

1. Verification does not depend on one generative model.
2. Core truth signal comes from retrieval + NLI + confidence fusion.
3. Pipeline has local-first design suitable for privacy-sensitive use.
4. Desktop launcher and extension integration are operational.
5. CI exists in `.github/workflows/ci.yml`.

## Gaps To Close for Research-Grade Reliability

1. Uncertainty module missing:
- No semantic entropy or self-consistency variance score yet.

2. Source reliability is static:
- Uses prior weights, but no dynamic trust calibration.

3. Bias module is heuristic:
- Current alerting is lexical cue-based, not classifier-backed.

4. Multi-hop verification not implemented:
- Single-hop retrieval/verification dominates.

5. Online deployment split not finalized:
- Docker exists, but frontend-on-Vercel architecture is not yet scaffolded.

## Deployment Readiness Snapshot

Current:
- GitHub remote configured and CI present.
- Docker artifacts exist at repo root.

Not yet complete:
- Dedicated Vercel frontend app (Next.js/Vite) with API integration.
- Hosted backend production profile and secrets/runbook.
- Store-distribution package for mobile app stores.

## Recommended Next Sequence

1. Add uncertainty/self-consistency scoring into confidence fusion.
2. Add classifier-backed bias detection.
3. Add architecture benchmark report with module-level ablations.
4. Finalize deployment split:
   - Backend: Dockerized Python API (Render/Fly/Railway/local)
   - Frontend: Vercel-hosted UI
5. Prepare store strategy only after mobile wrapper exists and backend SLA is stable.

## Monitoring Protocol (Use Per Release)

1. Run CI and local smoke tests.
2. Test extension flow on ChatGPT, Claude, Gemini.
3. Verify API health and analyze response contracts.
4. Check cold-start latency and first-token analysis delay.
5. Re-run benchmark subset and compare against previous release.
