# VeriFactAI Hardening Plan — Codex Audit Remediation (Approved)

## Context
Codex audit identified 3 High, 3 Medium, 1 Low findings. Core architecture is strong (positive finding). All fixes target local-first reliability on MacBook Air M4 (16GB), free-only defaults, and research-grade evaluation integrity.

---

## MANDATORY POLICY (Principal Additions)

### P1: Fallback Order by Mode
- **Query mode** (interactive / demo): Ollama → Anthropic → OpenAI → spaCy-only decomposition. Fallback is automatic and logged.
- **Verification mode** (benchmark / evaluation): NO fallback. Uses ONLY the configured primary provider. If primary fails, the sample is skipped and logged. This ensures benchmark results are deterministic and not contaminated by cross-provider behavior differences.

### P2: Hardware Model Defaults
- **Default model**: `qwen2.5:3b-instruct` — fits comfortably in 16GB, fast inference on M4 Metal
- **Upgrade profile** (optional, user-activated): `llama3.1:8b` — for users with 32GB+ or when running eval overnight
- Config key: `LLM_MODEL` in `.env`. No code change needed to switch — just edit `.env` and restart.

### P3: Report-Claim Policy
- **Core benchmark claims** (reported in abstract, results tables, comparison charts): MUST come from **fixed-response evaluation** — verification of pre-existing known-true and known-false text. No live LLM generation involved.
- **Stress-test claims** (reported in appendix / supplementary): May use live `verify_query()` mode. Must be clearly labeled as "Live Generation Stress Test" with a note that results include generation variability.
- Evaluation script enforces this by separating `evaluate_truthfulqa_fixed()` from `evaluate_truthfulqa_live()`.

---

## Fix 1: Runtime Determinism (HIGH)
**Problem**: No active `.env`, provider selection can silently fall back to shell/global values.

**Changes**:
- `verifactai/.env` — create from `.env.example` with local-safe defaults
- `verifactai/config.py` — default model to `qwen2.5:3b-instruct`, add `PerformanceProfile` with `interactive` and `eval` modes
- Reduce default `max_tokens` to 1024 for interactive, keep 2048 for eval

## Fix 2: Mac-Safe Performance Profile (HIGH)
**Problem**: llama3.1:8b + max_tokens 2048 can cause UI lag and thermal throttling on 16GB M4.

**Changes**:
- `verifactai/config.py` — add `PerformanceProfile` dataclass:
  - `interactive`: max_tokens=1024, top_k=3, embedding_batch_size=128
  - `eval`: max_tokens=2048, top_k=5, embedding_batch_size=256
- `verifactai/core/pipeline.py` — accept profile parameter

## Fix 3: Provider Fallback Chain (HIGH)
**Problem**: Single provider path — one outage blocks everything.

**Changes**:
- `verifactai/core/llm_client.py` — implement ordered fallback:
  - **Query mode**: ollama → anthropic (if key set) → openai (if key set) → return `None` (triggers spaCy-only decomposition)
  - **Eval mode**: primary provider only, no fallback, skip + log on failure
- Log every fallback transition: cause, old provider, new provider
- `LLMClient.generate()` gets a `strict: bool = False` parameter. `strict=True` disables fallback (used by evaluation).

## Fix 4: Evaluation Integrity (MEDIUM)
**Problem**: TruthfulQA mixes generation variability with verification scoring. HaluEval threshold hard-coded at 60.

**Changes**:
- `verifactai/evaluation/evaluate.py`:
  - Add `evaluate_truthfulqa_fixed()` — verifies pre-existing known-false answer strings from dataset (no LLM generation). This is the **core** benchmark.
  - Rename current function to `evaluate_truthfulqa_live()` — clearly labeled as stress test. Kept but not used for headline metrics.
  - Replace hard-coded `60` threshold with `cfg.confidence.hallucination_threshold` (default 0.60, tunable)
  - Add extended metrics: macro F1, per-class P/R/F1, retrieval recall@k, latency p50/p95
- `verifactai/evaluation/metrics.py` — add `latency_percentiles()` and `retrieval_recall_at_k()`
- `verifactai/config.py` — add `hallucination_threshold: float = 0.60` to `ConfidenceConfig`

## Fix 5: UX & Operational Polish (MEDIUM)
**Problem**: Error messages assume paid API. No startup diagnostics.

**Changes**:
- `verifactai/app.py` — update error guidance for local-first:
  - "Ensure `ollama serve` is running" instead of "check API key"
  - "Run `ollama pull qwen2.5:3b-instruct`" if model missing
- `verifactai/app.py` — add startup diagnostics in sidebar:
  - Ollama reachable: yes/no
  - Model available: yes/no
  - spaCy model: yes/no
  - FAISS index: yes/no (with file size)
- `verifactai/smoke_test.py` — one-command local smoke test:
  - Checks all prerequisites
  - Runs spaCy fallback decomposition (no LLM needed)
  - Runs retrieval on a test query (if index exists)
  - Reports pass/fail with clear messages

## Fix 6: Repo Hygiene (LOW)
**Problem**: `verifactai.log` and cache artifacts not in `.gitignore`.

**Changes**:
- `.gitignore` — add: `*.log`, `*.pyc`, `__pycache__/`, `.pytest_cache/`, `assets/evaluation/`, `.env`

---

## Files Modified
1. `verifactai/.env` — active config with safe local defaults
2. `verifactai/config.py` — performance profiles, lighter model, configurable threshold
3. `verifactai/core/llm_client.py` — fallback chain with strict mode + logging
4. `verifactai/core/pipeline.py` — profile-aware pipeline
5. `verifactai/evaluation/evaluate.py` — fixed vs live evaluation, extended metrics
6. `verifactai/evaluation/metrics.py` — latency + retrieval recall functions
7. `verifactai/app.py` — local-first errors, startup diagnostics
8. `verifactai/smoke_test.py` — new one-command test
9. `.gitignore` — complete artifact coverage

## Verification Commands
```bash
python smoke_test.py                              # prerequisite check + basic pipeline
pytest tests/ -v                                  # all unit tests pass
streamlit run app.py                              # dashboard with diagnostics
python evaluation/evaluate.py --benchmark sanity  # metrics output clean
```

## Acceptance Criteria
1. End-to-end works with zero paid API keys
2. Query mode survives provider failure via fallback chain
3. Eval mode is deterministic (no fallback, no generation variability in core metrics)
4. Demo responsive on MacBook Air M4 16GB
5. Core benchmark claims from fixed-response evaluation only
6. Live generation results clearly labeled as stress test
