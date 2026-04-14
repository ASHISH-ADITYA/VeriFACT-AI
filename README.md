# VeriFACT AI

**Verification & Truth Layer for AI**

Real-time hallucination detection for LLM outputs. Runs as a browser extension on ChatGPT and Claude, verifying claims against trusted sources with per-claim evidence and confidence scores.

---

## What It Does

VeriFACT AI sits between you and your AI assistant. It automatically:

1. **Decomposes** responses into atomic factual claims
2. **Retrieves** evidence from Wikipedia (369K indexed passages)
3. **Verifies** each claim using DeBERTa-v3 NLI with a specificity gate
4. **Scores** confidence using Bayesian fusion of 4 signals
5. **Annotates** output with color-coded verdicts and source citations

All processing runs locally. No data leaves your machine.

## Benchmark Results

| Benchmark | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| HaluEval (multi-sentence) | **93.8%** | **93.3%** | **100%** | **96.6%** |
| TruthfulQA (single-sentence) | 44.1% | 45.8% | 75.9% | 57.1% |

The large gap reflects benchmark format: HaluEval has multi-claim answers (real-world use case), while TruthfulQA has single-sentence answers that produce binary scores.

## Architecture

```
Browser Extension (ChatGPT/Claude)
        |
        v
Overlay Server (localhost:8765)
        |
        v
VeriFactPipeline
  |-- Claim Decomposer (Ollama / spaCy fallback)
  |-- Evidence Retriever (FAISS + sentence-transformers)
  |-- Verdict Engine (DeBERTa NLI + specificity gate)
  |-- Confidence Scorer (Bayesian fusion)
  |-- Annotator (HTML/JSON output)
        |
        v
Streamlit Dashboard (localhost:8501)
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Chrome browser (for extension)

### Setup

```bash
# Clone
git clone https://github.com/ASHISH-ADITYA/VeriFACT-AI.git
cd VeriFACT-AI

# Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r verifactai/requirements.txt
python -m spacy download en_core_web_sm

# Ollama model
ollama pull llama3.1:8b

# Environment config
cp verifactai/.env.example verifactai/.env
# Edit .env if needed (defaults work for local-only mode)

# Build knowledge index (first time only, ~5 min for dev, ~30 min for full)
cd verifactai
python data/build_index.py --wiki-only --max-articles 5000  # dev build
# python data/build_index.py --wiki-only                    # full build (200K articles)

# Verify everything works
python smoke_test.py
```

### Run

```bash
# Dashboard
cd verifactai
streamlit run app.py

# Overlay server (for browser extension)
python overlay_server.py
```

### Browser Extension

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select `verifactai/integrations/web_beacon_extension/`
5. Navigate to ChatGPT or Claude — the VF beacon appears

## Docker

```bash
docker build -t verifact-ai .
docker run -p 8501:8501 -p 8765:8765 verifact-ai
```

Note: Docker mode uses spaCy fallback for claim decomposition (no Ollama inside container). For full LLM-powered decomposition, run Ollama on the host and set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in your `.env`.

## Project Structure

```
verifactai/
  app.py                    # Streamlit dashboard
  overlay_server.py         # Extension API server
  smoke_test.py             # One-command validation
  config.py                 # Configuration + performance profiles
  core/
    pipeline.py             # End-to-end orchestrator
    claim_decomposer.py     # Atomic claim extraction
    evidence_retriever.py   # FAISS dense retrieval
    verdict_engine.py       # NLI + specificity gate
    annotator.py            # HTML/JSON output
    llm_client.py           # Multi-provider LLM with fallback
  evaluation/
    evaluate.py             # Benchmark runner
    metrics.py              # P/R/F1/ECE/latency
    plots.py                # Publication-quality plots
  integrations/
    web_beacon_extension/   # Chrome Manifest V3 extension
  data/
    build_index.py          # Corpus indexer
  desktop/                  # macOS desktop launcher
  utils/
    helpers.py              # Logging, retry, timing
    runtime_safety.py       # Thread safety for M4
```

## Key Technical Contribution

**Specificity Gate**: Standard NLI models conflate topical similarity with factual entailment. Our specificity gate requires both high NLI entailment AND high retrieval similarity for a "supported" verdict. This improved hallucination recall from 0.10 to 0.76 (7.6x) on TruthfulQA.

## Tech Stack

| Component | Technology |
|---|---|
| LLM inference | Ollama (local, free) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector search | FAISS |
| NLI model | cross-encoder/nli-deberta-v3-base |
| Dashboard | Streamlit |
| Extension | Chrome Manifest V3 |
| Hardware acceleration | Apple MPS (Metal) |

## Configuration

All config via `verifactai/.env`:

```
LLM_PROVIDER=ollama          # ollama | anthropic | openai
LLM_MODEL=llama3.1:8b        # any Ollama model
OLLAMA_BASE_URL=http://localhost:11434
```

Performance profiles in `config.py`:
- `interactive`: max_tokens=1024, top_k=3 (responsive demo)
- `eval`: max_tokens=2048, top_k=5 (full benchmark quality)

## Known Limitations

- English only
- Static corpus (no real-time web search)
- Single-sentence claims produce binary scores (limits TruthfulQA performance)
- Corpus coverage gaps for niche topics
- Extension DOM selectors may break if ChatGPT/Claude update their UI

## License

MIT

## Author

Aditya Ashish — Final Year CSE Capstone Project, 2026
