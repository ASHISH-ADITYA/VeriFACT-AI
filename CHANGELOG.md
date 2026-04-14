# Changelog

## [1.0.0] - 2026-04-14

### Added
- 5-stage verification pipeline: claim decomposition, evidence retrieval, NLI verdict, Bayesian confidence, annotated output
- Specificity gate for NLI that prevents topical-similarity false entailment
- Chrome extension (Manifest V3) with floating beacon for ChatGPT and Claude
- Overlay API server for extension communication
- Streamlit dashboard with color-coded annotations, claim cards, confidence histograms
- Multi-provider LLM client with ordered fallback chain (Ollama -> Anthropic -> OpenAI -> spaCy)
- Performance profiles: interactive (fast) and eval (thorough)
- Evaluation suite with TruthfulQA and HaluEval benchmarks
- Full Wikipedia Simple English index (369K vectors, 541 MB)
- macOS desktop launcher scripts
- Docker support
- Smoke test (8-point validation)
- Unit tests (11 tests)

### Benchmark Results
- HaluEval: 93.8% accuracy, 0.97 F1
- TruthfulQA: 0.76 hallucination recall (7.6x improvement over baseline)
