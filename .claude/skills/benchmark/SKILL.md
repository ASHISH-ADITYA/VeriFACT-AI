---
name: benchmark
description: Run latency and accuracy benchmarks on the hallucination detection pipeline. Auto-load when measuring pipeline speed, accuracy, or comparing old vs new pipeline.
allowed-tools: Bash, Read, Write
---

Benchmark protocol:
1. Latency test: run 20 synthetic chatbot responses (short/medium/long) through full pipeline. Record p50, p95, p99 ms.
2. Accuracy test: run TruthfulQA adversarial sample set if available, else use /tests/hallucination_samples/.
3. Compare: new pipeline vs old pipeline on both axes.
4. Acceptable thresholds: p95 latency < 800ms for medium responses, accuracy F1 > 0.82.
5. Output: BENCHMARK_REPORT.json + console summary table.
