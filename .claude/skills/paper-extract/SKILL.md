---
name: paper-extract
description: Extract NLI models, hallucination detection methods, and implementation blueprints from uploaded research papers in /architecture. Auto-load when working on hallucination detection, NLI, SelfCheckGPT, semantic entropy, or DeBERTa.
allowed-tools: Read, Glob, WebFetch
---

You are an AI research engineer. For every paper in /architecture/:

1. Extract: model names, architectures, dataset benchmarks, reported latency figures.
2. Map each method to one of: Claim Decomposition / Evidence Retrieval / NLI Verdict / Self-Consistency / Semantic Entropy / Constitutional/Reflexion.
3. Flag methods that can run fully locally (no paid API) and those compatible with MPS/CPU.
4. Produce a METHODS_MANIFEST.json with: method_name, source_paper, local_compatible, latency_class (fast/medium/slow), implementation_complexity.
5. Prioritise methods that are (a) local, (b) fast, (c) combinable for signal fusion.
