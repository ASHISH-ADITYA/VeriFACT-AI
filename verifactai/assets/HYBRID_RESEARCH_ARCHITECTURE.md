# VeriFact AI Hybrid Research Architecture (HRA-v1)

This document defines the verification architecture implemented in this repository and maps each stage to published research patterns that are commonly used in factuality and hallucination detection systems.

## 1) Architecture Overview

VeriFact AI uses a hybrid pipeline with six stages:

1. Claim Decomposition
- Convert long responses into atomic, checkable claims.
- Implementation: claim decomposition module in the pipeline.

2. Evidence Retrieval (RAG-style retrieval stage)
- Retrieve top-k passages from a trusted corpus (Wikipedia/PubMed/OpenStax).
- Implementation: FAISS + sentence-transformer retrieval.

3. Entailment-based Verification (NLI stage)
- For each claim/evidence pair, compute entailment/neutral/contradiction probabilities.
- Implementation: DeBERTa-family NLI cross-encoder verdict engine.

4. Confidence Fusion and Scoring
- Combine NLI confidence, retrieval relevance, source reliability, and cross-reference agreement.
- Implementation: weighted confidence module in config-driven scoring.

5. Correction + Evidence Attribution
- For contradicted claims, generate candidate correction text and expose evidence/source links.
- Implementation: annotator correction and JSON/HTML report generation.

6. Risk Signaling Layer
- Convert low-confidence/contradicted/unverifiable claims into user-facing hallucination and red-flag alerts.
- Add lexical bias-cue alerts for potentially biased phrasing as a separate warning channel.
- Implementation: overlay server alert builder and browser beacon chips.

## 2) Why this is research-backed

This architecture combines retrieval-augmented verification and NLI-style evidence checking, which are core patterns from factuality and fact-verification research.

- FEVER-style evidence-supported claim verification
- Retrieval-augmented generation/verification pipelines
- NLI-based factual consistency judgment
- Confidence calibration and uncertainty-aware outputs
- Self-check style reliability signaling for generated content

## 3) Core Paper Anchors (for your future research paper)

Use these as baseline anchors and verify final citation details before submission:

1. FEVER: a large-scale dataset and benchmark for fact extraction and verification (NAACL 2018).
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (NeurIPS 2020).
3. DeBERTa: Decoding-enhanced BERT with disentangled attention (ICLR 2021).
4. TruthfulQA: Measuring how models mimic human falsehoods (ACL 2022).
5. SelfCheckGPT: Zero-resource black-box hallucination detection for generated text (EMNLP 2023).

## 4) Current implementation mapping

- Claim decomposition: core pipeline stage 1.
- Retrieval: vector index + metadata retrieval stage.
- NLI verification: verdict engine stage.
- Confidence fusion: config confidence weights.
- Attribution: claim-level evidence + source URL.
- Risk signaling: overlay API alerts + browser beacon chips.

## 5) Limitations and next research upgrades

1. Bias detection is currently heuristic cue-based and should be upgraded to classifier-backed bias detection.
2. Add calibration curves and threshold optimization against held-out benchmark sets.
3. Add multi-hop evidence chains and contradiction rationale extraction.
4. Add provenance trust scoring beyond source-type priors.

## 6) Thesis/Paper framing suggestion

Name the method as:

- HRA-v1: Hybrid Retrieval-Augmented Entailment Architecture for Real-time Hallucination Signaling.

This gives you a concrete architecture identity for your paper, while staying faithful to implemented code paths.
