# Paper-to-Implementation Traceability Matrix

This matrix maps architecture papers in `/architecture` to concrete implementation status in VeriFact AI.

Legend:
- Implemented: method exists in runtime path.
- Partial: method is approximated or scoped-down.
- Planned: not implemented yet.

## Core Matrix

| Paper file | Method / claim extracted | Implementation status | Evidence in code |
|---|---|---|---|
| architecture/1.md | Hallucination taxonomy and benchmark-driven evaluation | Partial | Evaluation harness + report docs, but no full taxonomy classifier stack |
| architecture/2.md | Uncertainty-aware reliability framing | Partial | Uncertainty-aware confidence added in NLI engine |
| architecture/3.md | Domain-specific hallucination profiling (legal style) | Partial | Domain JSONL benchmark entrypoint added with legal starter dataset |
| architecture/4.md | RAG limits and groundedness distinction | Implemented | Retrieval + NLI + evidence-grounded verdict pipeline |
| architecture/5.md | Hybrid detection strategy across modules | Implemented | Claim decomposition + retrieval + NLI + confidence fusion |
| architecture/6.md | Learned discriminator for hallucination detection | Partial | Local TF-IDF + LogisticRegression discriminator module with save/load + overlay hook |
| architecture/7.md | Real-world interaction benchmark behavior | Planned | AuthenHallu-style dataset integration not present |
| architecture/8.md | TruthfulQA as truthfulness benchmark | Implemented | `evaluation/evaluate.py` TruthfulQA modes |
| architecture/9.md | RAG system design patterns | Implemented | FAISS retrieval pipeline in core |
| architecture/10.md | RAG model variants / stronger retrieval-generation coupling | Partial | Retrieval present; no end-to-end RAG-Token-style training |
| architecture/11.md | Reflexion self-reflection loop | Partial | Critique-revise loop added for contradiction corrections; no full actor-evaluator loop |
| architecture/12.md | Constitutional critique/revision safety loop | Partial | Constitutional critique-revise pass added for contradicted-claim corrections |
| architecture/AAA.md | SelfCheckGPT self-consistency checks | Implemented | Multi-sample sampled judgement loop with consistency distribution + uncertainty blending |
| architecture/s41586-024-07421-0.md | Semantic entropy uncertainty signal | Partial | Label entropy + rationale-cluster entropy integrated in SelfCheck pipeline |

## Newly Implemented in this upgrade

1. SelfCheck multi-sample consistency loop
- Added repeated sampled claim judgements with majority consistency, distribution tracking, and blend-back into claim confidence.

2. Semantic entropy utility for sampled judgements
- Added normalized entropy and disagreement estimators for claim-level uncertainty from sampled outputs.

3. Reflexion-style correction refinement
- Added a critique-and-revise loop for contradicted-claim corrections to reduce unsupported wording.

4. Constitutional critique layer
- Added a constitutional critique-and-revise pass to enforce no-fabrication and uncertainty-aware wording.

5. Uncertainty-aware confidence fusion (research-inspired)
- Existing entropy/disagreement uncertainty proxy remains active in the NLI verdict engine.

6. Optional classifier-assisted risk signaling
- Existing optional local risk classifier remains integrated with fail-soft behavior.

7. Discriminator + domain benchmark path
- Added free local discriminator training/inference module and domain benchmark runner.

## Evidence Files

- `core/verdict_engine.py`
- `config.py`
- `core/pipeline.py`
- `core/selfcheck.py`
- `core/semantic_entropy.py`
- `core/annotator.py`
- `overlay_server.py`
- `core/risk_classifier.py`
- `core/hallucination_discriminator.py`
- `integrations/web_beacon_extension/content.js`
- `evaluation/domain_benchmark.py`
- `evaluation/data/legal_domain.jsonl`

## Gap Closure Plan (to target 8-9/10 MVP)

1. Upgrade semantic entropy from label-level sampling to meaning-cluster entropy over generated explanations.
2. Expand discriminator from baseline TF-IDF to stronger encoder-based model with benchmark reporting.
3. Add larger real-world chat and legal domain datasets for robust significance testing.
