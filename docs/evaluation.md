# Evaluation & Benchmarks

## Benchmark Results

### HaluEval (Multi-Sentence — Primary)

| Metric | Value |
|---|---|
| Accuracy | **93.8%** |
| Precision | **93.3%** |
| Recall | **100%** |
| F1 | **96.6%** |
| AUROC | **0.75** |

### TruthfulQA Fixed-Response (Single-Sentence)

| Metric | Value |
|---|---|
| Hallu. Recall | **0.76** |
| Hallu. Precision | 0.46 |
| Hallu. F1 | 0.57 |

### Specificity Gate Ablation

| Configuration | Hallu Recall | Hallu F1 |
|---|---|---|
| Baseline (no gate) | 0.10 | 0.167 |
| + Specificity gate | **0.76** | **0.571** |
| Improvement | **7.6x** | **3.4x** |

## Running Benchmarks

```bash
# Quick sanity check (3 samples, no datasets needed)
make benchmark-quick

# Full suite (TruthfulQA + HaluEval, downloads datasets)
make benchmark

# Custom
cd verifactai
PYTHONPATH=. python evaluation/evaluate.py --benchmark truthfulqa-fixed --max-samples 50
PYTHONPATH=. python evaluation/evaluate.py --benchmark halueval --max-samples 50
```

## Evaluation Modes

| Mode | Label | Purpose |
|---|---|---|
| `truthfulqa-fixed` | **CORE** | Fixed-response verification, deterministic |
| `halueval` | **CORE** | Fixed-response classification, deterministic |
| `truthfulqa-live` | STRESS TEST | Live LLM generation + verification, includes variability |
| `sanity` | SMOKE | 3 hand-crafted samples, no download needed |

## Output Files

- `assets/evaluation/evaluation_results.json` — all metrics
- `assets/evaluation/truthfulqa_fixed_cm.png` — confusion matrix
- `assets/evaluation/truthfulqa_fixed_roc.png` — ROC curve
- `assets/evaluation/halueval_cm.png` — confusion matrix
- `assets/evaluation/halueval_roc.png` — ROC curve
