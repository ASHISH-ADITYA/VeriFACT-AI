"""Tests for evaluation metrics — precision, recall, ECE, latency."""

from __future__ import annotations

from evaluation.metrics import (
    binary_hallucination_metrics,
    expected_calibration_error,
    latency_percentiles,
    retrieval_recall_at_k,
)


class TestBinaryMetrics:

    def test_perfect_classification(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        m = binary_hallucination_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_all_wrong(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        m = binary_hallucination_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0
        assert m["recall"] == 0.0

    def test_with_scores_produces_auroc(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.8, 0.9]
        m = binary_hallucination_metrics(y_true, y_pred, y_scores)
        assert "auroc" in m
        assert m["auroc"] == 1.0

    def test_confusion_matrix_shape(self):
        m = binary_hallucination_metrics([0, 1, 0, 1], [0, 0, 1, 1])
        cm = m["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_macro_f1_present(self):
        m = binary_hallucination_metrics([0, 1], [0, 1])
        assert "macro_f1" in m


class TestCalibrationError:

    def test_perfect_calibration(self):
        # If 80% confident claims are correct 80% of the time → ECE ≈ 0
        confidences = [0.8] * 10
        accuracies = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  # 80% correct
        ece = expected_calibration_error(confidences, accuracies)
        assert ece < 0.05

    def test_empty_input(self):
        assert expected_calibration_error([], []) == 0.0

    def test_ece_bounded(self):
        ece = expected_calibration_error([0.9, 0.9], [0, 0])
        assert 0 <= ece <= 1


class TestLatencyPercentiles:

    def test_basic(self):
        result = latency_percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["p50"] == 3.0
        assert result["mean"] == 3.0
        assert result["p95"] > 4.0

    def test_empty(self):
        result = latency_percentiles([])
        assert result["p50"] == 0.0

    def test_single_value(self):
        result = latency_percentiles([7.5])
        assert result["p50"] == 7.5
        assert result["mean"] == 7.5


class TestRetrievalRecall:

    def test_all_found(self):
        assert retrieval_recall_at_k([True, True, True]) == 1.0

    def test_none_found(self):
        assert retrieval_recall_at_k([False, False]) == 0.0

    def test_partial(self):
        assert retrieval_recall_at_k([True, False, True, False]) == 0.5

    def test_empty(self):
        assert retrieval_recall_at_k([]) == 0.0
