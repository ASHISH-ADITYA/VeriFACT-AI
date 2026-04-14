"""Tests for configuration system — profiles, defaults, validation."""

from __future__ import annotations

from config import PROFILES, Config, Profile


class TestConfig:
    """Config loading and default values."""

    def test_config_loads(self):
        cfg = Config()
        assert cfg.llm.provider in ("ollama", "anthropic", "openai")

    def test_default_profile_is_interactive(self):
        cfg = Config()
        assert cfg.active_profile == Profile.INTERACTIVE

    def test_profiles_exist(self):
        assert Profile.INTERACTIVE in PROFILES
        assert Profile.EVAL in PROFILES

    def test_interactive_profile_values(self):
        p = PROFILES[Profile.INTERACTIVE]
        assert p.max_tokens <= 1024
        assert p.retrieval_top_k <= 5
        assert p.embedding_batch_size <= 256

    def test_eval_profile_values(self):
        p = PROFILES[Profile.EVAL]
        assert p.max_tokens >= 1024
        assert p.retrieval_top_k >= 3

    def test_apply_profile_changes_values(self):
        cfg = Config()
        cfg.apply_profile(Profile.EVAL)
        assert cfg.llm.max_tokens == PROFILES[Profile.EVAL].max_tokens
        assert cfg.retrieval.top_k == PROFILES[Profile.EVAL].retrieval_top_k

    def test_confidence_weights_sum_to_one(self):
        cfg = Config()
        w = cfg.confidence
        total = w.w_nli + w.w_retrieval + w.w_source + w.w_cross_ref
        # w_uncertainty is a 5th signal added by the uncertainty module;
        # include it if present
        if hasattr(w, "w_uncertainty"):
            total += w.w_uncertainty
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"

    def test_source_reliability_known_sources(self):
        cfg = Config()
        sr = cfg.confidence.source_reliability
        assert "wikipedia" in sr
        assert "pubmed" in sr
        assert sr["wikipedia"] == 1.0

    def test_hallucination_threshold_in_range(self):
        cfg = Config()
        assert 0.0 < cfg.confidence.hallucination_threshold < 1.0

    def test_nli_thresholds_ordered(self):
        cfg = Config()
        # Entailment threshold should be lower than contradiction
        # (easier to support than to contradict)
        assert cfg.nli.entailment_threshold <= cfg.nli.contradiction_threshold
