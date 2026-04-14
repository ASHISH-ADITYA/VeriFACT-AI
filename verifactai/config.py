"""
VeriFactAI Configuration Module.

Centralizes all tunable parameters, model paths, thresholds, and weights.
Loads secrets from .env; everything else has sensible production defaults.

Hardware target: MacBook Air M4, 16 GB RAM.
Default provider: Ollama (free, local).
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Resolve project root relative to this file
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Performance profiles — tuned for M4 16 GB
# ---------------------------------------------------------------------------


class Profile(str, Enum):
    """interactive = responsive demo; eval = full-quality benchmark."""

    INTERACTIVE = "interactive"
    EVAL = "eval"


class PerformanceProfile(BaseModel):
    max_tokens: int
    retrieval_top_k: int
    embedding_batch_size: int


PROFILES: dict[Profile, PerformanceProfile] = {
    Profile.INTERACTIVE: PerformanceProfile(
        max_tokens=1024,
        retrieval_top_k=3,
        embedding_batch_size=128,
    ),
    Profile.EVAL: PerformanceProfile(
        max_tokens=2048,
        retrieval_top_k=5,
        embedding_batch_size=256,
    ),
}


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """LLM provider settings."""

    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5:3b-instruct"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))
    max_tokens: int = 1024  # overridden by active profile
    retry_attempts: int = 3
    retry_base_delay: float = 1.0


class EmbeddingConfig(BaseModel):
    """Sentence-transformer embedding settings."""

    model_name: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    dimension: int = 384
    batch_size: int = 128  # overridden by active profile
    normalize: bool = True


class RetrievalConfig(BaseModel):
    """FAISS vector retrieval settings."""

    index_path: str = Field(
        default_factory=lambda: os.getenv(
            "FAISS_INDEX_PATH",
            str(_PROJECT_ROOT / "data" / "index" / "knowledge.index"),
        )
    )
    metadata_path: str = Field(
        default_factory=lambda: os.getenv(
            "METADATA_PATH",
            str(_PROJECT_ROOT / "data" / "metadata" / "chunks.jsonl"),
        )
    )
    top_k: int = 3  # overridden by active profile
    relevance_threshold: float = 0.30
    chunk_size: int = 200
    chunk_overlap: int = 50


class NLIConfig(BaseModel):
    """Natural Language Inference model settings."""

    model_name: str = Field(
        default_factory=lambda: os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
    )
    entailment_threshold: float = 0.65
    contradiction_threshold: float = 0.75
    max_input_length: int = 512


class ConfidenceConfig(BaseModel):
    """Bayesian confidence scoring weights."""

    w_nli: float = 0.32
    w_retrieval: float = 0.22
    w_source: float = 0.12
    w_cross_ref: float = 0.16
    w_uncertainty: float = 0.18

    verified_threshold: float = 0.75
    uncertain_lower: float = 0.40
    hallucination_threshold: float = (
        0.50  # for evaluation binary classification (tuned down from 0.60)
    )

    uncertainty_entropy_weight: float = 0.65
    uncertainty_disagreement_weight: float = 0.35

    source_reliability: dict[str, float] = Field(
        default_factory=lambda: {
            "wikipedia": 1.0,
            "pubmed": 1.0,
            "openstax": 0.95,
            "unknown": 0.50,
        }
    )


class SelfCheckConfig(BaseModel):
    """Self-consistency sampling and semantic-entropy controls."""

    enabled: bool = Field(default_factory=lambda: os.getenv("SELFCHECK_ENABLED", "1") == "1")
    samples: int = Field(default_factory=lambda: int(os.getenv("SELFCHECK_SAMPLES", "5")))
    min_valid_samples: int = Field(
        default_factory=lambda: int(os.getenv("SELFCHECK_MIN_VALID", "3"))
    )
    max_evidence: int = Field(default_factory=lambda: int(os.getenv("SELFCHECK_MAX_EVIDENCE", "2")))
    temperature_start: float = Field(
        default_factory=lambda: float(os.getenv("SELFCHECK_TEMP_START", "0.1"))
    )
    temperature_step: float = Field(
        default_factory=lambda: float(os.getenv("SELFCHECK_TEMP_STEP", "0.15"))
    )
    uncertainty_entropy_weight: float = Field(
        default_factory=lambda: float(os.getenv("SELFCHECK_ENTROPY_WEIGHT", "0.6"))
    )
    uncertainty_disagreement_weight: float = Field(
        default_factory=lambda: float(os.getenv("SELFCHECK_DISAGREEMENT_WEIGHT", "0.4"))
    )
    confidence_blend_weight: float = Field(
        default_factory=lambda: float(os.getenv("SELFCHECK_CONF_BLEND", "0.2"))
    )


class ReflexionConfig(BaseModel):
    """Critique-and-revise loop controls for contradiction corrections."""

    enabled: bool = Field(default_factory=lambda: os.getenv("REFLEXION_ENABLED", "1") == "1")
    max_rounds: int = Field(default_factory=lambda: int(os.getenv("REFLEXION_MAX_ROUNDS", "1")))


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class Config(BaseModel):
    """Top-level configuration container."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    nli: NLIConfig = Field(default_factory=NLIConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    selfcheck: SelfCheckConfig = Field(default_factory=SelfCheckConfig)
    reflexion: ReflexionConfig = Field(default_factory=ReflexionConfig)

    project_root: Path = _PROJECT_ROOT
    log_level: str = "INFO"
    active_profile: Profile = Profile.INTERACTIVE

    def apply_profile(self, profile: Profile | None = None) -> None:
        """Apply a performance profile, overriding relevant sub-config values."""
        p = PROFILES[profile or self.active_profile]
        self.llm.max_tokens = p.max_tokens
        self.retrieval.top_k = p.retrieval_top_k
        self.embedding.batch_size = p.embedding_batch_size


# Singleton
cfg = Config()
cfg.apply_profile()
