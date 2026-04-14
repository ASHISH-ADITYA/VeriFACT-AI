"""
Optional risk classifier for bias/toxicity-style red flags.

This module is intentionally optional and fail-soft:
- If model load fails, the pipeline falls back to heuristic alerts.
- Keeps local-first deployment functional without paid APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.helpers import logger


@dataclass
class RiskPrediction:
    label: str
    score: float


class RiskClassifier:
    """Thin wrapper over a local HuggingFace text-classifier pipeline."""

    def __init__(self, model_name: str = "unitary/toxic-bert") -> None:
        self.model_name = model_name
        self._pipeline = None
        self._available = False
        self._init_model()

    def _init_model(self) -> None:
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                truncation=True,
            )
            self._available = True
            logger.info(f"RiskClassifier loaded: {self.model_name}")
        except Exception as exc:
            self._pipeline = None
            self._available = False
            logger.warning(f"RiskClassifier unavailable ({self.model_name}): {exc}")

    @property
    def available(self) -> bool:
        return self._available and self._pipeline is not None

    def predict(self, text: str) -> Optional[RiskPrediction]:
        if not self.available:
            return None
        if not text or len(text.strip()) < 6:
            return None

        try:
            out = self._pipeline(text[:512])
            if not out:
                return None
            top = out[0]
            return RiskPrediction(
                label=str(top.get("label", "unknown")).lower(),
                score=float(top.get("score", 0.0)),
            )
        except Exception as exc:
            logger.warning(f"RiskClassifier prediction failed: {exc}")
            return None
