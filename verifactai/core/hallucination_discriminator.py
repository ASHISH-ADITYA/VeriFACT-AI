"""
Lightweight free hallucination discriminator (RelD-inspired path).

Provides a trainable text classifier using TF-IDF + LogisticRegression.
This is intentionally simple and local-first: no paid APIs, no external service.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class DiscriminatorPrediction:
    label: str
    score: float


class HallucinationDiscriminator:
    """Binary hallucination discriminator: 1=hallucinated, 0=grounded."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.pipeline: Pipeline | None = None

    def train(self, texts: list[str], labels: list[int]) -> None:
        if not texts or not labels or len(texts) != len(labels):
            raise ValueError("texts and labels must be non-empty and aligned")

        clf = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.98,
                    ),
                ),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=600,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        clf.fit(texts, labels)
        self.pipeline = clf

    def predict(self, text: str) -> DiscriminatorPrediction:
        if self.pipeline is None:
            raise RuntimeError("Discriminator model not loaded/trained")
        probs = self.pipeline.predict_proba([text])[0]
        hall_score = float(probs[1])
        label = "hallucinated" if hall_score >= 0.5 else "grounded"
        return DiscriminatorPrediction(label=label, score=hall_score)

    def save(self, path: str | None = None) -> str:
        if self.pipeline is None:
            raise RuntimeError("Cannot save untrained discriminator")
        target = Path(path or self.model_path or "assets/models/hallucination_discriminator.joblib")
        target.parent.mkdir(parents=True, exist_ok=True)
        dump(self.pipeline, target)
        return str(target)

    def load(self, path: str | None = None) -> bool:
        target = Path(path or self.model_path or "assets/models/hallucination_discriminator.joblib")
        if not target.exists():
            return False
        self.pipeline = load(target)
        return True
