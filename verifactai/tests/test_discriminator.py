from __future__ import annotations

from core.hallucination_discriminator import HallucinationDiscriminator


def test_discriminator_train_predict_roundtrip(tmp_path):
    texts = [
        "The Eiffel Tower is in Paris.",
        "Water boils at 100C at sea level.",
        "The moon is made entirely of cheese.",
        "Einstein invented the internet in 1700.",
    ]
    labels = [0, 0, 1, 1]

    model = HallucinationDiscriminator()
    model.train(texts, labels)

    pred = model.predict("The moon is made of cheese.")
    assert 0.0 <= pred.score <= 1.0
    assert pred.label in {"hallucinated", "grounded"}

    model_path = tmp_path / "reld.joblib"
    model.save(str(model_path))

    loaded = HallucinationDiscriminator(str(model_path))
    assert loaded.load() is True
    pred2 = loaded.predict("The Eiffel Tower is in Paris.")
    assert 0.0 <= pred2.score <= 1.0
