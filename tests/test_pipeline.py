import numpy as np
import pandas as pd
import pytest
from visa.pipeline import prepare, train, predict, TARGET, LABELS


def test_round_trip_and_unknown_category(tmp_path):
    labels = list(LABELS)
    frame = pd.DataFrame(
        {
            "numeric": np.arange(120),
            "category": ["a", "b"] * 60,
            TARGET: [labels[0], labels[1]] * 60,
        }
    )
    data, artifact = tmp_path / "data.csv", tmp_path / "model.joblib"
    frame.to_csv(data, index=False)
    metrics = train(data, artifact)
    assert metrics["split_rows"] == {"train": 72, "validation": 24, "test": 24}
    assert 0 <= metrics["threshold"] <= 1
    sample = pd.DataFrame({"numeric": [50], "category": ["unseen"]})
    sample.to_csv(data, index=False)
    result = predict(data, artifact, tmp_path / "result.csv")
    assert len(result) == 1
    assert result["predicted_label"].iloc[0] in labels
    sample.drop(columns="numeric").to_csv(data, index=False)
    with pytest.raises(ValueError, match="Missing feature"):
        predict(data, artifact, tmp_path / "invalid.csv")


def test_rejects_invalid_target():
    with pytest.raises(ValueError, match="must contain only"):
        prepare(pd.DataFrame({"x": [1], TARGET: ["unexpected"]}), training=True)
