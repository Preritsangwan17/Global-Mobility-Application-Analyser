import pandas as pd
import pytest
from visa.pipeline import prepare, TARGET, LABELS, DROP


def test_rejects_repeated_entity_before_splitting():
    frame = pd.DataFrame(
        {DROP[0]: ["same", "same"], "measurement": [1, 2], TARGET: list(LABELS)}
    )
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(frame, training=True)


def test_prediction_keeps_repeated_rows():
    frame = pd.DataFrame({DROP[0]: ["same", "same"], "measurement": [1, 1]})
    X, _ = prepare(frame)
    assert len(X) == 2 and DROP[0] not in X
