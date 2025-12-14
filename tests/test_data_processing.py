import pandas as pd
from src.data_processing import prepare_features

def test_prepare_features_columns():
    df = pd.DataFrame({
        "Amount": [100, 200],
        "Value": [1, 2],
        "is_high_risk": [0, 1]
    })

    X, y = prepare_features(df)

    assert "is_high_risk" not in X.columns
    assert len(y) == 2


def test_prepare_features_target_values():
    df = pd.DataFrame({
        "Amount": [50, 80],
        "is_high_risk": [1, 0]
    })

    _, y = prepare_features(df)
    assert set(y.unique()) == {0, 1}
