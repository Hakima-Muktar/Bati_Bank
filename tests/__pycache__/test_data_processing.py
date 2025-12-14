import pandas as pd
import pytest
from src.data_processing import prepare_features


def test_prepare_features_columns():
    """
    Test that target column is removed from features
    and returned separately.
    """
    df = pd.DataFrame({
        "Amount": [100, 200],
        "Value": [1, 2],
        "is_high_risk": [0, 1]
    })

    X, y = prepare_features(df)

    assert "is_high_risk" not in X.columns
    assert len(y) == 2


def test_prepare_features_target_values():
    """
    Test that target values remain binary and unchanged.
    """
    df = pd.DataFrame({
        "Amount": [50, 80],
        "is_high_risk": [1, 0]
    })

    _, y = prepare_features(df)

    assert set(y.unique()) == {0, 1}


def test_prepare_features_missing_target():
    """
    Test that function raises error when target is missing.
    """
    df = pd.DataFrame({
        "Amount": [10, 20],
        "Value": [1, 2]
    })

    with pytest.raises(ValueError):
        prepare_features(df)
