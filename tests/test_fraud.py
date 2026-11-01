import numpy as np
import pandas as pd
import pytest

from fraud.data import FEATURES, TARGET, validate_schema
from fraud.metrics import evaluate
from fraud.model import make_xgb, pos_weight, stratified_split
from fraud.threshold import apply_threshold, choose_threshold


def synthetic(n=2000, frac=0.02, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < frac).astype(int)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df.loc[y == 1, "V1"] += 3.0  # make fraud learnable
    df[TARGET] = y
    return df


# ---- data schema
def test_schema_ok():
    validate_schema(synthetic())


def test_schema_missing_column():
    with pytest.raises(ValueError):
        validate_schema(synthetic().drop(columns=["V7"]))


def test_schema_bad_target():
    df = synthetic()
    df.loc[0, TARGET] = 2
    with pytest.raises(ValueError):
        validate_schema(df)


def test_bundled_sample_schema():
    from pathlib import Path
    sample = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "sample.csv")
    validate_schema(sample)
    assert 150 <= len(sample) <= 300
    assert sample[TARGET].sum() > 0


# ---- split and model
def test_stratified_split_keeps_fraud_rate():
    df = synthetic(n=5000, frac=0.02)
    *_, y_tr, y_va, y_te = stratified_split(df)
    rate = df[TARGET].mean()
    for part in (y_tr, y_va, y_te):
        assert abs(part.mean() - rate) < 0.005
    assert len(y_tr) + len(y_va) + len(y_te) == len(df)


def test_pos_weight():
    assert pos_weight([0, 0, 0, 1]) == 3.0


def test_proba_in_unit_interval():
    df = synthetic()
    X_tr, _, X_te, y_tr, _, _ = stratified_split(df)
    p = make_xgb(n_estimators=20, scale_pos_weight=pos_weight(y_tr)).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    assert p.min() >= 0.0 and p.max() <= 1.0


# ---- threshold logic
def test_apply_threshold_is_inclusive():
    out = apply_threshold([0.1, 0.5, 0.9], 0.5)
    assert out.tolist() == [0, 1, 1]


def test_choose_threshold_separable():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.05, 0.1, 0.2, 0.3, 0.7, 0.9])
    t = choose_threshold(y, p)
    assert 0.3 < t <= 0.7
    assert apply_threshold(p, t).tolist() == y.tolist()


def test_choose_threshold_min_recall():
    y = np.array([0, 0, 1, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.45, 0.6, 0.8, 0.9])
    t = choose_threshold(y, p, min_recall=1.0)
    assert apply_threshold(p, t)[y == 1].all()


def test_choose_threshold_unreachable_recall():
    with pytest.raises(ValueError):
        choose_threshold([0, 1], [0.9, 0.1], min_recall=1.1)


def test_evaluate_counts_add_up():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.2, 0.7, 0.8, 0.3, 0.1, 0.9])
    m = evaluate(y, p, 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
