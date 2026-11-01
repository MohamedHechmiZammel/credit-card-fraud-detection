"""Smoke tests of the improved, cost-sensitive and cross-validation pipelines on synthetic data."""
import numpy as np
import pandas as pd

from fraud.cost import cost_report, pick_threshold_by_cost
from fraud.cv import cross_validate, summarize
from fraud.data import FEATURES, TARGET
from fraud.model import amount_sample_weight, make_dummy, make_logreg, make_xgb, pos_weight, stratified_split
from fraud.threshold import choose_threshold


def synthetic(n=4000, frac=0.03, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < frac).astype(int)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df.loc[y == 1, "V1"] += 3.0
    df["Amount"] = rng.gamma(2.0, 40.0, n)
    df[TARGET] = y
    return df


def test_improved_and_cost_pipeline_runs():
    X_tr, X_va, X_te, y_tr, y_va, y_te = stratified_split(synthetic())
    class_w = make_xgb(n_estimators=30, scale_pos_weight=pos_weight(y_tr)).fit(X_tr, y_tr)
    amount_w = make_xgb(n_estimators=30).fit(X_tr, y_tr, sample_weight=amount_sample_weight(y_tr, X_tr["Amount"]))
    for model in (class_w, amount_w):
        p_va, p_te = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
        thr_f1 = choose_threshold(y_va, p_va)
        thr_cost, _, _ = pick_threshold_by_cost(y_va, p_va, X_va["Amount"], false_alarm_cost=5.0)
        for thr in (0.5, thr_f1, thr_cost):
            r = cost_report(y_te, p_te, thr, X_te["Amount"], 5.0)
            assert r["cost"] >= 0 and r["saved"] == r["no_model_cost"] - r["cost"]
        # the learnable signal must beat approving everything
        assert cost_report(y_te, p_te, thr_cost, X_te["Amount"], 5.0)["saved"] > 0


def test_cross_validation_runs_and_summarizes():
    fitters = {
        "dummy": lambda X, y: make_dummy().fit(X, y),
        "logreg": lambda X, y: make_logreg().fit(X, y),
        "xgb": lambda X, y: make_xgb(n_estimators=20).fit(X, y),
    }
    folds = cross_validate(synthetic(), fitters, n_splits=3, fixed_threshold=("dummy",))
    # dummy: one rule, the others: f1 and cost
    assert len(folds) == 3 * (1 + 2 + 2)
    assert set(folds["fold"]) == {0, 1, 2}
    summary = {(r["model"], r["threshold_rule"]): r for r in summarize(folds)}
    assert summary[("dummy", "f1")]["recall_mean"] == 0.0
    xgb = summary[("xgb", "f1")]
    assert xgb["n_folds"] == 3 and xgb["recall_mean"] > 0.5 and xgb["f1_std"] >= 0
    assert xgb["pr_auc_mean"] > summary[("dummy", "f1")]["pr_auc_mean"]
