"""Stratified k-fold cross-validation with the threshold picked on an inner validation slice.

For every outer fold:
  1. the 4 training folds are split again, stratified: 80% fit the model, 20% pick thresholds
  2. thresholds (best F1, and minimum cost) are chosen on that inner 20% only
  3. the model is scored once on the held-out outer fold

The outer fold never touches model fitting or threshold choice. Out-of-fold predictions were
the alternative, but they mix thresholds picked on differently fitted models, so the inner
slice is simpler to reason about.
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .cost import cost_report, pick_threshold_by_cost
from .data import FEATURES, TARGET
from .metrics import evaluate
from .threshold import choose_threshold

METRICS = ["precision", "recall", "f1", "pr_auc", "roc_auc"]


def cross_validate(df: pd.DataFrame, fitters: dict, n_splits: int = 5, seed: int = 42,
                   false_alarm_cost: float = 5.0, inner_val_size: float = 0.2,
                   review_caught_fraud: bool = False, fixed_threshold: tuple = ("dummy",)) -> pd.DataFrame:
    """Return one row per (fold, model, threshold rule).

    `fitters` maps a model name to fit(X, y) -> estimator with predict_proba.
    Models listed in `fixed_threshold` always use 0.5 (the Dummy has nothing to tune).
    """
    X, y = df[FEATURES], df[TARGET]
    rows = []
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        X_out, y_out = X.iloc[tr], y.iloc[tr]
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_out, y_out, test_size=inner_val_size, stratify=y_out, random_state=seed)
        X_te, y_te = X.iloc[te], y.iloc[te]
        for name, fit in fitters.items():
            model = fit(X_fit, y_fit)
            p_val = model.predict_proba(X_val)[:, 1]
            p_te = model.predict_proba(X_te)[:, 1]
            rules = {"f1": 0.5 if name in fixed_threshold else choose_threshold(y_val, p_val)}
            if name not in fixed_threshold:
                rules["cost"] = pick_threshold_by_cost(
                    y_val, p_val, X_val["Amount"], false_alarm_cost, review_caught_fraud)[0]
            for rule, thr in rules.items():
                m = evaluate(y_te, p_te, thr)
                c = cost_report(y_te, p_te, thr, X_te["Amount"], false_alarm_cost, review_caught_fraud)
                rows.append({"fold": fold, "model": name, "threshold_rule": rule, **m,
                             "cost": c["cost"], "saved": c["saved"], "no_model_cost": c["no_model_cost"]})
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame) -> list[dict]:
    """Mean and sample standard deviation (ddof=1) across folds for each model and rule."""
    out = []
    cols = METRICS + ["fp", "fn", "cost", "saved"]
    for (model, rule), g in folds.groupby(["model", "threshold_rule"], sort=False):
        row = {"model": model, "threshold_rule": rule, "n_folds": int(len(g))}
        for c in cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else float("nan")
        out.append(row)
    return out

