"""Training helpers for the improved pipeline."""
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data import FEATURES, TARGET


def stratified_split(df, seed: int = 42):
    """60/20/20 train/validation/test, stratified on Class."""
    X, y = df[FEATURES], df[TARGET]
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def pos_weight(y) -> float:
    """negatives / positives, the usual scale_pos_weight."""
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def make_xgb(**params) -> XGBClassifier:
    base = dict(tree_method="hist", n_jobs=4, random_state=42, eval_metric="aucpr")
    base.update(params)
    return XGBClassifier(**base)


def amount_sample_weight(y, amounts, cap: float = 20.0, reference: float | None = None) -> np.ndarray:
    """Per-row training weights: legit rows 1, fraud rows min(1 + Amount / reference, cap).

    `reference` is the median Amount of the frauds in `y`, unless given. A 500 EUR fraud
    therefore counts more than a 5 EUR one, which pushes the model to catch the expensive ones.
    """
    y, amounts = np.asarray(y), np.asarray(amounts, dtype=float)
    fraud = y == 1
    if reference is None:
        reference = float(np.median(amounts[fraud])) if fraud.any() else 1.0
    reference = max(reference, 1e-9)
    return np.where(fraud, np.minimum(1.0 + amounts / reference, cap), 1.0)


def make_dummy() -> DummyClassifier:
    """Always predicts legitimate."""
    return DummyClassifier(strategy="constant", constant=0)


def make_logreg() -> Pipeline:
    """Scaler inside the pipeline, so it is fitted on training rows only."""
    return Pipeline([("scale", StandardScaler()),
                     ("logreg", LogisticRegression(class_weight="balanced", max_iter=1000))])


def load_xgb_params(path: Path | str) -> dict:
    """Best params of the original grid search. Fails loudly if the file is missing or empty."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/reproduce.py xgb` first.")
    params = json.loads(path.read_text()).get("best_params")
    if not params:
        raise ValueError(f"{path} has no best_params")
    return params


def class_weighted_params(improved: dict) -> dict:
    """Fixed XGBoost params of the chosen class-weighted model, from reports/improved.json."""
    ip = improved["improved_params"]
    return dict(learning_rate=ip["learning_rate"], max_depth=ip["max_depth"], n_estimators=ip["n_estimators_used"],
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=ip["scale_pos_weight"])
