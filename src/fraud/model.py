"""Training helpers for the improved pipeline."""
import numpy as np
from sklearn.model_selection import train_test_split
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
