"""Reproduce the original 2024 team pipeline.

Original recipe (from the notebook and the deck):
  - drop rows with NaN, drop nothing else
  - StandardScaler fitted on the WHOLE dataset (Time and Amount included)
  - train_test_split(test_size=0.3, random_state=42), not stratified
    (the deck's confusion matrix sums to 85,443 = 30% of 284,807; the notebook
    text says 0.25, the deck output only fits 0.3)
  - GridSearchCV(cv=5, scoring="roc_auc")
  - Decision Tree grid: max_depth [10,20,30], min_samples_split [2,5,10], criterion [gini, entropy]
  - XGBoost grid: n_estimators [50,100,200], max_depth [3,5,7],
    learning_rate [0.01,0.1,0.2], booster [gbtree, dart]

The scaler leaks test statistics into training. It is kept here on purpose to
match the original; the improved pipeline in train_improved.py fixes it.

Usage:
  python scripts/reproduce.py dt
  python scripts/reproduce.py dt --scoring accuracy   # sklearn's default scoring, see results.md
  python scripts/reproduce.py xgb            # full 54-combo grid
  python scripts/reproduce.py xgb --params '{"n_estimators":200,...}'   # skip search
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud.data import FEATURES, TARGET, load_data  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"

GRIDS = {
    "dt": {"max_depth": [10, 20, 30], "min_samples_split": [2, 5, 10], "criterion": ["gini", "entropy"]},
    "xgb": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2], "booster": ["gbtree", "dart"]},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["dt", "xgb"])
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--params", type=json.loads, default=None, help="skip the grid search, fit these params")
    ap.add_argument("--scoring", default="roc_auc", help="GridSearchCV scoring (the notebook used roc_auc)")
    ap.add_argument("--n-jobs", type=int, default=1, help="GridSearchCV workers")
    args = ap.parse_args()

    df = load_data().dropna()
    X = StandardScaler().fit_transform(df[FEATURES])  # whole dataset, as in the original
    y = df[TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=42)

    if args.model == "dt":
        est = DecisionTreeClassifier(random_state=42)
    else:
        est = XGBClassifier(random_state=42, tree_method="hist", n_jobs=4)

    t0 = time.time()
    if args.params is None:
        gs = GridSearchCV(est, GRIDS[args.model], scoring=args.scoring, cv=5, n_jobs=args.n_jobs, verbose=1)
        gs.fit(X_tr, y_tr)
        best, params, cv_auc = gs.best_estimator_, gs.best_params_, float(gs.best_score_)
    else:
        best = est.set_params(**args.params).fit(X_tr, y_tr)
        params, cv_auc = args.params, None
    elapsed = time.time() - t0

    proba = best.predict_proba(X_te)[:, 1]
    pred = best.predict(X_te)  # argmax, as in the notebook (proba == 0.5 ties go to class 0)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    result = {
        "model": args.model, "grid_searched": args.params is None, "best_params": params,
        "scoring": args.scoring, "cv_score": cv_auc, "test_size": args.test_size, "n_test": int(len(y_te)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred)), "recall": float(recall_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)), "roc_auc": float(roc_auc_score(y_te, proba)),
        "seconds": round(elapsed, 1),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"reproduction_{args.model}_{args.scoring}.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
