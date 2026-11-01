"""Stratified 5-fold cross-validation for the baselines and the chosen XGBoost configs.

No grid search. XGBoost parameters come from reports/improved.json (class-weighted model) and
reports/reproduction_xgb_accuracy.json (original params), both written by earlier steps.
Thresholds are picked on an inner 20% slice of each training fold, see src/fraud/cv.py.
Writes reports/cv.json.

  python scripts/cross_validate.py [--folds 5] [--false-alarm-cost 5]
"""
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud.cv import cross_validate, summarize  # noqa: E402
from fraud.data import load_data  # noqa: E402
from fraud.model import (  # noqa: E402
    amount_sample_weight, class_weighted_params, load_xgb_params, make_dummy, make_logreg, make_xgb,
)


def fit_plain(make, X, y):
    return make().fit(X, y)


def fit_xgb(params, weights, X, y):
    sw = amount_sample_weight(y, X["Amount"]) if weights == "amount" else None
    return make_xgb(**params).fit(X, y, sample_weight=sw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--false-alarm-cost", type=float, default=5.0, help="EUR per false alarm (assumption)")
    ap.add_argument("--review-caught-fraud", action="store_true")
    args = ap.parse_args()

    improved = json.loads((ROOT / "reports" / "improved.json").read_text())
    base_params = load_xgb_params(ROOT / "reports" / "reproduction_xgb_accuracy.json")
    cw_params = class_weighted_params(improved)
    fitters = {
        "Always legit (Dummy)": partial(fit_plain, make_dummy),
        "Logistic regression (balanced)": partial(fit_plain, make_logreg),
        "XGBoost, original params": partial(fit_xgb, base_params, "none"),
        "XGBoost, class-weighted": partial(fit_xgb, cw_params, "none"),
        "XGBoost, amount-weighted": partial(fit_xgb, {**cw_params, "scale_pos_weight": 1.0}, "amount"),
    }
    t0 = time.time()
    folds = cross_validate(load_data(), fitters, n_splits=args.folds, false_alarm_cost=args.false_alarm_cost,
                           review_caught_fraud=args.review_caught_fraud, fixed_threshold=("Always legit (Dummy)",))
    result = {
        "folds": args.folds, "seed": 42, "false_alarm_cost_eur": args.false_alarm_cost,
        "review_caught_fraud": args.review_caught_fraud,
        "threshold_selection": "inner stratified 20% of each training fold; Dummy fixed at 0.5",
        "std": "sample standard deviation across folds (ddof=1)",
        "xgb_class_weighted_params": cw_params,
        "xgb_original_params": base_params,
        "summary": summarize(folds),
        "per_fold": folds.to_dict(orient="records"),
        "seconds": round(time.time() - t0, 1),
    }
    (ROOT / "reports" / "cv.json").write_text(json.dumps(result, indent=2))
    for r in result["summary"]:
        print(f"{r['model']:34s} {r['threshold_rule']:5s} F1 {r['f1_mean']:.3f}+-{r['f1_std']:.3f} "
              f"recall {r['recall_mean']:.3f}+-{r['recall_std']:.3f} PR-AUC {r['pr_auc_mean']:.3f}+-{r['pr_auc_std']:.3f} "
              f"cost {r['cost_mean']:.0f}")
    print(f"done in {result['seconds']}s, wrote reports/cv.json")


if __name__ == "__main__":
    main()
