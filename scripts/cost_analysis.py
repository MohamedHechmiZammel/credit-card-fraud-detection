"""Cost-sensitive evaluation on the stratified 60/20/20 split used by train_improved.py.

Compares three XGBoost models (original params, class-weighted, amount-weighted) under three
threshold rules (0.5, best F1 on validation, minimum cost on validation) and reports the
money lost on the test set against approving everything. Thresholds are never picked on test.

The false-alarm cost is an assumption (default 5 EUR per flagged legitimate transaction).
Sensitivity is shown for several values. See src/fraud/cost.py for the cost model.

  python scripts/cost_analysis.py [--false-alarm-cost 5] [--sensitivity 1 5 20] [--weight-cap 20]
Writes reports/cost.json and reports/figures/cost_curve.png.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud.cost import cost_report, pick_threshold_by_cost  # noqa: E402
from fraud.data import load_data  # noqa: E402
from fraud.model import (  # noqa: E402
    amount_sample_weight, class_weighted_params, load_xgb_params, make_xgb, stratified_split,
)
from fraud.threshold import choose_threshold  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--false-alarm-cost", type=float, default=5.0, help="EUR per false alarm (assumption)")
    ap.add_argument("--sensitivity", type=float, nargs="+", default=[1.0, 5.0, 20.0])
    ap.add_argument("--weight-cap", type=float, default=20.0, help="cap on the amount-based fraud weight")
    ap.add_argument("--review-caught-fraud", action="store_true",
                    help="also charge one false-alarm cost for every caught fraud")
    args = ap.parse_args()
    review = args.review_caught_fraud

    reports = ROOT / "reports"
    improved = json.loads((reports / "improved.json").read_text())
    base_params = load_xgb_params(reports / "reproduction_xgb_accuracy.json")
    cw_params = class_weighted_params(improved)

    X_tr, X_va, X_te, y_tr, y_va, y_te = stratified_split(load_data())
    a_va, a_te = X_va["Amount"].to_numpy(), X_te["Amount"].to_numpy()

    base = make_xgb(**base_params).fit(X_tr, y_tr)
    class_w = joblib.load(ROOT / "models" / "xgb_fraud.joblib")  # the model the demo ships
    amount_w = make_xgb(**{**cw_params, "scale_pos_weight": 1.0}).fit(
        X_tr, y_tr, sample_weight=amount_sample_weight(y_tr, X_tr["Amount"], cap=args.weight_cap))
    models = {"XGBoost, original params": base, "XGBoost, class-weighted": class_w,
              "XGBoost, amount-weighted": amount_w}
    scores = {n: (m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]) for n, m in models.items()}

    fraud_te = a_te[y_te.to_numpy() == 1]
    out = {
        "currency": "EUR", "review_caught_fraud": review, "default_false_alarm_cost": args.false_alarm_cost,
        "amount_weight": {"rule": "fraud rows: min(1 + Amount / median fraud Amount in train, cap); legit rows: 1",
                          "cap": args.weight_cap, "scale_pos_weight": 1.0},
        "test": {"rows": len(y_te), "frauds": int(y_te.sum()), "fraud_amount_total": float(fraud_te.sum()),
                 "fraud_amount_median": float(np.median(fraud_te))},
        "by_false_alarm_cost": {},
    }
    curves = {}
    for fac in sorted(set(args.sensitivity) | {args.false_alarm_cost}):
        rows = []
        for name, (p_va, p_te) in scores.items():
            thr_cost, ths, costs = pick_threshold_by_cost(y_va, p_va, a_va, fac, review)
            if fac == args.false_alarm_cost:
                curves[name] = (ths, costs)
            for rule, thr in [("0.5", 0.5), ("best F1 on validation", choose_threshold(y_va, p_va)),
                              ("min cost on validation", thr_cost)]:
                rows.append({"model": name, "threshold_rule": rule,
                             **cost_report(y_te, p_te, thr, a_te, fac, review)})
        out["by_false_alarm_cost"][str(fac)] = rows
    (reports / "cost.json").write_text(json.dumps(out, indent=2))

    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for (name, (ths, costs)), color in zip(curves.items(), ["#888888", "#1b6ca8", "#2a9d8f"]):
        keep = ths < 1.0
        ax.plot(ths[keep][::-1], costs[keep][::-1], color=color, label=name)
    ax.set(xlabel="Threshold", ylabel="Validation cost (EUR)",
           title=f"Cost against threshold, {args.false_alarm_cost:g} EUR per false alarm", yscale="log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reports / "figures" / "cost_curve.png")

    for fac, rows in out["by_false_alarm_cost"].items():
        print(f"\nfalse alarm = {fac} EUR, no-model cost {rows[0]['no_model_cost']:.0f} EUR")
        for r in rows:
            print(f"  {r['model']:28s} {r['threshold_rule']:24s} thr {r['threshold']:.3f} cost {r['cost']:9.0f} "
                  f"saved {r['saved']:9.0f} FP {r['false_alarms']:4d} FN {r['missed_frauds']:3d}")


if __name__ == "__main__":
    main()
