"""Improved pipeline: stratified split, class weighting, threshold chosen on validation.

Steps
  1. stratified 60/20/20 train/validation/test split
  2. XGBoost with scale_pos_weight, small search scored by validation PR-AUC
  3. operating threshold = best F1 on the validation split (never the test set)
  4. report precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix on the test set
  5. baseline for comparison: the original best XGBoost params, no weighting, threshold 0.5,
     trained and tested on the same split
"""
import itertools
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud.data import FEATURES, TARGET, load_data  # noqa: E402
from fraud.metrics import evaluate  # noqa: E402
from fraud.model import make_xgb, pos_weight, stratified_split  # noqa: E402
from fraud.threshold import choose_threshold  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)


def main() -> None:
    df = load_data()
    X_tr, X_va, X_te, y_tr, y_va, y_te = stratified_split(df)
    ratio = pos_weight(y_tr)
    print(f"train {len(y_tr)} val {len(y_va)} test {len(y_te)}; test frauds {int(y_te.sum())}; neg/pos {ratio:.0f}")

    # Baseline: original best params (from reproduce.py xgb), no class weighting, threshold 0.5
    repro = ROOT / "reports" / "reproduction_xgb_accuracy.json"
    base_params = json.loads(repro.read_text())["best_params"] if repro.exists() else {}
    base = make_xgb(**base_params).fit(X_tr, y_tr)
    base_te = base.predict_proba(X_te)[:, 1]
    base_metrics = evaluate(y_te, base_te, 0.5)

    # Improved: search on validation PR-AUC
    grid = list(itertools.product([1.0, float(np.sqrt(ratio)), ratio], [3, 5, 7]))
    best = None
    for spw, depth in grid:
        m = make_xgb(n_estimators=400, learning_rate=0.1, max_depth=depth, scale_pos_weight=spw,
                     subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        score = average_precision_score(y_va, m.predict_proba(X_va)[:, 1])
        print(f"spw={spw:7.1f} depth={depth} val PR-AUC={score:.4f} trees={m.best_iteration + 1}")
        if best is None or score > best[0]:
            best = (score, spw, depth, m)
    val_pr_auc, spw, depth, model = best
    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    thr = choose_threshold(y_va, p_va)
    imp_metrics = evaluate(y_te, p_te, thr)
    imp_at_half = evaluate(y_te, p_te, 0.5)
    base_thr = choose_threshold(y_va, base.predict_proba(X_va)[:, 1])
    base_tuned = evaluate(y_te, base_te, base_thr)
    thr_r85 = choose_threshold(y_va, p_va, min_recall=0.85)
    imp_r85 = evaluate(y_te, p_te, thr_r85)
    print("baseline@0.5", base_metrics)
    print("improved@val-thr", imp_metrics)

    result = {
        "split": {"train": len(y_tr), "val": len(y_va), "test": len(y_te),
                  "test_frauds": int(y_te.sum()), "seed": 42, "stratified": True},
        "baseline_original_params_thr0.5": base_metrics,
        "baseline_original_params_val_threshold": base_tuned,
        "baseline_params": base_params,
        "improved": imp_metrics,
        "improved_at_thr0.5": imp_at_half,
        "improved_min_recall_0.85": imp_r85,
        "improved_params": {"scale_pos_weight": spw, "max_depth": depth,
                            "n_estimators_used": int(model.best_iteration + 1), "learning_rate": 0.1},
        "val_pr_auc": val_pr_auc,
    }
    (ROOT / "reports" / "improved.json").write_text(json.dumps(result, indent=2))
    joblib.dump(model, ROOT / "models" / "xgb_fraud.joblib")
    (ROOT / "models" / "threshold.json").write_text(json.dumps({"threshold": thr, "features": FEATURES}, indent=2))

    # Demo sample: 40 frauds + 160 legit transactions drawn from the TEST split
    te = X_te.copy()
    te[TARGET] = y_te
    sample = pd.concat([te[te[TARGET] == 1].sample(40, random_state=42),
                        te[te[TARGET] == 0].sample(160, random_state=42)])
    sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)
    sample.to_csv(ROOT / "data" / "sample.csv", index=False, float_format="%.6g")

    # ---- figures
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p, c in [("Original params, no weighting", base_te, "#888888"), ("Improved", p_te, "#1b6ca8")]:
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax.plot(rc, pr, color=c, label=f"{name} (PR-AUC {average_precision_score(y_te, p):.3f})")
    ax.scatter([imp_metrics["recall"]], [imp_metrics["precision"]], color="#d1495b", zorder=5,
               label=f"Chosen threshold {thr:.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test set)", ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "pr_curve.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p, c in [("Original params, no weighting", base_te, "#888888"), ("Improved", p_te, "#1b6ca8")]:
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, color=c, label=name)
    ax.plot([0, 1], [0, 1], ":", color="#bbbbbb")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "roc_curve.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = np.array([[imp_metrics["tn"], imp_metrics["fp"]], [imp_metrics["fn"], imp_metrics["tp"]]])
    ax.imshow(np.log10(cm + 1), cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > 1000 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["legit", "fraud"], yticklabels=["legit", "fraud"],
           xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix, threshold {thr:.2f}")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png"); plt.close(fig)

    pr, rc, th = precision_recall_curve(y_va, p_va)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(th, pr[:-1], label="precision"); ax.plot(th, rc[:-1], label="recall"); ax.plot(th, f1, label="F1")
    ax.axvline(thr, color="#d1495b", ls="--", label=f"chosen {thr:.2f}")
    ax.set(xlabel="Threshold", title="Threshold selection (validation set)", ylim=(0, 1.02)); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "threshold_selection.png"); plt.close(fig)

    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: t[1])[-12:]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.barh([a for a, _ in imp], [b for _, b in imp], color="#1b6ca8")
    ax.set(title="Top 12 feature importances (gain-based)", xlabel="importance")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png"); plt.close(fig)
    print("saved model, figures, sample.csv")


if __name__ == "__main__":
    main()
