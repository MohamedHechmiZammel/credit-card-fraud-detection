"""Build reports/results.md from reports/reproduction_*.json and reports/improved.json."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "reports"
dt_acc = json.loads((R / "reproduction_dt_accuracy.json").read_text())
dt = json.loads((R / "reproduction_dt_roc_auc.json").read_text())
xgb = json.loads((R / "reproduction_xgb_accuracy.json").read_text())
imp = json.loads((R / "improved.json").read_text())

ORIG = {
    "dt": {"tn": 85290, "fp": 17, "fn": 27, "tp": 109, "accuracy": 0.9994850368081645},
    "xgb": {"accuracy": 0.9996371850239341},
}


def pct(x):
    return f"{x:.4f}"


def recall_precision(m):
    return m["precision"], m["recall"], m["f1"]


lines = []
lines += [
    "# Results",
    "",
    "All numbers are produced by the scripts in `scripts/`. Test sets differ between the two sections, see notes.",
    "",
    "## 1. Reproduction of the original 2024 pipeline",
    "",
    "Original recipe: StandardScaler on the whole dataset, `train_test_split(test_size=0.3, random_state=42)`, "
    "GridSearchCV with 5-fold CV (scoring noted per row). 85,443 test rows with 136 frauds (same as the deck).",
    "",
    "| Model | Source | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |",
    "|---|---|---|---|---|---|---|---|---|---|",
    f"| Decision Tree | Original deck | {ORIG['dt']['tn']} | {ORIG['dt']['fp']} | {ORIG['dt']['fn']} | {ORIG['dt']['tp']} | "
    f"{ORIG['dt']['accuracy']:.5f} | {109/(109+17):.4f} | {109/136:.4f} | {2*109/(2*109+17+27):.4f} |",
    f"| Decision Tree | Reproduced (CV scoring = accuracy) | {dt_acc['tn']} | {dt_acc['fp']} | {dt_acc['fn']} | {dt_acc['tp']} | "
    f"{dt_acc['accuracy']:.5f} | {pct(dt_acc['precision'])} | {pct(dt_acc['recall'])} | {pct(dt_acc['f1'])} |",
    f"| Decision Tree | Reproduced (CV scoring = roc_auc, as in the notebook) | {dt['tn']} | {dt['fp']} | {dt['fn']} | {dt['tp']} | "
    f"{dt['accuracy']:.5f} | {pct(dt['precision'])} | {pct(dt['recall'])} | {pct(dt['f1'])} |",
    f"| XGBoost | Original deck | not shown | not shown | not shown | not shown | {ORIG['xgb']['accuracy']:.5f} | n/a | n/a | n/a |",
    f"| XGBoost | Reproduced (CV scoring = accuracy) | {xgb['tn']} | {xgb['fp']} | {xgb['fn']} | {xgb['tp']} | {xgb['accuracy']:.5f} | "
    f"{pct(xgb['precision'])} | {pct(xgb['recall'])} | {pct(xgb['f1'])} |",
    "",
    f"Decision Tree best params (accuracy scoring): `{dt_acc['best_params']}`, full 18-combination grid. "
    f"With roc_auc scoring the search picks `{dt['best_params']}` instead.",
    f"XGBoost best params (accuracy scoring): `{xgb['best_params']}`, "
    "full 54-combination grid with 5-fold CV (run on CPU in about "
    f"{xgb['seconds'] / 60:.0f} minutes, so no shortcut was needed).",
    "",
    "The Decision Tree row matches the deck exactly when the grid search uses scikit-learn's default scoring (accuracy). "
    "The notebook text says `scoring='roc_auc'`, but the deck's numbers only come out with the default, so the deck was "
    "probably produced by an earlier version of the code. The XGBoost accuracy does not match exactly "
    f"(deck 0.99964, reproduced {xgb['accuracy']:.5f}, a difference of about 5 transactions out of 85,443). "
    "A likely cause is a different XGBoost version or tree method (the original run is not recorded). "
    "The deck shows the XGBoost confusion matrix only as an image, so only its accuracy is a hard reference.",
    "",
    "## 2. Improved pipeline",
    "",
    f"Stratified 60/20/20 split (train {imp['split']['train']:,} / validation {imp['split']['val']:,} / "
    f"test {imp['split']['test']:,}, {imp['split']['test_frauds']} frauds in test). Class weighting through "
    f"`scale_pos_weight`, early stopping on validation PR-AUC, no scaler (trees do not need one). "
    f"The threshold is the best-F1 point on the validation split. Accuracy is shown last on purpose: "
    "predicting \"legit\" for everything already scores 99.83%.",
    "",
    "| Model | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | TN | FP | FN | TP | Accuracy |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for name, m in [
    ("XGBoost, original params, no weighting", imp["baseline_original_params_thr0.5"]),
    ("XGBoost, original params, threshold from validation", imp["baseline_original_params_val_threshold"]),
    ("Improved, threshold 0.5", imp["improved_at_thr0.5"]),
    ("**Improved, validation-chosen threshold (best F1)**", imp["improved"]),
    ("Improved, high-recall mode (threshold for recall >= 0.85 on validation)", imp["improved_min_recall_0.85"]),
]:
    lines.append(
        f"| {name} | {m['threshold']:.3f} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | "
        f"{pct(m['pr_auc'])} | {pct(m['roc_auc'])} | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} | {m['accuracy']:.5f} |"
    )
p = imp["improved_params"]
lines += [
    "",
    f"Improved model: `scale_pos_weight={p['scale_pos_weight']:.1f}`, `max_depth={p['max_depth']}`, "
    f"`learning_rate={p['learning_rate']}`, {p['n_estimators_used']} trees (early stopping). "
    f"Validation PR-AUC {imp['val_pr_auc']:.4f}.",
    "",
    "## Notes",
    "",
    "- Sections 1 and 2 use different test sets (random 30% versus stratified 20%), so compare rows within a section, not across.",
    "- The original scaler was fitted on all rows before splitting. That leaks test statistics. It is kept in section 1 only to match the deck.",
    "- With only ~100 frauds in the test set, one extra miss moves recall by about one point. Treat differences of a few points as noise.",
    "- Weighting did not raise PR-AUC here (0.866 versus 0.871 for the unweighted baseline), and that gap is inside the noise. "
    "What the rebuild adds is a leak-free split, honest metrics and a threshold that was picked without looking at the test set.",
    "- The high-recall row shows the cost of chasing recall: about 300 extra false alarms for 10 more frauds caught.",
    "- ROC-AUC looks excellent for almost any model on this data. PR-AUC and recall at a given precision are the more honest numbers.",
    "",
    "## Figures",
    "",
    "![PR curve](figures/pr_curve.png)",
    "![ROC curve](figures/roc_curve.png)",
    "![Confusion matrix](figures/confusion_matrix.png)",
    "![Threshold selection](figures/threshold_selection.png)",
    "![Feature importance](figures/feature_importance.png)",
]
(R / "results.md").write_text("\n".join(lines) + "\n")
print("wrote", R / "results.md")
