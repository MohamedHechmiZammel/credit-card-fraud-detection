"""Build reports/results.md from reports/reproduction_*.json, improved.json, cv.json and cost.json."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "reports"
dt_acc = json.loads((R / "reproduction_dt_accuracy.json").read_text())
dt = json.loads((R / "reproduction_dt_roc_auc.json").read_text())
xgb = json.loads((R / "reproduction_xgb_accuracy.json").read_text())
imp = json.loads((R / "improved.json").read_text())
cv = json.loads((R / "cv.json").read_text())
cost = json.loads((R / "cost.json").read_text())

ORIG = {
    "dt": {"tn": 85290, "fp": 17, "fn": 27, "tp": 109, "accuracy": 0.9994850368081645},
    "xgb": {"accuracy": 0.9996371850239341},
}


def pct(x):
    return f"{x:.4f}"


def pm(row, key):
    return f"{row[key + '_mean']:.3f} +- {row[key + '_std']:.3f}"


def cvrow(model, rule):
    return next(r for r in cv["summary"] if r["model"] == model and r["threshold_rule"] == rule)


lines = []
lines += [
    "# Results",
    "",
    "All numbers are produced by the scripts in `scripts/`. Section 1 and section 2 use different test sets, so never compare rows across them.",
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
    "## 2. Rebuilt evaluation (leak-free split)",
    "",
    f"Stratified 60/20/20 split (train {imp['split']['train']:,} / validation {imp['split']['val']:,} / "
    f"test {imp['split']['test']:,}, {imp['split']['test_frauds']} frauds in test). The class-weighted model uses "
    f"`scale_pos_weight`, early stopping on validation PR-AUC and no scaler (trees do not need one). "
    f"The threshold is the best-F1 point on the validation split. Accuracy is shown last on purpose: "
    "predicting \"legit\" for everything already scores 99.83%.",
    "",
    "| Model | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | TN | FP | FN | TP | Accuracy |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for name, m in [
    ("Always legit (Dummy)", imp["dummy_always_legit"]),
    ("Logistic regression (scaled, balanced), threshold 0.5", imp["logreg_balanced_thr0.5"]),
    ("Logistic regression (scaled, balanced), threshold from validation", imp["logreg_balanced_val_threshold"]),
    ("XGBoost, original params, no weighting", imp["baseline_original_params_thr0.5"]),
    ("XGBoost, original params, threshold from validation", imp["baseline_original_params_val_threshold"]),
    ("XGBoost, class-weighted, threshold 0.5", imp["improved_at_thr0.5"]),
    ("**XGBoost, class-weighted, threshold from validation (best F1)**", imp["improved"]),
    ("XGBoost, class-weighted, high-recall mode (threshold for recall >= 0.85 on validation)", imp["improved_min_recall_0.85"]),
]:
    lines.append(
        f"| {name} | {m['threshold']:.3f} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | "
        f"{pct(m['pr_auc'])} | {pct(m['roc_auc'])} | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} | {m['accuracy']:.5f} |"
    )
p = imp["improved_params"]
lines += [
    "",
    f"Class-weighted model: `scale_pos_weight={p['scale_pos_weight']:.1f}`, `max_depth={p['max_depth']}`, "
    f"`learning_rate={p['learning_rate']}`, {p['n_estimators_used']} trees (early stopping). "
    f"Validation PR-AUC {imp['val_pr_auc']:.4f}.",
    "",
    "## 3. Stratified 5-fold cross-validation",
    "",
    f"Fixed parameters, no grid search. Each fold: the 4 training folds are split again, 80% fit the model and a "
    f"stratified 20% slice picks the threshold; the held-out fold is scored once. Mean +- sample standard deviation "
    f"over {cv['folds']} folds. 'f1' rows use the best-F1 threshold, 'cost' rows the minimum-cost threshold at "
    f"{cv['false_alarm_cost_eur']:g} EUR per false alarm. Cost is per held-out fold of about 57,000 transactions.",
    "",
    "| Model | Threshold rule | Precision | Recall | F1 | PR-AUC | ROC-AUC | Cost (EUR) | Saved vs no model (EUR) |",
    "|---|---|---|---|---|---|---|---|---|",
]
for r in cv["summary"]:
    lines.append(
        f"| {r['model']} | {r['threshold_rule']} | " + " | ".join(pm(r, k) for k in ("precision", "recall", "f1", "pr_auc", "roc_auc"))
        + f" | {r['cost_mean']:,.0f} +- {r['cost_std']:,.0f} | {r['saved_mean']:,.0f} +- {r['saved_std']:,.0f} |")
no_model_cv = sum(f["no_model_cost"] for f in cv["per_fold"] if f["model"] == cv["summary"][0]["model"]) / cv["folds"]
lines += [
    "",
    f"Approving everything costs {no_model_cv:,.0f} EUR per fold on average (the sum of fraud Amounts in the fold).",
    "",
    "## 4. Cost-sensitive evaluation",
    "",
    f"Same split as section 2, amounts in {cost['currency']}. Cost = Amount of every missed fraud + false-alarm cost x "
    f"false alarms{' + the same cost for each caught fraud' if cost['review_caught_fraud'] else ''}. "
    f"The false-alarm cost is an assumption: {cost['default_false_alarm_cost']:g} EUR by default, with 1 and 20 shown as sensitivity. "
    f"The test set holds {cost['test']['frauds']} frauds worth {cost['test']['fraud_amount_total']:,.0f} EUR in total "
    f"(median {cost['test']['fraud_amount_median']:.2f} EUR), so approving everything costs that much. "
    f"All thresholds are picked on validation only. 'Amount-weighted' gives each fraud row the weight "
    f"min(1 + Amount / median fraud Amount, {cost['amount_weight']['cap']:g}) and each legitimate row 1, with no `scale_pos_weight`.",
    "",
]
for fac, rows in cost["by_false_alarm_cost"].items():
    star = " (default)" if float(fac) == cost["default_false_alarm_cost"] else ""
    lines += [
        f"### False alarm = {float(fac):g} EUR{star}",
        "",
        "| Model | Threshold rule | Threshold | Test cost (EUR) | Saved vs no model (EUR) | Missed fraud (EUR) | False alarms | Missed frauds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['threshold_rule']} | {r['threshold']:.3f} | {r['cost']:,.0f} | {r['saved']:,.0f} | "
            f"{r['missed_fraud_amount']:,.0f} | {r['false_alarms']} | {r['missed_frauds']} |")
    lines.append("")
cal = imp["calibration"]
lines += [
    "![Cost curve](figures/cost_curve.png)",
    "",
    "## 5. Calibration",
    "",
    "Brier score on the test set (lower is better). The score of a model that always predicts the training fraud rate is "
    f"{cal['brier_constant_prevalence']:.5f}.",
    "",
    "| Model | Brier score | Mean score | Observed fraud rate |",
    "|---|---|---|---|",
    f"| XGBoost, original params, no weighting | {cal['brier_original_params_unweighted']:.5f} | {cal['mean_score_original_params_unweighted']:.5f} | {cal['observed_fraud_rate']:.5f} |",
    f"| XGBoost, class-weighted | {cal['brier_class_weighted']:.5f} | {cal['mean_score_class_weighted']:.5f} | {cal['observed_fraud_rate']:.5f} |",
    f"| Logistic regression (balanced) | {cal['brier_logreg_balanced']:.5f} | n/a | {cal['observed_fraud_rate']:.5f} |",
    "",
    "Class weighting trades calibration for recall, so the demo labels its output as a fraud score and not as a probability. "
    "Isotonic calibration on validation was not applied; the demo only needs the ranking and a threshold.",
    "",
    "![Reliability curve](figures/calibration.png)",
    "",
    "## Notes",
    "",
    "- Sections 1 and 2 use different test sets (random 30% versus stratified 20%), so compare rows within a section, not across.",
    "- The original scaler was fitted on all rows before splitting. That leaks test statistics into training. It is kept in section 1 only to reproduce the deck.",
    "- With only ~100 frauds in the test set, one extra miss moves recall by about one point. Section 3 measures the spread: "
    f"the best-F1 class-weighted model has an F1 standard deviation of {cvrow('XGBoost, class-weighted', 'f1')['f1_std']:.3f} "
    f"and a recall standard deviation of {cvrow('XGBoost, class-weighted', 'f1')['recall_std']:.3f} across folds.",
    f"- Class-weighting did not clearly beat the unweighted baseline: test PR-AUC {imp['improved']['pr_auc']:.3f} versus "
    f"{imp['baseline_original_params_thr0.5']['pr_auc']:.3f}, and cross-validated PR-AUC {pm(cvrow('XGBoost, class-weighted', 'f1'), 'pr_auc')} "
    f"versus {pm(cvrow('XGBoost, original params', 'f1'), 'pr_auc')}. What the rebuild adds is a leak-free split, "
    "honest metrics and a threshold that was picked without looking at the test set.",
    f"- The high-recall row shows the cost of chasing recall: {imp['improved_min_recall_0.85']['fp'] - imp['improved']['fp']} extra false alarms "
    f"for {imp['improved_min_recall_0.85']['tp'] - imp['improved']['tp']} more frauds caught.",
    f"- A minimum-cost threshold picked on validation did not reliably beat the best-F1 one. Cross-validated cost for the "
    f"class-weighted model: {cvrow('XGBoost, class-weighted', 'cost')['cost_mean']:,.0f} EUR with the cost rule against "
    f"{cvrow('XGBoost, class-weighted', 'f1')['cost_mean']:,.0f} EUR with the F1 rule, and the standard deviation across folds is "
    f"about {cvrow('XGBoost, class-weighted', 'f1')['cost_std']:,.0f} EUR. A handful of large frauds per fold decides most of the cost, "
    "so cost differences of a few hundred EUR between models or rules are not evidence.",
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
