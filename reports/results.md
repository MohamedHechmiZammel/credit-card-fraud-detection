# Results

All numbers are produced by the scripts in `scripts/`. Section 1 and section 2 use different test sets, so never compare rows across them.

## 1. Reproduction of the original 2024 pipeline

Original recipe: StandardScaler on the whole dataset, `train_test_split(test_size=0.3, random_state=42)`, GridSearchCV with 5-fold CV (scoring noted per row). 85,443 test rows with 136 frauds (same as the deck).

| Model | Source | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| Decision Tree | Original deck | 85290 | 17 | 27 | 109 | 0.99949 | 0.8651 | 0.8015 | 0.8321 |
| Decision Tree | Reproduced (CV scoring = accuracy) | 85290 | 17 | 27 | 109 | 0.99949 | 0.8651 | 0.8015 | 0.8321 |
| Decision Tree | Reproduced (CV scoring = roc_auc, as in the notebook) | 85287 | 20 | 30 | 106 | 0.99941 | 0.8413 | 0.7794 | 0.8092 |
| XGBoost | Original deck | not shown | not shown | not shown | not shown | 0.99964 | n/a | n/a | n/a |
| XGBoost | Reproduced (CV scoring = accuracy) | 85300 | 7 | 29 | 107 | 0.99958 | 0.9386 | 0.7868 | 0.8560 |

Decision Tree best params (accuracy scoring): `{'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 10}`, full 18-combination grid. With roc_auc scoring the search picks `{'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 10}` instead.
XGBoost best params (accuracy scoring): `{'booster': 'gbtree', 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}`, full 54-combination grid with 5-fold CV (run on CPU in about 21 minutes, so no shortcut was needed).

The Decision Tree row matches the deck exactly when the grid search uses scikit-learn's default scoring (accuracy). The notebook text says `scoring='roc_auc'`, but the deck's numbers only come out with the default, so the deck was probably produced by an earlier version of the code. The XGBoost accuracy does not match exactly (deck 0.99964, reproduced 0.99958, a difference of about 5 transactions out of 85,443). A likely cause is a different XGBoost version or tree method (the original run is not recorded). The deck shows the XGBoost confusion matrix only as an image, so only its accuracy is a hard reference.

## 2. Rebuilt evaluation (leak-free split)

Stratified 60/20/20 split (train 170,883 / validation 56,962 / test 56,962, 98 frauds in test). The class-weighted model uses `scale_pos_weight`, early stopping on validation PR-AUC and no scaler (trees do not need one). The threshold is the best-F1 point on the validation split. Accuracy is shown last on purpose: predicting "legit" for everything already scores 99.83%.

| Model | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | TN | FP | FN | TP | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Always legit (Dummy) | 0.500 | 0.0000 | 0.0000 | 0.0000 | 0.0017 | 0.5000 | 56864 | 0 | 98 | 0 | 0.99828 |
| Logistic regression (scaled, balanced), threshold 0.5 | 0.500 | 0.0616 | 0.9184 | 0.1155 | 0.7179 | 0.9726 | 55494 | 1370 | 8 | 90 | 0.97581 |
| Logistic regression (scaled, balanced), threshold from validation | 1.000 | 0.8000 | 0.8163 | 0.8081 | 0.7179 | 0.9726 | 56844 | 20 | 18 | 80 | 0.99933 |
| XGBoost, original params, no weighting | 0.500 | 0.9737 | 0.7551 | 0.8506 | 0.8712 | 0.9745 | 56862 | 2 | 24 | 74 | 0.99954 |
| XGBoost, original params, threshold from validation | 0.394 | 0.9500 | 0.7755 | 0.8539 | 0.8712 | 0.9745 | 56860 | 4 | 22 | 76 | 0.99954 |
| XGBoost, class-weighted, threshold 0.5 | 0.500 | 0.8723 | 0.8367 | 0.8542 | 0.8657 | 0.9779 | 56852 | 12 | 16 | 82 | 0.99951 |
| **XGBoost, class-weighted, threshold from validation (best F1)** | 0.852 | 0.9286 | 0.7959 | 0.8571 | 0.8657 | 0.9779 | 56858 | 6 | 20 | 78 | 0.99954 |
| XGBoost, class-weighted, high-recall mode (threshold for recall >= 0.85 on validation) | 0.010 | 0.2268 | 0.8980 | 0.3621 | 0.8657 | 0.9779 | 56564 | 300 | 10 | 88 | 0.99456 |

Class-weighted model: `scale_pos_weight=24.0`, `max_depth=3`, `learning_rate=0.1`, 317 trees (early stopping). Validation PR-AUC 0.8301.

## 3. Stratified 5-fold cross-validation

Fixed parameters, no grid search. Each fold: the 4 training folds are split again, 80% fit the model and a stratified 20% slice picks the threshold; the held-out fold is scored once. Mean +- sample standard deviation over 5 folds. 'f1' rows use the best-F1 threshold, 'cost' rows the minimum-cost threshold at 5 EUR per false alarm. Cost is per held-out fold of about 57,000 transactions.

| Model | Threshold rule | Precision | Recall | F1 | PR-AUC | ROC-AUC | Cost (EUR) | Saved vs no model (EUR) |
|---|---|---|---|---|---|---|---|---|
| Always legit (Dummy) | f1 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.002 +- 0.000 | 0.500 +- 0.000 | 12,026 +- 3,986 | 0 +- 0 |
| Logistic regression (balanced) | f1 | 0.834 +- 0.027 | 0.754 +- 0.033 | 0.792 +- 0.023 | 0.724 +- 0.034 | 0.975 +- 0.009 | 3,501 +- 1,535 | 8,524 +- 2,676 |
| Logistic regression (balanced) | cost | 0.597 +- 0.304 | 0.813 +- 0.053 | 0.639 +- 0.234 | 0.724 +- 0.034 | 0.975 +- 0.009 | 3,656 +- 2,074 | 8,370 +- 2,959 |
| XGBoost, original params | f1 | 0.951 +- 0.023 | 0.789 +- 0.030 | 0.862 +- 0.020 | 0.850 +- 0.025 | 0.979 +- 0.012 | 3,664 +- 1,227 | 8,362 +- 2,885 |
| XGBoost, original params | cost | 0.760 +- 0.313 | 0.831 +- 0.047 | 0.752 +- 0.234 | 0.850 +- 0.025 | 0.979 +- 0.012 | 3,417 +- 1,773 | 8,609 +- 2,766 |
| XGBoost, class-weighted | f1 | 0.906 +- 0.023 | 0.815 +- 0.045 | 0.857 +- 0.024 | 0.857 +- 0.026 | 0.981 +- 0.009 | 3,095 +- 1,489 | 8,930 +- 2,834 |
| XGBoost, class-weighted | cost | 0.748 +- 0.242 | 0.825 +- 0.060 | 0.760 +- 0.142 | 0.857 +- 0.026 | 0.981 +- 0.009 | 3,319 +- 1,608 | 8,707 +- 2,815 |
| XGBoost, amount-weighted | f1 | 0.912 +- 0.031 | 0.797 +- 0.025 | 0.850 +- 0.022 | 0.843 +- 0.038 | 0.982 +- 0.009 | 3,042 +- 1,420 | 8,984 +- 2,968 |
| XGBoost, amount-weighted | cost | 0.675 +- 0.311 | 0.831 +- 0.046 | 0.699 +- 0.235 | 0.843 +- 0.038 | 0.982 +- 0.009 | 3,174 +- 2,134 | 8,851 +- 3,062 |

Approving everything costs 12,026 EUR per fold on average (the sum of fraud Amounts in the fold).

## 4. Cost-sensitive evaluation

Same split as section 2, amounts in EUR. Cost = Amount of every missed fraud + false-alarm cost x false alarms. The false-alarm cost is an assumption: 5 EUR by default, with 1 and 20 shown as sensitivity. The test set holds 98 frauds worth 10,645 EUR in total (median 10.34 EUR), so approving everything costs that much. All thresholds are picked on validation only. 'Amount-weighted' gives each fraud row the weight min(1 + Amount / median fraud Amount, 20) and each legitimate row 1, with no `scale_pos_weight`.

### False alarm = 1 EUR

| Model | Threshold rule | Threshold | Test cost (EUR) | Saved vs no model (EUR) | Missed fraud (EUR) | False alarms | Missed frauds |
|---|---|---|---|---|---|---|---|
| XGBoost, original params | 0.5 | 0.500 | 2,996 | 7,649 | 2,994 | 2 | 24 |
| XGBoost, original params | best F1 on validation | 0.394 | 2,791 | 7,854 | 2,787 | 4 | 22 |
| XGBoost, original params | min cost on validation | 0.001 | 1,612 | 9,033 | 1,294 | 318 | 9 |
| XGBoost, class-weighted | 0.5 | 0.500 | 1,945 | 8,700 | 1,933 | 12 | 16 |
| XGBoost, class-weighted | best F1 on validation | 0.852 | 2,399 | 8,246 | 2,393 | 6 | 20 |
| XGBoost, class-weighted | min cost on validation | 0.003 | 2,097 | 8,548 | 1,294 | 803 | 9 |
| XGBoost, amount-weighted | 0.5 | 0.500 | 1,943 | 8,702 | 1,935 | 8 | 18 |
| XGBoost, amount-weighted | best F1 on validation | 0.231 | 1,848 | 8,797 | 1,833 | 15 | 15 |
| XGBoost, amount-weighted | min cost on validation | 0.002 | 2,466 | 8,179 | 1,823 | 643 | 10 |

### False alarm = 5 EUR (default)

| Model | Threshold rule | Threshold | Test cost (EUR) | Saved vs no model (EUR) | Missed fraud (EUR) | False alarms | Missed frauds |
|---|---|---|---|---|---|---|---|
| XGBoost, original params | 0.5 | 0.500 | 3,004 | 7,641 | 2,994 | 2 | 24 |
| XGBoost, original params | best F1 on validation | 0.394 | 2,807 | 7,838 | 2,787 | 4 | 22 |
| XGBoost, original params | min cost on validation | 0.781 | 3,005 | 7,640 | 2,995 | 2 | 25 |
| XGBoost, class-weighted | 0.5 | 0.500 | 1,993 | 8,652 | 1,933 | 12 | 16 |
| XGBoost, class-weighted | best F1 on validation | 0.852 | 2,423 | 8,222 | 2,393 | 6 | 20 |
| XGBoost, class-weighted | min cost on validation | 0.167 | 1,957 | 8,688 | 1,832 | 25 | 14 |
| XGBoost, amount-weighted | 0.5 | 0.500 | 1,975 | 8,670 | 1,935 | 8 | 18 |
| XGBoost, amount-weighted | best F1 on validation | 0.231 | 1,908 | 8,737 | 1,833 | 15 | 15 |
| XGBoost, amount-weighted | min cost on validation | 0.021 | 2,186 | 8,459 | 1,831 | 71 | 13 |

### False alarm = 20 EUR

| Model | Threshold rule | Threshold | Test cost (EUR) | Saved vs no model (EUR) | Missed fraud (EUR) | False alarms | Missed frauds |
|---|---|---|---|---|---|---|---|
| XGBoost, original params | 0.5 | 0.500 | 3,034 | 7,611 | 2,994 | 2 | 24 |
| XGBoost, original params | best F1 on validation | 0.394 | 2,867 | 7,778 | 2,787 | 4 | 22 |
| XGBoost, original params | min cost on validation | 0.781 | 3,035 | 7,610 | 2,995 | 2 | 25 |
| XGBoost, class-weighted | 0.5 | 0.500 | 2,173 | 8,472 | 1,933 | 12 | 16 |
| XGBoost, class-weighted | best F1 on validation | 0.852 | 2,513 | 8,132 | 2,393 | 6 | 20 |
| XGBoost, class-weighted | min cost on validation | 0.167 | 2,332 | 8,313 | 1,832 | 25 | 14 |
| XGBoost, amount-weighted | 0.5 | 0.500 | 2,095 | 8,550 | 1,935 | 8 | 18 |
| XGBoost, amount-weighted | best F1 on validation | 0.231 | 2,133 | 8,512 | 1,833 | 15 | 15 |
| XGBoost, amount-weighted | min cost on validation | 0.925 | 2,332 | 8,313 | 2,272 | 3 | 24 |

![Cost curve](figures/cost_curve.png)

## 5. Calibration

Brier score on the test set (lower is better). The score of a model that always predicts the training fraud rate is 0.00172.

| Model | Brier score | Mean score | Observed fraud rate |
|---|---|---|---|
| XGBoost, original params, no weighting | 0.00041 | 0.00144 | 0.00172 |
| XGBoost, class-weighted | 0.00045 | 0.00198 | 0.00172 |
| Logistic regression (balanced) | 0.02205 | n/a | 0.00172 |

Class weighting trades calibration for recall, so the demo labels its output as a fraud score and not as a probability. Isotonic calibration on validation was not applied; the demo only needs the ranking and a threshold.

![Reliability curve](figures/calibration.png)

## Notes

- Sections 1 and 2 use different test sets (random 30% versus stratified 20%), so compare rows within a section, not across.
- The original scaler was fitted on all rows before splitting. That leaks test statistics into training. It is kept in section 1 only to reproduce the deck.
- With only ~100 frauds in the test set, one extra miss moves recall by about one point. Section 3 measures the spread: the best-F1 class-weighted model has an F1 standard deviation of 0.024 and a recall standard deviation of 0.045 across folds.
- Class-weighting did not clearly beat the unweighted baseline: test PR-AUC 0.866 versus 0.871, and cross-validated PR-AUC 0.857 +- 0.026 versus 0.850 +- 0.025. What the rebuild adds is a leak-free split, honest metrics and a threshold that was picked without looking at the test set.
- The high-recall row shows the cost of chasing recall: 294 extra false alarms for 10 more frauds caught.
- A minimum-cost threshold picked on validation did not reliably beat the best-F1 one. Cross-validated cost for the class-weighted model: 3,319 EUR with the cost rule against 3,095 EUR with the F1 rule, and the standard deviation across folds is about 1,489 EUR. A handful of large frauds per fold decides most of the cost, so cost differences of a few hundred EUR between models or rules are not evidence.
- ROC-AUC looks excellent for almost any model on this data. PR-AUC and recall at a given precision are the more honest numbers.

## Figures

![PR curve](figures/pr_curve.png)
![ROC curve](figures/roc_curve.png)
![Confusion matrix](figures/confusion_matrix.png)
![Threshold selection](figures/threshold_selection.png)
![Feature importance](figures/feature_importance.png)
