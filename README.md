# Credit card fraud detection

Detects fraudulent card transactions in the ULB credit card dataset (284,807 transactions, 492 frauds, 0.17% positive class). Compares a Decision Tree and XGBoost, then rebuilds the evaluation so that a model's quality is judged by precision, recall, F1 and PR-AUC instead of accuracy.

Originally built as a 5-person team project at ESPRIT (2024); this is my reproducible rebuild.

## What is in the repo

| Path | Content |
|---|---|
| `scripts/download_data.py` | Downloads and checks the dataset (no account needed) |
| `scripts/reproduce.py` | The original 2024 pipeline and both grid searches |
| `scripts/train_improved.py` | Stratified split, class weighting, threshold chosen on a validation split |
| `scripts/make_report.py` | Builds `reports/results.md` from the JSON results |
| `scripts/build_space.py` | Assembles `space/`, a ready-to-upload Hugging Face Space (nothing is uploaded) |
| `src/fraud/` | Data checks, model helpers, threshold logic, metrics |
| `app/app.py` | Gradio demo using `data/sample.csv` and `models/xgb_fraud.joblib` |
| `tests/` | pytest suite (schema, probabilities in [0, 1], threshold logic) |
| `reports/` | `results.md`, JSON results, figures |

## Dataset

ULB / Worldline credit card fraud data: two days of European card transactions from September 2013. Features are `Time`, `Amount`, and 28 anonymized PCA components `V1` to `V28`. The target is `Class` (1 = fraud).

`scripts/download_data.py` fetches `creditcard.csv` from the Hugging Face mirror `David-Egea/Creditcard-fraud-detection` and checks 284,807 rows and 492 frauds. (OpenML id 1597 has the same rows but no `Time` column, so it does not fit the original recipe.) The CSV is about 150 MB and is not committed.

License: the original data is released by the ULB Machine Learning Group under the Database Contents License (DbCL) v1.0. The Hugging Face mirror tags itself MIT, which does not change the original terms. Check the original terms before redistributing the data. Citation: Dal Pozzolo, Caelen, Johnson, Bontempi, "Calibrating Probability with Undersampling for Unbalanced Classification", IEEE SSCI 2015.

## How to run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # CPU only
python scripts/download_data.py
python scripts/reproduce.py dt  --scoring accuracy    # about 12 min on 4 cores
python scripts/reproduce.py xgb --scoring accuracy    # about 21 min, full 54-combination grid
python scripts/train_improved.py                      # about 1 min, writes model, figures, sample
python scripts/make_report.py                         # writes reports/results.md
python -m pytest
python app/app.py                                     # demo on http://127.0.0.1:7860
```

The tests and the demo do not need the full dataset.

## Results

Full tables and notes are in [reports/results.md](reports/results.md).

### Reproduction against the 2024 deck

Same preprocessing and split as the original: StandardScaler fitted on all rows, `train_test_split(test_size=0.3, random_state=42)`, 5-fold GridSearchCV, same grids. 85,443 test rows, 136 frauds.

| Model | Version | TN | FP | FN | TP | Accuracy |
|---|---|---|---|---|---|---|
| Decision Tree | Original deck | 85290 | 17 | 27 | 109 | 0.99949 |
| Decision Tree | Reproduced | 85290 | 17 | 27 | 109 | 0.99949 |
| XGBoost | Original deck | not shown | not shown | not shown | not shown | 0.99964 |
| XGBoost | Reproduced | 85300 | 7 | 29 | 107 | 0.99958 |

The Decision Tree matches the deck exactly. The XGBoost accuracy is 5 transactions short of the deck (the original XGBoost version and tree method were not recorded). Details on both are in `reports/results.md`.

### Improved evaluation

Stratified 60/20/20 split, 98 frauds in the test set. The threshold is chosen on the validation split (best F1) and applied once to the test set.

| Model | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|
| XGBoost, original params, no weighting | 0.50 | 0.974 | 0.755 | 0.851 | 0.871 | 0.975 | 2 | 24 |
| Improved XGBoost, class weighting | 0.50 | 0.872 | 0.837 | 0.854 | 0.866 | 0.978 | 12 | 16 |
| Improved XGBoost, validation threshold | 0.85 | 0.929 | 0.796 | 0.857 | 0.866 | 0.978 | 6 | 20 |

The gain is small and I do not want to oversell it. With 98 frauds in the test set, one extra missed fraud moves recall by one point, so F1 of 0.857 against 0.851 is a tie. Class weighting did not raise PR-AUC. What the rebuild changes is the method: no scaler leakage, a stratified split, a threshold picked without looking at the test set, and metrics that do not hide the minority class. Accuracy sits at 99.95% for every row above, which is why it is not the headline.

### Figures

| | |
|---|---|
| ![PR curve](reports/figures/pr_curve.png) | ![Confusion matrix](reports/figures/confusion_matrix.png) |
| ![Threshold selection](reports/figures/threshold_selection.png) | ![ROC curve](reports/figures/roc_curve.png) |

## Demo

`python app/app.py` opens a Gradio app. Pick one of 200 held-out transactions (40 frauds, 160 legitimate, from the test split), move the threshold slider, and read the fraud probability, the decision and the precision-recall curve. The sample has 20% frauds against 0.17% in reality, so its curve is illustrative only.

## Limitations

- About 100 frauds in the test set make every fraud metric noisy. A different seed can shift recall by several points.
- The data is from two days in 2013, and 28 features are anonymized PCA outputs. There is no way to check what the model relies on, and no guarantee it transfers to current card traffic.
- `Time` is seconds since the first transaction. A random split mixes past and future, which is optimistic. A time-based split would be stricter, and I did not run one.
- The original XGBoost grid includes `booster="dart"`. XGBoost 3.x marks it as deprecated. It never won the grid search here.
- The XGBoost reproduction is not exact (5 transactions on accuracy).
- Precision and recall depend on the cost of a false alarm against a missed fraud. The F1 threshold assumes both cost the same, which is rarely true in practice. The `min_recall` option in `choose_threshold` gives a high-recall mode, and it costs about 300 false alarms per 57,000 transactions in the test run.

## License

MIT for the code, see [LICENSE](LICENSE). The dataset keeps its own license, see above.
