"""Assemble the self-contained Hugging Face Space folder in space/ (does not upload anything)."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "space"
S.mkdir(exist_ok=True)
shutil.copy(ROOT / "app" / "app.py", S / "app.py")
shutil.copy(ROOT / "models" / "xgb_fraud.joblib", S / "model.joblib")
shutil.copy(ROOT / "models" / "threshold.json", S / "threshold.json")
shutil.copy(ROOT / "data" / "sample.csv", S / "sample.csv")
(S / "requirements.txt").write_text(
    "pandas==3.1.0\nnumpy==2.5.4\nscikit-learn==1.9.1\nxgboost==3.4.1\nmatplotlib==3.11.2\njoblib==1.6.0\n"
)
(S / "README.md").write_text("""---
title: Credit Card Fraud Detection
emoji: 💳
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: XGBoost fraud scoring with a validation-chosen threshold
---

# Credit card fraud detection demo

Pick one of 200 held-out sample transactions (40 frauds, 160 legitimate), move the decision threshold, and see the
fraud probability, the decision and the precision-recall curve. The model is XGBoost with class weighting trained on
the ULB credit card fraud dataset. The sample is enriched with frauds, so the curve is illustrative only.

Source code, training scripts, tests and results: see the project repository.
""")
print("space/ ready:", sorted(p.name for p in S.iterdir()))
