---
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
