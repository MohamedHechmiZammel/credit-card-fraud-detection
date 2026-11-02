---
title: Credit Card Fraud Detection
emoji: 💳
colorFrom: blue
colorTo: red
sdk: static
pinned: false
license: mit
short_description: XGBoost fraud scoring in plain JavaScript, no server
---

# Credit card fraud detection demo

Pick one of 200 held-out transactions (40 frauds, 160 legitimate) or edit any of the 30 features and see the
fraud score, the decision and each step of the computation. The model is an XGBoost classifier with class
weighting, trained on the ULB credit card fraud dataset.

The page loads the 317 trees from `model.json` and walks them in plain JavaScript (`score.js`), with no
libraries and no server. A row is flagged as fraud when its score is at or above the threshold in
`threshold.json` (0.852, chosen on validation data). The score is the raw model output, not a calibrated
probability, and the sample is enriched with frauds, so the fraud rate you see here is not the real one.

`verify.mjs` checks the JavaScript scorer against scores from the original Python model on all 200 rows:
largest difference 4.9e-7, same decision on every row. Run it with `node verify.mjs <reference.npy>`.

Source code, training scripts, tests and results: see the project repository.
