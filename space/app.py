"""Gradio demo: pick a sample transaction, see the fraud probability and the decision.

Works in two layouts:
  repo:   app/app.py with models/ and data/ next to it
  space:  app.py, model.joblib, threshold.json, sample.csv in one folder
"""
import json
from pathlib import Path

import gradio as gr
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _find(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"none of these exist: {[str(c) for c in candidates]}")


MODEL = joblib.load(_find(HERE / "model.joblib", ROOT / "models" / "xgb_fraud.joblib"))
META = json.loads(_find(HERE / "threshold.json", ROOT / "models" / "threshold.json").read_text())
SAMPLE = pd.read_csv(_find(HERE / "sample.csv", ROOT / "data" / "sample.csv"))
FEATURES = META["features"]
DEFAULT_THRESHOLD = round(float(META["threshold"]), 3)
PROBA = MODEL.predict_proba(SAMPLE[FEATURES])[:, 1]
LABELS = SAMPLE["Class"].to_numpy()


def pr_figure(threshold: float, index: int):
    precision, recall, thr = precision_recall_curve(LABELS, PROBA)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#1b6ca8", label=f"PR curve (AP {average_precision_score(LABELS, PROBA):.3f})")
    pred = PROBA >= threshold
    tp = int((pred & (LABELS == 1)).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(LABELS.sum()), 1)
    ax.scatter([r], [p], color="#d1495b", zorder=5, label=f"threshold {threshold:.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02), xlim=(0, 1.02),
           title="Precision-recall on the 200-row sample")
    ax.legend(loc="lower left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def score(index: int, threshold: float):
    index = int(index)
    row = SAMPLE.iloc[[index]]
    prob = float(PROBA[index])
    flagged = prob >= threshold
    decision = "FRAUD ALERT: block and review" if flagged else "Looks legitimate: approve"
    truth = "fraud" if int(row["Class"].iloc[0]) == 1 else "legitimate"
    detail = f"Fraud probability {prob:.4f} against threshold {threshold:.2f}. True label: {truth}."
    show = row[["Time", "Amount", "V1", "V2", "V3", "V4", "V14"]].round(3)
    return round(prob, 4), decision, detail, show, pr_figure(threshold, index)


with gr.Blocks(title="Credit card fraud detection") as demo:
    gr.Markdown(
        "# Credit card fraud detection\n"
        "XGBoost with class weighting, trained on the ULB credit card dataset. The decision threshold was "
        "chosen on a validation split. The 200 sample transactions come from the held-out test split and "
        "contain 20% frauds, far more than the real rate of 0.17%, so the curve below is only illustrative."
    )
    with gr.Row():
        with gr.Column():
            idx = gr.Slider(0, len(SAMPLE) - 1, value=0, step=1, label="Sample transaction")
            thr = gr.Slider(0.01, 0.99, value=DEFAULT_THRESHOLD, step=0.01, label="Decision threshold")
            btn = gr.Button("Score", variant="primary")
            prob = gr.Number(label="Fraud probability")
            decision = gr.Textbox(label="Decision")
            detail = gr.Textbox(label="Detail")
            row = gr.Dataframe(label="Selected transaction (a few features)")
        with gr.Column():
            plot = gr.Plot(label="Precision-recall curve")
    inputs, outputs = [idx, thr], [prob, decision, detail, row, plot]
    btn.click(score, inputs, outputs)
    demo.load(score, inputs, outputs)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
