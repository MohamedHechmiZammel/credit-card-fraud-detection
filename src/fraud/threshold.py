"""Operating threshold selection. Always call this on a validation split."""
import numpy as np
from sklearn.metrics import precision_recall_curve


def choose_threshold(y_val, proba_val, min_recall: float | None = None) -> float:
    """Pick the threshold with the best F1 on the validation data.

    If min_recall is given, only thresholds with recall >= min_recall are
    considered, and the one with the best precision among them is returned.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    # precision/recall have one more entry than thresholds; drop the last point
    precision, recall = precision[:-1], recall[:-1]
    if min_recall is not None:
        ok = recall >= min_recall
        if not ok.any():
            raise ValueError("no threshold reaches the requested recall")
        best = np.argmax(np.where(ok, precision, -1.0))
    else:
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        best = int(np.argmax(f1))
    return float(thresholds[best])


def apply_threshold(proba, threshold: float) -> np.ndarray:
    """Flag a transaction as fraud when probability >= threshold."""
    return (np.asarray(proba) >= threshold).astype(int)
