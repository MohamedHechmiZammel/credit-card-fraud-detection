"""Cost-sensitive evaluation: money lost to missed frauds plus the price of a false alarm.

Cost model (all amounts in EUR, the currency of the ULB dataset):

  cost = sum(Amount of every missed fraud)
         + false_alarm_cost * (number of legitimate transactions flagged)
         + review_cost_for_caught_fraud * (number of frauds caught)    [optional]

Assumptions, stated so they can be challenged:
  - A missed fraud loses its full Amount. Chargeback fees and customer churn are ignored.
  - A false alarm costs a flat `false_alarm_cost` (analyst time, customer friction).
  - A caught fraud loses nothing. With `review_caught_fraud=True` it also costs one
    `false_alarm_cost`, because every flagged transaction is reviewed by a person.
  - "No model" means approving everything, so its cost is the sum of all fraud Amounts.
"""
import numpy as np

from .threshold import apply_threshold


def transaction_cost(y_true, y_pred, amounts, false_alarm_cost: float,
                     review_caught_fraud: bool = False) -> float:
    """Total cost of a set of decisions. See the module docstring for the assumptions."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    amounts = np.asarray(amounts, dtype=float)
    missed = (y_true == 1) & (y_pred == 0)
    false_alarms = (y_true == 0) & (y_pred == 1)
    caught = (y_true == 1) & (y_pred == 1)
    cost = amounts[missed].sum() + false_alarm_cost * false_alarms.sum()
    if review_caught_fraud:
        cost += false_alarm_cost * caught.sum()
    return float(cost)


def no_model_cost(y_true, amounts) -> float:
    """Cost of approving every transaction: all fraud Amounts are lost."""
    return float(np.asarray(amounts, dtype=float)[np.asarray(y_true) == 1].sum())


def cost_curve(y_true, scores, amounts, false_alarm_cost: float, review_caught_fraud: bool = False):
    """Cost of every distinct threshold, from the highest score down, in O(n log n).

    A transaction is flagged when score >= threshold (same rule as apply_threshold).
    The first point is a threshold just above the top score, which flags nothing.
    Returns (thresholds, costs), both in descending threshold order.
    """
    y, s, a = np.asarray(y_true), np.asarray(scores, dtype=float), np.asarray(amounts, dtype=float)
    order = np.argsort(-s, kind="stable")
    y, s, a = y[order], s[order], a[order]
    caught_amount = np.cumsum(np.where(y == 1, a, 0.0))
    n_fp = np.cumsum(y == 0)
    n_tp = np.cumsum(y == 1)
    last_of_group = np.append(np.flatnonzero(s[:-1] != s[1:]), len(s) - 1)
    total_fraud = a[y == 1].sum()
    flag_cost = n_fp[last_of_group] + (n_tp[last_of_group] if review_caught_fraud else 0)
    costs = total_fraud - caught_amount[last_of_group] + false_alarm_cost * flag_cost
    thresholds = s[last_of_group]
    return (np.concatenate([[np.nextafter(s[0], np.inf)], thresholds]),
            np.concatenate([[total_fraud], costs]))


def pick_threshold_by_cost(y_true, scores, amounts, false_alarm_cost: float,
                           review_caught_fraud: bool = False):
    """Threshold that minimizes total cost on the given data (use a validation split).

    Returns (threshold, thresholds, costs). Ties go to the highest threshold, so the
    model flags as little as it can for the same money.
    """
    thresholds, costs = cost_curve(y_true, scores, amounts, false_alarm_cost, review_caught_fraud)
    best = int(np.argmin(costs))
    return float(thresholds[best]), thresholds, costs


def cost_report(y_true, scores, threshold: float, amounts, false_alarm_cost: float,
                review_caught_fraud: bool = False) -> dict:
    """Cost at a threshold, the no-model cost and the money saved between the two."""
    pred = apply_threshold(scores, threshold)
    y, a = np.asarray(y_true), np.asarray(amounts, dtype=float)
    cost = transaction_cost(y, pred, a, false_alarm_cost, review_caught_fraud)
    baseline = no_model_cost(y, a)
    return {
        "threshold": float(threshold),
        "cost": cost,
        "no_model_cost": baseline,
        "saved": baseline - cost,
        "missed_fraud_amount": float(a[(y == 1) & (pred == 0)].sum()),
        "false_alarms": int(((y == 0) & (pred == 1)).sum()),
        "missed_frauds": int(((y == 1) & (pred == 0)).sum()),
        "caught_frauds": int(((y == 1) & (pred == 1)).sum()),
    }
