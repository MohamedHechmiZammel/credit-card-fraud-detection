import numpy as np
import pytest

from fraud.cost import cost_curve, cost_report, no_model_cost, pick_threshold_by_cost, transaction_cost
from fraud.model import amount_sample_weight, load_xgb_params
from fraud.threshold import apply_threshold


def test_cost_arithmetic_hand_example():
    # frauds: 100 (caught), 50 (missed). false alarms: the 10 and the 30 transactions.
    y = [1, 1, 0, 0, 0]
    amounts = [100, 50, 10, 20, 30]
    pred = [1, 0, 1, 0, 1]
    assert transaction_cost(y, pred, amounts, false_alarm_cost=5) == 50 + 2 * 5
    assert transaction_cost(y, pred, amounts, false_alarm_cost=5, review_caught_fraud=True) == 50 + 3 * 5


def test_no_model_cost_is_all_fraud_amounts():
    assert no_model_cost([1, 1, 0, 0, 0], [100, 50, 10, 20, 30]) == 150
    assert transaction_cost([1, 1, 0], [0, 0, 0], [100, 50, 10], 5) == 150  # flagging nothing


def test_threshold_picks_the_minimum_cost_point():
    y = [1, 0, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    amounts = [100, 3, 50, 10, 10]
    thr, thresholds, costs = pick_threshold_by_cost(y, scores, amounts, false_alarm_cost=5)
    # flag nothing 150, >=0.9 50, >=0.8 55, >=0.7 5, >=0.2 10, >=0.1 15
    assert costs.tolist() == [150, 50, 55, 5, 10, 15]
    assert thr == 0.7
    assert costs.min() == transaction_cost(y, apply_threshold(scores, thr), amounts, 5)


def test_expensive_false_alarms_raise_the_threshold():
    y = [1, 0, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    amounts = [100, 3, 50, 10, 10]
    thr, _, costs = pick_threshold_by_cost(y, scores, amounts, false_alarm_cost=100)
    assert thr == 0.9 and costs.min() == 50  # only the top-scored fraud is worth one false alarm


def test_cost_curve_matches_brute_force_with_ties():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.1).astype(int)
    scores = np.round(rng.random(300), 1)  # many ties
    amounts = rng.uniform(1, 200, 300)
    for review in (False, True):
        thresholds, costs = cost_curve(y, scores, amounts, 7.0, review)
        for t, c in zip(thresholds, costs):
            assert c == pytest.approx(transaction_cost(y, apply_threshold(scores, t), amounts, 7.0, review))


def test_cost_report_saved_is_no_model_minus_cost():
    r = cost_report([1, 1, 0, 0, 0], [0.9, 0.2, 0.8, 0.1, 0.1], 0.5, [100, 50, 10, 20, 30], 5)
    assert r["cost"] == 50 + 5
    assert r["no_model_cost"] == 150 and r["saved"] == 95
    assert (r["false_alarms"], r["missed_frauds"], r["caught_frauds"]) == (1, 1, 1)


def test_amount_sample_weight():
    w = amount_sample_weight([0, 1, 1, 1], [999, 5, 10, 1000], cap=20)  # median fraud Amount = 10
    assert w.tolist() == [1.0, 1.5, 2.0, 20.0]


def test_load_xgb_params_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_xgb_params(tmp_path / "missing.json")
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(ValueError):
        load_xgb_params(empty)
    ok = tmp_path / "ok.json"
    ok.write_text('{"best_params": {"max_depth": 5}}')
    assert load_xgb_params(ok) == {"max_depth": 5}
