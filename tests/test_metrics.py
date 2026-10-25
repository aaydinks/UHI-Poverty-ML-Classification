import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from uhi_poverty import metrics

# 10 tracts, hand-tallied: TP=3, FN=1, TN=4, FP=2
Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
PRED = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
SCORE = np.array([0.9, 0.8, 0.7, 0.3, 0.1, 0.2, 0.4, 0.45, 0.6, 0.75])


def test_hand_computed_confusion_and_denominators():
    cm = metrics.confusion(Y, PRED)
    assert cm == {"TP": 3, "FP": 2, "TN": 4, "FN": 1}
    r = metrics.rates(cm)
    assert r["TPR"] == pytest.approx(3 / 4) and r["TPR_denominator"] == 4
    assert r["FNR"] == pytest.approx(1 / 4) and r["FNR_denominator"] == 4
    assert r["TNR"] == pytest.approx(4 / 6) and r["TNR_denominator"] == 6
    assert r["FPR"] == pytest.approx(2 / 6) and r["FPR_denominator"] == 6
    assert r["PPV"] == pytest.approx(3 / 5) and r["PPV_denominator"] == 5
    assert r["NPV"] == pytest.approx(4 / 5) and r["NPV_denominator"] == 5


def test_panel_matches_hand_values_and_sklearn():
    p = metrics.panel(Y, SCORE, PRED)
    assert p["n"] == 10
    assert p["prevalence"] == pytest.approx(0.4)
    assert p["accuracy"] == pytest.approx(7 / 10)
    assert p["balanced_accuracy"] == pytest.approx((3 / 4 + 4 / 6) / 2)
    assert p["precision"] == pytest.approx(3 / 5)
    assert p["recall"] == pytest.approx(3 / 4)
    assert p["specificity"] == pytest.approx(4 / 6)
    assert p["f1"] == pytest.approx(f1_score(Y, PRED))
    assert p["mcc"] == pytest.approx(matthews_corrcoef(Y, PRED))
    assert p["roc_auc"] == pytest.approx(roc_auc_score(Y, SCORE))
    assert p["pr_auc"] == pytest.approx(average_precision_score(Y, SCORE))
    assert p["brier"] == pytest.approx(brier_score_loss(Y, SCORE))


def test_ranking_metrics_with_ties_and_weights_match_sklearn():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    score = np.round(rng.uniform(size=500) * 0.6 + 0.4 * y, 1)  # many ties
    w = rng.integers(0, 4, 500).astype(float)
    p = metrics.panel(y, score, w=w)
    assert p["roc_auc"] == pytest.approx(roc_auc_score(y, score, sample_weight=w))
    assert p["pr_auc"] == pytest.approx(average_precision_score(y, score, sample_weight=w))


def test_constant_score_is_degenerate():
    p = metrics.panel(Y, np.ones(10), np.ones(10), constant_score=True)
    assert p["roc_auc"] == 0.5
    assert p["pr_auc"] == pytest.approx(0.4)
    assert p["mcc"] == 0.0
    assert p["brier"] == pytest.approx(0.6)
