"""Confusion counts, the metric panel, and UA cluster-bootstrap intervals.

Bootstrap resamples are drawn over urbanized areas (UAs), never over rows: each
replicate draws UAs with replacement and gives every tract the number of times
its UA was drawn as a weight. The draws depend only on the sorted set of UAs and
the seed, so two models scored on the same tracts get identical draws, which is
what makes the paired differences valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

B_DEFAULT = 1000
SEED = 37
THRESHOLD = 0.5

PANEL = [
    "n", "prevalence", "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
    "f1", "mcc", "roc_auc", "pr_auc", "brier",
]
RATES = {  # name: (numerator, denominator cells, plain-language denominator)
    "TPR": ("TP", ("TP", "FN"), "all actual positives"),
    "FNR": ("FN", ("TP", "FN"), "all actual positives"),
    "TNR": ("TN", ("TN", "FP"), "all actual negatives"),
    "FPR": ("FP", ("TN", "FP"), "all actual negatives"),
    "PPV": ("TP", ("TP", "FP"), "all predicted positives"),
    "NPV": ("TN", ("TN", "FN"), "all predicted negatives"),
}


def confusion(y, pred, w=None) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    tp = float(np.sum(w * (y == 1) * (pred == 1)))
    fp = float(np.sum(w * (y == 0) * (pred == 1)))
    tn = float(np.sum(w * (y == 0) * (pred == 0)))
    fn = float(np.sum(w * (y == 1) * (pred == 0)))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def _div(a, b):
    return a / b if b > 0 else np.nan


def rates(cm: dict) -> dict:
    out = {}
    for name, (num, den, _) in RATES.items():
        out[name] = _div(cm[num], sum(cm[d] for d in den))
        out[f"{name}_denominator"] = sum(cm[d] for d in den)
    return out


class _Ranker:
    """Weighted ROC-AUC and average precision from one sort of the scores."""

    def __init__(self, y, score):
        y = np.asarray(y, dtype=float)
        score = np.asarray(score, dtype=float)
        order = np.argsort(-score, kind="mergesort")
        s_sorted = score[order]
        # tie groups in descending score order
        new = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
        group_sorted = np.cumsum(new) - 1
        self.group = np.empty_like(group_sorted)
        self.group[order] = group_sorted
        self.n_groups = int(group_sorted[-1] + 1) if len(score) else 0
        self.y = y

    def scores(self, w):
        pos = np.bincount(self.group, weights=w * self.y, minlength=self.n_groups)
        neg = np.bincount(self.group, weights=w * (1 - self.y), minlength=self.n_groups)
        wp, wn = pos.sum(), neg.sum()
        if wp <= 0 or wn <= 0:
            return np.nan, np.nan
        # groups are in descending score order: negatives ranked below group g
        neg_below = wn - np.cumsum(neg)
        auc = float(np.sum(pos * (neg_below + 0.5 * neg)) / (wp * wn))
        tp = np.cumsum(pos)
        fp = np.cumsum(neg)
        precision = tp / np.maximum(tp + fp, 1e-300)
        recall = tp / wp
        ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
        return auc, ap


def panel_from_counts(cm: dict, y, score, w, ranker: _Ranker | None,
                      constant_score: bool = False) -> dict:
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    n = tp + fp + tn + fn
    pos, neg = tp + fn, tn + fp
    recall = _div(tp, pos)
    spec = _div(tn, neg)
    prec = _div(tp, tp + fp)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    prevalence = _div(pos, n)
    if constant_score:
        # A constant score carries no ranking: ROC-AUC 0.5 and PR-AUC = prevalence by definition.
        auc, ap = 0.5, prevalence
    else:
        auc, ap = ranker.scores(w) if ranker is not None else (np.nan, np.nan)
    brier = float(np.sum(w * (np.asarray(score) - np.asarray(y)) ** 2) / np.sum(w))
    return {
        "n": n, "prevalence": prevalence, "accuracy": _div(tp + tn, n),
        "balanced_accuracy": (recall + spec) / 2, "precision": prec, "recall": recall,
        "specificity": spec, "f1": f1, "mcc": mcc, "roc_auc": auc, "pr_auc": ap,
        "brier": brier,
    }


def panel(y, score, pred=None, w=None, constant_score: bool = False) -> dict:
    """The full metric panel (section 4.4) for one set of predictions."""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    pred = (score >= THRESHOLD).astype(float) if pred is None else np.asarray(pred, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    cm = confusion(y, pred, w)
    ranker = None if constant_score else _Ranker(y, score)
    return panel_from_counts(cm, y, score, w, ranker, constant_score)


# --------------------------------------------------------------------------- bootstrap


@lru_cache(maxsize=64)
def _draws(n_clusters: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_clusters, size=(B, n_clusters))
    counts = np.zeros((B, n_clusters), dtype=np.int32)
    for b in range(B):
        counts[b] = np.bincount(picks[b], minlength=n_clusters)
    counts.setflags(write=False)
    return counts


def cluster_weights(clusters, B: int = B_DEFAULT, seed: int = SEED):
    """(B, n_rows) generator of row weights from a UA cluster bootstrap."""
    clusters = np.asarray(clusters)
    uniq, code = np.unique(clusters, return_inverse=True)
    counts = _draws(len(uniq), B, seed)
    for b in range(B):
        yield counts[b][code].astype(float)


@dataclass
class Scored:
    """One model's out-of-fold predictions on a fixed set of tracts."""

    y: np.ndarray
    score: np.ndarray
    pred: np.ndarray
    constant_score: bool = False

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=float)
        self.score = np.asarray(self.score, dtype=float)
        self.pred = np.asarray(self.pred, dtype=float)
        self._ranker = None if self.constant_score else _Ranker(self.y, self.score)

    def at(self, w) -> dict:
        cm = confusion(self.y, self.pred, w)
        return panel_from_counts(cm, self.y, self.score, w, self._ranker, self.constant_score)


def bootstrap_panel(scored: Scored, clusters, B: int = B_DEFAULT, seed: int = SEED,
                    metrics: list[str] | None = None) -> pd.DataFrame:
    """Point estimate and percentile 95 % CI for each metric."""
    metrics = metrics or PANEL
    point = scored.at(np.ones_like(scored.y))
    reps = np.array([[scored.at(w)[m] for m in metrics]
                     for w in cluster_weights(clusters, B, seed)])
    lo = np.nanpercentile(reps, 2.5, axis=0)
    hi = np.nanpercentile(reps, 97.5, axis=0)
    return pd.DataFrame({"metric": metrics, "estimate": [point[m] for m in metrics],
                         "ci_low": lo, "ci_high": hi})


def paired_delta(a: Scored, b: Scored, clusters, metrics=("roc_auc", "mcc"),
                 B: int = B_DEFAULT, seed: int = SEED) -> pd.DataFrame:
    """Δ = a − b with a paired UA cluster-bootstrap CI (same draws for both)."""
    ones = np.ones_like(a.y)
    pa, pb = a.at(ones), b.at(ones)
    reps = []
    for w in cluster_weights(clusters, B, seed):
        ra, rb = a.at(w), b.at(w)
        reps.append([ra[m] - rb[m] for m in metrics])
    reps = np.array(reps)
    return pd.DataFrame({
        "metric": list(metrics),
        "a": [pa[m] for m in metrics],
        "b": [pb[m] for m in metrics],
        "delta": [pa[m] - pb[m] for m in metrics],
        "ci_low": np.nanpercentile(reps, 2.5, axis=0),
        "ci_high": np.nanpercentile(reps, 97.5, axis=0),
        "share_reps_above_0": np.mean(reps > 0, axis=0),
    })


def per_group(df: pd.DataFrame, group: str, min_n: int = 30, constant_score: bool = False,
              metrics=("n", "prevalence", "accuracy", "f1", "mcc", "roc_auc")) -> pd.DataFrame:
    """Metrics within each group. Groups below `min_n` or with one class get no metrics."""
    rows = []
    for g, sub in df.groupby(group, sort=True):
        n = len(sub)
        both = sub["y"].nunique() == 2
        row = {group: g, "n_tracts": n, "both_classes": both, "eligible": both and n >= min_n}
        if row["eligible"]:
            p = panel(sub["y"], sub["score"], sub["pred"], constant_score=constant_score)
            row.update({m: p[m] for m in metrics if m != "n"})
        rows.append(row)
    return pd.DataFrame(rows)


def fold_sd(df: pd.DataFrame, constant_score: bool = False, metrics=PANEL) -> dict:
    """Across-fold standard deviation of each metric (for the 5-fold schemes)."""
    vals = [panel(s["y"], s["score"], s["pred"], constant_score=constant_score)
            for _, s in df.groupby("fold")]
    return {m: float(np.nanstd([v[m] for v in vals], ddof=1)) for m in metrics}


# --------------------------------------------------------------------------- OOF runs

FIVE_FOLD = ("random5", "county5", "ua5")


def scored_run(task: str, config: str, model: str, scheme: str, drop_top: int = 0,
               df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, Scored]:
    """Cached out-of-fold predictions for one run, plus a `Scored` wrapper."""
    from . import validation

    o = validation.oof(task, config, model, scheme, drop_top=drop_top, df=df)
    return o, Scored(o["y"], o["score"], o["pred"], constant_score=(model == "majority"))


def evaluate(task: str, config: str, model: str, scheme: str, drop_top: int = 0,
             df: pd.DataFrame | None = None, B: int = B_DEFAULT, ci: bool = True,
             metrics: list[str] | None = None) -> pd.DataFrame:
    """Long table: one row per metric, with UA cluster-bootstrap CI and across-fold SD."""
    metrics = metrics or PANEL
    o, s = scored_run(task, config, model, scheme, drop_top, df)
    if ci:
        t = bootstrap_panel(s, o["ua"].to_numpy(), B=B, metrics=metrics)
    else:
        point = s.at(np.ones_like(s.y))
        t = pd.DataFrame({"metric": metrics, "estimate": [point[m] for m in metrics],
                          "ci_low": np.nan, "ci_high": np.nan})
    if scheme in FIVE_FOLD:
        sd = fold_sd(o, constant_score=(model == "majority"), metrics=metrics)
        t["fold_sd"] = [sd[m] for m in metrics]
    else:
        t["fold_sd"] = np.nan
    t.insert(0, "scheme", scheme)
    t.insert(0, "model", model)
    t.insert(0, "config", config)
    t.insert(0, "task", task)
    t["drop_top"] = drop_top
    t["n_folds"] = o["fold"].nunique()
    t["n_uas"] = o["ua"].nunique()
    return t
