"""2×2 association statistics shared by sections 2.2 and 1.5.

Rows are the exposure (`High_UHI`), columns the outcome (`High_Pov`); index 1 is
the positive class. Every function takes 0/1 arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

from .metrics import _draws

Z = stats.norm.ppf(0.975)


def table(x, y, w=None) -> np.ndarray:
    """[[n00, n01], [n10, n11]]: rows x = 0/1, columns y = 0/1."""
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    w = np.ones(len(x)) if w is None else np.asarray(w, dtype=float)
    t = np.zeros((2, 2))
    np.add.at(t, (x, y), w)
    return t


def odds_ratio(t: np.ndarray) -> float:
    return (t[1, 1] * t[0, 0]) / (t[1, 0] * t[0, 1])


def two_by_two(x, y) -> dict:
    """χ² (with and without Yates), φ / Cramér's V, OR (Woolf CI), RR and RD."""
    t = table(x, y)
    n = t.sum()
    chi2, p, _, expected = stats.chi2_contingency(t, correction=False)
    chi2_y, p_y, _, _ = stats.chi2_contingency(t, correction=True)
    phi = (t[1, 1] * t[0, 0] - t[1, 0] * t[0, 1]) / np.sqrt(
        t.sum(1).prod() * t.sum(0).prod())
    orr = odds_ratio(t)
    se_log_or = np.sqrt((1 / t).sum())
    p1 = t[1, 1] / t[1].sum()  # P(y = 1 | x = 1)
    p0 = t[0, 1] / t[0].sum()  # P(y = 1 | x = 0)
    rr = p1 / p0
    se_log_rr = np.sqrt(1 / t[1, 1] - 1 / t[1].sum() + 1 / t[0, 1] - 1 / t[0].sum())
    rd = p1 - p0
    se_rd = np.sqrt(p1 * (1 - p1) / t[1].sum() + p0 * (1 - p0) / t[0].sum())
    return {
        "n": int(n), "n00": int(t[0, 0]), "n01": int(t[0, 1]), "n10": int(t[1, 0]),
        "n11": int(t[1, 1]), "expected": expected.round(2).tolist(),
        "chi2": float(chi2), "df": 1, "p": float(p), "chi2_yates": float(chi2_y),
        "p_yates": float(p_y), "phi": float(phi), "cramers_v": float(abs(phi)),
        "odds_ratio": float(orr),
        "or_ci_low": float(np.exp(np.log(orr) - Z * se_log_or)),
        "or_ci_high": float(np.exp(np.log(orr) + Z * se_log_or)),
        "p_outcome_given_exposed": float(p1), "p_outcome_given_unexposed": float(p0),
        "risk_ratio": float(rr), "rr_ci_low": float(np.exp(np.log(rr) - Z * se_log_rr)),
        "rr_ci_high": float(np.exp(np.log(rr) + Z * se_log_rr)),
        "risk_difference": float(rd), "rd_ci_low": float(rd - Z * se_rd),
        "rd_ci_high": float(rd + Z * se_rd),
    }


def cmh(x, y, strata) -> dict:
    """Cochran–Mantel–Haenszel common OR across strata, with the Breslow–Day test."""
    df = pd.DataFrame({"x": np.asarray(x, int), "y": np.asarray(y, int), "s": np.asarray(strata)})
    tables = []
    for _, g in df.groupby("s"):
        t = table(g["x"], g["y"])
        # strata with an empty margin carry no information about the OR
        if (t.sum(0) > 0).all() and (t.sum(1) > 0).all():
            tables.append(t)
    st = StratifiedTable(tables)
    lo, hi = st.oddsratio_pooled_confint()
    null = st.test_null_odds(correction=False)
    bd = st.test_equal_odds()
    return {"n_strata_informative": len(tables), "cmh_or": float(st.oddsratio_pooled),
            "cmh_or_ci_low": float(lo), "cmh_or_ci_high": float(hi),
            "cmh_chi2": float(null.statistic), "cmh_p": float(null.pvalue),
            "breslow_day_chi2": float(bd.statistic), "breslow_day_p": float(bd.pvalue)}


def cluster_bootstrap_or(x, y, clusters, B: int = 1000, seed: int = 37) -> dict:
    """Percentile CI for the OR from a bootstrap that resamples clusters (UAs)."""
    x = np.asarray(x, int)
    y = np.asarray(y, int)
    uniq, code = np.unique(np.asarray(clusters), return_inverse=True)
    counts = _draws(len(uniq), B, seed)
    cell = x * 2 + y
    # per-UA cell counts, then each replicate is a weighted sum over UAs
    per_ua = np.zeros((len(uniq), 4))
    np.add.at(per_ua, (code, cell), 1)
    reps = counts @ per_ua  # (B, 4): n00, n01, n10, n11
    ors = (reps[:, 3] * reps[:, 0]) / (reps[:, 2] * reps[:, 1])
    return {"or_boot_ci_low": float(np.percentile(ors, 2.5)),
            "or_boot_ci_high": float(np.percentile(ors, 97.5)), "B": B}


def clustered_logit(x, y, clusters) -> dict:
    """Logistic regression of y on x with UA-clustered (sandwich) standard errors."""
    X = sm.add_constant(np.asarray(x, dtype=float))
    _, groups = np.unique(np.asarray(clusters), return_inverse=True)
    fit = sm.Logit(np.asarray(y, dtype=float), X).fit(disp=0, cov_type="cluster",
                                                      cov_kwds={"groups": groups})
    b, se = fit.params[1], fit.bse[1]
    return {"logit_or": float(np.exp(b)), "logit_or_ci_low": float(np.exp(b - Z * se)),
            "logit_or_ci_high": float(np.exp(b + Z * se)), "logit_cluster_se": float(se),
            "logit_p": float(fit.pvalues[1]), "n_clusters": int(groups.max() + 1)}


def full(x, y, clusters, B: int = 1000) -> dict:
    out = two_by_two(x, y)
    out.update(cmh(x, y, clusters))
    out.update(cluster_bootstrap_or(x, y, clusters, B=B))
    out.update(clustered_logit(x, y, clusters))
    return out
