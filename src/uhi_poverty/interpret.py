"""Out-of-fold attributions for the evaluated LightGBM models (sections 5.1-5.2).

The explained model is the evaluated model: the fixed-setting LightGBM of
`models.py`, refitted on each training fold of the UA-grouped 5-fold split and
explained on its held-out fold only, so no tract is explained by a model that saw
its urbanized area. Contributions are LightGBM's exact TreeSHAP values
(`pred_contrib=True`) in log-odds: for each tract they sum, with the bias term,
to the logit of its out-of-fold score.

For every fold model the module also records, on the held-out rows:

- mean |SHAP| per feature, and mean |sum of SHAP| per attribution group
  (`features.ATTRIBUTION_GROUPS`, which keep every exact identity together);
- grouped permutation importance: the drop in ROC-AUC when all columns of a group
  are permuted together, averaged over `N_REPEATS` draws;
- the direction of each feature: Spearman's rho between its value and its own
  SHAP value, and between its value and its group's summed SHAP value;
- ALE local effects (log-odds) for `water_frac`, on bin edges fixed from the whole
  sample so that folds can be pooled. It is the only interpreted column that is
  neither in an identity nor near-duplicated: moving one
  term of an identity, or `Area` while `Total_Count` and `log_aland` (Spearman
  >= 0.99) stay fixed, leaves the data, so those directions are read from group
  SHAP on observed rows instead.

Seed 37 is the attribution of record: its folds are those of the cached OOF
predictions, and `compute()` checks the contributions against them. Its `rows`
table has one row per tract, and its feature-named columns hold SHAP values, not
inputs (`feature_values()` returns the inputs aligned with it). Seeds 38 and
39 redraw the folds, giving 15 fold models for the stability checks.

Results are cached under `data/interim/interpret/` with a fingerprint, as in
`validation.oof()`. `uv run python -m uhi_poverty.interpret` (or `make interpret`)
fills the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import features, models, paths, validation

log = logging.getLogger("uhi_poverty.interpret")

TASKS = ("A", "B")
MODEL = "lgbm"
SCHEME = "ua5"
SEEDS = (37, 38, 39)
RECORD_SEED = 37
N_REPEATS = 5
ALE_BINS = 20
ALE_FEATURES = ["water_frac"]
ADDITIVITY_TOL = 1e-9
# A group's rank is interpreted only if it moves by at most this many places
# across the 15 fold models; a direction only if its sign agrees in all 15.
STABLE_RANK_SPREAD = 2
VERSION = 1
TABLES = ("rows", "importance", "permutation", "direction", "ale")


def config(task: str) -> str:
    return features.TASKS[task]["primary"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def contributions(estimator, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP values in log-odds, n × (p + 1); the last column is the bias."""
    c = np.asarray(estimator.predict(X, pred_contrib=True))
    if c.shape != (len(X), X.shape[1] + 1):
        raise AssertionError(f"unexpected contribution shape {c.shape}")
    return c


def group_sums(phi: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Per-row SHAP summed within each attribution group."""
    return pd.DataFrame({g: phi[members].sum(axis=1) for g, members in groups.items()},
                        index=phi.index)


def ale_edges(df: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    """Fixed ALE bin edges: quantiles of each feature over the whole sample (no labels)."""
    q = np.linspace(0, 1, ALE_BINS + 1)
    return {c: np.unique(np.quantile(df[c].dropna(), q)) for c in ALE_FEATURES if c in cols}


def _rho(x: np.ndarray, s: np.ndarray) -> float:
    ok = ~np.isnan(x)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(s[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], s[ok]).statistic)


def _permutation(pipe, Xte: pd.DataFrame, yte: np.ndarray, groups: dict[str, list[str]],
                 seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    base = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    rows = []
    for g, members in groups.items():
        idx = [Xte.columns.get_loc(c) for c in members]
        drops = []
        for _ in range(N_REPEATS):
            Xp = Xte.copy()
            Xp.iloc[:, idx] = Xte.iloc[rng.permutation(len(Xte)), idx].to_numpy()
            drops.append(base - roc_auc_score(yte, pipe.predict_proba(Xp)[:, 1]))
        rows.append({"group": g, "auc": base, "auc_drop": float(np.mean(drops)),
                     "auc_drop_sd": float(np.std(drops, ddof=1))})
    return rows


def _ale(estimator, Xte: pd.DataFrame, edges: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for feat, e in edges.items():
        x = Xte[feat].to_numpy()
        ok = ~np.isnan(x)
        # bin b covers (e[b-1], e[b]]; the lowest edge joins bin 1
        b = np.clip(np.searchsorted(e, x[ok], side="left"), 1, len(e) - 1)
        lo, hi = Xte[ok].copy(), Xte[ok].copy()
        lo[feat], hi[feat] = e[b - 1], e[b]
        diff = estimator.predict(hi, raw_score=True) - estimator.predict(lo, raw_score=True)
        for k in range(1, len(e)):
            m = b == k
            rows.append({"feature": feat, "bin": k, "lo": e[k - 1], "hi": e[k],
                         "n": int(m.sum()), "sum_diff": float(diff[m].sum())})
    return rows


def _fold(X, y, cols, groups, edges, seed, label, train_idx, test_idx):
    Xtr = pd.DataFrame(X[train_idx], columns=cols)
    Xte = pd.DataFrame(X[test_idx], columns=cols)
    pipe = models.make_model(MODEL).fit(Xtr, y[train_idx])
    est = pipe.named_steps["model"]
    c = contributions(est, Xte)
    phi = pd.DataFrame(c[:, :-1], columns=cols)
    gs = group_sums(phi, groups)
    tag = {"seed": seed, "fold": str(label), "n": len(test_idx)}
    importance = (
        [{**tag, "level": "feature", "name": f, "mean_abs": float(phi[f].abs().mean())}
         for f in cols]
        + [{**tag, "level": "group", "name": g, "mean_abs": float(gs[g].abs().mean())}
           for g in groups]
    )
    group_of = {f: g for g, members in groups.items() for f in members}
    direction = [{**tag, "feature": f, "group": group_of[f],
                  "rho_feature": _rho(Xte[f].to_numpy(), phi[f].to_numpy()),
                  "rho_group": _rho(Xte[f].to_numpy(), gs[group_of[f]].to_numpy())}
                 for f in cols]
    permutation = [{**tag, **r}
                   for r in _permutation(pipe, Xte, y[test_idx], groups, seed * 100 + int(label))]
    ale = [{**tag, **r} for r in _ale(est, Xte, edges)]
    rows = None
    if seed == RECORD_SEED:
        rows = phi.assign(bias=c[:, -1], _idx=test_idx, fold=str(label))
    return rows, importance, direction, permutation, ale


def _fingerprint(df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                 groups: dict[str, list[str]], edges: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256(validation._fingerprint(df, X, y, MODEL, SCHEME).encode())
    settings = {"version": VERSION, "seeds": SEEDS, "repeats": N_REPEATS, "groups": groups,
                "ale": {k: v.tolist() for k, v in edges.items()}}
    h.update(json.dumps(settings, sort_keys=True).encode())
    return h.hexdigest()


def compute(task: str, df: pd.DataFrame | None = None, n_jobs: int = -1,
            refresh: bool = False) -> dict:
    """All attribution tables for one task's primary configuration, cached."""
    df = validation.frame("primary", df)
    cfg = config(task)
    cols = features.config(cfg)
    groups = features.attribution_groups(cfg)
    y = df[features.TASKS[task]["target"]].astype(int).to_numpy()
    X = df[cols].to_numpy(dtype=np.float64)
    edges = ale_edges(df, cols)

    out_dir = paths.INTERPRET_DIR / f"{task}__{cfg}"
    meta_path = out_dir / "meta.json"
    fp = _fingerprint(df, X, y, groups, edges)
    if not refresh and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp and all((out_dir / f"{t}.parquet").exists()
                                                 for t in TABLES):
            return {"meta": meta, **{t: pd.read_parquet(out_dir / f"{t}.parquet")
                                     for t in TABLES}}
        log.info("%s: cache fingerprint changed, recomputing", task)

    t0 = time.time()
    jobs = [(seed, label, tr, te) for seed in SEEDS
            for label, tr, te in validation.splits(df, y, SCHEME, seed=seed)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fold)(X, y, cols, groups, edges, seed, label, tr, te)
        for seed, label, tr, te in jobs
    )
    rows = pd.concat([r[0] for r in results if r[0] is not None]).sort_values("_idx")
    if len(rows) != len(df) or rows["_idx"].duplicated().any():
        raise AssertionError(f"{task}: every tract must be explained exactly once")
    idx = rows.pop("_idx").to_numpy()
    rows.insert(0, "GEOID", df["GEOID"].to_numpy()[idx])
    rows.insert(1, "ua", df["ua"].to_numpy()[idx])
    rows.insert(2, "y", y[idx])
    rows.insert(3, "score", sigmoid(rows[cols].sum(axis=1) + rows["bias"]))
    rows = rows.reset_index(drop=True)
    tables = {
        "rows": rows,
        "importance": pd.DataFrame([r for res in results for r in res[1]]),
        "direction": pd.DataFrame([r for res in results for r in res[2]]),
        "permutation": pd.DataFrame([r for res in results for r in res[3]]),
        "ale": pd.DataFrame([r for res in results for r in res[4]]),
    }

    # The explained model must be the evaluated model: the contributions of seed 37
    # reproduce the cached out-of-fold scores of `validation.oof()`.
    o = validation.oof(task, cfg, MODEL, SCHEME, df=df)
    m = rows[["GEOID", "fold", "score", "bias"]].merge(
        o[["GEOID", "fold", "score"]], on="GEOID", suffixes=("", "_oof"), validate="one_to_one")
    if (m["fold"] != m["fold_oof"]).any():
        raise AssertionError(f"{task}: fold assignment differs from the cached OOF run")
    logit_oof = np.log(m["score_oof"] / (1 - m["score_oof"]))
    logit = rows.set_index("GEOID").loc[m["GEOID"], cols].sum(axis=1).to_numpy() + m["bias"]
    additivity = {"n": int(len(m)),
                  "max_abs_error_probability": float((m["score"] - m["score_oof"]).abs().max()),
                  "max_abs_error_logit": float(np.abs(logit - logit_oof).max())}
    if additivity["max_abs_error_probability"] > ADDITIVITY_TOL:
        raise AssertionError(f"{task}: contributions do not reproduce the OOF scores "
                             f"({additivity['max_abs_error_probability']:.2e})")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_parquet(out_dir / f"{name}.parquet", index=False)
    meta = {"task": task, "config": cfg, "fingerprint": fp, "features": cols,
            "groups": groups, "n_fold_models": len(jobs), "seeds": list(SEEDS),
            "record_seed": RECORD_SEED, "scheme": SCHEME, "n_repeats": N_REPEATS,
            "ale_bins": ALE_BINS, "additivity": additivity,
            "seconds": round(time.time() - t0, 1)}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    log.info("%s: %d fold models, %.0f s", task, len(jobs), meta["seconds"])
    return {"meta": meta, **tables}


def linear_contributions(pipe, X: pd.DataFrame) -> np.ndarray:
    """Exact SHAP values of the fixed logistic pipeline in log-odds, n × (p + 1).

    Every preprocessing step (median imputation, `log1p`, standardisation) acts on one
    column at a time, so the model is linear in one transformed copy of each input.
    With the training fold as background, interventional SHAP is then
    φ_j = β_j · (z_j − mean_train z_j) = β_j · z_j, since the scaler centres the
    training fold; the bias (last column) is the intercept.
    """
    pre, est = pipe[:-1], pipe.named_steps["model"]
    z = np.asarray(pre.transform(X), dtype=float)
    beta = np.asarray(est.coef_, dtype=float).ravel()
    if z.shape != (len(X), X.shape[1]) or beta.shape != (X.shape[1],):
        raise AssertionError("the logistic pipeline must keep one column per input")
    return np.column_stack([z * beta, np.full(len(X), float(est.intercept_[0]))])


def _logistic_fold(X, y, cols, label, train_idx, test_idx):
    Xtr = pd.DataFrame(X[train_idx], columns=cols)
    Xte = pd.DataFrame(X[test_idx], columns=cols)
    pipe = models.make_model("logistic").fit(Xtr, y[train_idx])
    c = linear_contributions(pipe, Xte)
    return pd.DataFrame(c[:, :-1], columns=cols).assign(bias=c[:, -1], _idx=test_idx,
                                                         fold=str(label))


def logistic_rows(task: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Out-of-fold SHAP of the fixed logistic regression, shaped like the `rows` table.

    Same split as the LightGBM attribution of record (UA-grouped 5-fold, seed 37).
    The contributions must reproduce the cached out-of-fold scores of
    `validation.oof(task, config, "logistic", "ua5")`; that run must already be cached.
    """
    df = validation.frame("primary", df)
    cfg = config(task)
    cols = features.config(cfg)
    y = df[features.TASKS[task]["target"]].astype(int).to_numpy()
    X = df[cols].to_numpy(dtype=np.float64)
    key = validation.run_key(task, cfg, "logistic", SCHEME)
    meta_path = paths.OOF_DIR / f"{key}.json"
    fp = validation._fingerprint(df, X, y, "logistic", SCHEME)
    if not meta_path.exists() or json.loads(meta_path.read_text()).get("fingerprint") != fp:
        raise RuntimeError(f"{key}: not in the out-of-fold cache (run `make engine`)")

    parts = [_logistic_fold(X, y, cols, label, tr, te)
             for label, tr, te in validation.splits(df, y, SCHEME, seed=RECORD_SEED)]
    rows = pd.concat(parts).sort_values("_idx")
    if len(rows) != len(df) or rows["_idx"].duplicated().any():
        raise AssertionError(f"{task}: every tract must be explained exactly once")
    idx = rows.pop("_idx").to_numpy()
    rows.insert(0, "GEOID", df["GEOID"].to_numpy()[idx])
    rows.insert(1, "ua", df["ua"].to_numpy()[idx])
    rows.insert(2, "y", y[idx])
    rows.insert(3, "score", sigmoid(rows[cols].sum(axis=1) + rows["bias"]))
    rows = rows.reset_index(drop=True)

    o = validation.oof(task, cfg, "logistic", SCHEME, df=df)
    m = rows[["GEOID", "fold", "score"]].merge(o[["GEOID", "fold", "score"]], on="GEOID",
                                                suffixes=("", "_oof"), validate="one_to_one")
    if (m["fold"] != m["fold_oof"]).any():
        raise AssertionError(f"{task}: fold assignment differs from the cached logistic run")
    err = float((m["score"] - m["score_oof"]).abs().max())
    if err > ADDITIVITY_TOL:
        raise AssertionError(f"{task}: logistic contributions do not reproduce the OOF "
                             f"scores ({err:.2e})")
    return rows


def feature_values(rows: pd.DataFrame, cols: list[str],
                   df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Input values aligned with `rows`, whose feature-named columns hold SHAP values."""
    fr = validation.frame("primary", df).set_index("GEOID")
    return fr.loc[rows["GEOID"], cols].set_axis(rows.index)


def group_ranks(importance: pd.DataFrame) -> pd.DataFrame:
    """Rank of every group within each fold model (1 = largest mean |sum of SHAP|)."""
    g = importance[importance.level == "group"].copy()
    g["rank"] = g.groupby(["seed", "fold"])["mean_abs"].rank(ascending=False, method="min")
    return g


def pooled_ale(ale: pd.DataFrame, feature: str, seed: int | None = RECORD_SEED,
               fold: str | None = None) -> pd.DataFrame:
    """Accumulated, centred ALE curve (log-odds) at the bin upper edges."""
    a = ale[ale.feature == feature]
    if seed is not None:
        a = a[a.seed == seed]
    if fold is not None:
        a = a[a.fold == fold]
    b = a.groupby("bin").agg(lo=("lo", "first"), hi=("hi", "first"), n=("n", "sum"),
                             sum_diff=("sum_diff", "sum")).reset_index()
    b["effect"] = np.where(b["n"] > 0, b["sum_diff"] / b["n"].clip(lower=1), 0.0)
    b["ale"] = b["effect"].cumsum()
    # centre so that the n-weighted mean over bin midpoints is zero
    mid = (b["ale"] + b["ale"].shift(fill_value=0.0)) / 2
    b["ale"] -= np.average(mid, weights=b["n"]) if b["n"].sum() else 0.0
    return b


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fill the attribution cache")
    parser.add_argument("--only", choices=TASKS, default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = validation.load_analysis()
    for task in [args.only] if args.only else TASKS:
        compute(task, df=df, refresh=args.refresh)


if __name__ == "__main__":
    main()
