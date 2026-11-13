"""Validation splitters and the cached out-of-fold (OOF) engine.

Every learned step is fitted inside the training fold: the model factories in
`models.py` are whole pipelines and are refitted per fold. Folds run in parallel
with joblib; estimators are single-threaded.

OOF predictions are cached at `data/interim/oof/<key>.parquet` with a sidecar
JSON holding a fingerprint of the exact inputs (rows, labels, features, groups,
model settings, library versions). A cache file whose fingerprint does not
match is recomputed, so a rebuilt sample can never be scored with stale
predictions.

`uv run python -m uhi_poverty.validation` (or `make engine`) fills the cache for
every run the analysis scripts need.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

from . import features, models, paths

log = logging.getLogger("uhi_poverty.validation")

SEED = 37
SCHEMES = ("random5", "county5", "ua5", "loco", "state", "region", "major_ua")
GROUPED = ("county5", "ua5", "loco", "state", "region", "major_ua")
SCHEME_LABELS = {
    "random5": "Random 5-fold (reference)",
    "county5": "County-grouped 5-fold",
    "ua5": "UA-grouped 5-fold",
    "loco": "Leave-one-city-out",
    "state": "Leave-one-state-out (UA-buffered)",
    "region": "Leave-one-region-out (UA-buffered)",
    "major_ua": "Each of the 20 largest UAs held out",
}
N_MAJOR_UA = 20

# --------------------------------------------------------------------------- data


def load_analysis() -> pd.DataFrame:
    return pd.read_parquet(paths.ANALYSIS_PARQUET)


def frame(name: str = "primary", df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows of a named frame, in a stable order (by GEOID), index reset."""
    df = load_analysis() if df is None else df
    mask = {"primary": "in_primary", "lpi": "in_lpi_frame"}[name]
    return df[df[mask]].sort_values("GEOID").reset_index(drop=True)


def top_uas(df: pd.DataFrame, k: int) -> list[str]:
    counts = df.groupby("ua").size().reset_index(name="n")
    counts = counts.sort_values(["n", "ua"], ascending=[False, True])
    return counts["ua"].head(k).tolist()


# --------------------------------------------------------------------------- splitters


def _buffered(df: pd.DataFrame, by: str):
    for value in sorted(df[by].unique()):
        test = np.flatnonzero(df[by].to_numpy() == value)
        held_uas = set(df["ua"].to_numpy()[test])
        train = np.flatnonzero(~df["ua"].isin(held_uas).to_numpy())
        yield value, train, test


def splits(df: pd.DataFrame, y: np.ndarray, scheme: str, seed: int = SEED):
    """List of (fold_label, train_idx, test_idx)."""
    n = len(df)
    idx = np.arange(n)
    if scheme == "random5":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return [(f, tr, te) for f, (tr, te) in enumerate(cv.split(idx, y))]
    if scheme in ("county5", "ua5"):
        groups = df["county_fips" if scheme == "county5" else "ua"].to_numpy()
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        return [(f, tr, te) for f, (tr, te) in enumerate(cv.split(idx, y, groups))]
    if scheme == "loco":
        groups = df["ua"].to_numpy()
        out = []
        for tr, te in LeaveOneGroupOut().split(idx, y, groups):
            out.append((groups[te[0]], tr, te))
        return out
    if scheme == "state":
        return list(_buffered(df, "state_fips"))
    if scheme == "region":
        return list(_buffered(df, "region"))
    if scheme == "major_ua":
        ua = df["ua"].to_numpy()
        return [(u, np.flatnonzero(ua != u), np.flatnonzero(ua == u))
                for u in top_uas(df, N_MAJOR_UA)]
    raise KeyError(scheme)


# --------------------------------------------------------------------------- engine


def _fit_fold(model: str, X: np.ndarray, y: np.ndarray, columns: list[str], train_idx,
              test_idx):
    Xtr = pd.DataFrame(X[train_idx], columns=columns)
    Xte = pd.DataFrame(X[test_idx], columns=columns)
    return models.fit_predict(model, Xtr, y[train_idx], Xte)


def run_key(task: str, config: str, model: str, scheme: str, drop_top: int = 0) -> str:
    key = f"{task}__{config}__{model}__{scheme}"
    return key + (f"__drop-top{drop_top}" if drop_top else "")


def _fingerprint(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, model: str,
                 scheme: str) -> str:
    h = hashlib.sha256()
    h.update("\n".join(df["GEOID"]).encode())
    h.update(np.ascontiguousarray(y, dtype=np.int8).tobytes())
    h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    h.update("\n".join(df["ua"].astype(str)).encode())
    h.update("\n".join(df["state_fips"] + df["county_fips"] + df["region"]).encode())
    settings = {"model": model, "scheme": scheme, "seed": SEED, "lgbm": models.LGBM_PARAMS,
                "logistic": models.LOGISTIC_PARAMS, "sklearn": sklearn.__version__,
                "lightgbm": lightgbm.__version__}
    h.update(json.dumps(settings, sort_keys=True).encode())
    return h.hexdigest()


def oof(task: str, config: str, model: str, scheme: str, drop_top: int = 0,
        df: pd.DataFrame | None = None, n_jobs: int = -1, refresh: bool = False) -> pd.DataFrame:
    """Out-of-fold predictions for one (task, config, model, scheme), cached."""
    df = frame("primary", df)
    if drop_top:
        df = df[~df["ua"].isin(top_uas(df, drop_top))].reset_index(drop=True)
    target = features.TASKS[task]["target"]
    cols = features.config(config)
    y = df[target].astype(int).to_numpy()
    X = df[cols].to_numpy(dtype=np.float64)

    key = run_key(task, config, model, scheme, drop_top)
    out_path = paths.OOF_DIR / f"{key}.parquet"
    meta_path = out_path.with_suffix(".json")
    fp = _fingerprint(df, X, y, model, scheme)
    if not refresh and out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp:
            return pd.read_parquet(out_path)
        log.info("%s: cache fingerprint changed, recomputing", key)

    fold_list = splits(df, y, scheme)
    t0 = time.time()
    if model == "majority":
        results = [_fit_fold(model, X, y, cols, tr, te) for _, tr, te in fold_list]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_fold)(model, X, y, cols, tr, te) for _, tr, te in fold_list
        )
    score = np.full(len(df), np.nan)
    fold = np.empty(len(df), dtype=object)
    n_tested = np.zeros(len(df), dtype=int)
    for (label, _, te), s in zip(fold_list, results, strict=True):
        score[te] = s
        fold[te] = str(label)
        n_tested[te] += 1
    if (n_tested > 1).any():
        raise AssertionError(f"{key}: a tract was tested more than once")
    keep = n_tested == 1
    out = pd.DataFrame({
        "GEOID": df["GEOID"], "ua": df["ua"], "state_fips": df["state_fips"],
        "region": df["region"], "fold": fold, "y": y, "score": score,
    })[keep].reset_index(drop=True)
    out["pred"] = (out["score"] >= models.THRESHOLD).astype(int)
    paths.OOF_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    meta = {"key": key, "fingerprint": fp, "n_rows": int(len(out)), "n_folds": len(fold_list),
            "features": cols, "seconds": round(time.time() - t0, 1)}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    log.info("%s: %d folds, %d rows, %.0f s", key, len(fold_list), len(out), meta["seconds"])
    return out


# --------------------------------------------------------------------------- planned runs

PRIMARY_RUNS = [(t, features.TASKS[t]["primary"], m, s)
                for t in ("A", "B") for m in models.MODELS for s in SCHEMES]
ABLATION_CONFIGS = ["A-full", "A-narrow", "A-noLST-keepUHI", "A-noLST", "A-noLST+pov",
                    "A-lst-only", "A-lst"]
ABLATION_SCHEMES = ("loco", "ua5", "random5")
ABLATION_RUNS = [("A", c, m, s) for c in ABLATION_CONFIGS for m in ("logistic", "lgbm")
                 for s in ABLATION_SCHEMES]
BASELINE_RUNS = [("B", c, m, s) for c in ("geo-only", "B-nonthermal")
                 for m in ("logistic", "lgbm") for s in ("loco", "ua5")]
METRO_DROPS = (1, 3, 10)
METRO_RUNS = [(t, features.TASKS[t]["primary"], m, "ua5", k)
              for t in ("A", "B") for m in ("logistic", "lgbm") for k in METRO_DROPS]


def planned_runs() -> list[tuple]:
    runs = [(*r, 0) for r in PRIMARY_RUNS + ABLATION_RUNS + BASELINE_RUNS] + METRO_RUNS
    seen, unique = set(), []
    for r in runs:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    # cheap runs first; leave-one-city-out LightGBM last
    cost = {"majority": 0, "logistic": 1, "lgbm": 2}
    return sorted(unique, key=lambda r: (r[3] == "loco", cost[r[2]]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fill the out-of-fold prediction cache")
    parser.add_argument("--only", default=None, help="substring filter on the run key")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_analysis()
    runs = planned_runs()
    if args.only:
        runs = [r for r in runs if args.only in run_key(*r)]
    log.info("%d runs planned", len(runs))
    for task, config, model, scheme, drop in runs:
        oof(task, config, model, scheme, drop_top=drop, df=df, refresh=args.refresh)


if __name__ == "__main__":
    main()
