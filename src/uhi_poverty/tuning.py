"""Nested tuning of preprocessing, hyperparameters and threshold, and the soft-voting ensemble.

Used by section 4.6. The preprocessing, the hyperparameters and the decision
threshold are all chosen on training cities only:

1. **Outer folds.** The UA-grouped 5-fold of `validation.splits` (seed 37). For
   outer fold k, an inner UA-grouped 5-fold over the training cities scores each
   Optuna trial by its pooled inner out-of-fold ROC-AUC. The best trial's
   preprocessing and parameters, and the thresholds τ_acc and τ_F1 chosen on its
   inner out-of-fold predictions, are fold k's settings. The model is refitted on
   all training cities and scored on fold k.
2. **Leave-one-city-out.** Each city lies in exactly one outer test fold k(c).
   It is scored by a model fitted on every other city with the settings of
   fold k(c), which were chosen without it.
3. **Random 5-fold** repeats step 1 with random inner and outer folds. It is a
   reference for random cross-validation, never a headline.

The ensembles average the three tuned boosters' probabilities, with equal
weights or with weights chosen on the same inner out-of-fold predictions.

Results are cached under `data/interim/tuned/`: per (task, model, tuning
scheme) a settings JSON and the inner out-of-fold scores, and per (task, model,
scheme) the outer predictions, each with a fingerprint of everything that went
into them. `uv run python -m uhi_poverty.tuning` (or `make tune`) fills the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import time

import catboost
import imblearn
import lightgbm
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost
from joblib import Parallel, delayed
from optuna.trial import TrialState
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import features, models, paths, validation

log = logging.getLogger("uhi_poverty.tuning")

SEED = 37
VERSION = 1  # bump when the search space or the procedure changes
N_TRIALS = {"logistic": 40, "lgbm": 60, "xgb": 60, "catboost": 40}
BATCH = 3  # trials asked from each fold's study per round (TPE with constant liar)
INNER_FOLDS = 5
TAU_GRID = np.round(np.linspace(0.01, 0.99, 99), 4)  # threshold grid
WEIGHT_STEP = 0.05
TUNE_SCHEME = {"ua5": "ua5", "loco": "ua5", "random5": "random5"}  # evaluation -> tuning
SCHEMES = ("ua5", "loco", "random5")
ALL_MODELS = models.TUNED_MODELS + models.ENSEMBLES

# Round 2. Every change follows from round 1's *inner*
# cross-validation only, never from its outer scores: the low-variance filter and
# PCA lost in every model and task and are dropped; scaling cannot change a tree
# without PCA, so boosters get none; parameters that the search pushed against
# the edge of their range get a wider range; there are more trials; each outer
# refit is averaged over five seeds; and a stacked ensemble (a logistic
# regression on the members' inner out-of-fold logits) joins the soft votes.
N_TRIALS_R2 = {"logistic": 60, "lgbm": 150, "xgb": 150, "catboost": 100}
BAG_SEEDS = {1: (SEED,), 2: (37, 38, 39, 40, 41)}
ENSEMBLES_R2 = models.ENSEMBLES + ("ens_stacked",)
ALL_MODELS_R2 = models.TUNED_MODELS + ENSEMBLES_R2
SCHEMES_R2 = ("ua5", "loco")


def n_trials_for(model: str, rnd: int = 1) -> int | None:
    return (N_TRIALS if rnd == 1 else N_TRIALS_R2).get(model)


optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------------------------------- search space


def suggest(trial, model: str, rnd: int = 1) -> tuple[dict, dict]:
    """(preprocessing, model parameters) for one trial."""
    if rnd == 2:
        return suggest_r2(trial, model)
    prep = {
        "engineer": trial.suggest_categorical("engineer", [False, True]),
        "variance": trial.suggest_categorical("variance", [False, True]),
        "corr": trial.suggest_categorical("corr", [False, True]),
        "scaler": trial.suggest_categorical("scaler", ["none", "minmax", "standard"]),
        "pca": trial.suggest_categorical("pca", ["none", "0.95", "0.99"]),
        "smote": trial.suggest_categorical("smote", [False, True]),
    }
    if model == "logistic":
        params = {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
    elif model == "lgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
    elif model == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    elif model == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }
    else:
        raise KeyError(model)
    return prep, params


def suggest_r2(trial, model: str) -> tuple[dict, dict]:
    """Round 2: the options inner CV never favoured are fixed off; edge-hit ranges widened."""
    prep = {
        "engineer": trial.suggest_categorical("engineer", [False, True]),
        "variance": False,
        "corr": trial.suggest_categorical("corr", [False, True]),
        "scaler": (trial.suggest_categorical("scaler", ["minmax", "standard"])
                   if model == "logistic" else "none"),
        "pca": "none",
        "smote": trial.suggest_categorical("smote", [False, True]),
    }
    if model == "logistic":
        params = {"C": trial.suggest_float("C", 1e-4, 100.0, log=True)}
    elif model == "lgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 3000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 800, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 100.0, log=True),
        }
    elif model == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 15.0),
        }
    elif model == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 200, 2500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 200.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 15.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        }
    else:
        raise KeyError(model)
    return prep, params


# --------------------------------------------------------------------------- folds


def inner_splits(df: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, scheme: str,
                 seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inner (train, validation) splits of the outer training rows, as absolute row indices."""
    train_idx = np.asarray(train_idx)
    sub_y = y[train_idx]
    pos = np.arange(len(train_idx))
    if scheme == "ua5":
        cv = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
        it = cv.split(pos, sub_y, df["ua"].to_numpy()[train_idx])
    elif scheme == "random5":
        cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
        it = cv.split(pos, sub_y)
    else:
        raise KeyError(scheme)
    return [(train_idx[a], train_idx[b]) for a, b in it]


def outer_fold_of_rows(n: int, outer) -> np.ndarray:
    """For every row, the index k of the outer fold that tests it."""
    k_of = np.full(n, -1, dtype=int)
    for k, (_, _, te) in enumerate(outer):
        k_of[te] = k
    if (k_of < 0).any():
        raise AssertionError("an outer fold partition leaves rows untested")
    return k_of


def evaluation_folds(df: pd.DataFrame, y: np.ndarray, scheme: str):
    """(label, train_idx, test_idx, k): k is the outer tuning fold whose settings apply."""
    outer = validation.splits(df, y, TUNE_SCHEME[scheme])
    if scheme != "loco":
        return [(label, tr, te, k) for k, (label, tr, te) in enumerate(outer)]
    k_of = outer_fold_of_rows(len(df), outer)
    out = []
    for label, tr, te in validation.splits(df, y, "loco"):
        ks = np.unique(k_of[te])
        if len(ks) != 1:
            raise AssertionError(f"city {label} is split across outer folds")
        out.append((label, tr, te, int(ks[0])))
    return out


# --------------------------------------------------------------------------- thresholds


def choose_tau(y: np.ndarray, score: np.ndarray, metric: str) -> float:
    """The grid threshold that maximises accuracy or F1 (first maximum)."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(score)[None, :] >= TAU_GRID[:, None]
    tp = (pred & y).sum(axis=1)
    fp = (pred & ~y).sum(axis=1)
    fn = (~pred & y).sum(axis=1)
    if metric == "accuracy":
        value = (tp + (~pred & ~y).sum(axis=1)) / len(y)
    elif metric == "f1":
        value = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    else:
        raise KeyError(metric)
    return float(TAU_GRID[int(np.argmax(value))])


# --------------------------------------------------------------------------- fitting


def _design(df: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = features.config(features.TASKS[task]["primary"])
    X = df[cols].to_numpy(dtype=np.float64)
    y = df[features.TASKS[task]["target"]].astype(int).to_numpy()
    return X, y, cols


def _fit_score(task: str, model: str, prep: dict, params: dict, X: np.ndarray, y: np.ndarray,
               cols: list[str], train_idx, test_idx, seeds: tuple[int, ...] = (SEED,)):
    """Test-row probabilities, or None if this combination of options cannot be fitted."""
    try:
        return models.fit_predict_tuned(task, model, prep, params,
                                        pd.DataFrame(X[train_idx], columns=cols), y[train_idx],
                                        pd.DataFrame(X[test_idx], columns=cols), seeds)
    except Exception as err:  # e.g. the filters leave no column for PCA
        log.warning("%s %s: fit failed (%s): %s", task, model, type(err).__name__, err)
        return None


def tune_folds(task: str, model: str, df: pd.DataFrame, tune_scheme: str,
               n_trials: int | None = None, n_jobs: int = -1,
               fixed_prep: dict | None = None, outer: list | None = None,
               rnd: int = 1) -> list[dict]:
    """Optuna search in every outer fold at once; returns one settings dict per outer fold.

    Each study only ever sees its outer fold's training rows: the inner splits are
    drawn from them, and τ is chosen on the inner out-of-fold scores. `outer`
    overrides the outer folds; the final models pass one fold holding every row.
    """
    X, y, cols = _design(df, task)
    n_trials = n_trials or n_trials_for(model, rnd)
    outer = outer if outer is not None else validation.splits(df, y, tune_scheme)
    inner = [inner_splits(df, y, tr, tune_scheme) for _, tr, _ in outer]
    studies = [optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=SEED + k, constant_liar=True)) for k in range(len(outer))]
    best = [{"auc": -np.inf} for _ in outer]
    history: list[list[dict]] = [[] for _ in outer]
    done = [0] * len(outer)
    t0 = time.time()
    with Parallel(n_jobs=n_jobs) as par:
        while min(done) < n_trials:
            asked = []
            for k, study in enumerate(studies):
                for _ in range(min(BATCH, n_trials - done[k])):
                    trial = study.ask()
                    prep, params = suggest(trial, model, rnd)
                    if fixed_prep:
                        prep.update(fixed_prep)
                    asked.append((k, trial, prep, params))
                    done[k] += 1
            jobs = [(i, f) for i, (k, *_) in enumerate(asked) for f in range(len(inner[k]))]
            scores = par(delayed(_fit_score)(task, model, asked[i][2], asked[i][3], X, y, cols,
                                             *inner[asked[i][0]][f]) for i, f in jobs)
            by_trial: dict[int, list] = {}
            for (i, f), s in zip(jobs, scores, strict=True):
                by_trial.setdefault(i, [None] * INNER_FOLDS)[f] = s
            for i, (k, trial, prep, params) in enumerate(asked):
                parts = by_trial[i]
                if any(p is None for p in parts):
                    studies[k].tell(trial, state=TrialState.FAIL)
                    history[k].append({"trial": trial.number, "auc": None, **prep, **params})
                    continue
                rows = np.concatenate([va for _, va in inner[k]])
                score = np.concatenate(parts)
                auc = float(roc_auc_score(y[rows], score))
                studies[k].tell(trial, auc)
                history[k].append({"trial": trial.number, "auc": auc, **prep, **params})
                if auc > best[k]["auc"]:
                    best[k] = {"auc": auc, "prep": prep, "params": params, "trial": trial.number,
                               "rows": rows, "score": score}
            log.info("%s %s %s: %d/%d trials per fold, best inner AUC %s, %.0f s", task, model,
                     tune_scheme, min(done), n_trials,
                     " ".join(f"{b['auc']:.4f}" for b in best), time.time() - t0)
    settings = []
    for k, (label, _, te) in enumerate(outer):
        b = best[k]
        if "rows" not in b:
            raise RuntimeError(f"{task} {model} {tune_scheme} fold {k}: every trial failed")
        if np.intersect1d(b["rows"], te).size:
            raise AssertionError("inner validation rows overlap the outer test fold")
        settings.append({
            "k": k, "label": str(label), "inner_auc": b["auc"], "trial": b["trial"],
            "prep": b["prep"], "params": b["params"],
            "tau_acc": choose_tau(y[b["rows"]], b["score"], "accuracy"),
            "tau_f1": choose_tau(y[b["rows"]], b["score"], "f1"),
            "inner_rows": b["rows"].tolist(), "inner_score": b["score"].tolist(),
            "history": history[k],
        })
    return settings


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ensemble_settings(booster_settings: dict[str, list[dict]], y: np.ndarray,
                      stacked: bool = False) -> dict:
    """Per outer fold: equal and inner-tuned soft-voting weights, and their thresholds.

    With `stacked`, also a logistic regression on the members' inner out-of-fold
    logits (four coefficients, fitted on tens of thousands of inner rows).
    """
    grid = [w for w in itertools.product(np.arange(0, 1 + 1e-9, WEIGHT_STEP), repeat=2)
            if w[0] + w[1] <= 1 + 1e-9]
    weights_grid = np.array([(a, b, 1 - a - b) for a, b in grid]).clip(0, 1)
    out = {"ens_equal": [], "ens_weighted": []}
    if stacked:
        out["ens_stacked"] = []
    n_folds = len(next(iter(booster_settings.values())))
    for k in range(n_folds):
        members = [booster_settings[m][k] for m in models.BOOSTERS]
        rows = np.asarray(members[0]["inner_rows"])
        for s in members[1:]:
            if not np.array_equal(np.asarray(s["inner_rows"]), rows):
                raise AssertionError("boosters were tuned on different inner splits")
        P = np.column_stack([s["inner_score"] for s in members])
        yk = y[rows]
        aucs = [roc_auc_score(yk, P @ w) for w in weights_grid]
        choice = {"ens_equal": np.full(len(members), 1 / len(members)),
                  "ens_weighted": weights_grid[int(np.argmax(aucs))]}
        for name, w in choice.items():
            s = P @ w
            out[name].append({
                "k": k, "weights": dict(zip(models.BOOSTERS, map(float, w), strict=True)),
                "inner_auc": float(roc_auc_score(yk, s)),
                "tau_acc": choose_tau(yk, s, "accuracy"), "tau_f1": choose_tau(yk, s, "f1"),
            })
        if stacked:
            meta = LogisticRegression(C=1.0, max_iter=1000).fit(_logit(P), yk)
            s = meta.predict_proba(_logit(P))[:, 1]
            out["ens_stacked"].append({
                "k": k, "weights": dict(zip(models.BOOSTERS, map(float, meta.coef_[0]),
                                            strict=True)),
                "intercept": float(meta.intercept_[0]),
                "inner_auc": float(roc_auc_score(yk, s)),
                "tau_acc": choose_tau(yk, s, "accuracy"), "tau_f1": choose_tau(yk, s, "f1"),
            })
    return out


def predict_folds(task: str, model: str, df: pd.DataFrame, scheme: str, settings: list[dict],
                  n_jobs: int = -1, seeds: tuple[int, ...] = (SEED,)) -> pd.DataFrame:
    """Outer predictions of one tuned model, with each fold's own settings and thresholds."""
    X, y, cols = _design(df, task)
    folds = evaluation_folds(df, y, scheme)
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(task, model, settings[k]["prep"], settings[k]["params"], X, y, cols,
                            tr, te, seeds) for _, tr, te, k in folds)
    score = np.full(len(df), np.nan)
    fold = np.empty(len(df), dtype=object)
    k_row = np.full(len(df), -1, dtype=int)
    for (label, _, te, k), s in zip(folds, scores, strict=True):
        if s is None:
            raise RuntimeError(f"{task} {model} {scheme}: refit failed in fold {label}")
        score[te], fold[te], k_row[te] = s, str(label), k
    return _frame(df, y, fold, k_row, score, settings)


def _frame(df, y, fold, k_row, score, settings) -> pd.DataFrame:
    tau_acc = np.array([settings[k]["tau_acc"] for k in k_row])
    tau_f1 = np.array([settings[k]["tau_f1"] for k in k_row])
    out = pd.DataFrame({
        "GEOID": df["GEOID"], "ua": df["ua"], "state_fips": df["state_fips"],
        "region": df["region"], "fold": fold, "settings_fold": k_row, "y": y, "score": score,
        "tau_acc": tau_acc, "tau_f1": tau_f1,
    })
    if out["score"].isna().any():
        raise AssertionError("a tract was never scored")
    out["pred"] = (out["score"] >= out["tau_acc"]).astype(int)  # the headline threshold
    out["pred_f1"] = (out["score"] >= out["tau_f1"]).astype(int)
    out["pred_05"] = (out["score"] >= models.THRESHOLD).astype(int)
    return out


def combine(member_preds: dict[str, pd.DataFrame], settings: list[dict]) -> pd.DataFrame:
    """Soft vote of the boosters' outer predictions with each fold's weights and thresholds."""
    base = member_preds[models.BOOSTERS[0]]
    for m in models.BOOSTERS[1:]:
        if not (member_preds[m]["GEOID"].equals(base["GEOID"])
                and member_preds[m]["settings_fold"].equals(base["settings_fold"])):
            raise AssertionError("member predictions are not aligned")
    k_row = base["settings_fold"].to_numpy()
    w = np.array([[settings[k]["weights"][m] for m in models.BOOSTERS] for k in k_row])
    P = np.column_stack([member_preds[m]["score"].to_numpy() for m in models.BOOSTERS])
    if "intercept" in settings[0]:  # stacked: logistic regression on the members' logits
        b = np.array([settings[k]["intercept"] for k in k_row])
        score = 1 / (1 + np.exp(-(b + (_logit(P) * w).sum(axis=1))))
    else:
        score = (P * w).sum(axis=1)
    df = base[["GEOID", "ua", "state_fips", "region"]]
    return _frame(df, base["y"].to_numpy(), base["fold"].to_numpy(), k_row, score, settings)


# --------------------------------------------------------------------------- cache


def _versions() -> dict:
    return {"sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__, "catboost": catboost.__version__,
            "optuna": optuna.__version__, "imblearn": imblearn.__version__}


def data_hash(df: pd.DataFrame, task: str) -> str:
    X, y, cols = _design(df, task)
    h = hashlib.sha256()
    h.update("\n".join(df["GEOID"]).encode())
    h.update(np.ascontiguousarray(y, dtype=np.int8).tobytes())
    h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    h.update("\n".join(cols).encode())
    h.update("\n".join(df["ua"].astype(str)).encode())
    return h.hexdigest()


def tuning_fingerprint(df: pd.DataFrame, task: str, model: str, tune_scheme: str,
                       rnd: int = 1) -> str:
    spec = {"version": VERSION, "task": task, "model": model, "scheme": tune_scheme,
            "seed": SEED, "batch": BATCH, "inner_folds": INNER_FOLDS,
            "n_trials": n_trials_for(model, rnd), "data": data_hash(df, task), **_versions()}
    if rnd > 1:  # round 1's fingerprint is left exactly as it was, so its cache stays valid
        spec.update({"round": rnd, "bag_seeds": BAG_SEEDS[rnd]})
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def settings_hash(settings: list[dict]) -> str:
    """Hash of what a prediction depends on: each fold's options, parameters, weights and τ."""
    keep = [{k: s.get(k) for k in ("prep", "params", "weights", "tau_acc", "tau_f1")}
            for s in settings]
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=float).encode()).hexdigest()


def _round_tag(rnd: int) -> str:
    return "" if rnd == 1 else f"__r{rnd}"


def _settings_path(task, model, tune_scheme, rnd: int = 1):
    return paths.TUNED_DIR / f"{task}__{model}{_round_tag(rnd)}__tune-{tune_scheme}.json"


def _pred_path(task, model, scheme, rnd: int = 1):
    return paths.TUNED_DIR / f"{task}__{model}{_round_tag(rnd)}__{scheme}.parquet"


def _read_cached(path, fp):
    if path.exists():
        meta = json.loads(path.read_text())
        if meta.get("fingerprint") == fp:
            return meta
    return None


def tuned_settings(task: str, model: str, tune_scheme: str, df: pd.DataFrame | None = None,
                   n_jobs: int = -1, refresh: bool = False, rnd: int = 1) -> list[dict]:
    """Per-outer-fold settings of one tuned model or ensemble, cached."""
    df = validation.frame("primary", df)
    if model in ENSEMBLES_R2:
        ensembles = models.ENSEMBLES if rnd == 1 else ENSEMBLES_R2
        members = {m: tuned_settings(task, m, tune_scheme, df, n_jobs, rnd=rnd)
                   for m in models.BOOSTERS}
        fp = hashlib.sha256((tune_scheme + _round_tag(rnd)
                             + "".join(settings_hash(members[m])
                                       for m in models.BOOSTERS)).encode()).hexdigest()
        path = _settings_path(task, model, tune_scheme, rnd)
        meta = None if refresh else _read_cached(path, fp)
        if meta is None:
            _, y, _ = _design(df, task)
            ens = ensemble_settings(members, y, stacked=rnd > 1)
            for name in ensembles:
                _write_json(_settings_path(task, name, tune_scheme, rnd),
                            {"fingerprint": fp, "settings": ens[name]})
            meta = {"settings": ens[model]}
        return meta["settings"]
    fp = tuning_fingerprint(df, task, model, tune_scheme, rnd)
    path = _settings_path(task, model, tune_scheme, rnd)
    meta = None if refresh else _read_cached(path, fp)
    if meta is None:
        t0 = time.time()
        settings = tune_folds(task, model, df, tune_scheme, n_jobs=n_jobs, rnd=rnd)
        meta = {"fingerprint": fp, "seconds": round(time.time() - t0, 1), "settings": settings}
        _write_json(path, meta)
    return meta["settings"]


def predictions(task: str, model: str, scheme: str, df: pd.DataFrame | None = None,
                n_jobs: int = -1, refresh: bool = False, rnd: int = 1) -> pd.DataFrame:
    """Outer predictions of one tuned model or ensemble under one scheme, cached."""
    df = validation.frame("primary", df)
    tune_scheme = TUNE_SCHEME[scheme]
    settings = tuned_settings(task, model, tune_scheme, df, n_jobs, rnd=rnd)
    # an ensemble's predictions also depend on every member's settings
    members = models.BOOSTERS if model in ENSEMBLES_R2 else ()
    fp = hashlib.sha256((tuning_fingerprint(df, task, model, tune_scheme, rnd) + scheme
                         + settings_hash(settings)
                         + "".join(settings_hash(tuned_settings(task, m, tune_scheme, df, n_jobs,
                                                                rnd=rnd))
                                   for m in members)).encode()).hexdigest()
    path = _pred_path(task, model, scheme, rnd)
    meta_path = path.with_suffix(".json")
    if not refresh and path.exists() and _read_cached(meta_path, fp) is not None:
        return pd.read_parquet(path)
    t0 = time.time()
    if model in ENSEMBLES_R2:
        members = {m: predictions(task, m, scheme, df, n_jobs, rnd=rnd) for m in models.BOOSTERS}
        out = combine(members, settings)
    else:
        out = predict_folds(task, model, df, scheme, settings, n_jobs, seeds=BAG_SEEDS[rnd])
    paths.TUNED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    _write_json(meta_path, {"fingerprint": fp, "n_rows": len(out),
                            "n_folds": int(out["fold"].nunique()),
                            "seconds": round(time.time() - t0, 1)})
    log.info("%s %s%s %s: %d folds, %.0f s", task, model, _round_tag(rnd), scheme,
             out["fold"].nunique(), time.time() - t0)
    return out


def _write_json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n")


# --------------------------------------------------------------------------- final models
#
# The nested runs above estimate how well the *procedure* does in unseen cities. A
# model to hand on is that same procedure applied once to every city: inner
# UA-grouped 5-fold over all 54,243 tracts chooses the options, parameters and τ,
# and the pipeline is refitted on all of them. It has no held-out score of its
# own; its expected performance is the nested leave-one-city-out estimate.


def final_settings(task: str, model: str, df: pd.DataFrame | None = None, n_jobs: int = -1,
                   refresh: bool = False) -> dict:
    """Settings chosen by inner UA-grouped CV over every tract, cached."""
    df = validation.frame("primary", df)
    if model in models.ENSEMBLES:
        members = {m: [final_settings(task, m, df, n_jobs)] for m in models.BOOSTERS}
        _, y, _ = _design(df, task)
        return ensemble_settings(members, y)[model][0]
    fp = tuning_fingerprint(df, task, model, "all")
    path = _settings_path(task, model, "all")
    meta = None if refresh else _read_cached(path, fp)
    if meta is None:
        t0 = time.time()
        everything = [("all", np.arange(len(df)), np.array([], dtype=int))]
        settings = tune_folds(task, model, df, "ua5", n_jobs=n_jobs, outer=everything)
        meta = {"fingerprint": fp, "seconds": round(time.time() - t0, 1), "settings": settings}
        _write_json(path, meta)
    return meta["settings"][0]


def fit_final(task: str, model: str, df: pd.DataFrame | None = None,
              n_jobs: int = -1) -> models.FinalModel:
    """The tuned model (or ensemble) refitted on every tract."""
    df = validation.frame("primary", df)
    X, y, cols = _design(df, task)
    Xdf = pd.DataFrame(X, columns=cols)
    s = final_settings(task, model, df, n_jobs)
    if model in models.ENSEMBLES:
        names = list(models.BOOSTERS)
        fitted = Parallel(n_jobs=min(n_jobs if n_jobs > 0 else 3, 3))(
            delayed(_fit_full)(task, m, final_settings(task, m, df), Xdf, y) for m in names)
        estimator = models.SoftVote(dict(zip(names, fitted, strict=True)), s["weights"])
        detail = {m: {k: final_settings(task, m, df)[k] for k in ("prep", "params")}
                  for m in names}
        detail["weights"] = s["weights"]
    else:
        estimator = _fit_full(task, model, s, Xdf, y)
        detail = {"prep": s["prep"], "params": s["params"]}
    card = {"task": task, "target": features.TASKS[task]["target"],
            "config": features.TASKS[task]["primary"], "model": model,
            "label": models.TUNED_LABELS[model], "n_train": int(len(y)),
            "positives": int(y.sum()), "inner_roc_auc": s["inner_auc"],
            "tau_acc": s["tau_acc"], "tau_f1": s["tau_f1"], "settings": detail,
            "versions": _versions(), "tuning_version": VERSION}
    return models.FinalModel(task, model, estimator, cols, s["tau_acc"], s["tau_f1"], card)


def _fit_full(task, model, s, Xdf, y):
    return models.make_tuned_pipeline(task, model, s["prep"], s["params"]).fit(Xdf, y)


# --------------------------------------------------------------------------- runs


def planned_runs(rnd: int = 1) -> list[tuple[str, str, str]]:
    """(task, model, scheme): the held-out-city schemes first, the random reference last."""
    if rnd == 2:  # round 1 already reports the random-fold reference
        return [(t, m, s) for s in SCHEMES_R2 for t in ("A", "B") for m in ALL_MODELS_R2]
    return [(t, m, s) for s in SCHEMES for t in ("A", "B") for m in ALL_MODELS]


def pilot(n_trials: int = 20, n_jobs: int = -1) -> pd.DataFrame:
    """Task B LightGBM under UA-grouped 5-fold, with and without the engineered terms."""
    df = validation.frame("primary")
    fixed = validation.oof("B", "B-env", "lgbm", "ua5", df=df)
    rows = [{"variant": "fixed settings, tau = 0.5",
             "roc_auc": roc_auc_score(fixed["y"], fixed["score"]),
             "accuracy": float((fixed["pred"] == fixed["y"]).mean())}]
    for engineer in (False, True):
        settings = tune_folds("B", "lgbm", df, "ua5", n_trials=n_trials, n_jobs=n_jobs,
                              fixed_prep={"engineer": engineer})
        o = predict_folds("B", "lgbm", df, "ua5", settings, n_jobs=n_jobs)
        rows.append({"variant": f"tuned, engineer={engineer}, tau_acc",
                     "roc_auc": roc_auc_score(o["y"], o["score"]),
                     "accuracy": float((o["pred"] == o["y"]).mean()),
                     "accuracy_tau_05": float((o["pred_05"] == o["y"]).mean()),
                     "inner_auc": np.mean([s["inner_auc"] for s in settings]),
                     "tau_acc": [s["tau_acc"] for s in settings]})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fill the tuned-model cache (section 4.6)")
    parser.add_argument("--only", default=None, help="substring filter on 'task__model__scheme'")
    parser.add_argument("--pilot", action="store_true", help="run the Task B LightGBM pilot")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--round", type=int, default=1, choices=(1, 2),
                        help="2 = the narrowed, widened, seed-bagged second round")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.pilot:
        pd.set_option("display.width", 200)
        print(pilot().to_string(index=False))
        return
    df = validation.frame("primary")
    runs = planned_runs(args.round)
    if args.only:
        runs = [r for r in runs if args.only in "__".join(r)]
    log.info("%d runs planned", len(runs))
    for task, model, scheme in runs:
        predictions(task, model, scheme, df, refresh=args.refresh, rnd=args.round)


if __name__ == "__main__":
    main()
