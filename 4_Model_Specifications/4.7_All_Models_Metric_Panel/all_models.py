"""Section 4.7: one metric panel for every model in the repository.

Every model evaluated anywhere in sections 3–6 is scored the same way here, from its
cached leave-one-city-out predictions (486 folds, every tract predicted by a model
that never saw its urbanized area):

- the fixed-setting primary models of section 4.4 (majority class, logistic regression,
  LightGBM; τ = 0.5);
- the six further algorithms of the algorithm comparison (decision tree, random forest,
  SVM, KNN, CatBoost, XGBoost; library defaults, τ = 0.5);
- the tuned models and soft-voting ensembles of section 4.6, at their own threshold τ_acc,
  chosen on the training cities;
- the reference and baseline feature sets of sections 3.2 and 4.5, including the LST
  reference model (`A-lst`, every LST column).

For each: class prevalence, accuracy, balanced accuracy, precision, recall, specificity,
F1, ROC-AUC and PR-AUC, each with a 95 % percentile interval from the UA cluster
bootstrap (B = 1,000, seed 37) that every other section uses. Nothing is refitted.

Checks: every model of a task is scored on the same tracts with the same labels, and
every number that another section also reports must match it exactly.
"""

from __future__ import annotations

import functools
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from uhi_poverty import features, metrics, models, paths, report, tuning, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
SCHEME = "loco"
METRICS = ["prevalence", "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
           "f1", "roc_auc", "pr_auc"]
METRIC_LABELS = {"prevalence": "Prevalence", "accuracy": "Accuracy",
                 "balanced_accuracy": "Balanced accuracy", "precision": "Precision",
                 "recall": "Recall", "specificity": "Specificity", "f1": "F1",
                 "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
SPEC = paths.ROOT / "4_Model_Specifications"
FIG = paths.ROOT / "6_Figures"
CHECKS = {  # source table -> path
    "4.4": SPEC / "4.4_Metric_Panel_with_Uncertainty" / "results" / "metric_panel.csv",
    "4.6": SPEC / "4.6_Tuned_Models_and_Ensemble" / "results" / "tuned_panel.csv",
    "3.2": paths.ROOT / "3_UHI_Model_Predictors" / "3.2_GBM_Without_LST" / "results"
    / "ablation.csv",
    "6": FIG / "results" / "algorithms_long.csv",
}
FIXED = {"majority": "Majority class", "logistic": "Logistic regression", "lgbm": "LightGBM"}
OTHER = ("dt", "rf", "svm", "knn", "catboost", "xgboost")
REFERENCE_A = ["A-lst", "A-lst-only", "A-full", "A-narrow", "A-noLST-keepUHI", "A-noLST+pov"]
BASELINE_B = ["geo-only", "B-nonthermal"]
GROUPS = {
    "primary": "Primary models (fixed settings, τ = 0.5)",
    "other": "Further algorithms (library defaults, τ = 0.5)",
    "tuned": "Tuned in nested cross-validation (τ_acc)",
    "reference": "Reference and baseline feature sets (τ = 0.5)",
}


@functools.cache
def _algorithms():
    spec = importlib.util.spec_from_file_location("algorithms", FIG / "algorithms.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def entries() -> list[dict]:
    """Every model: task, group, label, and how to load its predictions."""
    out = []
    for task in ("A", "B"):
        cfg = features.TASKS[task]["primary"]
        for m, label in FIXED.items():
            out.append({"task": task, "group": "primary", "label": label, "config": cfg,
                        "model": m, "source": "fixed", "threshold": "0.5"})
        alg = _algorithms()
        for m in OTHER:
            out.append({"task": task, "group": "other", "label": alg.ALGORITHM_LABELS[m],
                        "config": cfg, "model": m, "source": "algorithms", "threshold": "0.5"})
        for m in tuning.ALL_MODELS:
            out.append({"task": task, "group": "tuned", "label": models.TUNED_LABELS[m],
                        "config": cfg, "model": m, "source": "tuned", "threshold": "tau_acc"})
        for c in (REFERENCE_A if task == "A" else BASELINE_B):
            for m in ("logistic", "lgbm"):
                out.append({"task": task, "group": "reference",
                            "label": f"{FIXED[m]}, {c}" + (" (LST reference)" if c == "A-lst"
                                                           else ""),
                            "config": c, "model": m, "source": "fixed", "threshold": "0.5"})
    return out


def predictions(e: dict, df: pd.DataFrame) -> pd.DataFrame:
    """GEOID, ua, y, score and the class at the entry's threshold."""
    if e["source"] == "fixed":
        o = validation.oof(e["task"], e["config"], e["model"], SCHEME, df=df)
    elif e["source"] == "algorithms":
        alg = _algorithms()
        fr = validation.frame("primary", df)
        if e["model"] in alg.REPO_MODELS:
            o = alg.repo_oof(e["task"], e["model"], fr)
        else:
            # read the cache only; never refit here
            X, y, _ = alg._design(e["task"], fr)
            key = f"{e['task']}__{alg.PANELS[e['task']]}__{e['model']}__{SCHEME}"
            meta = alg.CACHE / f"{key}.json"
            if not meta.exists() or json.loads(meta.read_text()).get("fingerprint") != \
                    alg._fingerprint(fr, X, y, e["model"]):
                raise RuntimeError(f"{key}: not cached (run 6_Figures/algorithms.py)")
            o = alg.fitted_oof(e["task"], e["model"], fr)
    else:
        o = tuning.predictions(e["task"], e["model"], SCHEME, df=df)
    return o[["GEOID", "ua", "y", "score", "pred"]].sort_values("GEOID").reset_index(drop=True)


def _score(e: dict, o: pd.DataFrame) -> pd.DataFrame:
    s = metrics.Scored(o["y"], o["score"], o["pred"], constant_score=(e["model"] == "majority"))
    t = metrics.bootstrap_panel(s, o["ua"].to_numpy(), metrics=METRICS)
    return t.assign(**{k: e[k] for k in ("task", "group", "label", "config", "model",
                                         "threshold")}, n=len(o))


def score_all(df: pd.DataFrame) -> pd.DataFrame:
    es = entries()
    preds = [predictions(e, df) for e in es]
    for task in ("A", "B"):
        ref = next(p for e, p in zip(es, preds, strict=True) if e["task"] == task)
        for e, p in zip(es, preds, strict=True):
            if e["task"] == task and not (p["GEOID"].equals(ref["GEOID"])
                                          and p["y"].equals(ref["y"])):
                raise AssertionError(f"{e['label']}: scored on different tracts or labels")
    parts = Parallel(n_jobs=-1)(delayed(_score)(e, p) for e, p in zip(es, preds, strict=True))
    t = pd.concat(parts, ignore_index=True)
    cols = ["task", "group", "label", "config", "model", "threshold", "n", "metric", "estimate",
            "ci_low", "ci_high"]
    return t[cols]


# --------------------------------------------------------------------------- checks


def cross_check(t: pd.DataFrame) -> pd.DataFrame:
    """Largest absolute difference from every other section's numbers for the same model."""
    rows = []

    def compare(name, mine, ref, keys):
        m = mine.merge(ref, on=keys, suffixes=("", "_ref"), validate="one_to_one")
        if len(m) == 0:
            raise AssertionError(f"{name}: nothing to compare")
        diff = max((m[c] - m[f"{c}_ref"]).abs().max() for c in ("estimate", "ci_low", "ci_high"))
        rows.append({"source": name, "n_compared": len(m), "max_abs_diff": float(diff)})

    keys = ["task", "config", "model", "metric"]
    fixed = t[t.threshold.eq("0.5") & t.group.isin(["primary", "reference"])]
    ref = pd.read_csv(CHECKS["4.4"]).query("scheme == @SCHEME")
    compare("4.4 metric panel", fixed[fixed.group == "primary"][keys + ["estimate", "ci_low",
                                                                       "ci_high"]],
            ref[keys + ["estimate", "ci_low", "ci_high"]], keys)
    ref = pd.read_csv(CHECKS["3.2"]).query("scheme == @SCHEME & task == 'A'")
    mine = fixed[fixed.config.isin(REFERENCE_A)]
    compare("3.2 ablation", mine[keys + ["estimate", "ci_low", "ci_high"]],
            ref[keys + ["estimate", "ci_low", "ci_high"]], keys)
    ref = pd.read_csv(CHECKS["6"])
    mine = t[t.group.isin(["primary", "other"]) & (t.model != "majority")]
    compare("6 algorithm comparison", mine[keys + ["estimate", "ci_low", "ci_high"]],
            ref[keys + ["estimate", "ci_low", "ci_high"]], keys)
    ref = pd.read_csv(CHECKS["4.6"]).query("scheme == @SCHEME & threshold == 'tau_acc'")
    ref = ref.rename(columns={"entry": "model"})
    k2 = ["task", "model", "metric"]
    compare("4.6 tuned panel", t[t.group == "tuned"][k2 + ["estimate", "ci_low", "ci_high"]],
            ref[k2 + ["estimate", "ci_low", "ci_high"]], k2)
    check = pd.DataFrame(rows)
    if check["max_abs_diff"].max() > 1e-12:
        raise AssertionError(f"numbers differ from another section:\n{check}")
    return check


def sanity(t: pd.DataFrame) -> None:
    """Pipeline sanity checks; stops on failure."""
    w = t.pivot_table(index=["task", "label"], columns="metric", values="estimate")
    for task in ("A", "B"):
        prev = w.loc[task, "prevalence"]
        report.check(prev.max() - prev.min() < 1e-12, f"{task}: prevalence differs by model")
    # balanced accuracy is the mean of recall and specificity
    report.check(np.allclose(w["balanced_accuracy"], (w["recall"] + w["specificity"]) / 2),
                   "balanced accuracy != mean of recall and specificity")
    # F1 is the harmonic mean of precision and recall
    pr = w[["precision", "recall"]]
    report.check(np.allclose(w["f1"], 2 * pr.prod(axis=1) / pr.sum(axis=1)),
                   "F1 != harmonic mean of precision and recall")
    # every interval contains its estimate (the majority class has a degenerate ROC-AUC)
    report.check(((t.ci_low <= t.estimate + 1e-12) & (t.estimate <= t.ci_high + 1e-12)).all(),
                   "an estimate lies outside its interval")
    lst = w.loc[("A", "Logistic regression, A-lst (LST reference)")]
    lst_g = w.loc[("A", "LightGBM, A-lst (LST reference)")]
    best_a = w.loc["A"].drop(index=[i for i in w.loc["A"].index if "A-" in i])["roc_auc"]
    report.check(min(lst["roc_auc"], lst_g["roc_auc"]) > 0.99,
                   "the LST reference should be near-perfect (its inputs contain the target)")
    report.check(best_a.max() < min(lst["roc_auc"], lst_g["roc_auc"]),
                   "a model without LST input should not beat the LST reference")




def main() -> None:
    df = validation.load_analysis()
    t = score_all(df)
    t.to_csv(OUT / "all_models.csv", index=False)
    check = cross_check(t)
    check.to_csv(OUT / "cross_check.csv", index=False)
    sanity(t)
    print(check.to_string(index=False))


if __name__ == "__main__":
    main()
