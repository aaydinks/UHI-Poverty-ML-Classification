"""Section 4.6: nested tuning of the full pipeline, XGBoost, CatBoost and a soft-voting ensemble.

Reads the tuned-model cache (`make tune`, `src/uhi_poverty/tuning.py`) and the
fixed-setting predictions of sections 4.1-4.5, and scores both on the same
tracts. Every tuned choice (preprocessing, hyperparameters, ensemble weights and
the thresholds τ_acc and τ_F1) was made on training cities only. Writes
`results/tuned_panel`, `results/paired_deltas`, `results/search_choices`,
`results/mechanisms` and `results/tuned_vs_fixed.png`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from uhi_poverty import features, metrics, models, paths, plotting, report, tuning, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
TASKS = ("A", "B")
SCHEMES = ("loco", "ua5", "random5")
HEADLINE = "loco"
TARGETS = {"A": 0.90, "B": 0.75}
METRICS = ["accuracy", "f1", "roc_auc", "mcc", "balanced_accuracy", "precision", "recall"]
SHOW = ["accuracy", "f1", "roc_auc", "mcc", "balanced_accuracy"]
METRIC_LABELS = {"accuracy": "Accuracy", "f1": "F1", "roc_auc": "ROC-AUC", "mcc": "MCC",
                 "balanced_accuracy": "Balanced accuracy", "precision": "Precision",
                 "recall": "Recall"}
# (entry, source, model, label)
ENTRIES = [("fixed-logistic", "fixed", "logistic", "Logistic regression, fixed (4.4)"),
           ("fixed-lgbm", "fixed", "lgbm", "LightGBM, fixed (4.4)")] + [
    (m, "tuned", m, models.TUNED_LABELS[m]) for m in tuning.ALL_MODELS]
LABELS = {e: label for e, _, _, label in ENTRIES}
# threshold name -> prediction column in the tuned cache
THRESHOLDS = {"0.5": "pred_05", "tau_acc": "pred", "tau_f1": "pred_f1"}
THRESHOLD_LABELS = {"0.5": "τ = 0.5", "tau_acc": "τ_acc", "tau_f1": "τ_F1"}
PREP_OPTIONS = ["engineer", "variance", "corr", "scaler", "pca", "smote"]
PREP_LABELS = {"engineer": "Engineered terms", "variance": "Low-variance filter (0.1)",
               "corr": "Correlation filter (|r| > 0.9)", "scaler": "Scaling",
               "pca": "PCA", "smote": "SMOTE"}


def load(task: str, source: str, model: str, scheme: str, df: pd.DataFrame) -> pd.DataFrame:
    if source == "fixed":
        o = validation.oof(task, features.TASKS[task]["primary"], model, scheme, df=df)
        return o.assign(pred_05=o["pred"], pred_f1=np.nan)
    return tuning.predictions(task, model, scheme, df=df)


def _score(task, source, model, scheme, threshold) -> pd.DataFrame:
    o = load(task, source, model, scheme, validation.load_analysis())
    s = metrics.Scored(o["y"], o["score"], o[THRESHOLDS[threshold]])
    t = metrics.bootstrap_panel(s, o["ua"].to_numpy(), metrics=METRICS)
    return t.assign(task=task, scheme=scheme, threshold=threshold)


def panel(df: pd.DataFrame) -> pd.DataFrame:
    jobs = []
    for task in TASKS:
        for scheme in SCHEMES:
            for entry, source, model, _ in ENTRIES:
                load(task, source, model, scheme, df)  # warm the caches in this process
                for threshold in (["0.5"] if source == "fixed" else THRESHOLDS):
                    jobs.append((entry, (task, source, model, scheme, threshold)))
    parts = Parallel(n_jobs=-1)(delayed(_score)(*args) for _, args in jobs)
    for (entry, _), part in zip(jobs, parts, strict=True):
        part.insert(0, "entry", entry)
    t = pd.concat(parts, ignore_index=True)
    t["label"] = t["entry"].map(LABELS)
    return t[["task", "scheme", "entry", "label", "threshold", "metric", "estimate", "ci_low",
              "ci_high"]]


def comparisons() -> list[tuple[str, str, str, str, tuple[str, ...]]]:
    """(comparison, entry a, threshold a, entry b, metrics); b is at its own headline threshold."""
    out = []
    for m in tuning.ALL_MODELS:
        out.append((f"{LABELS[m]} − LightGBM, fixed", m, "tau_acc", "fixed-lgbm",
                    ("accuracy", "roc_auc")))
        out.append((f"{LABELS[m]} (τ_F1) − LightGBM, fixed", m, "tau_f1", "fixed-lgbm",
                    ("f1",)))
    for ens in models.ENSEMBLES:
        for m in models.BOOSTERS:
            out.append((f"{LABELS[ens]} − {LABELS[m]}", ens, "tau_acc", m,
                        ("accuracy", "roc_auc")))
    return out


def deltas(df: pd.DataFrame) -> pd.DataFrame:
    source = {e: (s, m) for e, s, m, _ in ENTRIES}
    frames = []
    for task in TASKS:
        for scheme in ("loco", "ua5"):
            for label, ea, ta, eb, mets in comparisons():
                oa = load(task, *source[ea], scheme, df)
                ob = load(task, *source[eb], scheme, df)
                if not oa["GEOID"].equals(ob["GEOID"]):
                    raise AssertionError("paired runs are not aligned")
                tb = "0.5" if source[eb][0] == "fixed" else "tau_acc"
                sa = metrics.Scored(oa["y"], oa["score"], oa[THRESHOLDS[ta]])
                sb = metrics.Scored(ob["y"], ob["score"], ob[THRESHOLDS[tb]])
                d = metrics.paired_delta(sa, sb, oa["ua"].to_numpy(), metrics=mets)
                d.insert(0, "comparison", label)
                d.insert(0, "scheme", scheme)
                d.insert(0, "task", task)
                frames.append(d)
    return pd.concat(frames, ignore_index=True)


def search_choices(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for tune_scheme in ("ua5", "random5"):
            for model in tuning.ALL_MODELS:
                for s in tuning.tuned_settings(task, model, tune_scheme, df):
                    row = {"task": task, "tuning_scheme": tune_scheme, "model": model,
                           "fold": s["k"], "inner_roc_auc": s["inner_auc"],
                           "tau_acc": s["tau_acc"], "tau_f1": s["tau_f1"]}
                    if model in models.ENSEMBLES:
                        row.update({f"w_{m}": s["weights"][m] for m in models.BOOSTERS})
                    else:
                        row.update(s["prep"])
                        row["params"] = json.dumps(s["params"], sort_keys=True)
                    rows.append(row)
    return pd.DataFrame(rows)


def mechanisms(df: pd.DataFrame) -> pd.DataFrame:
    """For each option value: the best inner ROC-AUC among trials using it, averaged over folds."""
    rows = []
    for task in TASKS:
        for model in models.TUNED_MODELS:
            settings = tuning.tuned_settings(task, model, "ua5", df)
            for option in PREP_OPTIONS:
                values = sorted({str(h[option]) for s in settings for h in s["history"]})
                for v in values:
                    best, n = [], 0
                    for s in settings:
                        aucs = [h["auc"] for h in s["history"]
                                if str(h[option]) == v and h["auc"] is not None]
                        n += len(aucs)
                        if aucs:
                            best.append(max(aucs))
                    chosen = sum(str(s["prep"][option]) == v for s in settings)
                    rows.append({"task": task, "model": model, "option": option, "value": v,
                                 "trials": n, "folds_with_trials": len(best),
                                 "mean_best_inner_roc_auc": np.mean(best) if best else np.nan,
                                 "folds_choosing": chosen})
    return pd.DataFrame(rows)


def class_balance(df: pd.DataFrame) -> dict:
    """Positive share of the training rows, across the outer folds of each scheme."""
    out = {}
    for task in TASKS:
        d = validation.frame("primary", df)
        y = d[features.TASKS[task]["target"]].astype(int).to_numpy()
        for scheme in ("ua5", "loco"):
            shares = [y[tr].mean() for _, tr, _ in validation.splits(d, y, scheme)]
            out[f"{task}_{scheme}"] = [float(min(shares)), float(max(shares))]
    return out


def plot(t: pd.DataFrame) -> None:
    plotting.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), sharey="row")
    for col, task in enumerate(TASKS):
        for row, metric in enumerate(("accuracy", "f1")):
            ax = axes[row, col]
            threshold = "tau_acc" if metric == "accuracy" else "tau_f1"
            labels, est, lo, hi, colors = [], [], [], [], []
            for entry, source, _, label in ENTRIES:
                th = "0.5" if source == "fixed" else threshold
                r = t[(t.task == task) & (t.scheme == HEADLINE) & (t.entry == entry)
                      & (t.threshold == th) & (t.metric == metric)].iloc[0]
                labels.append(label)
                est.append(r.estimate)
                lo.append(r.ci_low)
                hi.append(r.ci_high)
                colors.append(plotting.MUTED if source == "fixed" else (
                    plotting.ORANGE if entry in models.ENSEMBLES else plotting.BLUE))
            y = np.arange(len(labels))[::-1]
            est, lo, hi = map(np.asarray, (est, lo, hi))
            ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="none", ecolor=plotting.AXIS,
                        elinewidth=1.5, capsize=0)
            ax.scatter(est, y, c=colors, s=36, edgecolors=plotting.SURFACE, linewidths=1.5,
                       zorder=3)
            for yi, v in zip(y, est, strict=True):
                ax.text(v, yi + 0.3, f"{v:.3f}", ha="center", fontsize=7, color=plotting.INK_2)
            ax.axvline(TARGETS[task], color=plotting.INK_2, linewidth=1, linestyle="--")
            ax.set_yticks(y, labels)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(f"{METRIC_LABELS[metric]}, leave-one-city-out")
            if row == 0:
                ax.set_title(f"Task {task}: {'High_UHI' if task == 'A' else 'High_Pov'}")
    plotting.note(fig, "Grey: fixed settings at τ = 0.5. Blue: tuned, at the threshold chosen on "
                  "training cities (τ_acc for accuracy, τ_F1 for F1). Orange: soft-voting "
                  "ensembles. Bars: 95 % UA cluster-bootstrap CI. Dashed: the target (0.90 for "
                  "Task A, 0.75 for Task B).")
    fig.tight_layout()
    plotting.savefig(fig, OUT / "tuned_vs_fixed.png")


def main() -> None:
    df = validation.load_analysis()
    t = panel(df)
    t.to_csv(OUT / "tuned_panel.csv", index=False)
    d = deltas(df)
    report.write_table(d, OUT / "paired_deltas")
    report.write_table(search_choices(df), OUT / "search_choices")
    report.write_table(mechanisms(df), OUT / "mechanisms")
    report.write_json(class_balance(df), OUT / "class_balance.json")
    plot(t)




SMALL_GAIN = 0.02


if __name__ == "__main__":
    main()
