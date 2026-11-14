"""Section 4.5: the models against simple baselines, side by side.

Under leave-one-city-out and UA-grouped 5-fold: the majority class, logistic
regression and LightGBM on each task's primary features, plus reference
baselines (Task A: `LST_urb_day_CT` alone; Task B: location only, and the
non-thermal features without any heat variable). Paired UA cluster-bootstrap
differences in ROC-AUC and MCC for LightGBM − logistic, LightGBM − majority,
and the reference comparisons.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uhi_poverty import metrics, models, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
SCHEMES = ("loco", "ua5")
ENTRIES = {  # task: [(config, model, label)]
    "A": [("A-noLST", "majority", "Majority class"),
          ("A-lst-only", "logistic", "LST-only, logistic"),
          ("A-lst-only", "lgbm", "LST-only, LightGBM"),
          ("A-noLST", "logistic", "Logistic regression (A-noLST)"),
          ("A-noLST", "lgbm", "LightGBM (A-noLST)")],
    "B": [("B-env", "majority", "Majority class"),
          ("geo-only", "logistic", "Location only, logistic"),
          ("geo-only", "lgbm", "Location only, LightGBM"),
          ("B-nonthermal", "logistic", "No heat variables, logistic"),
          ("B-nonthermal", "lgbm", "No heat variables, LightGBM"),
          ("B-env", "logistic", "Logistic regression (B-env)"),
          ("B-env", "lgbm", "LightGBM (B-env)")],
}
COMPARISONS = {  # task: [(label, (config, model), (config, model))]
    "A": [("LightGBM − logistic", ("A-noLST", "lgbm"), ("A-noLST", "logistic")),
          ("LightGBM − majority class", ("A-noLST", "lgbm"), ("A-noLST", "majority")),
          ("LightGBM (A-noLST) − LightGBM (LST only)", ("A-noLST", "lgbm"), ("A-lst-only", "lgbm"))],
    "B": [("LightGBM − logistic", ("B-env", "lgbm"), ("B-env", "logistic")),
          ("LightGBM − majority class", ("B-env", "lgbm"), ("B-env", "majority")),
          ("LightGBM (B-env) − LightGBM (location only)", ("B-env", "lgbm"), ("geo-only", "lgbm")),
          ("LightGBM (B-env) − LightGBM (no heat variables)", ("B-env", "lgbm"),
           ("B-nonthermal", "lgbm"))],
}
KEY = ["prevalence", "accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc", "pr_auc"]


def side_by_side(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, entries in ENTRIES.items():
        for scheme in SCHEMES:
            for config, model, label in entries:
                t = metrics.evaluate(task, config, model, scheme, df=df, metrics=KEY)
                t = t.set_index("metric")
                row = {"task": task, "scheme": scheme, "baseline": label, "config": config,
                       "model": model}
                for m in KEY:
                    row[m] = t.loc[m, "estimate"]
                for m in ("roc_auc", "mcc"):
                    row[f"{m}_ci_low"] = t.loc[m, "ci_low"]
                    row[f"{m}_ci_high"] = t.loc[m, "ci_high"]
                rows.append(row)
    return pd.DataFrame(rows)


def deltas(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for task, comps in COMPARISONS.items():
        for scheme in SCHEMES:
            for label, (ca, ma), (cb, mb) in comps:
                oa, sa = metrics.scored_run(task, ca, ma, scheme, df=df)
                ob, sb = metrics.scored_run(task, cb, mb, scheme, df=df)
                if not oa["GEOID"].equals(ob["GEOID"]):
                    raise AssertionError("paired runs are not aligned")
                d = metrics.paired_delta(sa, sb, oa["ua"].to_numpy(), metrics=("roc_auc", "mcc"))
                d.insert(0, "comparison", label)
                d.insert(0, "scheme", scheme)
                d.insert(0, "task", task)
                frames.append(d)
    return pd.concat(frames, ignore_index=True)


def plot(t: pd.DataFrame) -> None:
    plotting.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6))
    for ax, task in zip(axes, ("A", "B"), strict=True):
        sub = t[(t.task == task) & (t.scheme == "loco")].iloc[::-1]
        y = np.arange(len(sub))
        colors = [plotting.MODEL_COLORS[m] for m in sub["model"]]
        ax.errorbar(sub["roc_auc"], y, xerr=[sub["roc_auc"] - sub["roc_auc_ci_low"],
                                             sub["roc_auc_ci_high"] - sub["roc_auc"]],
                    fmt="none", ecolor=plotting.AXIS, elinewidth=1.5, capsize=0)
        ax.scatter(sub["roc_auc"], y, c=colors, s=40, edgecolors=plotting.SURFACE,
                   linewidths=1.5, zorder=3)
        for yi, v in zip(y, sub["roc_auc"], strict=True):
            ax.text(v, yi + 0.3, f"{v:.3f}", ha="center", fontsize=7, color=plotting.INK_2)
        ax.set_yticks(y, sub["baseline"])
        ax.axvline(0.5, color=plotting.AXIS, linewidth=1)
        ax.set_xlabel("ROC-AUC, leave-one-city-out")
        ax.set_title(f"Task {task}: {'High_UHI' if task == 'A' else 'High_Pov'}")
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=plotting.MODEL_COLORS[m],
                          label=models.MODEL_LABELS[m]) for m in models.MODELS]
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3)
    plotting.note(fig, "Bars: 95 % UA cluster-bootstrap CI (B = 1,000). Colour = learner.")
    fig.tight_layout()
    plotting.savefig(fig, OUT / "baselines.png")


def main() -> None:
    df = validation.load_analysis()
    t = side_by_side(df)
    report.write_table(t, OUT / "baselines")
    d = deltas(df)
    report.write_table(d, OUT / "paired_deltas")
    plot(t)


if __name__ == "__main__":
    main()
