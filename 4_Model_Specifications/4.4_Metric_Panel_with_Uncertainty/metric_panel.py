"""Section 4.4: the full metric panel with uncertainty, every model × task × scheme.

Each metric gets a percentile 95 % CI from a UA cluster bootstrap (B = 1,000,
seed 37) on the pooled out-of-fold predictions, plus the across-fold SD for the
5-fold schemes. Writes `results/metric_panel.csv` and `results/forest_plot.png`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from uhi_poverty import features, metrics, models, paths, plotting, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
TASKS = ("A", "B")
SHOW = ["n", "prevalence", "accuracy", "balanced_accuracy", "precision", "recall",
        "specificity", "f1", "roc_auc", "pr_auc", "mcc", "brier"]


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    jobs = [(t, features.TASKS[t]["primary"], m, s)
            for t in TASKS for m in models.MODELS for s in validation.SCHEMES]
    # warm the cache in this process first, so workers only read it
    for job in jobs:
        validation.oof(*job, df=df)
    parts = Parallel(n_jobs=-1)(delayed(metrics.evaluate)(*job) for job in jobs)
    return pd.concat(parts, ignore_index=True)


def forest(panel: pd.DataFrame) -> None:
    plotting.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.2), sharey=True)
    order = list(validation.SCHEMES)[::-1]
    for i, task in enumerate(TASKS):
        for j, metric in enumerate(("roc_auc", "mcc")):
            ax = axes[i, j]
            for k, model in enumerate(models.MODELS):
                s = panel[(panel.task == task) & (panel.model == model) & (panel.metric == metric)]
                s = s.set_index("scheme").loc[order]
                y = np.arange(len(order)) + (k - 1) * 0.24
                ax.errorbar(s["estimate"], y,
                            xerr=[s["estimate"] - s["ci_low"], s["ci_high"] - s["estimate"]],
                            fmt="o", color=plotting.MODEL_COLORS[model], markersize=5,
                            markeredgecolor=plotting.SURFACE, elinewidth=1.5, capsize=0,
                            label=models.MODEL_LABELS[model])
            ax.set_yticks(range(len(order)), [validation.SCHEME_LABELS[s] for s in order])
            ax.set_title(f"{features.TASKS[task]['label']} · "
                         f"{'ROC-AUC' if metric == 'roc_auc' else 'MCC'}")
            ax.grid(axis="y", visible=False)
            ax.axvline(0.5 if metric == "roc_auc" else 0, color=plotting.AXIS, linewidth=1)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=3)
    plotting.note(fig, "Points: pooled out-of-fold estimate; bars: 95 % UA cluster-bootstrap CI "
                       "(B = 1,000).")
    fig.tight_layout()
    plotting.savefig(fig, OUT / "forest_plot.png")


def main() -> None:
    df = validation.load_analysis()
    panel = run_all(df)
    panel.to_csv(OUT / "metric_panel.csv", index=False)
    forest(panel)


if __name__ == "__main__":
    main()
