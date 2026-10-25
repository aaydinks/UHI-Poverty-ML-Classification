"""Section 4.4: the full metric panel with uncertainty, every model × task × scheme.

Each metric gets a percentile 95 % CI from a UA cluster bootstrap (B = 1,000,
seed 37) on the pooled out-of-fold predictions, plus the across-fold SD for the
5-fold schemes. Writes `results/metric_panel.{csv,md}`, `results/forest_plot.png`
and the README.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from uhi_poverty import features, metrics, models, paths, plotting, report, validation

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


def md(panel: pd.DataFrame) -> str:
    out = ["# Metric panel with 95 % UA cluster-bootstrap CIs\n",
           "Pooled out-of-fold predictions, τ = 0.5, B = 1,000. Cells: estimate [95 % CI]; "
           "± across-fold SD for 5-fold schemes. Majority class: ROC-AUC = 0.5 and PR-AUC = "
           "prevalence by definition.\n"]
    for task in TASKS:
        sub = panel[panel.task == task]
        rows = []
        for (model, scheme), g in sub.groupby(["model", "scheme"], sort=False):
            g = g.set_index("metric")
            row = {"model": models.MODEL_LABELS[model],
                   "scheme": validation.SCHEME_LABELS[scheme],
                   "n": f"{int(g.loc['n', 'estimate']):,}",
                   "prevalence": report.fmt_num(g.loc["prevalence", "estimate"])}
            for m in SHOW[2:]:
                cell = report.fmt_ci(g.loc[m, "estimate"], g.loc[m, "ci_low"], g.loc[m, "ci_high"])
                if not np.isnan(g.loc[m, "fold_sd"]):
                    cell += f" ±{g.loc[m, 'fold_sd']:.3f}"
                row[m] = cell
            rows.append(row)
        out.append(f"## {features.TASKS[task]['label']} ({features.TASKS[task]['primary']})\n")
        out.append(report.md_table(pd.DataFrame(rows)))
    return "\n".join(out)


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
    report.write_text(md(panel), OUT / "metric_panel.md")
    forest(panel)
    write_readme()


def write_readme() -> None:
    p = pd.read_csv(OUT / "metric_panel.csv")
    f = report.fmt_num

    def g(task, model, scheme, m):
        r = p[(p.task == task) & (p.model == model) & (p.scheme == scheme) & (p.metric == m)].iloc[0]
        return r

    def ci(task, model, scheme, m):
        r = g(task, model, scheme, m)
        return report.fmt_ci(r.estimate, r.ci_low, r.ci_high)

    rows = []
    for task in TASKS:
        for model in models.MODELS:
            rows.append(f"| {task} | {models.MODEL_LABELS[model]} | "
                        + " | ".join(ci(task, model, "loco", m) for m in
                                     ("roc_auc", "pr_auc", "mcc", "f1", "balanced_accuracy"))
                        + " |")
    text = f"""# 4.4 Full metric panel with uncertainty

## Purpose

This section gathers everything needed to judge the classifiers in one table: class
prevalence, the full set of classification metrics rather than F1 alone, and confidence
intervals.

## Method

`metric_panel.py` scores every model (majority class, logistic regression, LightGBM) × task
(A: `High_UHI` on `A-noLST`; B: `High_Pov` on `B-env`) × validation scheme (7 schemes) from
the shared out-of-fold predictions. For each it reports n and prevalence; accuracy, balanced
accuracy, precision, recall, specificity and F1 at τ = 0.5; ROC-AUC and PR-AUC; MCC and the
Brier score. Each metric has a percentile 95 % CI from a cluster bootstrap that resamples
urbanized areas (B = 1,000, seed 37), so the intervals respect the nesting of tracts in cities.
The 5-fold schemes also report the across-fold SD.

## Results

Headline scheme, leave-one-city-out (estimate [95 % CI]):

| Task | Model | ROC-AUC | PR-AUC | MCC | F1 | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The majority class scores below chance on MCC and balanced accuracy under leave-one-city-out
because its constant prediction flips toward the class each held-out city has less of
([`../4.3_Confusion_Matrices_and_Denominators/README.md`](../4.3_Confusion_Matrices_and_Denominators/README.md)).
Prevalence is {f(g('A', 'lgbm', 'loco', 'prevalence').estimate)} for Task A and
{f(g('B', 'lgbm', 'loco', 'prevalence').estimate)} for Task B, so PR-AUC has a no-skill floor near
0.5. Every model × scheme row is in [`results/metric_panel.md`](results/metric_panel.md) (long
format: [`results/metric_panel.csv`](results/metric_panel.csv)); the forest plot is
[`results/forest_plot.png`](results/forest_plot.png).
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
