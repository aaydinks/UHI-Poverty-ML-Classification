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
    report.write_table(t, OUT / "baselines", digits=3,
                       title="Models and baselines (tau = 0.5; 95 % UA cluster-bootstrap CIs)")
    d = deltas(df)
    report.write_table(d, OUT / "paired_deltas", digits=3,
                       title="Paired differences, Δ = a − b (95 % UA cluster-bootstrap CI)")
    plot(t)
    write_readme()


def write_readme() -> None:
    t = pd.read_csv(OUT / "baselines.csv")
    d = pd.read_csv(OUT / "paired_deltas.csv")
    f = report.fmt_num

    def row(task, label, scheme="loco"):
        return t[(t.task == task) & (t.baseline == label) & (t.scheme == scheme)].iloc[0]

    def delta(task, comp, metric="roc_auc", scheme="loco"):
        r = d[(d.task == task) & (d.comparison == comp) & (d.metric == metric)
              & (d.scheme == scheme)].iloc[0]
        return f"{f(r.delta)} [{f(r.ci_low)}, {f(r.ci_high)}]"

    lines = []
    for task, entries in ENTRIES.items():
        for _, _, label in entries:
            r = row(task, label)
            lines.append(f"| {task} | {label} | {f(r.roc_auc)} [{f(r.roc_auc_ci_low)}, "
                         f"{f(r.roc_auc_ci_high)}] | {f(r.mcc)} | {f(r.balanced_accuracy)} | "
                         f"{f(r.pr_auc)} |")
    dlines = []
    for task, comps in COMPARISONS.items():
        for label, _, _ in comps:
            dlines.append(f"| {task} | {label} | {delta(task, label)} | "
                          f"{delta(task, label, 'mcc')} |")
    lst = row("A", "LST-only, LightGBM")
    heat = d[(d.task == "B") & (d.comparison == "LightGBM (B-env) − LightGBM (no heat variables)")
             & (d.metric == "roc_auc") & (d.scheme == "loco")].iloc[0]
    heat_note = (", an improvement the data cannot distinguish from zero" if heat.ci_low <= 0
                 else "")
    text = f"""# 4.5 Baselines

## Purpose

This section tests whether the gradient-boosted models beat simpler alternatives: a
majority-class baseline, a plain statistical model, and, for the heat task, a model that only
sees land-surface temperature.

## Method

`baselines.py` compares, under leave-one-city-out and UA-grouped 5-fold:

- the majority class (`DummyClassifier("most_frequent")`);
- logistic regression (median imputation → `log1p` on area and pixel counts → standardisation);
- LightGBM (fixed settings);
- reference baselines: for Task A, `LST_urb_day_CT` alone (an LST-only model); for
  Task B, location only (centroid latitude/longitude + Census region) and the same
  non-thermal features as `A-noLST` with no heat variable (`B-nonthermal`).

Paired differences use the same UA bootstrap draws for both models (B = 1,000).

A constant score carries no ranking, so the majority class's ROC-AUC is 0.5 and its PR-AUC
equals the prevalence by definition. Its threshold metrics are computed from its predictions as
they are. With a median split the two classes differ by one tract, so holding out a city tips
the training majority toward the class that city has less of. Pooled over leave-one-city-out
folds, the majority class is therefore worse than chance (section 4.3), which makes
"LightGBM − majority" differences in MCC larger than LightGBM's gain over a no-skill rule
(MCC 0). That gain is simply LightGBM's own MCC.

## Results

Leave-one-city-out (UA-grouped 5-fold is in [`results/baselines.md`](results/baselines.md)):

| Task | Model | ROC-AUC [95 % CI] | MCC | Balanced accuracy | PR-AUC |
|---|---|---:|---:|---:|---:|
{chr(10).join(lines)}

Paired differences, leave-one-city-out ([`results/paired_deltas.md`](results/paired_deltas.md)):

| Task | Comparison | Δ ROC-AUC [95 % CI] | Δ MCC [95 % CI] |
|---|---|---:|---:|
{chr(10).join(dlines)}

`LST_urb_day_CT` is one of the two terms of the target (`UHI_annual_day` = `LST_urb_day_CT` −
`LST_rur_day`, section 3.1), yet on its own it predicts high SUHI poorly in held-out cities
(ROC-AUC {f(lst.roc_auc)}). SUHI is measured against each city's own rural reference, so the same
tract temperature can be a strong heat island in one city and none in another. The leak-free
`A-noLST` model, which sees no temperature at all, does far better (Δ ROC-AUC
{delta('A', 'LightGBM (A-noLST) − LightGBM (LST only)')}).

**Heat and poverty.** Adding every SUHI and LST variable to the non-thermal features
changes LightGBM's poverty ROC-AUC by {delta('B', 'LightGBM (B-env) − LightGBM (no heat variables)')}{heat_note}
under leave-one-city-out: the model without any heat variable reaches ROC-AUC
{f(row('B', 'No heat variables, LightGBM').roc_auc)}, against {f(row('B', 'LightGBM (B-env)').roc_auc)}
with them. Nearly all of the poverty model's skill comes from vegetation, elevation and built
form, the same inputs that predict high SUHI in Task A (ROC-AUC
{f(row('A', 'LightGBM (A-noLST)').roc_auc)}).

Figure: [`results/baselines.png`](results/baselines.png).
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
