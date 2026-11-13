"""Section 3.2: the gradient-boosted UHI model without any LST or SUHI input.

Trains LightGBM (and logistic regression) on `A-noLST`, which contains no
`LST_*` and no `UHI_*` column, and runs the ablation ladder
A-full → A-narrow → A-noLST-keepUHI → A-noLST → A-noLST+pov, plus two LST
references (`A-lst-only`, one urban-LST column; `A-lst`, every LST column), under leave-one-city-out (primary), UA-grouped 5-fold and random
5-fold (for comparison only). Writes the full predictor list of every
configuration to `results/feature_lists.csv`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uhi_poverty import features, metrics, models, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
LADDER = validation.ABLATION_CONFIGS
SCHEMES = validation.ABLATION_SCHEMES
LEARNED = ("logistic", "lgbm")
SHOW = ("roc_auc", "f1", "mcc")


def ablation(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for config in LADDER:
        for model in LEARNED:
            for scheme in SCHEMES:
                frames.append(metrics.evaluate("A", config, model, scheme, df=df,
                                               metrics=["n", "prevalence", *SHOW, "pr_auc",
                                                        "balanced_accuracy"]))
    t = pd.concat(frames, ignore_index=True)
    t["n_features"] = t["config"].map(lambda c: len(features.CONFIGS[c]))
    return t


def wide(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, model, scheme), g in t.groupby(["config", "model", "scheme"], sort=False):
        g = g.set_index("metric")
        row = {"config": config, "n_features": int(g["n_features"].iloc[0]),
               "model": models.MODEL_LABELS[model], "scheme": validation.SCHEME_LABELS[scheme]}
        for m in ("roc_auc", "f1", "mcc", "pr_auc", "balanced_accuracy"):
            row[m] = report.fmt_ci(g.loc[m, "estimate"], g.loc[m, "ci_low"], g.loc[m, "ci_high"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot(t: pd.DataFrame) -> None:
    plotting.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.4), sharey=True)
    order = LADDER[::-1]
    for ax, metric in zip(axes, SHOW, strict=True):
        for j, model in enumerate(LEARNED):
            s = t[(t.metric == metric) & (t.model == model) & (t.scheme == "loco")]
            s = s.set_index("config").loc[order]
            y = np.arange(len(order)) + (j - 0.5) * 0.28
            ax.errorbar(s["estimate"], y,
                        xerr=[s["estimate"] - s["ci_low"], s["ci_high"] - s["estimate"]],
                        fmt="o", color=plotting.MODEL_COLORS[model], markersize=6,
                        markeredgecolor=plotting.SURFACE, elinewidth=1.5, capsize=0,
                        label=models.MODEL_LABELS[model])
        ax.set_title({"roc_auc": "ROC-AUC", "f1": "F1", "mcc": "MCC"}[metric])
        ax.set_yticks(range(len(order)), [f"{c} ({len(features.CONFIGS[c])})" for c in order])
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=2)
    fig.suptitle("Task A ablation, leave-one-city-out", x=0.01, ha="left", y=1.1,
                 fontweight="bold", fontsize=10)
    plotting.note(fig, "Number of predictors in brackets. Bars: 95 % UA cluster-bootstrap CI.")
    plotting.savefig(fig, OUT / "ablation.png")


def main() -> None:
    df = validation.load_analysis()
    features.feature_list_table().to_csv(OUT / "feature_lists.csv", index=False)
    t = ablation(df)
    t.to_csv(OUT / "ablation.csv", index=False)
    w = wide(t)
    report.write_text("# Task A ablation ladder (estimate [95 % UA cluster-bootstrap CI])\n\n"
                      + report.md_table(w)
                      + "\nτ = 0.5, B = 1,000. Random 5-fold is shown for comparison only.\n",
                      OUT / "ablation.md")
    plot(t)
    write_readme()


def write_readme() -> None:
    t = pd.read_csv(OUT / "ablation.csv")
    fl = pd.read_csv(OUT / "feature_lists.csv")
    f = report.fmt_num

    def v(config, model, scheme, metric):
        r = t[(t.config == config) & (t.model == model) & (t.scheme == scheme)
              & (t.metric == metric)].iloc[0]
        return r

    def ci(config, model, scheme, metric):
        r = v(config, model, scheme, metric)
        return report.fmt_ci(r.estimate, r.ci_low, r.ci_high)

    lines = []
    for config in LADDER:
        lines.append(
            f"| `{config}` | {len(features.CONFIGS[config])} | "
            f"{features.CONFIG_NOTES[config]} | {f(v(config, 'lgbm', 'random5', 'roc_auc').estimate)} | "
            f"{ci(config, 'lgbm', 'loco', 'roc_auc')} | {f(v(config, 'lgbm', 'loco', 'f1').estimate)} | "
            f"{f(v(config, 'lgbm', 'loco', 'mcc').estimate)} |")
    nolst = fl[fl.config == "A-noLST"]
    fam = nolst.groupby("family", sort=False)["feature"].apply(lambda s: ", ".join(f"`{x}`" for x in s))
    famlines = "\n".join(f"- **{k}** ({len(nolst[nolst.family == k])}): {x}" for k, x in fam.items())
    text = f"""# 3.2 Gradient-boosted model without LST

## Purpose

This section fits the UHI model without the target's components or their close proxies, and
shows, through an ablation ladder and two LST references, what each family of inputs adds.

## Method

- `gbm_no_lst.py` trains LightGBM (and logistic regression) on **`A-noLST`**: vegetation
  (NDVI), urban–rural vegetation contrast (ΔNDVI), elevation (DEM), tract morphology and
  tract geometry. It contains **no `LST_*` column and no `UHI_*` column**, because every
  `UHI_*` column is an LST difference (section 3.1).
- Evaluation: leave-one-city-out (primary, shared with section 4), UA-grouped 5-fold, and random
  5-fold for comparison only. Fixed hyperparameters, τ = 0.5, every learned step fitted inside
  the training folds.
- Ablation ladder, each rung removing only what its name says (the two geometry columns are
  in every rung except the LST references): `A-full` → `A-narrow` → `A-noLST-keepUHI` →
  `A-noLST` → `A-noLST+pov`, plus `A-lst-only` (one urban-LST column) and the **LST
  reference `A-lst`** (all 24 `LST_*` columns and nothing else).
- [`results/feature_lists.csv`](results/feature_lists.csv) lists every predictor of every
  configuration in this repository, with its family.

## Results

LightGBM, Task A (`High_UHI`), leave-one-city-out unless stated
([`results/ablation.md`](results/ablation.md) has logistic regression, UA-grouped 5-fold, F1,
MCC, PR-AUC and balanced accuracy with CIs):

| Configuration | Predictors | Contents | ROC-AUC, random 5-fold | ROC-AUC, leave-one-city-out [95 % CI] | F1 | MCC |
|---|---:|---|---:|---:|---:|---:|
{chr(10).join(lines)}

- With the target's own components (`A-full`) the model is close to perfect, because it can
  subtract `LST_rur_day` from `LST_urb_day_CT`.
- Dropping just those two columns (`A-narrow`) barely changes anything: the other `LST_*` and
  `UHI_*` columns still encode the target.
- Dropping all 24 `LST_*` columns but keeping the `UHI_*` columns (`A-noLST-keepUHI`) still
  leaks, because the seasonal and night-time SUHI columns are differences built from the same
  temperatures.
- **`A-noLST` is the leak-free model**: ROC-AUC {ci('A-noLST', 'lgbm', 'loco', 'roc_auc')} for
  LightGBM and {ci('A-noLST', 'logistic', 'loco', 'roc_auc')} for logistic regression in held-out
  cities. It measures how well high surface urban heat can be predicted from vegetation,
  elevation and built form alone.
- The LST reference (`A-lst`, temperature columns only) reaches ROC-AUC
  {ci('A-lst', 'lgbm', 'loco', 'roc_auc')} for LightGBM and
  {ci('A-lst', 'logistic', 'loco', 'roc_auc')} for logistic regression. It is a reference, not
  a usable predictor: the label is the difference of two of its columns, which logistic
  regression can represent exactly.
- Adding the ACS poverty rate (`A-noLST+pov`) changes leave-one-city-out ROC-AUC from
  {f(v('A-noLST', 'lgbm', 'loco', 'roc_auc').estimate)} to
  {f(v('A-noLST+pov', 'lgbm', 'loco', 'roc_auc').estimate)}.

**Predictors of `A-noLST`** ({len(nolst)}):

{famlines}

Figure: [`results/ablation.png`](results/ablation.png).
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
