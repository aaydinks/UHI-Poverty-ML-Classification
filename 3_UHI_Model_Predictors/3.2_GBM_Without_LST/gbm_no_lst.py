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

from uhi_poverty import features, metrics, models, paths, plotting, validation

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
    plot(t)


if __name__ == "__main__":
    main()
