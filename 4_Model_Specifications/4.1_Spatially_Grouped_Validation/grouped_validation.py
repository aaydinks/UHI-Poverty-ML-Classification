"""Section 4.1: spatially grouped validation for both tasks and all three models.

Runs every scheme except the random reference (county5, ua5, loco, state,
region, major_ua) from the cached out-of-fold predictions and writes pooled
metrics plus per-UA (leave-one-city-out), per-state and per-region results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uhi_poverty import features, geo, metrics, models, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
SCHEMES = [s for s in validation.SCHEMES if s != "random5"]
TASKS = ("A", "B")
LEARNED = ("logistic", "lgbm")


def pooled(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        cfg = features.TASKS[task]["primary"]
        for model in models.MODELS:
            for scheme in SCHEMES:
                o, s = metrics.scored_run(task, cfg, model, scheme, df=df)
                p = s.at(np.ones_like(s.y))
                rows.append({"task": task, "config": cfg, "model": model, "scheme": scheme,
                             "n_folds": o["fold"].nunique(), **{m: p[m] for m in metrics.PANEL}})
    return pd.DataFrame(rows)


def per_group(df: pd.DataFrame, scheme: str, group: str, names: pd.Series | None) -> pd.DataFrame:
    frames = []
    for task in TASKS:
        cfg = features.TASKS[task]["primary"]
        for model in LEARNED:
            o, _ = metrics.scored_run(task, cfg, model, scheme, df=df)
            g = metrics.per_group(o, group, min_n=30)
            g.insert(0, "model", model)
            g.insert(0, "task", task)
            frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    if names is not None:
        out.insert(3, "name", out[group].map(names))
    return out


def summarise_groups(g: pd.DataFrame, unit: str) -> pd.DataFrame:
    rows = []
    for (task, model), sub in g.groupby(["task", "model"]):
        e = sub[sub["eligible"]]
        auc = e["roc_auc"]
        rows.append({"task": task, "model": model, "unit": unit, "n_units": len(sub),
                     "n_eligible": len(e), "auc_median": auc.median(), "auc_p25": auc.quantile(.25),
                     "auc_p75": auc.quantile(.75), "auc_min": auc.min(), "auc_max": auc.max(),
                     "share_auc_above_0.5": float((auc > 0.5).mean()),
                     "share_auc_at_least_0.7": float((auc >= 0.7).mean()),
                     "mean_across_units_auc": auc.mean(), "mean_across_units_mcc": e["mcc"].mean()})
    return pd.DataFrame(rows)


def plot_per_ua(g: pd.DataFrame) -> None:
    plotting.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    rng = np.random.default_rng(37)
    for ax, task in zip(axes, TASKS, strict=True):
        for i, model in enumerate(LEARNED):
            v = g[(g["task"] == task) & (g["model"] == model) & g["eligible"]]["roc_auc"].dropna()
            x = i + rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(x, v, s=9, color=plotting.MODEL_COLORS[model], alpha=0.55, linewidths=0)
            q = v.quantile([0.25, 0.5, 0.75])
            ax.plot([i - 0.28, i + 0.28], [q[0.5]] * 2, color=plotting.INK, linewidth=2)
            ax.plot([i, i], [q[0.25], q[0.75]], color=plotting.INK, linewidth=1)
            ax.text(i + 0.31, q[0.5], f"{q[0.5]:.2f}", va="center", fontsize=8,
                    color=plotting.INK)
        ax.axhline(0.5, color=plotting.AXIS, linewidth=1)
        ax.set_xticks(range(len(LEARNED)), [models.MODEL_LABELS[m] for m in LEARNED])
        ax.set_xlim(-0.5, len(LEARNED) - 0.3)
        ax.set_title(features.TASKS[task]["label"])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("ROC-AUC within the held-out UA")
    fig.suptitle("Leave-one-city-out: discrimination inside each held-out urbanized area",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", y=1.02)
    plotting.note(fig, "One dot per UA with >= 30 tracts and both classes; bar = median, "
                       "line = interquartile range.")
    plotting.savefig(fig, OUT / "per_ua_auc_loco.png")


def plot_pooled(p: pd.DataFrame) -> None:
    plotting.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    order = SCHEMES[::-1]
    for ax, task in zip(axes, TASKS, strict=True):
        for j, model in enumerate(LEARNED):
            sub = p[(p["task"] == task) & (p["model"] == model)].set_index("scheme").loc[order]
            y = np.arange(len(order)) + (j - 0.5) * 0.25
            ax.scatter(sub["roc_auc"], y, s=36, color=plotting.MODEL_COLORS[model],
                       edgecolors=plotting.SURFACE, linewidths=1.5, zorder=3,
                       label=models.MODEL_LABELS[model])
        ax.set_yticks(range(len(order)), [validation.SCHEME_LABELS[s] for s in order])
        ax.set_title(features.TASKS[task]["label"])
        ax.set_xlabel("Pooled out-of-fold ROC-AUC")
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    plotting.savefig(fig, OUT / "pooled_auc_by_scheme.png")


def main() -> None:
    df = validation.load_analysis()
    primary = validation.frame("primary", df)
    ua_names = primary.groupby("ua")["ua_name"].first()
    state_names = pd.Series({f: n for f, (_, n, _, _) in geo.STATES.items()})

    p = pooled(df)
    report.write_table(p, OUT / "pooled_metrics")
    loco = per_group(df, "loco", "ua", ua_names)
    loco.to_csv(OUT / "per_ua_loco.csv", index=False)
    state = per_group(df, "state", "state_fips", state_names)
    state.to_csv(OUT / "per_state.csv", index=False)
    region = per_group(df, "region", "region", None)
    region.to_csv(OUT / "per_region.csv", index=False)
    summary = pd.concat([summarise_groups(loco, "UA (leave-one-city-out)"),
                         summarise_groups(state, "state (leave-one-state-out)")],
                        ignore_index=True)
    report.write_table(summary, OUT / "per_group_summary")
    reg = region[["task", "model", "region", "n_tracts", "prevalence", "accuracy", "f1", "mcc",
                  "roc_auc"]]
    report.write_table(reg, OUT / "per_region")
    plot_per_ua(loco)
    plot_pooled(p)


if __name__ == "__main__":
    main()
