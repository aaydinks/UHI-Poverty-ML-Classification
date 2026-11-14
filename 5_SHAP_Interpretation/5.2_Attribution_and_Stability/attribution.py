"""Section 5.2: out-of-fold attributions, grouped importance and their stability.

From the cached attributions (`interpret.compute`): mean |SHAP| per feature and
mean |sum of SHAP| per attribution group, pooled over the seed-37 held-out folds;
grouped permutation importance (drop in ROC-AUC); each group's rank in every one
of the 15 fold models (seeds 37-39 × 5 folds); and the pairwise Spearman
correlation of group importances across those models. Writes the tables, a
grouped-importance figure and a SHAP summary of the top features.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uhi_poverty import features, interpret, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
DELTAS = paths.ROOT / "4_Model_Specifications" / "4.5_Baselines" / "results" / "paired_deltas.csv"
TOP = 10
N_POINTS = 2500


def group_table(task: str, r: dict) -> pd.DataFrame:
    m = r["meta"]
    gs = interpret.group_sums(r["rows"][m["features"]], m["groups"]).abs().mean()
    ranks = interpret.group_ranks(r["importance"]).groupby("name")["rank"]
    perm = r["permutation"].groupby("group")["auc_drop"]
    t = pd.DataFrame({
        "task": task, "group": gs.index, "n_features": [len(m["groups"][g]) for g in gs.index],
        "mean_abs_shap": gs.to_numpy(), "share": (gs / gs.sum()).to_numpy(),
    })
    t["rank"] = t["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    t["rank_min"] = t.group.map(ranks.min()).astype(int)
    t["rank_max"] = t.group.map(ranks.max()).astype(int)
    t["rank_stable"] = (t.rank_max - t.rank_min) <= interpret.STABLE_RANK_SPREAD
    t["perm_auc_drop"] = t.group.map(perm.mean())
    t["perm_auc_drop_min"] = t.group.map(perm.min())
    t["perm_auc_drop_max"] = t.group.map(perm.max())
    return t.sort_values("rank").reset_index(drop=True)


def feature_table(task: str, r: dict) -> pd.DataFrame:
    m = r["meta"]
    s = r["rows"][m["features"]].abs().mean().sort_values(ascending=False)
    group_of = {f: g for g, members in m["groups"].items() for f in members}
    return pd.DataFrame({"task": task, "rank": np.arange(1, len(s) + 1), "feature": s.index,
                         "group": [group_of[f] for f in s.index],
                         "mean_abs_shap": s.to_numpy()})


def stability(task: str, r: dict) -> dict:
    g = r["importance"][r["importance"].level == "group"]
    piv = g.pivot_table(index="name", columns=["seed", "fold"], values="mean_abs")
    rho = [piv[a].corr(piv[b], method="spearman") for a, b in combinations(piv.columns, 2)]
    return {"task": task, "fold_models": piv.shape[1], "groups": piv.shape[0],
            "mean_pairwise_spearman": float(np.mean(rho)),
            "min_pairwise_spearman": float(np.min(rho))}


def plot_groups(gt: pd.DataFrame, res: dict) -> None:
    plotting.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6),
                             gridspec_kw={"width_ratios": [6, 12]})
    for ax, task in zip(axes, interpret.TASKS, strict=True):
        t = gt[gt.task == task].iloc[::-1]
        y = np.arange(len(t))
        colors = [plotting.BLUE if s else plotting.AXIS for s in t.rank_stable]
        ax.barh(y, t.mean_abs_shap, color=colors, height=0.62)
        folds = res[task]["importance"].query("level == 'group'")
        for yi, g in zip(y, t.group, strict=True):
            v = folds.loc[folds.name == g, "mean_abs"]
            ax.scatter(v, np.full(len(v), yi), s=6, color=plotting.INK_2, zorder=3,
                       linewidths=0)
        ax.set_yticks(y, t.group)
        ax.set_xlabel("mean |sum of SHAP| within group (log-odds)")
        ax.set_title(features.TASKS[task]["label"])
        ax.grid(axis="y", visible=False)
    fig.suptitle("Grouped attribution, out of fold (UA-grouped 5-fold)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    plotting.note(fig, "Bars: pooled seed-37 held-out folds. Dots: each of the 15 fold models "
                       "(seeds 37–39). Blue = rank moves by at most "
                       f"{interpret.STABLE_RANK_SPREAD} places across the 15 models; grey = "
                       "unstable, not interpreted.")
    fig.tight_layout()
    plotting.savefig(fig, OUT / "group_importance.png")


def plot_summary(ft: pd.DataFrame, res: dict, df: pd.DataFrame) -> None:
    plotting.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"wspace": 0.6})
    rng = np.random.default_rng(interpret.RECORD_SEED)
    cmap = plotting.suhi_cmap()
    for ax, task in zip(axes, interpret.TASKS, strict=True):
        rows = res[task]["rows"]
        top = ft[ft.task == task].head(TOP).feature.tolist()[::-1]
        values = interpret.feature_values(rows, top, df)
        take = rng.choice(len(rows), size=min(N_POINTS, len(rows)), replace=False)
        for yi, f in enumerate(top):
            phi = rows[f].to_numpy()[take]
            pct = values[f].rank(pct=True).to_numpy()[take]
            jitter = rng.uniform(-0.28, 0.28, len(take))
            ax.scatter(phi, yi + jitter, c=pct, cmap=cmap, vmin=0, vmax=1, s=3, linewidths=0,
                       rasterized=True)
        ax.axvline(0, color=plotting.AXIS, lw=0.8, zorder=0)
        ax.set_yticks(range(len(top)), top)
        ax.set_xlabel("SHAP value (log-odds of the positive class)")
        ax.set_title(f"{features.TASKS[task]['label']}: top {TOP} features")
        ax.grid(axis="y", visible=False)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("feature value (percentile)")
    fig.suptitle("Per-feature SHAP values, out of fold", x=0.01, ha="left", fontweight="bold",
                 fontsize=10)
    plotting.note(fig, f"{N_POINTS:,} random tracts per task. Credit within an exact identity "
                       "(e.g. a SUHI column and its two LST terms) is split arbitrarily; read "
                       "importance from the grouped figure.")
    plotting.savefig(fig, OUT / "shap_summary.png")


def main() -> None:
    df = validation.load_analysis()
    res = {t: interpret.compute(t, df=df) for t in interpret.TASKS}
    gt = pd.concat([group_table(t, r) for t, r in res.items()], ignore_index=True)
    ft = pd.concat([feature_table(t, r) for t, r in res.items()], ignore_index=True)
    st = pd.DataFrame([stability(t, r) for t, r in res.items()])
    report.write_table(gt, OUT / "group_importance")
    report.write_table(ft, OUT / "feature_importance")
    report.write_table(st, OUT / "stability")
    plot_groups(gt, res)
    plot_summary(ft, res, df)


if __name__ == "__main__":
    main()
