"""Section 5.2: out-of-fold attributions, grouped importance and their stability.

From the cached attributions (`interpret.compute`): mean |SHAP| per feature and
mean |sum of SHAP| per attribution group, pooled over the seed-37 held-out folds;
grouped permutation importance (drop in ROC-AUC); each group's rank in every one
of the 15 fold models (seeds 37-39 × 5 folds); and the pairwise Spearman
correlation of group importances across those models. Writes the tables, a
grouped-importance figure, a SHAP summary of the top features, and the README.
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


def prose(group: str) -> str:
    """'Heat: winter night' -> 'winter-night heat'; 'Tract size' -> 'tract size'."""
    kind, _, when = group.partition(": ")
    return f"{when.replace(' ', '-')} {kind.lower()}" if when else group.lower()


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
    report.write_table(gt, OUT / "group_importance", digits=3,
                       title="Grouped attribution and permutation importance, out of fold",
                       note="`mean_abs_shap`: mean over tracts of |sum of SHAP within the group| "
                            "(log-odds), seed-37 held-out folds. `rank_min`/`rank_max`: across "
                            "the 15 fold models. `perm_auc_drop`: mean drop in held-out ROC-AUC "
                            "when the group's columns are permuted together (15 fold models × "
                            f"{interpret.N_REPEATS} draws).\n")
    report.write_table(ft, OUT / "feature_importance", digits=3,
                       title="Per-feature mean |SHAP|, out of fold (log-odds)")
    report.write_table(st, OUT / "stability", digits=3,
                       title="Agreement of group importances across the 15 fold models")
    plot_groups(gt, res)
    plot_summary(ft, res, df)
    write_readme()


def write_readme() -> None:
    gt = pd.read_csv(OUT / "group_importance.csv")
    st = pd.read_csv(OUT / "stability.csv").set_index("task")
    d = pd.read_csv(DELTAS)
    heat = d[(d.task == "B") & (d.metric == "roc_auc")
             & (d.comparison == "LightGBM (B-env) − LightGBM (no heat variables)")]
    heat = heat.set_index("scheme")
    f = report.fmt_num

    def block(task):
        t = gt[gt.task == task]
        return "\n".join(
            f"| {r.rank} | {r.group} | {r.n_features} | {f(r.mean_abs_shap)} | "
            f"{report.fmt_pct(r.share)} | {r.rank_min}–{r.rank_max} | "
            f"{'yes' if r.rank_stable else 'no'} | {f(r.perm_auc_drop)} |"
            for r in t.itertuples())

    a, b = gt[gt.task == "A"], gt[gt.task == "B"]
    b_heat = b[b.group.str.startswith("Heat")]
    b_heat_share = b_heat.share.sum()
    b_stable = b[b.rank_stable].group.tolist()
    # The prose below states these outcomes; fail rather than publish it if they change.
    for t in (a, b):
        report.require(t.iloc[0].rank_min == t.iloc[0].rank_max == 1,
                       f"{t.iloc[0].group} no longer ranks first in every fold model")
    report.require(not b_heat.rank_stable.any(), "a heat group now holds a stable rank")
    head = ("| Rank | Group | Columns | Mean \\|Σ SHAP\\| | Share | Rank across 15 models | "
            "Stable | Δ ROC-AUC when permuted |\n|---:|---|---:|---:|---:|---|---|---:|")
    text = f"""# 5.2 Attribution and stability

## Purpose

Ranks the input groups by their contribution to the two evaluated models' out-of-fold
predictions, and checks whether those ranks hold across refits. SHAP values describe
contributions to a fitted model's predictions, not causes.

## Method

- `attribution.py` reads the out-of-fold attributions of 5.1 for the two evaluated models:
  Task A, `High_UHI` from `A-noLST` (no temperature input), and Task B, `High_Pov` (ACS
  poverty rate) from `B-env`.
- Importance per attribution group: the mean over tracts of |sum of the group's SHAP values|,
  pooled over the held-out folds. Groups keep every exact identity whole (5.1).
- Grouped permutation importance: the drop in held-out ROC-AUC when all of a group's columns are
  permuted together, a check that does not depend on SHAP.
- Stability: the folds were redrawn with seeds 38 and 39, giving 15 fold models. A group's rank
  is interpreted only if it moves by at most {interpret.STABLE_RANK_SPREAD} places across all 15
  (this rule was fixed before the results were seen).

## Results

**Task A (`High_UHI`, no temperature input)**

{head}
{block("A")}

**Task B (`High_Pov`, ACS poverty)**

{head}
{block("B")}

Group importances agree closely across the 15 fold models (mean pairwise Spearman
{f(st.loc['A', 'mean_pairwise_spearman'])} on Task A and
{f(st.loc['B', 'mean_pairwise_spearman'])} on Task B; minimum
{f(st.loc['A', 'min_pairwise_spearman'])} and {f(st.loc['B', 'min_pairwise_spearman'])};
[`results/stability.md`](results/stability.md)).

- **Task A.** {a.iloc[0].group} ranks first in every fold model
  ({report.fmt_pct(a.iloc[0].share)} of the grouped attribution), followed by
  {prose(a.iloc[1].group)}. With every temperature input removed, the model relies on
  vegetation and tract geometry.
- **Task B.** {b.iloc[0].group} ranks first in every fold model
  ({report.fmt_pct(b.iloc[0].share)}). The six heat groups together carry
  {report.fmt_pct(b_heat_share)} of the grouped attribution, but none of them holds a stable
  rank (ranks {int(b_heat.rank_min.min())}–{int(b_heat.rank_max.max())}). The stable groups are
  {", ".join(prose(g) for g in b_stable[:-1])} and {prose(b_stable[-1])}.

**Attribution is not added value.** A heat group can receive attribution because it substitutes
for correlated non-thermal inputs. Removing every heat variable from the poverty model changes
its ROC-AUC by {f(heat.loc['ua5', 'delta'])} [{f(heat.loc['ua5', 'ci_low'])},
{f(heat.loc['ua5', 'ci_high'])}] under UA-grouped 5-fold and by {f(heat.loc['loco', 'delta'])}
[{f(heat.loc['loco', 'ci_low'])}, {f(heat.loc['loco', 'ci_high'])}] under leave-one-city-out
(section 4.5). The heat attributions in Task B therefore do not show that heat carries
information about poverty beyond vegetation, elevation and tract geometry.

Figures: [`results/group_importance.png`](results/group_importance.png),
[`results/shap_summary.png`](results/shap_summary.png) (per-feature values; the credit within an
identity is arbitrary). Per-feature means are in
[`results/feature_importance.md`](results/feature_importance.md).
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
