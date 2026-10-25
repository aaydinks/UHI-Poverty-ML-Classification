"""SHAP beeswarm, top 10 features per task.

One beeswarm per task, the top 10 features by mean |SHAP|, points coloured from
low to high feature value. The data are the cached out-of-fold TreeSHAP of the
two evaluated LightGBM models (section 5.1), Task A `High_UHI` from `A-noLST` (no
temperature input) and Task B `High_Pov` from `B-env` (ACS poverty rate). Only
the record seed's attribution is read; nothing is refitted. SHAP values are in
log-odds of the positive class.

The script stops if its top 10 differ from `5.2/results/feature_importance.csv`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uhi_poverty import interpret, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
SOURCE = paths.ROOT / "5_SHAP_Interpretation" / "5.2_Attribution_and_Stability" / "results"
TOP = 10
N_POINTS = 3000
N_SWARM_BINS = 60
SUPTITLE = "SHAP feature attribution — top 10 features per task"
TITLES = {"A": "(A)  High surface UHI (no temperature input)",
          "B": "(B)  High poverty (ACS rate)"}
XLABEL = "SHAP value (impact on model output, log-odds)"
FOOTNOTE = (f"Out-of-fold TreeSHAP of the evaluated LightGBM models (UA-grouped 5-fold, seed "
            f"{interpret.RECORD_SEED}); {N_POINTS:,} random tracts per task. Colour: the feature's "
            "percentile among all tracts. Credit within an exact identity (e.g. a dNDVI column and "
            "its two NDVI terms) is split arbitrarily between its members; read importance from "
            "the grouped attribution of section 5.2.")


def beeswarm(ax, phi: pd.DataFrame, pct: pd.DataFrame, rng: np.random.Generator,
             labels: list[str] | None = None) -> None:
    """One row per column of `phi` (first column on top), jittered by local density.

    `pct` holds each point's feature-value percentile (0-1, NaN for a missing value).
    """
    cmap = plotting.suhi_cmap()
    k = phi.shape[1]
    lo, hi = np.nanmin(phi.to_numpy()), np.nanmax(phi.to_numpy())
    edges = np.linspace(lo, hi, N_SWARM_BINS + 1)
    for i, f in enumerate(phi.columns):
        y0 = k - 1 - i
        x = phi[f].to_numpy()
        b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, N_SWARM_BINS - 1)
        counts = np.bincount(b, minlength=N_SWARM_BINS)
        width = 0.4 * counts[b] / counts.max()
        y = y0 + rng.uniform(-1, 1, len(x)) * width
        c = pct[f].to_numpy()
        missing = np.isnan(c)
        order = rng.permutation(len(x))
        ax.scatter(x[order], y[order], c=np.where(missing, 0.5, c)[order], cmap=cmap, vmin=0,
                   vmax=1, s=3, linewidths=0, rasterized=True)
        if missing.any():
            ax.scatter(x[missing], y[missing], color=plotting.MUTED, s=3, linewidths=0,
                       rasterized=True)
    ax.axvline(0, color=plotting.INK_2, lw=0.8, zorder=0)
    ax.set_yticks(range(k), (labels or list(phi.columns))[::-1])
    ax.set_ylim(-0.6, k - 0.4)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=plotting.GRID, lw=0.5, ls=":")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def colorbar(fig, axes) -> None:
    sm = plt.cm.ScalarMappable(cmap=plotting.suhi_cmap())
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.015, aspect=40)
    cb.set_ticks([0, 1], labels=["Low", "High"])
    cb.set_label("Feature value", labelpad=-6)
    cb.outline.set_visible(False)


def load_task(task: str, df: pd.DataFrame) -> dict:
    """The record seed's per-tract SHAP values and inputs for one task's top features."""
    cfg = interpret.config(task)
    src = paths.INTERPRET_DIR / f"{task}__{cfg}"
    meta = json.loads((src / "meta.json").read_text())
    report.require(meta["record_seed"] == interpret.RECORD_SEED, f"{task}: not the record seed")
    rows = pd.read_parquet(src / "rows.parquet")
    report.require(len(rows) == meta["additivity"]["n"], f"{task}: cache row count")
    cols = meta["features"]
    group_of = {f: g for g, members in meta["groups"].items() for f in members}
    imp = rows[cols].abs().mean().sort_values(ascending=False)
    top = pd.DataFrame({"task": task, "rank": np.arange(1, TOP + 1), "feature": imp.index[:TOP],
                        "group": [group_of[f] for f in imp.index[:TOP]],
                        "mean_abs_shap": imp.to_numpy()[:TOP]})
    names = top.feature.tolist()
    pct = interpret.feature_values(rows, names, df).rank(pct=True)
    take = np.random.default_rng(interpret.RECORD_SEED).choice(
        len(rows), size=min(N_POINTS, len(rows)), replace=False)
    return {"top": top, "phi": rows[names].iloc[take], "pct": pct.iloc[take],
            "n_tracts": len(rows), "config": cfg}


def check_against_5_2(top: pd.DataFrame) -> pd.DataFrame:
    fi = pd.read_csv(SOURCE / "feature_importance.csv")
    for task, t in top.groupby("task"):
        ref = fi[fi.task == task].sort_values("rank").head(TOP)
        if ref.feature.tolist() != t.feature.tolist() or \
                not np.allclose(ref.mean_abs_shap, t.mean_abs_shap):
            raise AssertionError(f"{task}: top {TOP} differ from 5.2/results/feature_importance")
    return top.assign(matches_5_2=True)


def load(df: pd.DataFrame | None = None) -> dict:
    df = validation.load_analysis() if df is None else df
    return {t: load_task(t, df) for t in interpret.TASKS}


def draw(fig, axes, data: dict, titles: dict | None = None) -> None:
    """The two panels on `axes` (Task A, Task B), with a shared colour bar."""
    titles = titles or TITLES
    for ax, task in zip(axes, interpret.TASKS, strict=True):
        beeswarm(ax, data[task]["phi"], data[task]["pct"],
                 np.random.default_rng(interpret.RECORD_SEED))
        ax.set_xlabel(XLABEL)
        ax.set_title(titles[task])
    colorbar(fig, axes)


def plot(data: dict) -> None:
    plotting.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"wspace": 0.55})
    draw(fig, axes, data)
    fig.suptitle(SUPTITLE, x=0.01, ha="left", fontweight="bold", fontsize=11)
    plotting.note(fig, FOOTNOTE, y=-0.0)
    plotting.savefig(fig, OUT / "shap_beeswarm.png")


def write_readme() -> None:
    t = pd.read_csv(OUT / "shap_top_features.csv")
    f = report.fmt_num

    def block(task):
        return "\n".join(f"| {r.rank} | `{r.feature}` | {r.group} | {f(r.mean_abs_shap)} |"
                         for r in t[t.task == task].itertuples())

    a, b = t[t.task == "A"], t[t.task == "B"]
    # The prose below states these outcomes; fail rather than publish it if they change.
    report.require(a.mean_abs_shap.iloc[0] > 2 * a.mean_abs_shap.iloc[1],
                   "panel A's top feature no longer has twice the next one's mean |SHAP|")
    report.require(b.mean_abs_shap.iloc[0] < 1.5 * b.mean_abs_shap.iloc[2],
                   "panel B's top three are no longer close")
    head = "| Rank | Feature | Attribution group | Mean \\|SHAP\\| (log-odds) |\n|---:|---|---|---:|"
    text = f"""# SHAP beeswarm

The beeswarm shows the SHAP values of the top 10 features per task for the two evaluated models
([`results/shap_beeswarm.png`](results/shap_beeswarm.png)):

- **Panel A, `High_UHI` without temperature input** (`A-noLST`): the model uses no `LST_*` or
  `UHI_*` column, because the urban and rural LST terms define the label (section 3.1).
- **Panel B, `High_Pov` from the ACS poverty rate** (`B-env`): positive SHAP raises the
  predicted probability of *high* poverty.
- SHAP values are LightGBM's exact TreeSHAP in log-odds, computed out of fold under the
  UA-grouped 5-fold split (section 5.1, seed {interpret.RECORD_SEED}) on the national sample;
  each panel shows {N_POINTS:,} randomly chosen tracts. Colour is the feature's percentile among
  all tracts.
- `shap_beeswarm.py` reads the cached attributions and stops if its ranking differs from
  [`5.2/results/feature_importance.csv`](../5_SHAP_Interpretation/5.2_Attribution_and_Stability/results/feature_importance.csv);
  it matches.

**Panel A**

{head}
{block("A")}

**Panel B**

{head}
{block("B")}

`{a.iloc[0].feature}` leads panel A with a mean |SHAP| of {f(a.iloc[0].mean_abs_shap)}, more than
twice the next feature. Panel B is flatter: its top three features lie between
{f(b.mean_abs_shap.iloc[2])} and {f(b.mean_abs_shap.iloc[0])}.

**Reading the figure.** Several features belong to exact identities (a dNDVI column and its two
NDVI terms, or `Area`, `Total_Count` and `Urban_Count`), and TreeSHAP splits the credit between
the members of an identity arbitrarily. Rank per feature is therefore not interpretable on its
own; the grouped attribution and its stability across 15 fold models are in
[section 5.2](../5_SHAP_Interpretation/5.2_Attribution_and_Stability/README.md). SHAP values
describe the fitted model's predictions, not causal effects.
"""
    report.write_text(text, HERE / "README_shap_beeswarm.md")


def main() -> None:
    data = load()
    top = check_against_5_2(pd.concat([d["top"] for d in data.values()], ignore_index=True))
    report.write_table(top, OUT / "shap_top_features", digits=3,
                       title="SHAP beeswarm: top 10 features per task by mean |SHAP|",
                       note="Out-of-fold TreeSHAP (log-odds), all tracts of the primary frame, "
                            f"seed {interpret.RECORD_SEED}. `matches_5_2`: same feature and value "
                            "at the same rank in 5.2/results/feature_importance.csv.\n")
    plot(data)
    write_readme()


if __name__ == "__main__":
    main()
