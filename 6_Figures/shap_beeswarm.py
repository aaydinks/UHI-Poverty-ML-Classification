"""SHAP beeswarm, top 10 features per task, LightGBM (top row) and logistic regression (bottom row).

One beeswarm per task and model, the top 10 features by mean |SHAP|, points coloured
from low to high feature value. Both models are the fixed-setting models evaluated
under the UA-grouped 5-fold split, explained out of fold on the same folds (seed 37):

- LightGBM, a gradient-boosted tree ensemble: the cached TreeSHAP of section 5.1
  (Task A `High_UHI` from `A-noLST`, no temperature input; Task B `High_Pov` from
  `B-env`, ACS poverty rate). Only the record seed's attribution is read; nothing is
  refitted. The script stops if its top 10 differ from `5.2/results/feature_importance.csv`.
- Logistic regression: exact linear SHAP (`interpret.logistic_rows`), checked against
  the cached out-of-fold logistic scores.

SHAP values are in log-odds of the positive class. Features whose SHAP values are
offset by another top-10 member of the same attribution group (Pearson r <= -0.9
between the two features' SHAP values) are marked with a dagger.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uhi_poverty import features, interpret, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
SOURCE = paths.ROOT / "5_SHAP_Interpretation" / "5.2_Attribution_and_Stability" / "results"
TOP = 10
N_POINTS = 3000
N_SWARM_BINS = 60
CLIP_Q = (0.005, 0.995)  # logistic panels: x-range from these quantiles of the panel's SHAP
OFFSET_R = -0.9  # SHAP correlation at or below which two members of a group offset each other
MODELS = ("lgbm", "logistic")
MODEL_LABELS = {"lgbm": "LightGBM (gradient-boosted tree ensemble)",
                "logistic": "Logistic regression"}
TASK_LABELS = {"A": "High surface UHI (no temperature input)",
               "B": "High poverty (ACS rate)"}
LETTERS = {("lgbm", "A"): "A", ("lgbm", "B"): "B", ("logistic", "A"): "C",
           ("logistic", "B"): "D"}
SUPTITLE = "SHAP feature attribution — top 10 features per task"
XLABEL = "SHAP value (impact on model output, log-odds)"
FOOTNOTE = (f"Out-of-fold SHAP of the fixed-setting models (UA-grouped 5-fold, seed "
            f"{interpret.RECORD_SEED}); {N_POINTS:,} random tracts per panel. LightGBM: exact "
            "TreeSHAP. Logistic regression: exact linear SHAP (coefficient × standardised value, "
            "training-fold background). Colour: the feature's percentile among all tracts. "
            "x-scales differ between panels. † Offset by another feature of the same exact "
            f"identity or attribution group (SHAP correlation ≤ {OFFSET_R}); their joint "
            "effect is the group total. Logistic panels are cut at the 0.5th and "
            "99.5th percentiles of their SHAP values; the numbers at the panel edges count the points beyond.")


def beeswarm(ax, phi: pd.DataFrame, pct: pd.DataFrame, rng: np.random.Generator,
             labels: list[str] | None = None, clip: tuple[float, float] | None = None) -> None:
    """One row per column of `phi` (first column on top), jittered by local density.

    `pct` holds each point's feature-value percentile (0-1, NaN for a missing value).
    With `clip`, points outside (lo, hi) are left out and counted at the row's edge.
    """
    cmap = plotting.suhi_cmap()
    k = phi.shape[1]
    lo, hi = clip or (np.nanmin(phi.to_numpy()), np.nanmax(phi.to_numpy()))
    edges = np.linspace(lo, hi, N_SWARM_BINS + 1)
    for i, f in enumerate(phi.columns):
        y0 = k - 1 - i
        x = phi[f].to_numpy()
        keep = (x >= lo) & (x <= hi)
        if clip:
            pad = 0.01 * (hi - lo)
            for n, xe, ha, txt in [((x < lo).sum(), lo - pad, "left", r"$\leftarrow$ {}"),
                                   ((x > hi).sum(), hi + pad, "right", r"{} $\rightarrow$")]:
                if n:
                    ax.text(xe, y0 + 0.28, txt.format(n), ha=ha, va="center", fontsize=6.5,
                            color=plotting.INK_2)
        x = x[keep]
        b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, N_SWARM_BINS - 1)
        counts = np.bincount(b, minlength=N_SWARM_BINS)
        width = 0.4 * counts[b] / counts.max()
        y = y0 + rng.uniform(-1, 1, len(x)) * width
        c = pct[f].to_numpy()[keep]
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
    if clip:
        ax.set_xlim(lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=plotting.GRID, lw=0.5, ls=":")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def colorbar(fig, axes) -> None:
    sm = plt.cm.ScalarMappable(cmap=plotting.suhi_cmap())
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.015, aspect=30)
    cb.set_ticks([0, 1], labels=["Low", "High"])
    cb.set_label("Feature value", labelpad=-6)
    cb.outline.set_visible(False)


# --------------------------------------------------------------------------- data


def model_rows(task: str, model: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-tract SHAP values of one model (record seed) and the task's metadata."""
    cfg = interpret.config(task)
    src = paths.INTERPRET_DIR / f"{task}__{cfg}"
    meta = json.loads((src / "meta.json").read_text())
    report.check(meta["record_seed"] == interpret.RECORD_SEED, f"{task}: not the record seed")
    if model == "lgbm":
        rows = pd.read_parquet(src / "rows.parquet")
        report.check(len(rows) == meta["additivity"]["n"], f"{task}: cache row count")
    else:
        rows = interpret.logistic_rows(task, df)
    report.check(meta["features"] == features.config(cfg), f"{task}: feature list")
    return rows, meta


def offset_pairs(phi: pd.DataFrame, group_of: dict[str, str]) -> list[tuple[str, str, float]]:
    """Pairs of top features in one attribution group whose SHAP values offset each other."""
    cols = list(phi.columns)
    r = phi.corr()
    return [(a, b, float(r.loc[a, b])) for i, a in enumerate(cols) for b in cols[i + 1:]
            if group_of[a] == group_of[b] and r.loc[a, b] <= OFFSET_R]


def load_task(task: str, model: str, df: pd.DataFrame) -> dict:
    """Top features, the plotted sample, group importance and offsetting pairs."""
    rows, meta = model_rows(task, model, df)
    cols, groups = meta["features"], meta["groups"]
    group_of = {f: g for g, members in groups.items() for f in members}
    imp = rows[cols].abs().mean().sort_values(ascending=False)
    names = list(imp.index[:TOP])
    pairs = offset_pairs(rows[names], group_of)
    flagged = {f for p in pairs for f in p[:2]}
    top = pd.DataFrame({"task": task, "model": model, "rank": np.arange(1, TOP + 1),
                        "feature": names, "group": [group_of[f] for f in names],
                        "mean_abs_shap": imp.to_numpy()[:TOP],
                        "offset_in_group": [f in flagged for f in names]})
    g = interpret.group_sums(rows[cols], groups).abs().mean().sort_values(ascending=False)
    grp = pd.DataFrame({"task": task, "model": model, "group": g.index,
                        "n_features": [len(groups[k]) for k in g.index],
                        "mean_abs_group_shap": g.to_numpy(),
                        "rank": np.arange(1, len(g) + 1)})
    pct = interpret.feature_values(rows, names, df).rank(pct=True)
    take = np.random.default_rng(interpret.RECORD_SEED).choice(
        len(rows), size=min(N_POINTS, len(rows)), replace=False)
    return {"top": top, "groups": grp, "pairs": pairs, "phi": rows[names].iloc[take],
            "pct": pct.iloc[take], "labels": [f + (" †" if f in flagged else "") for f in names]}


def check_against_5_2(top: pd.DataFrame) -> None:
    fi = pd.read_csv(SOURCE / "feature_importance.csv")
    for task, t in top[top.model == "lgbm"].groupby("task"):
        ref = fi[fi.task == task].sort_values("rank").head(TOP)
        if ref.feature.tolist() != t.feature.tolist() or \
                not np.allclose(ref.mean_abs_shap, t.mean_abs_shap):
            raise AssertionError(f"{task}: top {TOP} differ from 5.2/results/feature_importance")


def load(df: pd.DataFrame | None = None) -> dict:
    df = validation.load_analysis() if df is None else df
    return {(m, t): load_task(t, m, df) for m in MODELS for t in interpret.TASKS}


# --------------------------------------------------------------------------- figure


def plot(data: dict) -> None:
    plotting.apply_style()
    fig = plt.figure(figsize=(12.5, 10.6))
    rows = fig.subfigures(2, 1, hspace=0.04)
    for sub, model in zip(rows, MODELS, strict=True):
        axes = sub.subplots(1, 2, gridspec_kw={"wspace": 0.6})
        for ax, task in zip(axes, interpret.TASKS, strict=True):
            d = data[(model, task)]
            clip = (tuple(np.nanquantile(d["phi"].to_numpy(), CLIP_Q))
                    if model == "logistic" else None)
            beeswarm(ax, d["phi"], d["pct"], np.random.default_rng(interpret.RECORD_SEED),
                     labels=d["labels"], clip=clip)
            ax.set_xlabel(XLABEL)
            ax.set_title(f"({LETTERS[(model, task)]})  {TASK_LABELS[task]}")
        colorbar(sub, axes)
        sub.suptitle(MODEL_LABELS[model], x=0.01, ha="left", fontweight="bold", fontsize=11,
                     color=plotting.INK)
    fig.suptitle(SUPTITLE, x=0.01, y=1.035, ha="left", fontweight="bold", fontsize=12)
    plotting.note(fig, FOOTNOTE, y=-0.0)
    plotting.savefig(fig, OUT / "shap_beeswarm.png")




def main() -> None:
    data = load()
    top = pd.concat([d["top"] for d in data.values()], ignore_index=True)
    check_against_5_2(top)
    top["matches_5_2"] = np.where(top.model == "lgbm", "yes", "–")
    report.write_table(top, OUT / "shap_top_features")
    groups = pd.concat([d["groups"] for d in data.values()], ignore_index=True)
    report.write_table(groups, OUT / "shap_group_importance")
    plot(data)


if __name__ == "__main__":
    main()
