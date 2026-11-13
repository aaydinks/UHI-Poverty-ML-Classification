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
            "effect is the group total in the README. Logistic panels are cut at the 0.5th and "
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
    report.require(meta["record_seed"] == interpret.RECORD_SEED, f"{task}: not the record seed")
    if model == "lgbm":
        rows = pd.read_parquet(src / "rows.parquet")
        report.require(len(rows) == meta["additivity"]["n"], f"{task}: cache row count")
    else:
        rows = interpret.logistic_rows(task, df)
    report.require(meta["features"] == features.config(cfg), f"{task}: feature list")
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


# --------------------------------------------------------------------------- README


def plain(group: str) -> str:
    """'Vegetation: annual' -> 'annual vegetation'; 'Water share' -> 'water share'."""
    head, _, tail = group.partition(": ")
    return f"{tail} {head.lower()}" if tail else group.lower()


def comparison(top: pd.DataFrame, grp: pd.DataFrame) -> tuple[str, str, str, str]:
    """The group table per task and the two comparison paragraphs, checked against the data."""
    f = report.fmt_num

    def gtab(task):
        w = grp[grp.task == task].pivot(index="group", columns="model",
                                        values=["mean_abs_group_shap", "rank"])
        w = w.sort_values(("rank", "lgbm"))
        lines = [f"| {g} | {int(w.loc[g, ('rank', 'lgbm')])} | "
                 f"{f(w.loc[g, ('mean_abs_group_shap', 'lgbm')])} | "
                 f"{int(w.loc[g, ('rank', 'logistic')])} | "
                 f"{f(w.loc[g, ('mean_abs_group_shap', 'logistic')])} |" for g in w.index]
        return ("| Attribution group | LightGBM rank | LightGBM mean \\|Σ SHAP\\| | "
                "Logistic rank | Logistic mean \\|Σ SHAP\\| |\n|---|---:|---:|---:|---:|\n"
                + "\n".join(lines))

    def rank(task, model, group):
        g = grp[(grp.task == task) & (grp.model == model)].set_index("group")
        return int(g.loc[group, "rank"]), float(g.loc[group, "mean_abs_group_shap"])

    def order(task, model):
        g = grp[(grp.task == task) & (grp.model == model)].sort_values("rank")
        return g.group.tolist()

    def feats(task, model):
        return top[(top.task == task) & (top.model == model)].reset_index(drop=True)

    # Task A
    la, ra = order("A", "lgbm"), order("A", "logistic")
    fa_l, fa_r = feats("A", "lgbm"), feats("A", "logistic")
    report.require(la[:2] == ra[:2] == ["Vegetation: summer", "Tract size"],
                   "A: both models should rank summer vegetation, then tract size, first")
    report.require(fa_l.mean_abs_shap[0] > 2 * fa_l.mean_abs_shap[1],
                   "A: LightGBM's top feature no longer has twice the next one's mean |SHAP|")
    pair = set(fa_r.feature[:2])
    report.require(pair == {"Total_Count", "Area"} and fa_r.offset_in_group[:2].all(),
                   "A: logistic's top two should be the offsetting Total_Count/Area pair")
    size_r = rank("A", "logistic", "Tract size")[1]
    report.require(size_r < fa_r.mean_abs_shap[1], "A: the tract-size group no longer "
                   "contributes less than either of its two leading members")
    elev_l, elev_r = rank("A", "lgbm", "Elevation"), rank("A", "logistic", "Elevation")
    dem_r = fa_r[fa_r.feature == "DEM_urb_CT_act"]
    report.require(len(dem_r) == 1 and bool(dem_r.offset_in_group.iloc[0])
                   and elev_r[0] > elev_l[0], "A: elevation offsetting no longer holds")
    shared_a = sorted(set(fa_l.feature) & set(fa_r.feature))
    text_a = (
        f"**UHI (panels A and C).** The two models agree on what matters. Both rank the "
        f"summer-vegetation group first and tract size second, and the third place goes to "
        f"{plain(la[2])} for LightGBM and to {plain(ra[2])} for logistic regression. They "
        f"differ in how the vegetation signal is carried. LightGBM sends it almost entirely "
        f"through `DelNDVI_summer` (mean |SHAP| {f(fa_l.mean_abs_shap[0])}, more than twice the "
        f"next feature). Logistic regression spreads it over the urban NDVI terms and the "
        f"dNDVI columns. Logistic regression's two largest single features, `Total_Count` "
        f"({f(fa_r.mean_abs_shap[0])}) and `Area` ({f(fa_r.mean_abs_shap[1])}), largely cancel "
        f"each other: the whole tract-size group adds {f(size_r)}, less than either one alone. "
        f"The same happens to the elevation pair: `DEM_urb_CT_act` is logistic regression's "
        f"third feature, but the elevation group ranks {elev_r[0]} of {len(ra)} "
        f"(LightGBM: {elev_l[0]}). Features in both top 10s: "
        + ", ".join(f"`{s}`" for s in shared_a) + ".")

    # Task B
    lb, rb = order("B", "lgbm"), order("B", "logistic")
    fb_l = feats("B", "lgbm")
    report.require(lb[0] == "Tract size", "B: LightGBM's top group should be tract size")
    report.require(rb[:2] == ["Heat: annual night", "Heat: winter night"],
                   "B: logistic's top two groups should be annual and winter night heat")
    an_l, wn_l = rank("B", "lgbm", "Heat: annual night"), rank("B", "lgbm", "Heat: winter night")
    an_r, wn_r = rank("B", "logistic", "Heat: annual night"), rank("B", "logistic",
                                                                     "Heat: winter night")
    report.require(min(an_l[0], wn_l[0]) > 5, "B: LightGBM should rank night heat low")
    report.require(fb_l.mean_abs_shap[0] < 1.5 * fb_l.mean_abs_shap[2],
                   "B: LightGBM's top three are no longer close")
    veg_l = [g for g in lb[1:4]]
    report.require(all(g.startswith("Vegetation") for g in veg_l),
                   "B: LightGBM's groups 2-4 should be vegetation")
    veg_r = [i + 1 for i, g in enumerate(rb) if g.startswith("Vegetation")]
    shared_b = sorted(set(fb_l.feature) & set(feats("B", "logistic").feature))
    text_b = (
        f"**Poverty (panels B and D).** Here the models disagree. LightGBM's largest group is "
        f"tract size, followed by the three vegetation groups (winter, summer, annual), and its "
        f"attribution is flat: its top three features lie between "
        f"{f(fb_l.mean_abs_shap[2])} and {f(fb_l.mean_abs_shap[0])}. Logistic regression "
        f"leans on night-time surface heat. Its two largest groups are annual-night heat "
        f"({f(an_r[1])}) and winter-night heat ({f(wn_r[1])}), which LightGBM ranks "
        f"{an_l[0]} and {wn_l[0]} of {len(lb)}. Vegetation follows in logistic regression "
        f"(ranks {', '.join(map(str, veg_r))}), and tract size ranks "
        f"{rank('B', 'logistic', 'Tract size')[0]}. Read this as a difference in what each "
        f"model can use: a linear model fits one slope per column, so it picks up the broad "
        f"poverty gradient carried by the many night-LST columns, while the tree ensemble "
        f"gets its discrimination from thresholds and interactions on tract size and "
        f"vegetation. Features in both top 10s: "
        + (", ".join(f"`{s}`" for s in shared_b) if shared_b else "none") + ".")
    return gtab("A"), gtab("B"), text_a, text_b


def write_readme() -> None:
    t = pd.read_csv(OUT / "shap_top_features.csv")
    g = pd.read_csv(OUT / "shap_group_importance.csv")
    f = report.fmt_num

    def block(task, model):
        return "\n".join(
            f"| {r.rank} | `{r.feature}`{' †' if r.offset_in_group else ''} | {r.group} | "
            f"{f(r.mean_abs_shap)} |"
            for r in t[(t.task == task) & (t.model == model)].itertuples())

    gtab_a, gtab_b, text_a, text_b = comparison(t, g)
    head = "| Rank | Feature | Attribution group | Mean \\|SHAP\\| (log-odds) |\n|---:|---|---|---:|"
    text = f"""# SHAP beeswarm

The beeswarm shows the SHAP values of the top 10 features per task for two of the evaluated
models ([`results/shap_beeswarm.png`](results/shap_beeswarm.png)). The top row is LightGBM, a
gradient-boosted tree ensemble; the bottom row is logistic regression. Both use the fixed
settings of `models.py`.

- **Panels A and C, `High_UHI` without temperature input** (`A-noLST`): the models use no
  `LST_*` or `UHI_*` column, because the urban and rural LST terms define the label
  (section 3.1).
- **Panels B and D, `High_Pov` from the ACS poverty rate** (`B-env`): positive SHAP raises the
  predicted probability of *high* poverty.
- Both models are explained out of fold under the UA-grouped 5-fold split (seed
  {interpret.RECORD_SEED}), on the same folds, on the national sample. Each panel shows
  {N_POINTS:,} randomly chosen tracts. Colour is the feature's percentile among all tracts.
  The x-scales differ between panels.
- **LightGBM** values are exact TreeSHAP in log-odds (section 5.1). `shap_beeswarm.py` reads
  the cached attributions and stops if its ranking differs from
  [`5.2/results/feature_importance.csv`](../5_SHAP_Interpretation/5.2_Attribution_and_Stability/results/feature_importance.csv);
  it matches.
- **Logistic regression** values are exact linear SHAP. Every preprocessing step (median
  imputation, `log1p` on the size counts, standardisation) acts on one column, so with the
  training fold as background a feature's SHAP value is its coefficient times its
  standardised value, and the intercept is the base value
  (`interpret.logistic_rows`). The contributions reproduce the cached out-of-fold logistic
  scores exactly.
- **†** marks a top-10 feature whose SHAP values are offset by another top-10 feature of the
  same attribution group (correlation of their SHAP values ≤ {OFFSET_R}). Exact identities
  (a dNDVI column and its two NDVI terms; `Area`, `Total_Count` and `Urban_Count`) stay
  collinear in logistic regression too, and its L2 penalty can give two members large
  coefficients of opposite sign. Their individual rows then overstate what the model uses,
  and only their sum is meaningful. The comparison below is therefore made on attribution
  groups (mean |sum of SHAP| within a group), which is exact for both models.

## LightGBM vs logistic regression

{text_a}

{gtab_a}

{text_b}

{gtab_b}

**Reading the two rows.** Logistic-regression rows always show one clean colour gradient,
because each SHAP value is a linear function of one (transformed) feature. LightGBM rows can
mix colours, which signals non-linear effects and interactions. Compare ranks across the two
rows, not magnitudes: logistic regression's per-feature values are inflated by offsetting
coefficients on collinear columns. SHAP values describe each fitted model's predictions, not
causal effects. The grouped attribution of LightGBM and its stability across 15 fold models
are in [section 5.2](../5_SHAP_Interpretation/5.2_Attribution_and_Stability/README.md).

## Top 10 features

**Panel A — LightGBM, UHI**

{head}
{block("A", "lgbm")}

**Panel B — LightGBM, poverty**

{head}
{block("B", "lgbm")}

**Panel C — logistic regression, UHI**

{head}
{block("A", "logistic")}

**Panel D — logistic regression, poverty**

{head}
{block("B", "logistic")}

All values: [`results/shap_top_features.csv`](results/shap_top_features.csv) and
[`results/shap_group_importance.csv`](results/shap_group_importance.csv).
"""
    report.write_text(text, HERE / "README_shap_beeswarm.md")


def main() -> None:
    data = load()
    top = pd.concat([d["top"] for d in data.values()], ignore_index=True)
    check_against_5_2(top)
    top["matches_5_2"] = np.where(top.model == "lgbm", "yes", "–")
    report.write_table(top, OUT / "shap_top_features", digits=3,
                       title="SHAP beeswarm: top 10 features per task and model by mean |SHAP|",
                       note="Out-of-fold SHAP (log-odds), all tracts of the primary frame, UA-grouped "
                            f"5-fold, seed {interpret.RECORD_SEED}. `offset_in_group`: offset by "
                            "another top-10 feature of the same attribution group (SHAP "
                            f"correlation ≤ {OFFSET_R}). `matches_5_2`: same feature and value at "
                            "the same rank in 5.2/results/feature_importance.csv (LightGBM).\n")
    groups = pd.concat([d["groups"] for d in data.values()], ignore_index=True)
    report.write_table(groups, OUT / "shap_group_importance", digits=3,
                       title="SHAP by attribution group, per task and model",
                       note="Mean over tracts of |sum of SHAP within the group| (log-odds); "
                            "rank 1 = largest.\n")
    plot(data)
    write_readme()


if __name__ == "__main__":
    main()
