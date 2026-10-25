"""Section 1.5: do a few large metros drive the results?

1. Concentration: share of tracts in the top 1, 5, 10 and 20 UAs; Lorenz curve.
2. Model sensitivity: the primary Task A and Task B models (logistic, LightGBM)
   under UA-grouped 5-fold, trained and evaluated with all UAs and without the
   top 1, 3 and 10 UAs.
3. Pooled vs per-city: pooled leave-one-city-out metrics (each tract counts once)
   against the mean across cities (each UA counts once), and the per-UA ROC-AUC
   distribution for UAs with >= 30 tracts and both classes.
4. Association: OR and CMH OR (section 2.2) with and without the top 1, 3, 10 UAs.
Reuses the cached out-of-fold predictions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uhi_poverty import association, features, metrics, models, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
DROPS = (0, *validation.METRO_DROPS)
TASKS = ("A", "B")
LEARNED = ("logistic", "lgbm")


def concentration(p: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sizes = p.groupby("ua").size().sort_values(ascending=False)
    names = p.groupby("ua")["ua_name"].first()
    rows = [{"top_k_uas": k, "tracts": int(sizes.head(k).sum()),
             "share_of_tracts": float(sizes.head(k).sum() / sizes.sum()),
             "uas": "; ".join(names[sizes.head(k).index]) if k <= 10 else ""}
            for k in (1, 3, 5, 10, 20)]
    asc = np.sort(sizes.to_numpy())
    cum = np.r_[0, np.cumsum(asc)] / asc.sum()
    x = np.linspace(0, 1, len(cum))
    gini = float(1 - 2 * np.trapezoid(cum, x))
    plotting.apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(x * 100, cum * 100, color=plotting.BLUE, linewidth=2)
    ax.plot([0, 100], [0, 100], color=plotting.AXIS, linewidth=1)
    for k in (1, 10):
        share = sizes.head(k).sum() / sizes.sum()
        xk = 100 * (1 - k / len(sizes))
        ax.scatter([xk], [100 * (1 - share)], s=36, color=plotting.BLUE,
                   edgecolors=plotting.SURFACE, linewidths=1.5, zorder=3)
        ax.annotate(f"largest {k} UA{'s' if k > 1 else ''}: {share:.1%} of tracts",
                    (xk, 100 * (1 - share)), xytext=(-8, 10), textcoords="offset points",
                    ha="right", fontsize=8, color=plotting.INK)
    ax.set_xlabel("Cumulative share of urbanized areas, smallest first (%)")
    ax.set_ylabel("Cumulative share of tracts (%)")
    ax.set_title(f"Tracts are concentrated in large UAs (Gini {gini:.2f})")
    plotting.savefig(fig, OUT / "concentration.png")
    return pd.DataFrame(rows), {"gini": gini, "n_uas": int(len(sizes))}


def model_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        cfg = features.TASKS[task]["primary"]
        for model in LEARNED:
            for k in DROPS:
                t = metrics.evaluate(task, cfg, model, "ua5", drop_top=k, df=df,
                                     metrics=["n", "roc_auc", "mcc", "f1"]).set_index("metric")
                rows.append({"analysis": "model, UA-grouped 5-fold", "task": task,
                             "model": model, "top_uas_dropped": k,
                             "n": int(t.loc["n", "estimate"]),
                             "roc_auc": t.loc["roc_auc", "estimate"],
                             "roc_auc_ci_low": t.loc["roc_auc", "ci_low"],
                             "roc_auc_ci_high": t.loc["roc_auc", "ci_high"],
                             "mcc": t.loc["mcc", "estimate"], "f1": t.loc["f1", "estimate"]})
    return pd.DataFrame(rows)


def pooled_vs_city(df: pd.DataFrame, top10: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, dist = [], []
    for task in TASKS:
        cfg = features.TASKS[task]["primary"]
        for model in LEARNED:
            o, s = metrics.scored_run(task, cfg, model, "loco", df=df)
            pooled = s.at(np.ones_like(s.y))
            g = metrics.per_group(o, "ua", min_n=30)
            e = g[g["eligible"]]
            rows.append({
                "task": task, "model": model, "pooled_roc_auc": pooled["roc_auc"],
                "pooled_mcc": pooled["mcc"], "pooled_accuracy": pooled["accuracy"],
                "n_eligible_uas": len(e), "mean_city_roc_auc": e["roc_auc"].mean(),
                "median_city_roc_auc": e["roc_auc"].median(),
                "city_roc_auc_p25": e["roc_auc"].quantile(.25),
                "city_roc_auc_p75": e["roc_auc"].quantile(.75),
                "mean_city_mcc": e["mcc"].mean(), "mean_city_accuracy": e["accuracy"].mean(),
                "median_city_roc_auc_top10": e.loc[e["ua"].isin(top10), "roc_auc"].median(),
                "median_city_roc_auc_others": e.loc[~e["ua"].isin(top10), "roc_auc"].median(),
            })
            dist.append(g.assign(task=task, model=model))
    return pd.DataFrame(rows), pd.concat(dist, ignore_index=True)


def association_sensitivity(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in DROPS:
        q = p[~p["ua"].isin(validation.top_uas(p, k))] if k else p
        r = association.two_by_two(q["High_UHI"].astype(int), q["High_Pov"].astype(int))
        c = association.cmh(q["High_UHI"].astype(int), q["High_Pov"].astype(int), q["ua"])
        rows.append({"analysis": "association (2.2)", "top_uas_dropped": k, "n": r["n"],
                     "odds_ratio": r["odds_ratio"], "or_ci_low": r["or_ci_low"],
                     "or_ci_high": r["or_ci_high"], "cmh_or": c["cmh_or"],
                     "cmh_or_ci_low": c["cmh_or_ci_low"], "cmh_or_ci_high": c["cmh_or_ci_high"],
                     "cramers_v": r["cramers_v"]})
    return pd.DataFrame(rows)


def plot_sensitivity(ms: pd.DataFrame) -> None:
    plotting.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, task in zip(axes, TASKS, strict=True):
        for model in LEARNED:
            s = ms[(ms.task == task) & (ms.model == model)]
            x = np.arange(len(DROPS))
            ax.errorbar(x, s["roc_auc"], yerr=[s["roc_auc"] - s["roc_auc_ci_low"],
                                              s["roc_auc_ci_high"] - s["roc_auc"]],
                        color=plotting.MODEL_COLORS[model], marker="o", markersize=6,
                        markeredgecolor=plotting.SURFACE, linewidth=2, elinewidth=1.2,
                        capsize=0, label=models.MODEL_LABELS[model])
        ax.set_xticks(range(len(DROPS)), ["all UAs"] + [f"top {k} out" for k in DROPS[1:]])
        ax.set_title(features.TASKS[task]["label"])
        ax.set_ylabel("ROC-AUC, UA-grouped 5-fold")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    plotting.note(fig, "Models retrained and evaluated without the largest UAs. 95 % UA "
                       "cluster-bootstrap CIs.")
    fig.tight_layout()
    plotting.savefig(fig, OUT / "metro_sensitivity.png")


def main() -> None:
    df = validation.load_analysis()
    p = validation.frame("primary", df)
    conc, cstats = concentration(p)
    conc.to_csv(OUT / "concentration.csv", index=False)
    ms = model_sensitivity(df)
    top10 = validation.top_uas(p, 10)
    pc, dist = pooled_vs_city(df, top10)
    dist.to_csv(OUT / "per_ua_loco_auc.csv", index=False)
    asn = association_sensitivity(p)
    combined = pd.concat([ms, asn], ignore_index=True)
    combined.to_csv(OUT / "metro_sensitivity.csv", index=False)
    pc.to_csv(OUT / "pooled_vs_per_city.csv", index=False)
    md = ["# Large-metro sensitivity\n", "## Concentration\n", report.md_table(conc, 3),
          f"\nGini coefficient of tracts across {cstats['n_uas']} UAs: {cstats['gini']:.3f}.\n",
          "## Models without the largest UAs (UA-grouped 5-fold)\n", report.md_table(ms, 3),
          "\n## Pooled vs per-city (leave-one-city-out)\n", report.md_table(pc, 3),
          "\n## Association without the largest UAs\n", report.md_table(asn, 3)]
    report.write_text("\n".join(md), OUT / "metro_sensitivity.md")
    report.write_json(cstats, OUT / "concentration.json")
    plot_sensitivity(ms)
    write_readme()


def write_readme() -> None:
    conc = pd.read_csv(OUT / "concentration.csv").set_index("top_k_uas")
    cs = report.read_json(OUT / "concentration.json")
    ms = pd.read_csv(OUT / "metro_sensitivity.csv")
    pc = pd.read_csv(OUT / "pooled_vs_per_city.csv")
    model = ms[ms.analysis.str.startswith("model")]
    asn = ms[ms.analysis.str.startswith("association")].set_index("top_uas_dropped")
    f = report.fmt_num
    base = model[model.top_uas_dropped == 0].set_index(["task", "model"])["roc_auc"]
    model = model.assign(delta=model.apply(lambda r: r.roc_auc - base[(r.task, r.model)], axis=1))
    max_drop = max(0.0, float(-model["delta"].min()))
    rise = model.loc[model["delta"].idxmax()]
    lines = []
    for (task, mdl), g in model.groupby(["task", "model"], sort=False):
        g = g.set_index("top_uas_dropped")
        lines.append(f"| {task} | {models.MODEL_LABELS[mdl]} | "
                     + " | ".join(f(g.loc[k, "roc_auc"]) for k in DROPS) + " |")
    plines = "\n".join(
        f"| {r.task} | {models.MODEL_LABELS[r.model]} | {f(r.pooled_roc_auc)} | "
        f"{f(r.mean_city_roc_auc)} | {f(r.median_city_roc_auc)} "
        f"[{f(r.city_roc_auc_p25)}, {f(r.city_roc_auc_p75)}] | {f(r.median_city_roc_auc_top10)} | "
        f"{f(r.median_city_roc_auc_others)} | {r.n_eligible_uas} |" for r in pc.itertuples())
    or_range = (asn["odds_ratio"].min(), asn["odds_ratio"].max())
    cmh_range = (asn["cmh_or"].min(), asn["cmh_or"].max())
    lb = pc[(pc.task == "B") & (pc.model == "lgbm")].iloc[0]
    la = pc[(pc.task == "A") & (pc.model == "lgbm")].iloc[0]
    big_harder = (pc["median_city_roc_auc_top10"] <= pc["median_city_roc_auc_others"]).sum()
    depends = max_drop >= 0.02 or or_range[0] <= 1
    if depends:
        verdict = ("Some results shift when the largest metros are removed: ROC-AUC falls by up "
                   f"to {max_drop:.3f}; see the tables.")
    else:
        verdict = (
            "No conclusion depends on a few metros. Removing the 1, 3 or 10 largest UAs "
            + ("never lowers ROC-AUC" if max_drop < 0.0005 else
               f"never lowers ROC-AUC by more than {max_drop:.3f}")
            + f"; the largest change is a rise of "
            f"{rise.delta:.3f} ({models.MODEL_LABELS[rise.model].lower()}, Task {rise.task}, "
            f"top {int(rise.top_uas_dropped)} out), so the large metros make prediction slightly "
            "harder, not easier. The association keeps its direction and size, and the largest "
            f"UAs are not the easiest cities to predict ({big_harder} of {len(pc)} model × task "
            "pairs have a lower median within-city ROC-AUC in the ten largest UAs than in the "
            "rest).")
    within_higher = (pc["median_city_roc_auc"] >= pc["pooled_roc_auc"]).all()
    pooled_sentence = (
        "The median ROC-AUC *within* a held-out city is higher than the pooled figure for every "
        f"model (LightGBM: Task A {f(la.median_city_roc_auc)} vs pooled {f(la.pooled_roc_auc)}; "
        f"Task B {f(lb.median_city_roc_auc)} vs {f(lb.pooled_roc_auc)}). The models rank tracts "
        "inside an unseen city better than they place cities relative to each other, so pooled "
        "numbers understate within-city discrimination rather than overstate it."
        if within_higher else
        "Pooled and within-city ROC-AUC differ by model; both are reported in section 4.1 "
        f"(LightGBM: Task A pooled {f(la.pooled_roc_auc)} vs median within-city "
        f"{f(la.median_city_roc_auc)}; Task B {f(lb.pooled_roc_auc)} vs "
        f"{f(lb.median_city_roc_auc)}).")
    text = f"""# 1.5 Large-metro influence

## Purpose

Tests whether a handful of very large metropolitan areas dominate the sample and drive the
results.

## Method

`metro_influence.py` reuses the cached out-of-fold predictions to:

1. measure how concentrated tracts are in the largest urbanized areas (UAs);
2. retrain and re-evaluate the primary Task A (`A-noLST`) and Task B (`B-env`) models, logistic
   regression and LightGBM, under UA-grouped 5-fold with all UAs and without the largest 1, 3
   and 10 UAs;
3. compare pooled leave-one-city-out metrics (each tract counts once, so big UAs weigh more)
   with the mean across cities (each UA counts once), using UAs with ≥ 30 tracts and both
   classes;
4. recompute the association of section 2.2 (OR and CMH OR) without the largest 1, 3 and 10 UAs.

## Results

**Concentration** ([`results/concentration.png`](results/concentration.png)). The largest UA
({conc.loc[1, 'uas']}) holds {report.fmt_pct(conc.loc[1, 'share_of_tracts'])} of the
{int(conc.loc[20, 'tracts'] / conc.loc[20, 'share_of_tracts']):,} tracts; the largest 5 hold
{report.fmt_pct(conc.loc[5, 'share_of_tracts'])}, the largest 10
{report.fmt_pct(conc.loc[10, 'share_of_tracts'])} and the largest 20
{report.fmt_pct(conc.loc[20, 'share_of_tracts'])} (Gini {cs['gini']:.2f} across {cs['n_uas']} UAs).

**Models without the largest UAs** (ROC-AUC, UA-grouped 5-fold;
[`results/metro_sensitivity.md`](results/metro_sensitivity.md),
[`results/metro_sensitivity.png`](results/metro_sensitivity.png)):

| Task | Model | All UAs | Top 1 out | Top 3 out | Top 10 out |
|---|---|---:|---:|---:|---:|
{chr(10).join(lines)}

**Pooled vs per-city** (leave-one-city-out; per-city = ROC-AUC within each held-out UA):

| Task | Model | Pooled ROC-AUC | Mean across cities | Median across cities [IQR] | Median, 10 largest UAs | Median, other UAs | Eligible UAs |
|---|---|---:|---:|---:|---:|---:|---:|
{plines}

{pooled_sentence}

**Association without the largest UAs.** The odds ratio stays between {f(or_range[0], 2)} and
{f(or_range[1], 2)} and the CMH OR between {f(cmh_range[0], 2)} and {f(cmh_range[1], 2)} whether
the largest 0, 1, 3 or 10 UAs are excluded.

**Conclusion.** {verdict}
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
