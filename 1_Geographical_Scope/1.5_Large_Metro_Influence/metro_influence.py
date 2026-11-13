"""Section 1.5: the models and the association without the largest urbanized areas.

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
    report.write_json(cstats, OUT / "concentration.json")
    plot_sensitivity(ms)


if __name__ == "__main__":
    main()
