"""Section 4.2: random 5-fold against held-out validation, and spatial dependence.

1. Optimism: random 5-fold against UA-grouped 5-fold, leave-one-state-out, the
   20 largest UAs held out, and leave-one-city-out. Δ = random − held-out, with
   paired UA cluster-bootstrap CIs on the same tracts.
2. Spatial dependence: Moran's I (k = 8 nearest neighbours, 999 permutations)
   and the share of validation tracts with a nearest neighbour in training.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uhi_poverty import features, metrics, models, paths, plotting, report, spatial, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
HELD_OUT = ("ua5", "state", "major_ua", "loco")
TASKS = ("A", "B")
LEARNED = ("logistic", "lgbm")
MORAN_VARS = {"UHI_annual_day": "Daytime SUHI (°C)", "pov_rate": "ACS poverty rate",
              "NDVI_urb_CT": "Tract NDVI", "DelNDVI_summer": "Summer ΔNDVI (urban − rural)"}
K = 8


def optimism(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        cfg = features.TASKS[task]["primary"]
        for model in LEARNED:
            rand, _ = metrics.scored_run(task, cfg, model, "random5", df=df)
            for scheme in HELD_OUT:
                held, _ = metrics.scored_run(task, cfg, model, scheme, df=df)
                m = held.merge(rand[["GEOID", "score", "pred"]], on="GEOID",
                               suffixes=("_held", "_rand"), validate="one_to_one")
                a = metrics.Scored(m["y"], m["score_rand"], m["pred_rand"])
                b = metrics.Scored(m["y"], m["score_held"], m["pred_held"])
                d = metrics.paired_delta(a, b, m["ua"].to_numpy(), metrics=("roc_auc", "mcc", "f1"))
                d.insert(0, "n_tracts", len(m))
                d.insert(0, "held_out_scheme", scheme)
                d.insert(0, "model", model)
                d.insert(0, "task", task)
                rows.append(d.rename(columns={"a": "random5", "b": "held_out"}))
    return pd.concat(rows, ignore_index=True)


def morans(df: pd.DataFrame, nbrs: np.ndarray) -> pd.DataFrame:
    rows = []
    for col, label in MORAN_VARS.items():
        r = spatial.morans_i(df[col].to_numpy(), nbrs, permutations=999, seed=37)
        rows.append({"variable": col, "description": label, **r})
    return pd.DataFrame(rows)


def neighbour_share(df: pd.DataFrame, nbrs: np.ndarray) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        y = df[features.TASKS[task]["target"]].astype(int).to_numpy()
        for scheme in validation.SCHEMES:
            folds = validation.splits(df, y, scheme)
            r = spatial.neighbour_in_train_share(nbrs, [(tr, te) for _, tr, te in folds])
            rows.append({"task": task, "scheme": scheme, "scheme_label":
                         validation.SCHEME_LABELS[scheme], **r})
    return pd.DataFrame(rows)


def plot_optimism(t: pd.DataFrame) -> None:
    plotting.apply_style()
    sub = t[t["metric"] == "roc_auc"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    order = list(HELD_OUT)[::-1]
    for ax, task in zip(axes, TASKS, strict=True):
        for j, model in enumerate(LEARNED):
            s = sub[(sub.task == task) & (sub.model == model)].set_index("held_out_scheme").loc[order]
            y = np.arange(len(order)) + (j - 0.5) * 0.28
            ax.errorbar(s["delta"], y, xerr=[s["delta"] - s["ci_low"], s["ci_high"] - s["delta"]],
                        fmt="o", color=plotting.MODEL_COLORS[model], markersize=6,
                        markeredgecolor=plotting.SURFACE, elinewidth=1.5, capsize=0,
                        label=models.MODEL_LABELS[model])
        ax.axvline(0, color=plotting.AXIS, linewidth=1)
        ax.set_yticks(range(len(order)), [validation.SCHEME_LABELS[s] for s in order])
        ax.set_title(features.TASKS[task]["label"])
        ax.set_xlabel("Δ ROC-AUC, random 5-fold − held-out")
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    plotting.note(fig, "Paired on the same tracts; 95 % UA cluster-bootstrap intervals "
                       "(B = 1,000).")
    plotting.savefig(fig, OUT / "optimism_auc.png")


def plot_neighbours(n: pd.DataFrame) -> None:
    plotting.apply_style()
    sub = n[n.task == "A"].set_index("scheme").loc[list(validation.SCHEMES)[::-1]]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    y = np.arange(len(sub))
    ax.barh(y, sub["share_with_train_neighbour"] * 100, height=0.55, color=plotting.BLUE)
    for yi, v in zip(y, sub["share_with_train_neighbour"] * 100, strict=True):
        ax.text(v + 1, yi, f"{v:.1f} %", va="center", fontsize=8, color=plotting.INK)
    ax.set_yticks(y, sub["scheme_label"])
    ax.set_xlim(0, 112)
    ax.set_xlabel("Validation tracts with ≥ 1 of their 8 nearest neighbours in training (%)")
    ax.set_title("Random folds leave almost every test tract next to a training tract")
    ax.grid(axis="y", visible=False)
    plotting.savefig(fig, OUT / "train_neighbour_share.png")


def main() -> None:
    df = validation.frame("primary")
    nbrs, dist = spatial.knn(df["lat"], df["lon"], k=K)
    t = optimism(df)
    report.write_table(t, OUT / "optimism")
    mi = morans(df, nbrs)
    report.write_table(mi, OUT / "morans_i")
    ns = neighbour_share(df, nbrs)
    report.write_table(ns, OUT / "train_neighbour_share")
    report.write_json({"k": K, "median_km_to_kth_neighbour": float(np.median(dist[:, -1])),
                       "n": len(df)}, OUT / "knn.json")
    plot_optimism(t)
    plot_neighbours(ns)


if __name__ == "__main__":
    main()
