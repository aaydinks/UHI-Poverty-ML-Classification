"""The pipeline diagram.

A flowchart in two columns. Left: the four data sources, data synthesis, the
median-split targets of Task A (high UHI) and Task B (high poverty), the
predictor sets and the chi-squared association analysis. Right: the machine
learning models, leave-one-city-out validation, the evaluation metrics and SHAP
interpretation. The dashed box inside the models is the secondary tuned models.

Drawn at print size (7 in wide). Every number in the diagram is read from the
built sample or the shared modules and written with its source to
`results/pipeline.csv`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from uhi_poverty import features, metrics, models, paths, plotting, report

OUT = paths.results_dir(__file__)

WIDTH, HEIGHT = 7.0, 4.74  # inches; the axes use inches, so font sizes are print sizes
SERIF = ["Times New Roman", "Times", "DejaVu Serif"]
LINE = "#000000"
INNER_FC, INNER_EC = "#f5f5f5", "#666666"

# fill and edge of each stage
BLUE = ("#dae8fc", "#6c8ebf")
GREEN = ("#d5e8d4", "#82b366")
YELLOW = ("#fff2cc", "#d6b656")
ROSE = ("#f8cecc", "#b85450")
PURPLE = ("#e1d5e7", "#9673a6")
TEAL = ("#b0e3e6", "#0e8088")
ORANGE = ("#ffe6cc", "#d79b00")
LIME = ("#cdeb8b", "#36393d")

TITLE_FS, BODY_FS = 8.6, 7.6

PAPER_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
                 "roc_auc", "pr_auc"]


# --------------------------------------------------------------------------- primitives

def text(ax, x, y, s, fs=BODY_FS, weight="normal", ha="center", va="center", **kw) -> None:
    ax.text(x, y, s, fontsize=fs, fontweight=weight, ha=ha, va=va, color=LINE,
            fontfamily=SERIF, linespacing=1.25, zorder=6, **kw)


def rect(ax, x0, y0, x1, y1, colors, *, radius=0.04, lw=0.8, ls="-", shadow=True,
         z=2) -> None:
    fc, ec = colors
    if shadow:
        ax.add_patch(FancyBboxPatch((x0 + 0.035, y0 + 0.035), x1 - x0, y1 - y0,
                                    boxstyle=f"round,pad=0,rounding_size={radius}",
                                    fc="#000000", ec="none", alpha=0.12, zorder=z - 0.5))
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={radius}", fc=fc, ec=ec,
                                lw=lw, ls=ls, zorder=z))


def container(ax, x0, y0, x1, y1, colors, title, header=0.27) -> float:
    """A stage box with its title in a header band; returns the y below the band."""
    rect(ax, x0, y0, x1, y1, colors)
    ax.plot([x0, x1], [y0 + header] * 2, color=colors[1], lw=0.8, zorder=3)
    text(ax, (x0 + x1) / 2, y0 + header / 2 + 0.01, title, fs=TITLE_FS, weight="bold")
    return y0 + header


def inner(ax, x0, y0, x1, y1, label, fs=BODY_FS, weight="normal", ls="-") -> None:
    rect(ax, x0, y0, x1, y1, (INNER_FC, INNER_EC), radius=0.05, lw=0.7, ls=ls, shadow=False,
         z=4)
    text(ax, (x0 + x1) / 2, (y0 + y1) / 2, label, fs=fs, weight=weight)


def bullets(ax, x, y, items, fs=BODY_FS, step=0.165) -> None:
    for k, item in enumerate(items):
        text(ax, x, y + k * step, "•", fs=fs, ha="left")
        text(ax, x + 0.11, y + k * step, item, fs=fs, ha="left")


def arrow(ax, points) -> None:
    """A black polyline through `points` with a filled head at the last point."""
    for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1], strict=True):
        ax.plot([x0, x1], [y0, y1], color=LINE, lw=0.9, solid_capstyle="butt", zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=8,
                                 color=LINE, lw=0.9, shrinkA=0, shrinkB=0, zorder=5))


def cylinder(ax, x0, y0, w, h, label, colors=ROSE) -> None:
    """A dashed database cylinder with a centred label."""
    fc, ec = colors
    e = 0.11  # height of the end ellipses
    kw = {"ec": ec, "lw": 0.8, "ls": (0, (3, 2))}
    ax.add_patch(Rectangle((x0, y0 + e / 2), w, h - e, fc=fc, ec="none", zorder=2))
    ax.add_patch(Ellipse((x0 + w / 2, y0 + h - e / 2), w, e, fc=fc, ec="none", zorder=2))
    ax.add_patch(Arc((x0 + w / 2, y0 + h - e / 2), w, e, theta1=0, theta2=180, zorder=3,
                     **kw))
    ax.plot([x0, x0], [y0 + e / 2, y0 + h - e / 2], color=ec, lw=0.8, ls=(0, (3, 2)),
            zorder=3)
    ax.plot([x0 + w, x0 + w], [y0 + e / 2, y0 + h - e / 2], color=ec, lw=0.8,
            ls=(0, (3, 2)), zorder=3)
    ax.add_patch(Ellipse((x0 + w / 2, y0 + e / 2), w, e, fc=fc, zorder=3, **kw))
    text(ax, x0 + w / 2, y0 + h / 2 + e / 4, label, fs=7.4, weight="bold")


def median_icon(ax, x0, y0, w, h) -> None:
    """A small histogram cut at its median, like a thumbnail of the target split."""
    rect(ax, x0, y0, x0 + w, y0 + h, ("#ffffff", INNER_EC), radius=0.03, lw=0.6,
         shadow=False, z=4)
    heights = [0.15, 0.32, 0.6, 0.88, 1.0, 0.8, 0.55, 0.34, 0.2, 0.1]
    pad, base = 0.07, y0 + h - 0.08
    bw = (w - 2 * pad) / len(heights)
    for k, v in enumerate(heights):
        on = k >= 4
        ax.add_patch(Rectangle((x0 + pad + k * bw, base - v * (h - 0.24)), bw * 0.86,
                               v * (h - 0.24), fc="#e3863b" if on else "#7ea6d8", ec="none",
                               zorder=5))
    xm = x0 + pad + 4 * bw - bw * 0.07
    ax.plot([xm, xm], [y0 + 0.08, base], color=LINE, lw=0.7, ls=(0, (2, 1.5)), zorder=6)
    ax.plot([x0 + pad, x0 + w - pad], [base, base], color=LINE, lw=0.6, zorder=6)
    text(ax, xm, y0 + 0.07, "median", fs=5.6, style="italic", va="center")


# --------------------------------------------------------------------------- content

def values() -> dict:
    med = report.read_json(paths.MEDIANS_JSON)
    missing = set(PAPER_METRICS) - set(metrics.PANEL)
    if missing:
        raise AssertionError(f"metrics not in the panel: {missing}")
    return {
        "n_tracts": med["n"]["primary"],
        "n_uas": med["n"]["primary_uas"],
        "uhi_cut": med["cuts"]["High_UHI"]["value"],
        "pov_cut": med["cuts"]["High_Pov"]["value"],
        "n_features_A": len(features.config(features.TASKS["A"]["primary"])),
        "n_features_B": len(features.config(features.TASKS["B"]["primary"])),
        "config_A": features.TASKS["A"]["primary"],
        "config_B": features.TASKS["B"]["primary"],
        "tau": models.THRESHOLD,
        "bootstrap_B": metrics.B_DEFAULT,
        "n_metrics": len(PAPER_METRICS),
    }


def facts(v: dict) -> pd.DataFrame:
    """Every number the diagram shows, with its source."""
    med = f"data/{paths.MEDIANS_JSON.relative_to(paths.DATA).as_posix()}"
    rows = [
        ("Analysis sample: tracts", v["n_tracts"], f"{v['n_tracts']:,}", med),
        ("Analysis sample: urbanized areas (cities)", v["n_uas"], f"{v['n_uas']:,}", med),
        ("Task A cut (UHI_annual_day, national median, °C)", v["uhi_cut"],
         f"{v['uhi_cut']:.2f} °C", med),
        ("Task B cut (ACS poverty rate, national median)", v["pov_cut"], f"{v['pov_cut']:.1%}",
         med),
        (f"Predictors in {v['config_A']} (Task A)", v["n_features_A"], str(v["n_features_A"]),
         "uhi_poverty.features.CONFIGS"),
        (f"Predictors in {v['config_B']} (Task B)", v["n_features_B"], str(v["n_features_B"]),
         "uhi_poverty.features.CONFIGS"),
        ("Decision threshold of the headline models", v["tau"], f"{v['tau']}",
         "uhi_poverty.models.THRESHOLD"),
        ("Leave-one-city-out folds (one per city)", v["n_uas"], f"{v['n_uas']:,}", med),
        ("Training cities per fold", v["n_uas"] - 1, f"{v['n_uas'] - 1:,}", med),
        ("Metrics shown", v["n_metrics"], str(v["n_metrics"]), "pipeline.PAPER_METRICS"),
        ("Bootstrap resamples of cities", v["bootstrap_B"], f"{v['bootstrap_B']:,}",
         "uhi_poverty.metrics.B_DEFAULT"),
        ("ACS vintage", "2015–2019", "2015–2019", "data/README.md (S2)"),
        ("Gazetteer vintage", 2019, "2019", "data/README.md (S4, S6)"),
    ]
    return pd.DataFrame(rows, columns=["element", "value", "shown_as", "source"])


def draw(fig, v: dict) -> None:
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")

    L0, L1 = 0.08, 3.28  # left column
    R0, R1 = 3.72, 6.92  # right column

    # ---- data sources
    names = ["SUHI\nDatabase", "Census\nGazetteer", "ACS\n2015–2019", "HUD Low\nPoverty\nIndex"]
    cw, cgap, ctop, ch = 0.7, 0.1, 0.04, 0.64
    cx0 = (L0 + L1) / 2 - (4 * cw + 3 * cgap) / 2
    for k, name in enumerate(names):
        cylinder(ax, cx0 + k * (cw + cgap), ctop, cw, ch, name)

    # ---- data synthesis (a process box: double side bars)
    s0, s1 = 0.94, 1.98
    rect(ax, L0, s0, L1, s1, BLUE)
    for x in (L0 + 0.13, L1 - 0.13):
        ax.plot([x, x], [s0, s1], color=BLUE[1], lw=0.8, zorder=3)
    text(ax, L0 + 0.25, s0 + 0.17, "Data Synthesis:", fs=TITLE_FS, weight="bold", ha="left")
    bullets(ax, L0 + 0.45, s0 + 0.39, [
        "Join the four sources on the 2010 tract ID",
        "Remove non-urban tracts and missing values",
        "Remove tracts with fewer than 50 people",
        f"{v['n_tracts']:,} tracts in {v['n_uas']} urbanized areas",
    ])
    ybus = (ctop + ch + s0) / 2
    for k in range(4):
        xc = cx0 + k * (cw + cgap) + cw / 2
        ax.plot([xc, xc], [ctop + ch, ybus], color=LINE, lw=0.9, zorder=1)
    ax.plot([cx0 + cw / 2, cx0 + 3 * (cw + cgap) + cw / 2], [ybus, ybus], color=LINE, lw=0.9,
            zorder=1)
    arrow(ax, [((L0 + L1) / 2, ybus), ((L0 + L1) / 2, s0)])

    # ---- target definition
    t0, t1 = 2.22, 3.22
    y = container(ax, L0, t0, L1, t1, GREEN, "Target Definition (National Medians)")
    median_icon(ax, L0 + 0.1, y + 0.12, 0.82, t1 - y - 0.22)
    bx0 = L0 + 1.04
    bw = (L1 - 0.1 - bx0 - 0.08) / 2
    inner(ax, bx0, y + 0.12, bx0 + bw, t1 - 0.1,
          f"Task A:\nHigh UHI\n≥ {v['uhi_cut']:.2f} °C", weight="normal")
    inner(ax, bx0 + bw + 0.08, y + 0.12, L1 - 0.1, t1 - 0.1,
          f"Task B:\nHigh Poverty\n≥ {v['pov_cut']:.1%}")
    arrow(ax, [((L0 + L1) / 2, s1), ((L0 + L1) / 2, t0)])

    # ---- predictor sets (left) and association (right), bottom of the left column
    b0, b1 = 3.48, 4.66
    pm = 1.62
    rect(ax, L0, b0 + 0.2, pm, b1, YELLOW)
    rect(ax, L0, b0, L0 + 0.98, b0 + 0.3, YELLOW, radius=0.02)
    text(ax, L0 + 0.49, b0 + 0.15, "Predictors", fs=TITLE_FS, weight="bold")
    text(ax, (L0 + pm) / 2, (b0 + 0.3 + b1) / 2,
         f"Task A: {v['n_features_A']} predictors\n(no temperature data)\n\n"
         f"Task B: {v['n_features_B']} predictors\n(all variables)", fs=BODY_FS)
    a0 = pm + 0.14
    y = container(ax, a0, b0, L1, b1, ROSE, "Association Analysis")
    bullets(ax, a0 + 0.09, y + 0.15, [
        "χ² test of independence",
        "Odds ratio, Cramér's V",
        "95% CI by city resampling",
        "Within-city odds ratio",
        "Alternative cut-offs",
    ], step=0.163)
    ysplit = (t1 + b0) / 2
    ax.plot([(L0 + L1) / 2] * 2, [t1, ysplit], color=LINE, lw=0.9, zorder=1)
    ax.plot([(L0 + pm) / 2, (a0 + L1) / 2], [ysplit, ysplit], color=LINE, lw=0.9, zorder=1)
    arrow(ax, [((L0 + pm) / 2, ysplit), ((L0 + pm) / 2, b0)])
    arrow(ax, [((a0 + L1) / 2, ysplit), ((a0 + L1) / 2, b0)])

    # ---- dotted separator between the two columns
    m0, m1 = 0.3, 1.72
    mx = (R0 + R1) / 2
    xgap = (L1 + R0) / 2
    ax.plot([xgap, xgap], [0.04, b1 + 0.04], color=LINE, lw=1.0, ls=(0, (1, 2)), zorder=1)

    # ---- machine learning models
    y = container(ax, R0, m0, R1, m1, PURPLE,
                  f"Machine Learning Models (Task A, Task B; τ = {v['tau']})")
    iw = (R1 - R0 - 0.2 - 2 * 0.1) / 3
    for k, name in enumerate(["LightGBM", "Logistic\nRegression", "Majority\nClass"]):
        x0 = R0 + 0.1 + k * (iw + 0.1)
        inner(ax, x0, y + 0.12, x0 + iw, y + 0.6, name, fs=8.0)
    inner(ax, R0 + 0.1, y + 0.72, R1 - 0.1, m1 - 0.1,
          "Tuned (secondary): XGBoost, LightGBM, CatBoost,\nSoft-voting Ensemble", fs=7.2,
          ls=(0, (3, 2)))

    # ---- leave-one-city-out validation
    v0, v1 = 2.0, 3.0
    y = container(ax, R0, v0, R1, v1, TEAL, "Leave-One-City-Out Validation")
    w3 = (R1 - R0 - 0.2 - 2 * 0.28) / 3
    labels = [f"Train:\n{v['n_uas'] - 1} cities", "Test:\n1 held-out city",
              f"Repeat for\nall {v['n_uas']} cities"]
    for k, label in enumerate(labels):
        x0 = R0 + 0.1 + k * (w3 + 0.28)
        inner(ax, x0, y + 0.12, x0 + w3, v1 - 0.1, label)
        if k:
            arrow(ax, [(x0 - 0.24, (y + 0.12 + v1 - 0.1) / 2), (x0 - 0.02, (y + 0.12 + v1 - 0.1) / 2)])
    arrow(ax, [(mx, m1), (mx, v0)])

    # ---- results
    r0, r1 = 3.32, 4.66
    e1 = R0 + 1.68
    y = container(ax, R0, r0, e1, r1, ORANGE, "Evaluation Metrics")
    bullets(ax, R0 + 0.1, y + 0.17, [
        "Accuracy, Balanced acc.",
        "Precision, Recall",
        "Specificity, F1",
        "ROC-AUC, PR-AUC",
        "95% CI by city resampling",
    ], step=0.17)
    s0r = e1 + 0.14
    y = container(ax, s0r, r0, R1, r1, GREEN, "Interpretation")
    bullets(ax, s0r + 0.1, y + 0.17, [
        "SHAP (LightGBM)",
        "Out of fold, by city",
        "Grouped variables",
        "Rank stable over 15 refits",
    ], step=0.17)
    ysplit = (v1 + r0) / 2
    ax.plot([mx, mx], [v1, ysplit], color=LINE, lw=0.9, zorder=1)
    ax.plot([(R0 + e1) / 2, (s0r + R1) / 2], [ysplit, ysplit], color=LINE, lw=0.9, zorder=1)
    arrow(ax, [((R0 + e1) / 2, ysplit), ((R0 + e1) / 2, r0)])
    arrow(ax, [((s0r + R1) / 2, ysplit), ((s0r + R1) / 2, r0)])


def main() -> None:
    plotting.apply_style()
    plt.rcParams["figure.facecolor"] = plt.rcParams["savefig.facecolor"] = "#ffffff"
    v = values()
    facts(v).to_csv(OUT / "pipeline.csv", index=False)
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    draw(fig, v)
    plotting.savefig(fig, OUT / "pipeline.png")


if __name__ == "__main__":
    main()
