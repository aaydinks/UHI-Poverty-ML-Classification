"""The pipeline diagram.

A top-to-bottom flow in six stages (data, sample, outcomes, models, validation,
results) and two lanes: prediction, where Task A (high UHI) and Task B (high
poverty) share the models and leave-one-city-out validation, and association,
the chi-squared analysis of the same two median labels. Dashed elements are
secondary: the tuned models and the HUD Low Poverty Index.

Drawn at print size (7 in wide). Every number in the diagram is read from the
built sample or the shared modules and written with its source to
`results/pipeline.csv`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from uhi_poverty import features, metrics, models, paths, plotting, report

OUT = paths.results_dir(__file__)

WIDTH, HEIGHT = 7.0, 6.23  # inches; the axes use inches, so font sizes are print sizes

# lanes (x) and stages (y), in inches
GUTTER = 0.78
LANE_A = (0.92, 2.86)
LANE_B = (3.00, 4.94)
LANE_C = (5.22, 6.98)
PRED = (LANE_A[0], LANE_B[1])
STAGES = [  # name, top, bottom
    ("Data", 0.42, 1.00),
    ("Sample", 1.28, 1.80),
    ("Outcomes", 2.08, 2.86),
    ("Models", 3.14, 4.02),
    ("Validation", 4.30, 5.12),
    ("Results", 5.40, 6.16),
]

WHITE = "#ffffff"
EDGE = "#9a9890"
ACCENT = plotting.BLUE
ACCENT_FILL = "#eef4fc"
TITLE_FS, BODY_FS, SMALL_FS = 7.6, 6.9, 6.4

PAPER_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
                 "roc_auc", "pr_auc"]


# --------------------------------------------------------------------------- primitives

def font() -> list[str]:
    return list(plt.rcParams["font.sans-serif"])


def text(ax, x, y, s, fs=BODY_FS, color=plotting.INK_2, weight="normal", ha="center",
         va="center", **kw) -> None:
    ax.text(x, y, s, fontsize=fs, color=color, fontweight=weight, ha=ha, va=va,
            fontfamily=font(), linespacing=1.3, zorder=4, **kw)


def node(ax, x0, y0, x1, y1, title=None, body=None, *, fill=WHITE, edge=EDGE, lw=0.6,
         ls="-", title_color=plotting.INK, body_color=plotting.INK_2, title_fs=TITLE_FS,
         body_fs=BODY_FS) -> None:
    """A box with a bold title over grey body text, both centred."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.05", fc=fill, ec=edge,
                                lw=lw, ls=ls, zorder=2))
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    if title and body:
        n = body.count("\n") + 1
        line = body_fs / 72 * 1.3
        top = ym - (TITLE_FS / 72 * 1.25 + n * line) / 2
        text(ax, xm, top, title, fs=title_fs, color=title_color, weight="bold", va="top")
        text(ax, xm, top + title_fs / 72 * 1.35, body, fs=body_fs, color=body_color, va="top")
    elif title:
        text(ax, xm, ym, title, fs=title_fs, color=title_color, weight="bold")
    elif body:
        text(ax, xm, ym, body, fs=body_fs, color=body_color)


def arrow(ax, points, ls="-", color=plotting.INK_2) -> None:
    """A polyline through `points` with an arrowhead at the last point."""
    for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1], strict=True):
        ax.plot([x0, x1], [y0, y1], color=color, lw=0.7, ls=ls, solid_capstyle="butt",
                zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=6.5,
                                 color=color, lw=0.7, ls=ls, shrinkA=0, shrinkB=0.5,
                                 zorder=1))


def down(ax, x, y0, y1, **kw) -> None:
    arrow(ax, [(x, y0), (x, y1)], **kw)


def stage(name) -> tuple[float, float]:
    return next((t, b) for s, t, b in STAGES if s == name)


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
        ("SUHI database reference year", 2020, "2020", "data/README.md (S1)"),
    ]
    return pd.DataFrame(rows, columns=["element", "value", "shown_as", "source"])


def draw(fig, v: dict) -> None:
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")

    # lane headers and the divider between the two analyses
    for (x0, x1), label in ((PRED, "Prediction"), (LANE_C, "Association")):
        text(ax, (x0 + x1) / 2, 0.14, label, fs=8.6, color=plotting.INK, weight="bold")
        ax.plot([x0, x1], [0.27, 0.27], color=plotting.INK, lw=0.8, zorder=1)
    xdiv = (LANE_B[1] + LANE_C[0]) / 2
    ax.plot([xdiv, xdiv], [0.42, HEIGHT - 0.06], color=plotting.GRID, lw=0.8, zorder=0)

    # stage gutter
    for k, (name, top, bottom) in enumerate(STAGES, start=1):
        ym = (top + bottom) / 2
        ax.add_patch(plt.Circle((0.14, ym), 0.085, fc=plotting.INK, ec="none", zorder=3))
        text(ax, 0.14, ym, str(k), fs=6.8, color=WHITE, weight="bold")
        text(ax, 0.28, ym, name, fs=7.4, color=plotting.INK, ha="left")

    # 1 Data
    top, bottom = stage("Data")
    w = (PRED[1] - PRED[0] - 2 * 0.14) / 3
    sources = [
        ("SUHI database", "Chakraborty et al. (2020)\nLST, vegetation, elevation"),
        ("Census Gazetteer 2019", "tract area and water share\n2010 tract boundaries"),
        ("ACS 2015–2019", "5-year estimates\npoverty rate (B17001)"),
    ]
    src_x = []
    for k, (title, body) in enumerate(sources):
        x0 = PRED[0] + k * (w + 0.14)
        node(ax, x0, top, x0 + w, bottom, title, body)
        src_x.append(x0 + w / 2)
    node(ax, LANE_C[0], top, LANE_C[1], bottom, "HUD Low Poverty Index",
         "robustness outcome only", ls=(0, (3, 2)))

    # 2 Sample
    top2, bottom2 = stage("Sample")
    sx0, sx1 = 1.62, 4.24
    node(ax, sx0, top2, sx1, bottom2, "Analysis sample",
         f"{v['n_tracts']:,} census tracts in {v['n_uas']} urbanized areas (cities)\n"
         "50 states and DC, joined on the 2010 tract ID",
         fill=ACCENT_FILL, edge=ACCENT, lw=0.8)
    ymerge = (bottom + top2) / 2
    for x in src_x:
        ax.plot([x, x], [bottom, ymerge], color=plotting.INK_2, lw=0.7, zorder=1)
    ax.plot([src_x[0], src_x[-1]], [ymerge, ymerge], color=plotting.INK_2, lw=0.7, zorder=1)
    down(ax, (sx0 + sx1) / 2, ymerge, top2)

    # 3 Outcomes
    top3, bottom3 = stage("Outcomes")
    tasks = [
        (LANE_A, "Task A: high UHI",
         f"annual daytime UHI ≥ {v['uhi_cut']:.2f} °C (median)\n"
         f"{v['n_features_A']} predictors, no temperature variable"),
        (LANE_B, "Task B: high poverty",
         f"poverty rate ≥ {v['pov_cut']:.1%} (median)\n"
         f"{v['n_features_B']} predictors, including LST and UHI"),
    ]
    ysplit = (bottom2 + top3) / 2
    ax.plot([(sx0 + sx1) / 2] * 2, [bottom2, ysplit], color=plotting.INK_2, lw=0.7, zorder=1)
    ax.plot([sum(LANE_A) / 2, sum(LANE_B) / 2], [ysplit, ysplit], color=plotting.INK_2, lw=0.7,
            zorder=1)
    for (x0, x1), title, body in tasks:
        node(ax, x0, top3, x1, bottom3, title, body)
        down(ax, (x0 + x1) / 2, ysplit, top3)

    cx = sum(LANE_C) / 2
    node(ax, LANE_C[0], top3, LANE_C[1], bottom3, "High UHI × high poverty",
         "2 × 2 table of the same\ntwo median labels")
    arrow(ax, [(sx1, (top2 + bottom2) / 2), (cx - 0.25, (top2 + bottom2) / 2),
               (cx - 0.25, top3)])
    down(ax, cx + 0.25, bottom, top3, ls=(0, (3, 2)), color=plotting.MUTED)

    # 4 Models
    top4, bottom4 = stage("Models")
    hx0, hx1 = PRED[0], 3.70
    ax.add_patch(FancyBboxPatch((hx0, top4), hx1 - hx0, bottom4 - top4,
                                boxstyle="round,pad=0,rounding_size=0.05", fc=WHITE, ec=EDGE,
                                lw=0.6, zorder=2))
    text(ax, (hx0 + hx1) / 2, top4 + 0.14, "Headline models", fs=TITLE_FS, color=plotting.INK,
         weight="bold")
    text(ax, (hx0 + hx1) / 2, top4 + 0.30, f"settings fixed in advance, threshold {v['tau']}",
         fs=BODY_FS)
    pills = ["LightGBM\n(main model)", "Logistic\nregression", "Majority-class\nbaseline"]
    pw = (hx1 - hx0 - 0.2 - 2 * 0.08) / 3
    for k, label in enumerate(pills):
        x0 = hx0 + 0.1 + k * (pw + 0.08)
        node(ax, x0, top4 + 0.44, x0 + pw, bottom4 - 0.09, body=label, body_fs=BODY_FS,
             fill=ACCENT_FILL if k == 0 else "#f4f3f0",
             edge=ACCENT if k == 0 else EDGE, body_color=plotting.INK)
    tx0 = 3.84
    node(ax, tx0, top4, PRED[1], bottom4, "Tuned models",
         "secondary analysis\nOptuna search,\nnested by city;\nsoft-voting ensemble",
         ls=(0, (3, 2)), body_fs=SMALL_FS)
    ymerge4 = (bottom3 + top4) / 2
    for x0, x1 in (LANE_A, LANE_B):
        ax.plot([(x0 + x1) / 2] * 2, [bottom3, ymerge4], color=plotting.INK_2, lw=0.7, zorder=1)
    ax.plot([sum(LANE_A) / 2, sum(LANE_B) / 2], [ymerge4, ymerge4], color=plotting.INK_2,
            lw=0.7, zorder=1)
    down(ax, (hx0 + hx1) / 2, ymerge4, top4)
    arrow(ax, [(sum(LANE_B) / 2, ymerge4), ((tx0 + PRED[1]) / 2, ymerge4),
               ((tx0 + PRED[1]) / 2, top4)], ls=(0, (3, 2)), color=plotting.MUTED)

    node(ax, LANE_C[0], top4, LANE_C[1], bottom4, "χ² test of independence",
         "odds ratio and\nCramér's V as effect sizes")
    down(ax, cx, bottom3, top4)

    # 5 Validation
    top5, bottom5 = stage("Validation")
    node(ax, PRED[0], top5, PRED[1], bottom5)
    text(ax, PRED[0] + 0.14, top5 + 0.15, "Leave-one-city-out", fs=TITLE_FS,
         color=plotting.INK, weight="bold", ha="left")
    # one row of cities, one held out
    n_cells, cell, gap = 10, 0.13, 0.035
    gx0, gy = PRED[0] + 0.14, top5 + 0.30
    held = 3
    for k in range(n_cells):
        x = gx0 + k * (cell + gap)
        if k == 7:
            text(ax, x + cell / 2, gy + cell / 2, "…", fs=7, color=plotting.INK_2)
            continue
        on = k == held
        ax.add_patch(FancyBboxPatch((x, gy), cell, cell, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=ACCENT if on else "#e4e3de", ec="none", zorder=3))
    hx = gx0 + held * (cell + gap) + cell / 2
    text(ax, hx, gy + cell + 0.11, "test city", fs=SMALL_FS, color=ACCENT, weight="bold")
    gx1 = gx0 + n_cells * (cell + gap) - gap
    text(ax, gx1 + 0.2, gy + cell / 2 + 0.02,
         f"each of the {v['n_uas']} cities is held out in turn,\n"
         f"predicted by a model fitted on\nthe other {v['n_uas'] - 1} cities",
         fs=BODY_FS, ha="left")
    text(ax, PRED[0] + 0.14, bottom5 - 0.12,
         "every fitted step (imputation, scaling, tuning, threshold) sees the training cities only",
         fs=SMALL_FS, ha="left", style="italic")
    down(ax, (hx0 + hx1) / 2, bottom4, top5)
    down(ax, (tx0 + PRED[1]) / 2, bottom4, top5, ls=(0, (3, 2)), color=plotting.MUTED)

    node(ax, LANE_C[0], top5, LANE_C[1], bottom5, "Tracts clustered by city",
         f"95% CI from {v['bootstrap_B']:,} city resamples;\nodds ratio within cities\n(Mantel–Haenszel)")
    down(ax, cx, bottom4, top5)

    # 6 Results
    top6, bottom6 = stage("Results")
    node(ax, LANE_A[0], top6, LANE_A[1], bottom6, "Performance",
         f"{v['n_metrics']} metrics (accuracy, F1, ROC-AUC, …)\n"
         f"on all {v['n_tracts']:,} held-out tracts;\n"
         f"95% CI from {v['bootstrap_B']:,} city resamples")
    node(ax, LANE_B[0], top6, LANE_B[1], bottom6, "Interpretation",
         "SHAP values of the LightGBM models,\nout of fold, summed over\ngroups of related variables")
    for x0, x1 in (LANE_A, LANE_B):
        down(ax, (x0 + x1) / 2, bottom5, top6)
    node(ax, LANE_C[0], top6, LANE_C[1], bottom6, "Sensitivity",
         "other cut-offs, the Low Poverty\nIndex, and without the\nlargest cities")
    down(ax, cx, bottom5, top6)


def main() -> None:
    plotting.apply_style()
    v = values()
    facts(v).to_csv(OUT / "pipeline.csv", index=False)
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    draw(fig, v)
    plotting.savefig(fig, OUT / "pipeline.png")


if __name__ == "__main__":
    main()
