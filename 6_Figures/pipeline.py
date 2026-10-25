"""The pipeline diagram.

Three columns (Datasets → Models → Validation, rounded boxes, a fold grid with
the held-out part highlighted): the sources of `data/README.md`, the two tasks
with their median labels and feature configurations (`features.CONFIGS`), the
three fixed-setting models at τ = 0.5, leave-one-city-out as the headline scheme,
the metric panel with UA cluster-bootstrap CIs, out-of-fold TreeSHAP, and the
tuned models of section 4.6 as one dashed, secondary box.

Every number in the diagram is read from the built sample or the shared modules
and written with its source to `results/pipeline.csv`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from uhi_poverty import features, metrics, models, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
WIDTH = 100  # canvas units; 10 units = 1 inch
HEIGHT = 99
COLS = (3.0, 33.8, 61.8, 97.0)
HEADER = (3.0, 9.6)

BOX_FC = plotting.NEUTRAL
BOX_EC = plotting.AXIS
ACCENT_FC = plotting.SEQ_BLUE[0]
ACCENT_INK = plotting.SEQ_BLUE[5]
HELD_OUT = plotting.AQUA

METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.", "precision": "Precision",
    "recall": "Recall", "specificity": "Specificity", "f1": "F1", "mcc": "MCC",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier",
}


# --------------------------------------------------------------------------- primitives

def font() -> list[str]:
    """The style's font list as concrete families, so missing glyphs (→) fall back."""
    return list(plt.rcParams["font.sans-serif"])


def canvas(fig, height: float):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return ax


def frame(ax, bottom: float, headers: tuple[str, str, str]) -> None:
    """The rounded outer frame, the header band and the column rules."""
    x0, x1 = COLS[0], COLS[-1]
    outer = FancyBboxPatch((x0, HEADER[0]), x1 - x0, bottom - HEADER[0],
                           boxstyle="round,pad=0,rounding_size=1.6", fc=plotting.SURFACE,
                           ec=BOX_EC, lw=0.9, zorder=0)
    ax.add_patch(outer)
    band = Rectangle((x0, HEADER[0]), x1 - x0, HEADER[1] - HEADER[0], fc=BOX_FC, ec="none",
                     zorder=0.5)
    ax.add_patch(band)
    band.set_clip_path(outer)
    ax.plot([x0, x1], [HEADER[1]] * 2, color=BOX_EC, lw=0.9, zorder=1)
    for x in COLS[1:-1]:
        ax.plot([x, x], [HEADER[0], bottom], color=BOX_EC, lw=0.9, zorder=1)
    for (a, b), text in zip(zip(COLS[:-1], COLS[1:], strict=True), headers, strict=True):
        ax.text((a + b) / 2, sum(HEADER) / 2, text, ha="center", va="center", fontsize=15,
                color=plotting.INK, fontfamily=font())


def box(ax, x0, y0, x1, y1, text, fs=9.0, fc=BOX_FC, ec=BOX_EC, color=plotting.INK,
        weight="normal", ls="-", radius=0.8, lw=0.7, linespacing=1.5) -> None:
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={radius}", fc=fc, ec=ec,
                                lw=lw, ls=ls, zorder=2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs,
            color=color, fontweight=weight, linespacing=linespacing, zorder=3,
            fontfamily=font())


def chips(ax, x0, x1, y0, y1, labels, gap=1.0, fs=9.0, **kw) -> None:
    """A row of boxes filling [x0, x1], each as wide as its label needs."""
    weights = [len(s) + 4 for s in labels]
    room = x1 - x0 - gap * (len(labels) - 1)
    x = x0
    for label, wgt in zip(labels, weights, strict=True):
        w = room * wgt / sum(weights)
        box(ax, x, y0, x + w, y1, label, fs=fs, **kw)
        x += w + gap


def title(ax, x, y, text, fs=10.5, weight="normal", color=plotting.INK) -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight=weight, color=color,
            fontfamily=font())


def dashed(ax, x0, x1, y) -> None:
    ax.plot([x0, x1], [y, y], color=BOX_EC, lw=0.8, ls=(0, (3, 3)), zorder=1)


def rule(ax, x0, x1, y) -> None:
    ax.plot([x0, x1], [y, y], color=plotting.MUTED, lw=0.8, zorder=1)


def arrow(ax, x0, y0, x1, y1) -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops={"arrowstyle": "->", "color": plotting.MUTED, "lw": 1.3,
                            "shrinkA": 0, "shrinkB": 0, "mutation_scale": 14})


def bracket(ax, ys, x_from, x_mid, y_out, x_to) -> None:
    """Lines from several box edges to one vertical rule, then an arrow onwards."""
    for y in ys:
        ax.plot([x_from, x_mid], [y, y], color=plotting.MUTED, lw=0.8, zorder=1)
    ax.plot([x_mid, x_mid], [min(ys), max(ys)], color=plotting.MUTED, lw=0.8, zorder=1)
    arrow(ax, x_mid, y_out, x_to, y_out)


def fold_grid(ax, labels, highlight, x0, y0, cell_w=5.6, cell_h=5.1, gap=0.6, ncols=5,
              fs=10.5) -> float:
    """The fold squares; returns the y below the grid."""
    for k, label in enumerate(labels):
        r, c = divmod(k, ncols)
        x, y = x0 + c * (cell_w + gap), y0 + r * (cell_h + gap)
        on = k in highlight
        box(ax, x, y, x + cell_w, y + cell_h, label, fs=fs, fc=HELD_OUT if on else BOX_FC,
            ec=HELD_OUT if on else BOX_EC, radius=0.6)
    rows = -(-len(labels) // ncols)
    return y0 + rows * cell_h + (rows - 1) * gap


def legend_row(ax, x, y, fc, ec, text, fs=10.5) -> None:
    box(ax, x, y - 0.75, x + 2.1, y + 0.75, "", fc=fc, ec=ec, radius=0.3)
    ax.text(x + 3.0, y, text, ha="left", va="center", fontsize=fs, color=plotting.INK,
            fontfamily=font())


def model_stack(ax, y0, names, target_text, x0=36.5, x1=47.9, xe=51.2, xe1=60.3, h=4.2,
                gap=0.9, fs=11.0, target_fs=10.5, **target_kw) -> float:
    """Three model boxes joined by a bracket to one box on the right; returns the y below."""
    centres = []
    for k, name in enumerate(names):
        y = y0 + k * (h + gap)
        box(ax, x0, y, x1, y + h, name, fs=fs, linespacing=1.15)
        centres.append(y + h / 2)
    mid = (centres[0] + centres[-1]) / 2
    xm = (x1 + xe) / 2
    bracket(ax, centres, x1, xm, mid, xe)
    box(ax, xe, mid - 2.7, xe1, mid + 2.7, target_text, fs=target_fs, **target_kw)
    return y0 + len(names) * h + (len(names) - 1) * gap


# --------------------------------------------------------------------------- content

def values() -> dict:
    med = report.read_json(paths.MEDIANS_JSON)
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
        "n_major_ua": validation.N_MAJOR_UA,
        "lgbm_trees": models.LGBM_PARAMS["n_estimators"],
        "metrics": [m for m in metrics.PANEL if m in METRIC_LABELS],
    }


def facts(v: dict) -> pd.DataFrame:
    """Every number the diagram shows, with its source."""
    med = f"data/{paths.MEDIANS_JSON.relative_to(paths.DATA).as_posix()}"
    rows = [
        ("Analysis sample: tracts", v["n_tracts"], f"{v['n_tracts']:,}", med),
        ("Analysis sample: urbanized areas", v["n_uas"], f"{v['n_uas']:,}", med),
        ("Poverty target cut (ACS poverty rate, national median)", v["pov_cut"],
         f"{v['pov_cut']:.1%}", med),
        ("UHI target cut (UHI_annual_day, national median, °C)", v["uhi_cut"],
         f"{v['uhi_cut']:.2f} °C", med),
        (f"Predictors in {v['config_B']} (poverty task)", v["n_features_B"],
         str(v["n_features_B"]), "uhi_poverty.features.CONFIGS"),
        (f"Predictors in {v['config_A']} (UHI task)", v["n_features_A"],
         str(v["n_features_A"]), "uhi_poverty.features.CONFIGS"),
        ("Decision threshold τ", v["tau"], f"{v['tau']}", "uhi_poverty.models.THRESHOLD"),
        ("Leave-one-city-out folds (one per UA)", v["n_uas"], f"{v['n_uas']:,}", med),
        ("Training UAs per leave-one-city-out fold", v["n_uas"] - 1, f"{v['n_uas'] - 1:,}",
         med),
        ("County- and UA-grouped folds", 5, "5", "uhi_poverty.validation.SCHEMES"),
        ("Largest UAs held out in turn", v["n_major_ua"], str(v["n_major_ua"]),
         "uhi_poverty.validation.N_MAJOR_UA"),
        ("Cluster-bootstrap resamples of UAs", v["bootstrap_B"], f"{v['bootstrap_B']:,}",
         "uhi_poverty.metrics.B_DEFAULT"),
        ("Metrics in the panel", len(v["metrics"]), str(len(v["metrics"])),
         "uhi_poverty.metrics.PANEL"),
        ("TreeSHAP folds (UA-grouped)", 5, "5", "uhi_poverty.interpret"),
        ("ACS vintage", "2015–2019", "2015–2019", "data/README.md (S2)"),
        ("Gazetteer / boundary vintage", 2019, "2019", "data/README.md (S4, S6)"),
        ("SUHI database reference year", 2020, "2020", "data/README.md (S1)"),
    ]
    return pd.DataFrame(rows, columns=["element", "value", "shown_as", "source"])


def draw(fig, height: float = HEIGHT, v: dict | None = None) -> None:
    """The pipeline diagram into a Figure or SubFigure."""
    v = values() if v is None else v
    ax = canvas(fig, height)
    bottom = HEIGHT - 3
    frame(ax, bottom, ("Datasets", "Models", "Validation"))
    muted = {"color": plotting.INK_2, "ec": BOX_EC, "ls": (0, (3, 2))}

    # Datasets
    x0, x1 = 5.3, 31.5
    sources = [
        "Surface UHI database\n(Chakraborty et al. 2020)\n(SUHI, LST, NDVI, DEM, morphology)",
        "2019 Gazetteer and cartographic\nboundaries, 2010 tracts\n(centroids, land and water area)",
        "ACS 2015–2019 5-year\npoverty rate (B17001)\n(poverty outcome)",
    ]
    y = 23.0
    for text in sources:
        box(ax, x0, y, x1, y + 8.6, text, fs=10, radius=1.0)
        y += 10.4
    sample_y = (y + 3.2, y + 12.4)
    arrow(ax, (x0 + x1) / 2, y - 1.8, (x0 + x1) / 2, sample_y[0])
    box(ax, x0, sample_y[0], x1, sample_y[1],
        f"Analysis sample\n{v['n_tracts']:,} tracts in {v['n_uas']} UAs\n(50 states + DC)",
        fs=10, fc=ACCENT_FC, ec=plotting.BLUE, color=ACCENT_INK, radius=1.0)
    arrow(ax, x1, sum(sample_y) / 2, 36.0, sum(sample_y) / 2)
    box(ax, x0, sample_y[1] + 4.5, x1, sample_y[1] + 10.5,
        "HUD Low Poverty Index\n(robustness outcome only)", fs=9.5, fc=plotting.SURFACE,
        radius=1.0, **muted)

    # Models
    mx0, mx1 = 36.5, 60.3
    names = ["Majority class", "Logistic\nregression", "LightGBM"]
    compared = {"fc": ACCENT_FC, "ec": plotting.BLUE, "color": ACCENT_INK}
    y = 12.0
    for heading, rows in (
        ("Pred. Poverty", [
            f"Target: ACS poverty rate ≥ national median\n({v['pov_cut']:.1%}) → High_Pov",
            f"Features: {v['config_B']} ({v['n_features_B']}; all SUHI columns)",
            "Fixed settings, no tuning",
            f"Threshold τ = {v['tau']}",
        ]),
        ("Pred. UHI", [
            f"Target: UHI_annual_day ≥ national median\n({v['uhi_cut']:.2f} °C) → High_UHI",
            f"Features: {v['config_A']} ({v['n_features_A']}; no LST_* or UHI_*)",
            "Fixed settings, no tuning",
            f"Threshold τ = {v['tau']}",
        ]),
    ):
        title(ax, (mx0 + mx1) / 2, y, heading, weight="bold", color=plotting.INK_2, fs=11)
        dashed(ax, mx0, mx1, y + 1.6)
        y += 2.4
        for text in rows:
            h = 2.7 + 1.5 * text.count("\n")
            box(ax, mx0, y, mx1, y + h, text, fs=8.6, fc=plotting.SURFACE, radius=0.6,
                linespacing=1.3)
            y += h + 0.6
        y = model_stack(ax, y + 0.6, names, "Compared\nside by side", **compared)
        if heading == "Pred. Poverty":
            rule(ax, 35.0, 60.6, y + 1.6)
            y += 4.0
    box(ax, mx0, y + 2.6, mx1, bottom - 2.0,
        "Tuned models (section 4.6)\nNested Optuna search over XGBoost,\n"
        "LightGBM, CatBoost and a soft-voting\nensemble; τ chosen on training cities only",
        fs=8.6, fc=plotting.SURFACE, radius=1.0, linespacing=1.4, **muted)

    # Validation
    cx0, cx1 = 63.5, 94.4
    cm = (cx0 + cx1) / 2
    title(ax, cm, 11.6, "Leave-one-city-out (headline)")
    labels = ["1", "2", "3", "4", "5", "6", "7", "…", f"{v['n_uas'] - 1}", f"{v['n_uas']}"]
    y = fold_grid(ax, labels, {2}, cx0, 13.2, cell_w=5.7, cell_h=4.4, gap=0.6, fs=10)
    legend_row(ax, cx0, y + 2.0, HELD_OUT, HELD_OUT, "Held-out urbanized area (1 per fold)",
               fs=10)
    legend_row(ax, cx0, y + 4.4, BOX_FC, BOX_EC, f"Training UAs (the other {v['n_uas'] - 1})",
               fs=10)
    title(ax, cm, y + 7.0, f"{v['n_uas']} folds; no city in both train and test", fs=9.5,
          color=plotting.INK_2)
    y += 8.8
    dashed(ax, cx0 - 0.5, cx1 + 1.5, y)
    title(ax, cm, y + 2.4, "Also reported")
    chips(ax, cx0, cx1, y + 4.0, y + 7.2, ["County 5-fold", "UA 5-fold", "State hold-out"],
          fs=9)
    chips(ax, cx0, cx1, y + 7.9, y + 11.1, ["Region hold-out", f"{v['n_major_ua']} largest UAs"],
          fs=9)
    title(ax, cm, y + 12.6, "state and region hold-outs drop UAs that cross the border",
          fs=8, color=plotting.INK_2)
    y += 14.2
    dashed(ax, cx0 - 0.5, cx1 + 1.5, y)
    title(ax, cm, y + 2.4, "Preprocessing (inside each fold)")
    chips(ax, cx0, cx1, y + 4.0, y + 7.2, ["Logistic: impute, log1p, scale", "LightGBM: raw inputs"],
          fs=8.6)
    y += 9.0
    dashed(ax, cx0 - 0.5, cx1 + 1.5, y)
    title(ax, cm, y + 2.4, "Evaluation metrics")
    names_m = [METRIC_LABELS[m] for m in v["metrics"]]
    for k, row in enumerate((names_m[:4], names_m[4:7], names_m[7:])):
        chips(ax, cx0, cx1, y + 4.0 + 3.9 * k, y + 7.2 + 3.9 * k, row, fs=9)
    y += 4.0 + 3.9 * 3
    title(ax, cm, y + 0.6, f"95 % CI: UA cluster bootstrap (B = {v['bootstrap_B']:,})", fs=9,
          color=plotting.INK_2)
    y += 2.4
    dashed(ax, cx0 - 0.5, cx1 + 1.5, y)
    title(ax, cm, y + 2.4, "Interpretability")
    box(ax, cx0 + 2, y + 4.0, cx1 - 2, y + 7.2, "Out-of-fold TreeSHAP (LightGBM, UA 5-fold)",
        fs=9)
    box(ax, cx0 + 2, y + 7.9, cx1 - 2, y + 11.1, "Attribution grouped by exact identity", fs=9)


def write_readme(f: pd.DataFrame) -> None:
    rows = "\n".join(f"| {e} | {s} | `{src}` |"
                     for e, s, src in zip(f.element, f.shown_as, f.source, strict=True))
    text = f"""# Pipeline diagram

`pipeline.py` draws the pipeline diagram in three columns (Datasets → Models → Validation). Its numbers
are read from the built sample (`median_values.json`) and the shared modules, so they cannot
drift from the analysis.

## What the diagram shows

- **Datasets.** The SUHI database (Chakraborty et al. 2020), the 2019 Gazetteer and
  cartographic boundaries on 2010 tracts, and the ACS 2015–2019 poverty rate (table B17001)
  feed one national analysis sample. The HUD Low Poverty Index is a robustness outcome only
  (dashed).
- **Models.** Poverty: ACS poverty rate ≥ the national median → `High_Pov`, on `B-env`.
  UHI: `UHI_annual_day` ≥ the national median → `High_UHI`, on `A-noLST`, which drops every
  `LST_*` and `UHI_*` column, because the target is computed from LST (section 3.1). The
  majority class, logistic regression and LightGBM run with settings fixed in advance and
  τ = 0.5. They are compared, not blended, and no threshold is tuned on test data.
- **Validation.** Leave-one-city-out is the headline: the fold grid holds urbanized areas, one
  of which is held out per fold. County- and UA-grouped 5-fold, UA-buffered state and region
  hold-outs and the 20 largest UAs are also reported.
- **Preprocessing.** Logistic regression gets median imputation, log1p on the skewed size
  columns and standardisation, fitted inside each fold; LightGBM gets the raw inputs.
- **Metrics.** The full panel, each with a 95 % UA cluster-bootstrap CI.
- **Interpretability.** Out-of-fold TreeSHAP for LightGBM under UA-grouped 5-fold,
  attributed to groups that keep each exact identity whole (section 5.2).
- **Tuned models (section 4.6).** Dashed and secondary: a nested Optuna search over XGBoost,
  LightGBM, CatBoost and a soft-voting ensemble, with τ chosen on training cities only. The
  diagram gives no numbers for them; they are in section 4.6.

## Numbers shown

| Element | Shown as | Source |
|---|---|---|
{rows}

All values: [`results/pipeline.csv`](results/pipeline.csv).
Figure: [`results/pipeline.png`](results/pipeline.png).
"""
    report.write_text(text, HERE / "README_pipeline.md")


def main() -> None:
    plotting.apply_style()
    v = values()
    f = facts(v)
    f.to_csv(OUT / "pipeline.csv", index=False)
    fig = plt.figure(figsize=(WIDTH / 10, HEIGHT / 10))
    draw(fig, HEIGHT, v)
    plotting.savefig(fig, OUT / "pipeline.png")
    write_readme(f)


if __name__ == "__main__":
    main()
