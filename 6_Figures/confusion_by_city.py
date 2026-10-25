"""Confusion matrices in the ten largest held-out cities, for the UHI and poverty models.

A 2 × 5 grid of confusion matrices, each cell's count with its share of the
panel's tracts, the panel's F1 · Accuracy / MCC above it, and a two-line header
with the F1 spread across panels. The ten panels are the ten largest urbanized
areas (UAs) by tract count, each held out in turn: the cached leave-one-city-out
predictions of the headline LightGBM (Task A `A-noLST`, Task B `B-env`), at
τ = 0.5. The header carries the pooled leave-one-city-out F1 over all 54,243
tracts with its UA cluster-bootstrap CI, which must equal the 4.4 panel.

Writes `results/confusion_by_city_uhi.png`, `results/confusion_by_city_poverty.png`,
`results/confusion_by_city_panels.{csv,md}`, `results/confusion_by_city_pooled.csv` and
`README_confusion_by_city.md`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from uhi_poverty import metrics, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
METRIC_PANEL = (paths.ROOT / "4_Model_Specifications" / "4.4_Metric_Panel_with_Uncertainty"
                / "results" / "metric_panel.csv")
N_CITIES = 10
N_JOBS = 6
FIGURES = {
    "A": {"panels": "UHI", "config": "A-noLST", "task_label": "Urban heat island (high SUHI)",
          "classes": ("Low SUHI", "High SUHI"), "png": "confusion_by_city_uhi.png"},
    "B": {"panels": "Poverty", "config": "B-env", "task_label": "Poverty (ACS poverty rate)",
          "classes": ("Low poverty", "High poverty"), "png": "confusion_by_city_poverty.png"},
}
FIGSIZE = (14.5, 7.6)
# The UHI panels are drawn in blues and the poverty panels in oranges.
SEQ_ORANGE = ["#fde7da", "#f9c3a3", "#f49a6c", plotting.ORANGE, "#c4501f", "#8f3510",
              "#5e2209"]


def cmap_for(task: str) -> LinearSegmentedColormap:
    if task == "A":
        return plotting.sequential_cmap()
    return LinearSegmentedColormap.from_list("seq_orange", SEQ_ORANGE, N=256)


def principal_city(ua_name: str) -> str:
    """'Dallas--Fort Worth--Arlington, TX' -> 'Dallas'."""
    return ua_name.split(",")[0].split("--")[0].strip()


def spread(f1) -> str:
    f1 = np.asarray(f1, dtype=float)
    return (f"mean {f1.mean():.3f} · SD {f1.std(ddof=1):.3f} · "
            f"range [{f1.min():.3f}, {f1.max():.3f}]")


# --------------------------------------------------------------------------- drawing


def draw_grid(fig, panels: list[dict], classes: tuple[str, str], cmap, header: str,
              cell_text, vmax: float, note: str | None = None) -> None:
    """A 2 × 5 grid of confusion matrices on `fig` (a Figure or SubFigure).

    Each panel dict has `cm` ([[TN, FP], [FN, TP]]), `title` (bold) and optionally
    `subtitle`. Cell shading is the count's share of the panel's tracts on one
    scale (0 to `vmax`) for every panel; `cell_text(count, share)` gives the label.
    """
    axes = fig.subplots(2, 5, gridspec_kw={"hspace": 0.78, "wspace": 0.32, "top": 0.79, "bottom": 0.08, "left": 0.04, "right": 0.99})
    for ax, p in zip(axes.flat, panels, strict=True):
        cm = np.asarray(p["cm"], dtype=float)
        share = cm / cm.sum()
        ax.imshow(share, cmap=cmap, vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                ink = "white" if share[i, j] > 0.45 * vmax else plotting.INK
                ax.text(j, i, cell_text(int(cm[i, j]), share[i, j]), ha="center",
                        va="center", fontsize=8.5, fontweight="bold", color=ink,
                        linespacing=1.3)
        ax.set_xticks([0, 1], classes, fontsize=7.5)
        ax.set_yticks([0, 1], classes, rotation=90, va="center", fontsize=7.5)
        ax.set_xlabel("Predicted class", fontsize=8, labelpad=2)
        ax.set_ylabel("Actual class", fontsize=8, labelpad=2)
        ax.tick_params(length=0)
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
        sub = p.get("subtitle")
        n_sub = sub.count("\n") + 1 if sub else 0
        ax.set_title(p["title"], loc="center", fontsize=9, pad=5 + 10.8 * n_sub)
        if sub:
            ax.annotate(sub, (0.5, 1.0), xytext=(0, 3), xycoords="axes fraction",
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.2,
                        color=plotting.INK_2, linespacing=1.25)
    fig.suptitle(header, fontsize=11, fontweight="bold", y=0.985, va="top", linespacing=1.4)
    if note:
        fig.text(0.01, 0.0, note, ha="left", va="top", fontsize=7, color=plotting.INK_2)


def cell_count_share_of_city(count: int, share: float) -> str:
    return f"{count:,}\n{100 * share:.1f} % of city"


# --------------------------------------------------------------------------- data


def city_panels(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """One row per held-out city: counts, rates with denominators, F1, Acc, MCC."""
    cfg = FIGURES[task]["config"]
    meta = paths.OOF_DIR / f"{validation.run_key(task, cfg, 'lgbm', 'loco')}.json"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} missing: run `make engine` first")
    o = validation.oof(task, cfg, "lgbm", "loco", df=df, n_jobs=N_JOBS)
    names = df.groupby("ua")["ua_name"].first()
    rows = []
    for rank, ua in enumerate(validation.top_uas(df, N_CITIES), 1):
        sub = o[o["ua"] == ua]
        cm = metrics.confusion(sub["y"], sub["pred"])
        r = metrics.rates(cm)
        p = metrics.panel(sub["y"], sub["score"], sub["pred"])
        rows.append({
            "task": task, "rank": rank, "ua": ua, "ua_name": names[ua],
            "city": principal_city(names[ua]), "n": len(sub),
            **{k: int(v) for k, v in cm.items()},
            "TPR": r["TPR"], "TPR_denominator": int(r["TPR_denominator"]),
            "TNR": r["TNR"], "TNR_denominator": int(r["TNR_denominator"]),
            "f1": p["f1"], "accuracy": p["accuracy"], "mcc": p["mcc"],
        })
    return pd.DataFrame(rows)


def pooled(df: pd.DataFrame, task: str) -> dict:
    """Pooled leave-one-city-out F1 with its UA cluster-bootstrap CI, checked against 4.4."""
    cfg = FIGURES[task]["config"]
    o, s = metrics.scored_run(task, cfg, "lgbm", "loco", df=df)
    b = metrics.bootstrap_panel(s, o["ua"].to_numpy(), metrics=["f1", "mcc", "accuracy"])
    b = b.set_index("metric")
    ref = pd.read_csv(METRIC_PANEL)
    ref = ref[(ref.task == task) & (ref.config == cfg) & (ref.model == "lgbm")
              & (ref.scheme == "loco")].set_index("metric")
    for m in b.index:
        got = b.loc[m, ["estimate", "ci_low", "ci_high"]].to_numpy(dtype=float)
        want = ref.loc[m, ["estimate", "ci_low", "ci_high"]].to_numpy(dtype=float)
        if np.abs(got - want).max() > 1e-9:
            raise AssertionError(f"Task {task} {m}: {got} differs from the 4.4 panel {want}")
    return {"task": task, "config": cfg, "model": "lgbm", "scheme": "loco", "n": len(o),
            "n_uas": int(o["ua"].nunique()), "threshold": metrics.THRESHOLD,
            **{f"{m}_{c}": float(b.loc[m, c]) for m in b.index
               for c in ("estimate", "ci_low", "ci_high")}}


# --------------------------------------------------------------------------- figure


def panels_for(t: pd.DataFrame) -> list[dict]:
    out = []
    for r in t.itertuples():
        out.append({
            "cm": [[r.TN, r.FP], [r.FN, r.TP]],
            "title": f"{r.city} · n = {r.n:,}",
            "subtitle": (f"F1 {r.f1:.3f} · Acc {r.accuracy:.3f} · MCC {r.mcc:.3f}\n"
                         f"TPR {r.TPR:.3f} = {r.TP:,} / {r.TPR_denominator:,} actual high\n"
                         f"TNR {r.TNR:.3f} = {r.TN:,} / {r.TNR_denominator:,} actual low"),
        })
    return out


def header_for(t: pd.DataFrame, p: pd.Series) -> str:
    task = p["task"]
    return (f"Leave-one-city-out — {FIGURES[task]['task_label']}, 10 largest urbanized areas\n"
            f"Pooled F1 {p.f1_estimate:.3f} [95 % CI {p.f1_ci_low:.3f}, {p.f1_ci_high:.3f}] "
            f"over {int(p.n):,} tracts · F1 across the 10 cities: {spread(t.f1)}")


NOTE = ("Each panel is one urbanized area held out from training (the model is fitted on the "
        "other 485). Cells: count and its share of the city's tracts (the four cells sum to "
        "100 %). TPR = TP / actual high; TNR = TN / actual low. LightGBM, τ = 0.5. "
        "CI: UA cluster bootstrap, B = 1,000.")


def draw_figure(fig, task: str, t: pd.DataFrame | None = None,
                p: pd.Series | None = None, note: bool = True) -> None:
    """Draw the figure for `task` on `fig`, from the results CSVs by default."""
    if t is None:
        t = pd.read_csv(OUT / "confusion_by_city_panels.csv", dtype={"ua": str})
    if p is None:
        p = pd.read_csv(OUT / "confusion_by_city_pooled.csv").set_index("task").loc[task]
        p["task"] = task
    t = t[t.task == task].sort_values("rank")
    shares = t[["TN", "FP", "FN", "TP"]].to_numpy() / t[["n"]].to_numpy()
    vmax = float(np.ceil(shares.max() * 10) / 10)
    draw_grid(fig, panels_for(t), FIGURES[task]["classes"], cmap_for(task), header_for(t, p),
              cell_count_share_of_city, vmax, NOTE if note else None)


# --------------------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = validation.frame("primary")
    t = pd.concat([city_panels(df, task) for task in FIGURES], ignore_index=True)
    p = pd.DataFrame([pooled(df, task) for task in FIGURES])
    for _, r in t.iterrows():
        if r.TN + r.FP + r.FN + r.TP != r.n:
            raise AssertionError(f"{r.city}: confusion matrix does not sum to n")
    t.to_csv(OUT / "confusion_by_city_panels.csv", index=False)
    report.write_text(
        "# Confusion matrices by city: the ten held-out cities\n\n"
        + report.md_table(t.drop(columns=["ua_name"]), digits=3)
        + "\nLeave-one-city-out LightGBM, τ = 0.5. TPR denominator = actual high; "
          "TNR denominator = actual low.\n",
        OUT / "confusion_by_city_panels.md")
    p.to_csv(OUT / "confusion_by_city_pooled.csv", index=False)
    plotting.apply_style()
    for task, spec in FIGURES.items():
        fig = plt.figure(figsize=FIGSIZE)
        draw_figure(fig, task, t, p.set_index("task", drop=False).loc[task])
        plotting.savefig(fig, OUT / spec["png"])
    write_readme()
    print(t[["task", "city", "n", "f1", "accuracy", "mcc"]].to_string(index=False))
    print(p.to_string(index=False))


def write_readme() -> None:
    t = pd.read_csv(OUT / "confusion_by_city_panels.csv", dtype={"ua": str})
    p = pd.read_csv(OUT / "confusion_by_city_pooled.csv").set_index("task")
    f = report.fmt_num
    rows = []
    for r in t.itertuples():
        rows.append(f"| {FIGURES[r.task]['panels']} | {r.rank} | {r.city} | {r.n:,} | {r.TN:,} | "
                    f"{r.FP:,} | {r.FN:,} | {r.TP:,} | {f(r.TPR)} ({r.TP:,}/{r.TPR_denominator:,}) "
                    f"| {f(r.TNR)} ({r.TN:,}/{r.TNR_denominator:,}) | {f(r.f1)} | {f(r.accuracy)} "
                    f"| {f(r.mcc)} |")
    head = []
    for task, spec in FIGURES.items():
        x, s = p.loc[task], t[t.task == task]
        head.append(f"| {spec['panels']} | {task} `{spec['config']}` | {int(x.n):,} | "
                    f"{report.fmt_ci(x.f1_estimate, x.f1_ci_low, x.f1_ci_high)} | "
                    f"{report.fmt_ci(x.mcc_estimate, x.mcc_ci_low, x.mcc_ci_high)} | "
                    f"{spread(s.f1)} |")
    report.require(abs(p.loc["A", "f1_estimate"] - 0.819) < 5e-4
                   and abs(p.loc["B", "f1_estimate"] - 0.725) < 5e-4,
                   "pooled F1 no longer 0.819 / 0.725")
    cities = ", ".join(t[t.task == "A"].sort_values("rank").city)
    text = f"""# Confusion matrices by city

Two figures, one for the UHI model and one for the poverty model, each show ten confusion
matrices in a 2 × 5 grid, with the count and percentage in each cell, F1 · Accuracy / MCC above
each panel, and the F1 spread in the header.

## Design

- **Panels.** Each panel is one of the ten largest urbanized areas by tract count
  ({cities}), held out from training under leave-one-city-out validation. The
  model never saw a tract of the city it is scored on.
- **Model and labels.** The cached leave-one-city-out predictions of the headline LightGBM
  (`validation.oof(task, config, "lgbm", "loco")`): Task A `A-noLST` (no LST or SUHI
  predictors) for `High_UHI`, Task B `B-env` for `High_Pov`, both at τ = 0.5. Poverty is the ACS
  poverty rate against the national median.
- **Denominators** (per
  [`4.3 DEFINITIONS.md`](../4_Model_Specifications/4.3_Confusion_Matrices_and_Denominators/DEFINITIONS.md)).
  The cell percentage is the count over the city's tracts, and the cell says so; the four cells
  sum to 100 %. TPR (TP / actual high) and TNR (TN / actual low) are in each panel's subtitle
  with their numerators and denominators.
- **Header.** The pooled leave-one-city-out F1 over all {int(p.loc['A', 'n']):,} tracts and
  {int(p.loc['A', 'n_uas'])} UAs, with its 95 % UA cluster-bootstrap CI (B = 1,000), then the mean,
  SD and range of F1 across the ten cities. The script checks the pooled estimate and CI
  against
  [`4.4 metric_panel.csv`](../4_Model_Specifications/4.4_Metric_Panel_with_Uncertainty/results/metric_panel.csv)
  to 1e-9.
- **Shading** is the cell's share of the city's tracts on one scale for all ten panels, so
  panels can be compared. The UHI panels are blue and the poverty panels orange.

## Results

| Panels | Task | Tracts | Pooled F1 [95 % CI] | Pooled MCC [95 % CI] | F1 across the 10 cities |
|---|---|---:|---|---|---|
{chr(10).join(head)}

| Panels | Rank | City | n | TN | FP | FN | TP | TPR | TNR | F1 | Acc | MCC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Figures: [`results/confusion_by_city_uhi.png`](results/confusion_by_city_uhi.png),
[`results/confusion_by_city_poverty.png`](results/confusion_by_city_poverty.png). Tables:
[`results/confusion_by_city_panels.csv`](results/confusion_by_city_panels.csv),
[`results/confusion_by_city_pooled.csv`](results/confusion_by_city_pooled.csv).

Rebuild: `uv run python 6_Figures/confusion_by_city.py` (reads the
out-of-fold cache that `make engine` fills).
"""
    report.write_text(text, HERE / "README_confusion_by_city.md")


if __name__ == "__main__":
    main()
