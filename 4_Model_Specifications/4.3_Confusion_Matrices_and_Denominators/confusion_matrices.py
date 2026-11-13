"""Section 4.3: confusion matrices with every rate's denominator stated.

For each model × task under leave-one-city-out and UA-grouped 5-fold: raw
counts, row-normalised rates (TPR, FNR, TNR, FPR) and column-normalised rates
(PPV, NPV), each with its denominator. Writes `results/confusion_matrices.csv`,
`results/majority_folds.csv` and one annotated figure per task.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from uhi_poverty import (
    features,
    metrics,
    models,
    paths,
    plotting,
    report,
    validation,
)

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
SCHEMES = ("loco", "ua5")
TASKS = ("A", "B")
CLASS_NAMES = {"A": ("low SUHI", "high SUHI"), "B": ("low poverty", "high poverty")}


def table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        cfg = features.TASKS[task]["primary"]
        for scheme in SCHEMES:
            for model in models.MODELS:
                o, _ = metrics.scored_run(task, cfg, model, scheme, df=df)
                cm = metrics.confusion(o["y"], o["pred"])
                n = sum(cm.values())
                if n != len(o):
                    raise AssertionError("confusion matrix does not sum to n")
                r = metrics.rates(cm)
                row = {"task": task, "scheme": scheme, "model": model, "n": int(n),
                       **{k: int(v) for k, v in cm.items()}}
                for name in metrics.RATES:
                    row[name] = r[name]
                    row[f"{name}_denominator"] = int(r[f"{name}_denominator"])
                row["accuracy"] = (cm["TP"] + cm["TN"]) / n
                row["balanced_accuracy"] = (r["TPR"] + r["TNR"]) / 2
                rows.append(row)
    return pd.DataFrame(rows)


def plot(t: pd.DataFrame, task: str) -> None:
    plotting.apply_style()
    neg, pos = CLASS_NAMES[task]
    fig, axes = plt.subplots(len(models.MODELS), len(SCHEMES), figsize=(7.4, 9.0))
    cmap = plotting.sequential_cmap()
    for i, model in enumerate(models.MODELS):
        for j, scheme in enumerate(SCHEMES):
            ax = axes[i, j]
            r = t[(t.task == task) & (t.scheme == scheme) & (t.model == model)].iloc[0]
            cells = [[("TN", r.TN, r.TNR, r.TNR_denominator, "actual " + neg),
                      ("FP", r.FP, r.FPR, r.FPR_denominator, "actual " + neg)],
                     [("FN", r.FN, r.FNR, r.FNR_denominator, "actual " + pos),
                      ("TP", r.TP, r.TPR, r.TPR_denominator, "actual " + pos)]]
            vals = [[c[2] for c in row] for row in cells]
            ax.imshow(vals, cmap=cmap, vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    name, count, rate, den, what = cells[a][b]
                    ink = "white" if rate > 0.55 else plotting.INK
                    ax.text(b, a - 0.12, f"{name}  {count:,}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=ink)
                    ax.text(b, a + 0.18, f"{rate * 100:.1f} % of {den:,}\n{what}",
                            ha="center", va="center", fontsize=7, color=ink)
            ax.set_xticks([0, 1], [f"pred. {neg}", f"pred. {pos}"])
            ax.set_yticks([0, 1], [f"actual {neg}", f"actual {pos}"], rotation=90,
                          va="center")
            ax.tick_params(length=0)
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_title(f"{models.MODEL_LABELS[model]} · {validation.SCHEME_LABELS[scheme]}",
                         fontsize=9)
    fig.suptitle(f"{features.TASKS[task]['label']}: counts and row-normalised rates "
                 f"(n = {int(t[t.task == task].n.iloc[0]):,})", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    plotting.note(fig, "Shading = the cell's rate within its actual class (row). τ = 0.5.")
    fig.tight_layout()
    plotting.savefig(fig, OUT / f"confusion_task{task}.png")


def main() -> None:
    df = validation.load_analysis()
    t = table(df)
    report.write_table(t, OUT / "confusion_matrices")
    for task in TASKS:
        plot(t, task)
    rows = []
    for task in TASKS:
        for scheme in SCHEMES:
            o, _ = metrics.scored_run(task, features.TASKS[task]["primary"], "majority", scheme,
                                      df=df)
            g = o.groupby("fold")["pred"].first()
            rows.append({"task": task, "scheme": scheme, "n_folds": len(g),
                         "folds_predicting_positive": int(g.sum()),
                         "accuracy": float((o["y"] == o["pred"]).mean())})
    report.write_table(pd.DataFrame(rows), OUT / "majority_folds")


if __name__ == "__main__":
    main()
