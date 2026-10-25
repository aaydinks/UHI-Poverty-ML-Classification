"""Section 4.3: confusion matrices with every rate's denominator stated.

For each model × task under leave-one-city-out and UA-grouped 5-fold: raw
counts, row-normalised rates (TPR, FNR, TNR, FPR) and column-normalised rates
(PPV, NPV), each with its denominator. Writes `DEFINITIONS.md`,
`results/confusion_matrices.{csv,md}`, `results/majority_folds.{csv,md}`, one
annotated figure per task, and the README.
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


TERMS = """# Confusion-matrix terms and denominators

Every rate in this repository states its denominator. With the median split the two classes
are almost equal in size, so a rate computed over the wrong denominator can be off by a
factor of about two without looking implausible.

| Term | Definition | Denominator |
|---|---|---|
| TP, FP, TN, FN | Raw counts of true positives, false positives, true negatives, false negatives | – |
| TPR / recall / sensitivity | TP / (TP + FN) | all actual positives |
| FNR | FN / (TP + FN) | all actual positives |
| TNR / specificity | TN / (TN + FP) | all actual negatives |
| FPR | FP / (TN + FP) | all actual negatives |
| PPV / precision | TP / (TP + FP) | all predicted positives |
| NPV | TN / (TN + FN) | all predicted negatives |
| Accuracy | (TP + TN) / n | all tracts |
| Balanced accuracy | (TPR + TNR) / 2 | – |
| F1 | 2TP / (2TP + FP + FN) | – |
| MCC | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | – |

TPR + FNR = 1 and TNR + FPR = 1 (each pair shares a row denominator). PPV and NPV share no
denominator with the row rates.

Every confusion matrix is reported as counts together with rates within the actual class
(TPR, FNR, TNR, FPR), with PPV and NPV given separately over their own denominators
([`results/confusion_matrices.md`](results/confusion_matrices.md)).
"""


def main() -> None:
    df = validation.load_analysis()
    t = table(df)
    show = t[["task", "scheme", "model", "n", "TP", "FP", "TN", "FN", "TPR", "FNR", "TNR", "FPR",
              "PPV", "NPV", "accuracy", "balanced_accuracy"]]
    report.write_table(t, OUT / "confusion_matrices", digits=3,
                       title="Confusion matrices and rates (tau = 0.5)")
    report.write_text("# Confusion matrices and rates (τ = 0.5)\n\n" + report.md_table(show)
                      + "\nDenominators: TPR, FNR = actual positives; TNR, FPR = actual "
                        "negatives; PPV = predicted positives; NPV = predicted negatives. "
                        "`confusion_matrices.csv` lists every denominator.\n",
                      OUT / "confusion_matrices.md")
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
    report.write_table(pd.DataFrame(rows), OUT / "majority_folds", digits=3,
                       title="Majority-class baseline: which class each fold predicts")
    report.write_text(TERMS, HERE / "DEFINITIONS.md")
    write_readme()


def write_readme() -> None:
    t = pd.read_csv(OUT / "confusion_matrices.csv")
    mf = pd.read_csv(OUT / "majority_folds.csv").set_index(["task", "scheme"])
    f = report.fmt_num

    def r(task, model, scheme="loco"):
        return t[(t.task == task) & (t.model == model) & (t.scheme == scheme)].iloc[0]

    lines = []
    for task in TASKS:
        for model in ("logistic", "lgbm"):
            x = r(task, model)
            lines.append(
                f"| {task} | {models.MODEL_LABELS[model]} | {x.TP:,} | {x.FP:,} | {x.TN:,} | "
                f"{x.FN:,} | {f(x.TPR)} ({x.TP:,}/{x.TPR_denominator:,}) | "
                f"{f(x.FPR)} ({x.FP:,}/{x.FPR_denominator:,}) | "
                f"{f(x.PPV)} ({x.TP:,}/{x.PPV_denominator:,}) |")
    m = r("A", "majority")
    text = f"""# 4.3 Confusion matrices and denominators

## Purpose

This section reports the full confusion matrix of every model and states what each rate is
divided by. With the median split the two classes are almost equal in size, so a rate over the
wrong denominator can be off by a factor of about two without looking implausible.

## Method

- [`DEFINITIONS.md`](DEFINITIONS.md) defines every term with its denominator.
- `confusion_matrices.py` writes, for every model × task under leave-one-city-out and
  UA-grouped 5-fold, the four counts, all six rates and each rate's denominator
  ([`results/confusion_matrices.csv`](results/confusion_matrices.csv)), and one annotated
  figure per task. Each matrix is checked to sum to n.

## Results

Leave-one-city-out, n = {m.n:,} tracts (τ = 0.5); rate (numerator/denominator):

| Task | Model | TP | FP | TN | FN | TPR | FPR | PPV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

The majority-class baseline predicts one class for every tract in a fold: under
leave-one-city-out on Task A it yields TP = {m.TP:,}, FP = {m.FP:,}, TN = {m.TN:,},
FN = {m.FN:,}. The national median split leaves the two classes within one tract of each other,
so holding out a city tips the training majority toward the class that city has *less* of:
under leave-one-city-out the majority class predicts "positive" in {mf.loc[('A', 'loco'), 'folds_predicting_positive']}
of {mf.loc[('A', 'loco'), 'n_folds']} folds on Task A and its accuracy falls to
{f(mf.loc[('A', 'loco'), 'accuracy'])} ({f(mf.loc[('B', 'loco'), 'accuracy'])} on Task B;
[`results/majority_folds.md`](results/majority_folds.md)). Pooled over folds, the baseline is
therefore worse than chance, not merely equal to it. The no-skill references to compare models
against are balanced accuracy 0.5, MCC 0 and ROC-AUC 0.5.

Figures: [`results/confusion_taskA.png`](results/confusion_taskA.png),
[`results/confusion_taskB.png`](results/confusion_taskB.png). UA-grouped 5-fold matrices are
in the same CSV.
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
