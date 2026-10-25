"""Section 4.6: nested tuning of the full pipeline, XGBoost, CatBoost and a soft-voting ensemble.

Reads the tuned-model cache (`make tune`, `src/uhi_poverty/tuning.py`) and the
fixed-setting predictions of sections 4.1-4.5, and scores both on the same
tracts. Every tuned choice (preprocessing, hyperparameters, ensemble weights and
the thresholds τ_acc and τ_F1) was made on training cities only. Writes
`results/tuned_panel`, `results/paired_deltas`, `results/search_choices`,
`results/mechanisms`, `results/tuned_vs_fixed.png` and the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from uhi_poverty import features, metrics, models, paths, plotting, report, tuning, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
TASKS = ("A", "B")
SCHEMES = ("loco", "ua5", "random5")
HEADLINE = "loco"
TARGETS = {"A": 0.90, "B": 0.75}
METRICS = ["accuracy", "f1", "roc_auc", "mcc", "balanced_accuracy", "precision", "recall"]
SHOW = ["accuracy", "f1", "roc_auc", "mcc", "balanced_accuracy"]
METRIC_LABELS = {"accuracy": "Accuracy", "f1": "F1", "roc_auc": "ROC-AUC", "mcc": "MCC",
                 "balanced_accuracy": "Balanced accuracy", "precision": "Precision",
                 "recall": "Recall"}
# (entry, source, model, label)
ENTRIES = [("fixed-logistic", "fixed", "logistic", "Logistic regression, fixed (4.4)"),
           ("fixed-lgbm", "fixed", "lgbm", "LightGBM, fixed (4.4)")] + [
    (m, "tuned", m, models.TUNED_LABELS[m]) for m in tuning.ALL_MODELS]
LABELS = {e: label for e, _, _, label in ENTRIES}
# threshold name -> prediction column in the tuned cache
THRESHOLDS = {"0.5": "pred_05", "tau_acc": "pred", "tau_f1": "pred_f1"}
THRESHOLD_LABELS = {"0.5": "τ = 0.5", "tau_acc": "τ_acc", "tau_f1": "τ_F1"}
PREP_OPTIONS = ["engineer", "variance", "corr", "scaler", "pca", "smote"]
PREP_LABELS = {"engineer": "Engineered terms", "variance": "Low-variance filter (0.1)",
               "corr": "Correlation filter (|r| > 0.9)", "scaler": "Scaling",
               "pca": "PCA", "smote": "SMOTE"}


def load(task: str, source: str, model: str, scheme: str, df: pd.DataFrame) -> pd.DataFrame:
    if source == "fixed":
        o = validation.oof(task, features.TASKS[task]["primary"], model, scheme, df=df)
        return o.assign(pred_05=o["pred"], pred_f1=np.nan)
    return tuning.predictions(task, model, scheme, df=df)


def _score(task, source, model, scheme, threshold) -> pd.DataFrame:
    o = load(task, source, model, scheme, validation.load_analysis())
    s = metrics.Scored(o["y"], o["score"], o[THRESHOLDS[threshold]])
    t = metrics.bootstrap_panel(s, o["ua"].to_numpy(), metrics=METRICS)
    return t.assign(task=task, scheme=scheme, threshold=threshold)


def panel(df: pd.DataFrame) -> pd.DataFrame:
    jobs = []
    for task in TASKS:
        for scheme in SCHEMES:
            for entry, source, model, _ in ENTRIES:
                load(task, source, model, scheme, df)  # warm the caches in this process
                for threshold in (["0.5"] if source == "fixed" else THRESHOLDS):
                    jobs.append((entry, (task, source, model, scheme, threshold)))
    parts = Parallel(n_jobs=-1)(delayed(_score)(*args) for _, args in jobs)
    for (entry, _), part in zip(jobs, parts, strict=True):
        part.insert(0, "entry", entry)
    t = pd.concat(parts, ignore_index=True)
    t["label"] = t["entry"].map(LABELS)
    return t[["task", "scheme", "entry", "label", "threshold", "metric", "estimate", "ci_low",
              "ci_high"]]


def comparisons() -> list[tuple[str, str, str, str, tuple[str, ...]]]:
    """(comparison, entry a, threshold a, entry b, metrics); b is at its own headline threshold."""
    out = []
    for m in tuning.ALL_MODELS:
        out.append((f"{LABELS[m]} − LightGBM, fixed", m, "tau_acc", "fixed-lgbm",
                    ("accuracy", "roc_auc")))
        out.append((f"{LABELS[m]} (τ_F1) − LightGBM, fixed", m, "tau_f1", "fixed-lgbm",
                    ("f1",)))
    for ens in models.ENSEMBLES:
        for m in models.BOOSTERS:
            out.append((f"{LABELS[ens]} − {LABELS[m]}", ens, "tau_acc", m,
                        ("accuracy", "roc_auc")))
    return out


def deltas(df: pd.DataFrame) -> pd.DataFrame:
    source = {e: (s, m) for e, s, m, _ in ENTRIES}
    frames = []
    for task in TASKS:
        for scheme in ("loco", "ua5"):
            for label, ea, ta, eb, mets in comparisons():
                oa = load(task, *source[ea], scheme, df)
                ob = load(task, *source[eb], scheme, df)
                if not oa["GEOID"].equals(ob["GEOID"]):
                    raise AssertionError("paired runs are not aligned")
                tb = "0.5" if source[eb][0] == "fixed" else "tau_acc"
                sa = metrics.Scored(oa["y"], oa["score"], oa[THRESHOLDS[ta]])
                sb = metrics.Scored(ob["y"], ob["score"], ob[THRESHOLDS[tb]])
                d = metrics.paired_delta(sa, sb, oa["ua"].to_numpy(), metrics=mets)
                d.insert(0, "comparison", label)
                d.insert(0, "scheme", scheme)
                d.insert(0, "task", task)
                frames.append(d)
    return pd.concat(frames, ignore_index=True)


def search_choices(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for tune_scheme in ("ua5", "random5"):
            for model in tuning.ALL_MODELS:
                for s in tuning.tuned_settings(task, model, tune_scheme, df):
                    row = {"task": task, "tuning_scheme": tune_scheme, "model": model,
                           "fold": s["k"], "inner_roc_auc": s["inner_auc"],
                           "tau_acc": s["tau_acc"], "tau_f1": s["tau_f1"]}
                    if model in models.ENSEMBLES:
                        row.update({f"w_{m}": s["weights"][m] for m in models.BOOSTERS})
                    else:
                        row.update(s["prep"])
                        row["params"] = json.dumps(s["params"], sort_keys=True)
                    rows.append(row)
    return pd.DataFrame(rows)


def mechanisms(df: pd.DataFrame) -> pd.DataFrame:
    """For each option value: the best inner ROC-AUC among trials using it, averaged over folds."""
    rows = []
    for task in TASKS:
        for model in models.TUNED_MODELS:
            settings = tuning.tuned_settings(task, model, "ua5", df)
            for option in PREP_OPTIONS:
                values = sorted({str(h[option]) for s in settings for h in s["history"]})
                for v in values:
                    best, n = [], 0
                    for s in settings:
                        aucs = [h["auc"] for h in s["history"]
                                if str(h[option]) == v and h["auc"] is not None]
                        n += len(aucs)
                        if aucs:
                            best.append(max(aucs))
                    chosen = sum(str(s["prep"][option]) == v for s in settings)
                    rows.append({"task": task, "model": model, "option": option, "value": v,
                                 "trials": n, "folds_with_trials": len(best),
                                 "mean_best_inner_roc_auc": np.mean(best) if best else np.nan,
                                 "folds_choosing": chosen})
    return pd.DataFrame(rows)


def class_balance(df: pd.DataFrame) -> dict:
    """Positive share of the training rows, across the outer folds of each scheme."""
    out = {}
    for task in TASKS:
        d = validation.frame("primary", df)
        y = d[features.TASKS[task]["target"]].astype(int).to_numpy()
        for scheme in ("ua5", "loco"):
            shares = [y[tr].mean() for _, tr, _ in validation.splits(d, y, scheme)]
            out[f"{task}_{scheme}"] = [float(min(shares)), float(max(shares))]
    return out


def plot(t: pd.DataFrame) -> None:
    plotting.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), sharey="row")
    for col, task in enumerate(TASKS):
        for row, metric in enumerate(("accuracy", "f1")):
            ax = axes[row, col]
            threshold = "tau_acc" if metric == "accuracy" else "tau_f1"
            labels, est, lo, hi, colors = [], [], [], [], []
            for entry, source, _, label in ENTRIES:
                th = "0.5" if source == "fixed" else threshold
                r = t[(t.task == task) & (t.scheme == HEADLINE) & (t.entry == entry)
                      & (t.threshold == th) & (t.metric == metric)].iloc[0]
                labels.append(label)
                est.append(r.estimate)
                lo.append(r.ci_low)
                hi.append(r.ci_high)
                colors.append(plotting.MUTED if source == "fixed" else (
                    plotting.ORANGE if entry in models.ENSEMBLES else plotting.BLUE))
            y = np.arange(len(labels))[::-1]
            est, lo, hi = map(np.asarray, (est, lo, hi))
            ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="none", ecolor=plotting.AXIS,
                        elinewidth=1.5, capsize=0)
            ax.scatter(est, y, c=colors, s=36, edgecolors=plotting.SURFACE, linewidths=1.5,
                       zorder=3)
            for yi, v in zip(y, est, strict=True):
                ax.text(v, yi + 0.3, f"{v:.3f}", ha="center", fontsize=7, color=plotting.INK_2)
            ax.axvline(TARGETS[task], color=plotting.INK_2, linewidth=1, linestyle="--")
            ax.set_yticks(y, labels)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(f"{METRIC_LABELS[metric]}, leave-one-city-out")
            if row == 0:
                ax.set_title(f"Task {task}: {'High_UHI' if task == 'A' else 'High_Pov'}")
    plotting.note(fig, "Grey: fixed settings at τ = 0.5. Blue: tuned, at the threshold chosen on "
                  "training cities (τ_acc for accuracy, τ_F1 for F1). Orange: soft-voting "
                  "ensembles. Bars: 95 % UA cluster-bootstrap CI. Dashed: the target (0.90 for "
                  "Task A, 0.75 for Task B).")
    fig.tight_layout()
    plotting.savefig(fig, OUT / "tuned_vs_fixed.png")


def wide(t: pd.DataFrame) -> str:
    out = ["# Tuned and fixed models, every scheme and threshold\n",
           "Cells: estimate [95 % UA cluster-bootstrap CI], B = 1,000. Fixed models are scored at "
           "τ = 0.5 only. τ_acc and τ_F1 were chosen on inner out-of-fold predictions of the "
           "training cities.\n"]
    for task in TASKS:
        for scheme in SCHEMES:
            sub = t[(t.task == task) & (t.scheme == scheme)]
            rows = []
            for (entry, threshold), g in sub.groupby(["entry", "threshold"], sort=False):
                g = g.set_index("metric")
                row = {"model": LABELS[entry], "threshold": THRESHOLD_LABELS[threshold]}
                for m in SHOW:
                    row[METRIC_LABELS[m]] = report.fmt_ci(g.loc[m, "estimate"],
                                                          g.loc[m, "ci_low"], g.loc[m, "ci_high"])
                rows.append(row)
            out.append(f"## {features.TASKS[task]['label']}, "
                       f"{validation.SCHEME_LABELS[scheme]}\n")
            out.append(report.md_table(pd.DataFrame(rows)))
    return "\n".join(out)


def main() -> None:
    df = validation.load_analysis()
    t = panel(df)
    t.to_csv(OUT / "tuned_panel.csv", index=False)
    report.write_text(wide(t), OUT / "tuned_panel.md")
    d = deltas(df)
    report.write_table(d, OUT / "paired_deltas", digits=3,
                       title="Paired differences, Δ = a − b (95 % UA cluster-bootstrap CI)")
    report.write_table(search_choices(df), OUT / "search_choices", digits=3,
                       title="What inner cross-validation chose in each outer fold")
    report.write_table(mechanisms(df), OUT / "mechanisms", digits=4,
                       title="Each pipeline option: best inner ROC-AUC among the trials using "
                             "it (UA-grouped tuning, mean over the 5 outer folds)")
    report.write_json(class_balance(df), OUT / "class_balance.json")
    plot(t)
    write_readme()


# --------------------------------------------------------------------------- README


def write_readme() -> None:
    t = pd.read_csv(OUT / "tuned_panel.csv", dtype={"threshold": str})
    d = pd.read_csv(OUT / "paired_deltas.csv")
    ch = pd.read_csv(OUT / "search_choices.csv")
    mech = pd.read_csv(OUT / "mechanisms.csv")
    balance = report.read_json(OUT / "class_balance.json")
    f = report.fmt_num

    def cell(task, entry, metric, threshold, scheme=HEADLINE):
        r = t[(t.task == task) & (t.scheme == scheme) & (t.entry == entry)
              & (t.threshold == threshold) & (t.metric == metric)].iloc[0]
        return r

    def ci(task, entry, metric, threshold, scheme=HEADLINE):
        r = cell(task, entry, metric, threshold, scheme)
        return f"{f(r.estimate)} [{f(r.ci_low)}, {f(r.ci_high)}]"

    def head_th(entry, metric):
        if entry.startswith("fixed"):
            return "0.5"
        return "tau_f1" if metric == "f1" else "tau_acc"

    def best(task, metric, scheme=HEADLINE):
        rows = [(cell(task, e, metric, head_th(e, metric), scheme).estimate, e)
                for e, *_ in ENTRIES]
        return max(rows)

    def delta(task, comparison, metric, scheme=HEADLINE):
        r = d[(d.task == task) & (d.comparison == comparison) & (d.metric == metric)
              & (d.scheme == scheme)].iloc[0]
        return r

    def dtext(task, comparison, metric, scheme=HEADLINE):
        r = delta(task, comparison, metric, scheme)
        return f"{f(r.delta)} [{f(r.ci_low)}, {f(r.ci_high)}]"

    lines = []
    for task in TASKS:
        for entry, *_ in ENTRIES:
            lines.append(f"| {task} | {LABELS[entry]} | {ci(task, entry, 'accuracy', head_th(entry, 'accuracy'))} | "
                         f"{ci(task, entry, 'f1', head_th(entry, 'f1'))} | "
                         f"{ci(task, entry, 'roc_auc', head_th(entry, 'accuracy'))} | "
                         f"{f(cell(task, entry, 'accuracy', head_th(entry, 'accuracy'), 'ua5').estimate)} | "
                         f"{f(cell(task, entry, 'accuracy', head_th(entry, 'accuracy'), 'random5').estimate)} |")

    target_lines = []
    for task in TASKS:
        name = "High SUHI (Task A)" if task == "A" else "High poverty (Task B)"
        acc, acc_e = best(task, "accuracy")
        f1v, f1_e = best(task, "f1")
        target = TARGETS[task]
        ra, rr = best(task, "accuracy", "random5")
        target_lines.append(
            f"- **{name}, target {target:.2f}.** Best leave-one-city-out accuracy "
            f"{f(acc)} ({LABELS[acc_e]}): {'met' if acc >= target else 'not met'}. "
            f"Best F1 {f(f1v)} ({LABELS[f1_e]}): {'met' if f1v >= target else 'not met'}. "
            f"Under random 5-fold, which leaves almost every tract beside a "
            f"training tract (section 4.2), the best accuracy is {f(ra)} ({LABELS[rr]}).")

    dlines = []
    for task in TASKS:
        for m in tuning.ALL_MODELS:
            comp = f"{LABELS[m]} − LightGBM, fixed"
            dlines.append(f"| {task} | {LABELS[m]} | {dtext(task, comp, 'accuracy')} | "
                          f"{dtext(task, comp, 'roc_auc')} | "
                          f"{dtext(task, f'{LABELS[m]} (τ_F1) − LightGBM, fixed', 'f1')} |")

    ens_lines = []
    for task in TASKS:
        for ens in models.ENSEMBLES:
            parts = [dtext(task, f"{LABELS[ens]} − {LABELS[m]}", "accuracy")
                     for m in models.BOOSTERS]
            ens_lines.append(f"| {task} | {LABELS[ens]} | " + " | ".join(parts) + " |")

    def wins(task):
        out = []
        for m in tuning.ALL_MODELS:
            r = delta(task, f"{LABELS[m]} − LightGBM, fixed", "accuracy")
            r2 = delta(task, f"{LABELS[m]} − LightGBM, fixed", "roc_auc")
            if r.ci_low > 0 or r2.ci_low > 0:
                out.append(LABELS[m])
        return out

    def f1_wins_models(task):
        return [m for m in tuning.ALL_MODELS
                if delta(task, f"{LABELS[m]} (τ_F1) − LightGBM, fixed", "f1").ci_low > 0]

    def f1_wins(task):
        return [LABELS[m] for m in f1_wins_models(task)]

    def ens_beats_members(task, ens):
        return all(delta(task, f"{LABELS[ens]} − {LABELS[m]}", "accuracy").ci_low > 0
                   for m in models.BOOSTERS)

    # how often the inner CV picked each mechanism, over the boosters' 5 outer folds
    boost = ch[(ch.tuning_scheme == "ua5") & ch.model.isin(models.BOOSTERS)]
    n_fold_models = len(boost)

    def share(option, value):
        return int((boost[option].astype(str) == value).sum())

    taus = ch[(ch.tuning_scheme == "ua5") & ~ch.model.isin(models.ENSEMBLES)]
    mech_lines = []
    for task in TASKS:
        for option in PREP_OPTIONS:
            sub = mech[(mech.task == task) & (mech.option == option)
                       & mech.model.isin(models.BOOSTERS)]
            parts = []
            for v, g in sub.groupby("value", sort=True):
                parts.append(f"{v}: {f(g['mean_best_inner_roc_auc'].mean(), 4)} "
                             f"({int(g['folds_choosing'].sum())}/15)")
            mech_lines.append(f"| {task} | {PREP_LABELS[option]} | {'; '.join(parts)} |")

    win_a, win_b = wins("A"), wins("B")
    # the F1 sentence says the gains come from the threshold, not from ranking
    report.require(all(abs(delta(t, f"{LABELS[m]} − LightGBM, fixed", "roc_auc").delta) < 0.01
                       for t in TASKS for m in f1_wins_models(t)),
                   "an F1 gain comes with a ROC-AUC change of 0.01 or more; update the README text")
    best_a, best_ae = best("A", "accuracy")
    best_b, best_be = best("B", "accuracy")
    fixed_a = cell("A", "fixed-lgbm", "accuracy", "0.5").estimate
    fixed_b = cell("B", "fixed-lgbm", "accuracy", "0.5").estimate
    ens_note = {task: [LABELS[e] for e in models.ENSEMBLES if ens_beats_members(task, e)]
                for task in TASKS}

    def winners_text(w):
        # labels contain commas ("XGBoost, tuned"), so separate them with semicolons
        return "; ".join(w) if w else "none"

    text = f"""# 4.6 Tuned models and the soft-voting ensemble

## Purpose

Tests whether a fully tuned pipeline (scaling, PCA, SMOTE, variance and collinearity filters,
engineered terms, tuned XGBoost, LightGBM and CatBoost, a soft-voting ensemble of the three,
and threshold optimisation) raises accuracy and F1 in held-out cities, against targets of 0.90
for high SUHI and 0.75 for high poverty. Sections 4.1–4.5 use fixed settings and τ = 0.5,
declared in advance; the tuned models sit beside them and change none of their results.

## Method

`src/uhi_poverty/tuning.py` assembles the pipeline from its parts, and inner cross-validation
chooses every option:

| Step | Implementation | Search options |
|---|---|---|
| Scaling | `MinMaxScaler` / `StandardScaler` | none, min-max, standard |
| Low-variance filter | `VarianceThreshold(0.1)` | on, off |
| Collinearity filter | drop the later column of each pair with \\|r\\| > 0.9 | on, off |
| PCA | PCA after imputation and scaling | off, 95 % or 99 % of variance |
| Class rebalancing | `imblearn` SMOTE inside the training fold | on, off |
| Engineered terms | Task B: {len(features.ENGINEERED_B)} seasonal, day-night, city-contrast, squared and interaction terms of UHI; Task A: the same terms on the vegetation difference (no temperature) | on, off |
| Models | `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`, plus logistic regression | model parameters (Optuna TPE) |
| Hyperparameter search | Optuna, {tuning.N_TRIALS['lgbm']} trials for LightGBM and XGBoost, {tuning.N_TRIALS['catboost']} for CatBoost, {tuning.N_TRIALS['logistic']} for logistic regression, per outer fold | objective: pooled inner ROC-AUC |
| Soft-voting ensemble | mean of the three tuned boosters' probabilities | equal weights; weights on a {tuning.WEIGHT_STEP} grid chosen on inner predictions |
| Decision threshold | a 99-point grid on **inner** out-of-fold predictions | τ_acc (best accuracy), τ_F1 (best F1) |

The interaction terms multiply `UHI_annual_day` by `Area`, `DEM_rur` and `DEM_urb_CT`, the
first three numeric non-identifier columns of the SUHI file. Task A never sees an `LST_*` or
`UHI_*` column (section 3.2), so its engineered terms are built on `DelNDVI` instead; there
is no day/night vegetation, so the two day-night terms have no Task A analogue.

**Nesting.** For each of the five UA-grouped outer folds, an inner UA-grouped 5-fold over the
training cities scores every trial. The best trial's options and parameters, and the τ_acc and
τ_F1 that maximise accuracy and F1 on its inner out-of-fold predictions, are that fold's
settings. Under leave-one-city-out, each city is predicted by a model fitted on all other
cities with the settings of the outer fold that held it out, which were chosen without it.
Random 5-fold repeats the procedure with random folds, as a reference only.
`tests/test_tuning.py` checks that no tuning step sees an outer test row.

**SMOTE.** The median split leaves the classes balanced. The positive share of the training
cities ranges from {f(balance['A_loco'][0])} to {f(balance['A_loco'][1])} (Task A) and
{f(balance['B_loco'][0])} to {f(balance['B_loco'][1])} (Task B) across leave-one-city-out
folds, so SMOTE has almost nothing to rebalance.

## Results

Leave-one-city-out, the headline (estimate [95 % UA cluster-bootstrap CI]). Tuned models are
scored at τ_acc for accuracy and ROC-AUC and at τ_F1 for F1; fixed models at τ = 0.5. Every
threshold and scheme is in [`results/tuned_panel.md`](results/tuned_panel.md):

| Task | Model | Accuracy | F1 | ROC-AUC | Accuracy, UA-grouped 5-fold | Accuracy, random 5-fold (reference) |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

**Targets.**

{chr(10).join(target_lines)}

**Against the fixed LightGBM** (paired, leave-one-city-out;
[`results/paired_deltas.md`](results/paired_deltas.md)):

| Task | Tuned model | Δ accuracy | Δ ROC-AUC | Δ F1 (τ_F1) |
|---|---|---:|---:|---:|
{chr(10).join(dlines)}

Tuned models whose accuracy or ROC-AUC gain has a 95 % interval above zero: Task A,
{winners_text(win_a)}; Task B, {winners_text(win_b)}. Gains in F1 at τ_F1 do have intervals
above zero for Task A ({winners_text(f1_wins('A'))}) and Task B
({winners_text(f1_wins('B'))}): most of them come from moving the threshold, which trades
specificity for recall, rather than from better ranking. The best leave-one-city-out accuracy
moves from {f(fixed_a)} to {f(best_a)} for Task A ({LABELS[best_ae]}) and from {f(fixed_b)} to
{f(best_b)} for Task B ({LABELS[best_be]}).

**The ensemble against its members** (Δ accuracy, leave-one-city-out):

| Task | Ensemble | − LightGBM | − XGBoost | − CatBoost |
|---|---|---|---|---|
{chr(10).join(ens_lines)}

An ensemble beats all three of its members with a 95 % interval above zero for Task A:
{winners_text(ens_note['A'])}; for Task B: {winners_text(ens_note['B'])}.

**What the inner cross-validation chose** ([`results/search_choices.md`](results/search_choices.md),
[`results/mechanisms.md`](results/mechanisms.md)). Each cell gives, for each option value, the
best inner ROC-AUC among trials using it (mean over boosters and folds), and in brackets how
many of the 15 booster × fold settings chose it:

| Task | Option | Value: best inner ROC-AUC (chosen) |
|---|---|---|
{chr(10).join(mech_lines)}

Across the {n_fold_models} booster × fold settings of both tasks, PCA was chosen
{n_fold_models - share('pca', 'none')} times, SMOTE {share('smote', 'True')} times and the
engineered terms {share('engineer', 'True')} times. The chosen τ_acc ranged from
{f(taus['tau_acc'].min(), 2)} to {f(taus['tau_acc'].max(), 2)} and τ_F1 from
{f(taus['tau_f1'].min(), 2)} to {f(taus['tau_f1'].max(), 2)}.

{gap_text(best_a - fixed_a, best_b - fixed_b, best_a >= TARGETS['A'] and best_b >= TARGETS['B'])}

Figure: [`results/tuned_vs_fixed.png`](results/tuned_vs_fixed.png).
"""
    report.write_text(text, HERE / "README.md")


SMALL_GAIN = 0.02


def gap_text(gain_a: float, gain_b: float, targets_met: bool) -> str:
    f = report.fmt_num
    if targets_met:
        return "**Both targets are met under leave-one-city-out.**"
    size = ("Tuning moves the best leave-one-city-out accuracy by less than "
            f"{SMALL_GAIN:.2f} on both tasks ({f(gain_a)} for Task A, {f(gain_b)} for Task B)"
            if max(gain_a, gain_b) < SMALL_GAIN else
            f"Tuning moves the best leave-one-city-out accuracy by {f(gain_a)} for Task A and "
            f"{f(gain_b)} for Task B")
    return (f"**Why the targets are not reached.** {size}. Leave-one-city-out ROC-AUC bounds what "
            "any threshold can reach, and none of the options raises it far above the fixed "
            "models.")


if __name__ == "__main__":
    main()
