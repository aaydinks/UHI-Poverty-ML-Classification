"""Section 4.6, round 2: a second tuning round designed from round 1's inner evidence.

Round 2 (`python -m uhi_poverty.tuning --round 2`) keeps the nesting of round 1 and
changes only what round 1's *inner* cross-validation pointed to: the low-variance
filter and PCA are dropped, boosters get no scaler, ranges that the search pushed
against are widened, the trial budget grows, every outer refit is averaged over
five seeds, and a stacked ensemble joins the soft votes. This script scores round 2
against round 1 and the fixed LightGBM on the same tracts and writes
`results/round2_*` and `README_round2.md`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from uhi_poverty import features, metrics, models, paths, report, tuning, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
TASKS = ("A", "B")
SCHEMES = ("loco", "ua5")
TARGETS = {"A": 0.90, "B": 0.75}
METRICS = ["accuracy", "f1", "roc_auc", "mcc", "balanced_accuracy"]
LABELS = {**models.TUNED_LABELS,
          "ens_stacked": "Stacked ensemble (logistic regression on member logits)"}


def load(task: str, model: str, scheme: str, rnd: int, df: pd.DataFrame) -> pd.DataFrame:
    if rnd == 0:  # the fixed LightGBM of 4.4
        o = validation.oof(task, features.TASKS[task]["primary"], "lgbm", scheme, df=df)
        return o.assign(pred_f1=o["pred"])
    return tuning.predictions(task, model, scheme, df=df, rnd=rnd)


def scored(o: pd.DataFrame, metric: str) -> metrics.Scored:
    return metrics.Scored(o["y"], o["score"], o["pred_f1" if metric == "f1" else "pred"])


def panel(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    entries = [("fixed-lgbm", "lgbm", 0)] + [(f"r1-{m}", m, 1) for m in tuning.ALL_MODELS] + [
        (f"r2-{m}", m, 2) for m in tuning.ALL_MODELS_R2]
    for task in TASKS:
        for scheme in SCHEMES:
            for entry, model, rnd in entries:
                o = load(task, model, scheme, rnd, df)
                for metric in METRICS:
                    t = metrics.bootstrap_panel(scored(o, metric), o["ua"].to_numpy(),
                                                metrics=[metric])
                    rows.append({"task": task, "scheme": scheme, "entry": entry, "model": model,
                                 "round": rnd, "metric": metric,
                                 "estimate": t["estimate"].iloc[0],
                                 "ci_low": t["ci_low"].iloc[0], "ci_high": t["ci_high"].iloc[0]})
    return pd.DataFrame(rows)


def deltas(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for task in TASKS:
        for scheme in SCHEMES:
            fixed = load(task, "lgbm", scheme, 0, df)
            for m in tuning.ALL_MODELS_R2:
                r2 = load(task, m, scheme, 2, df)
                pairs = [("round 2 − fixed LightGBM", fixed)]
                if m in tuning.ALL_MODELS:
                    pairs.append(("round 2 − round 1", load(task, m, scheme, 1, df)))
                for label, other in pairs:
                    if not r2["GEOID"].equals(other["GEOID"]):
                        raise AssertionError("paired runs are not aligned")
                    for metric in ("accuracy", "f1", "roc_auc"):
                        d = metrics.paired_delta(scored(r2, metric), scored(other, metric),
                                                 r2["ua"].to_numpy(), metrics=(metric,))
                        d.insert(0, "comparison", label)
                        d.insert(0, "model", m)
                        d.insert(0, "scheme", scheme)
                        d.insert(0, "task", task)
                        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def choices(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for m in tuning.ALL_MODELS_R2:
            for s in tuning.tuned_settings(task, m, "ua5", df, rnd=2):
                row = {"task": task, "model": m, "fold": s["k"], "inner_roc_auc": s["inner_auc"],
                       "tau_acc": s["tau_acc"], "tau_f1": s["tau_f1"]}
                if "prep" in s:
                    row.update({k: s["prep"][k] for k in ("engineer", "corr", "smote", "scaler")})
                    row.update(s["params"])
                else:
                    row.update({f"w_{k}": v for k, v in s["weights"].items()})
                    if "intercept" in s:
                        row["intercept"] = s["intercept"]
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = validation.load_analysis()
    t = panel(df)
    t.to_csv(OUT / "round2_panel.csv", index=False)
    report.write_table(deltas(df), OUT / "round2_deltas", digits=3,
                       title="Round 2, paired differences (95 % UA cluster-bootstrap CI)")
    report.write_table(choices(df), OUT / "round2_choices", digits=3,
                       title="Round 2: what inner cross-validation chose in each outer fold")
    write_readme()


def write_readme() -> None:
    t = pd.read_csv(OUT / "round2_panel.csv")
    d = pd.read_csv(OUT / "round2_deltas.csv")
    mech = pd.read_csv(OUT / "mechanisms.csv")
    f = report.fmt_num

    def est(task, entry, metric, scheme="loco"):
        return t[(t.task == task) & (t.entry == entry) & (t.metric == metric)
                 & (t.scheme == scheme)].iloc[0]

    def ci(task, entry, metric, scheme="loco"):
        r = est(task, entry, metric, scheme)
        return f"{f(r.estimate)} [{f(r.ci_low)}, {f(r.ci_high)}]"

    def dl(task, m, comparison, metric):
        r = d[(d.task == task) & (d.model == m) & (d.comparison == comparison)
              & (d.metric == metric) & (d.scheme == "loco")]
        if r.empty:
            return "–"
        r = r.iloc[0]
        return f"{f(r.delta)} [{f(r.ci_low)}, {f(r.ci_high)}]"

    def best(task, metric, prefix):
        sub = t[(t.task == task) & (t.metric == metric) & (t.scheme == "loco")
                & t.entry.str.startswith(prefix)]
        r = sub.loc[sub.estimate.idxmax()]
        return r.estimate, r.model

    rows = []
    for task in TASKS:
        rows.append(f"| {task} | LightGBM, fixed (4.4) | {ci(task, 'fixed-lgbm', 'accuracy')} | "
                    f"{ci(task, 'fixed-lgbm', 'f1')} | {ci(task, 'fixed-lgbm', 'roc_auc')} | – | – |")
        for m in tuning.ALL_MODELS_R2:
            e = f"r2-{m}"
            rows.append(f"| {task} | {LABELS[m]} | {ci(task, e, 'accuracy')} | {ci(task, e, 'f1')} | "
                        f"{ci(task, e, 'roc_auc')} | {dl(task, m, 'round 2 − round 1', 'accuracy')} | "
                        f"{dl(task, m, 'round 2 − fixed LightGBM', 'accuracy')} |")

    gains = {}
    for task in TASKS:
        sub = d[(d.task == task) & (d.scheme == "loco") & (d.comparison == "round 2 − round 1")]
        gains[task] = sub
    better = {task: sorted({LABELS[r.model] for r in gains[task].itertuples() if r.ci_low > 0})
              for task in TASKS}
    worse = {task: sorted({LABELS[r.model] for r in gains[task].itertuples() if r.ci_high < 0})
             for task in TASKS}

    def names(x):
        return "; ".join(x) if x else "none"

    target_lines = []
    for task in TASKS:
        a1, m1 = best(task, "accuracy", "r1-")
        a2, m2 = best(task, "accuracy", "r2-")
        f2, mf = best(task, "f1", "r2-")
        target = TARGETS[task]
        name = "High SUHI (Task A)" if task == "A" else "High poverty (Task B)"
        target_lines.append(
            f"- **{name}, target {target:.2f}.** Best leave-one-city-out accuracy {f(a2)} "
            f"({LABELS[m2]}), against {f(a1)} in round 1 ({LABELS[m1]}): "
            f"{'met' if a2 >= target else 'not met'}. Best F1 {f(f2)} ({LABELS[mf]}): "
            f"{'met' if f2 >= target else 'not met'}.")

    def evidence(task, option, value):
        r = mech[(mech.task == task) & (mech.option == option) & (mech.value.astype(str) == value)
                 & mech.model.isin(models.BOOSTERS)]
        return f"{f(r['mean_best_inner_roc_auc'].mean(), 4)} ({int(r['folds_choosing'].sum())}/15)"

    text = f"""# 4.6, round 2: a second tuning round

## Why a second round, and how it was designed

Round 1 ([README](README.md)) left the fixed models' leave-one-city-out ROC-AUC almost unchanged.
Round 2 asks whether a more focused search does better. Every change follows from round 1's
**inner** cross-validation (the training cities of each outer fold), never from its outer scores,
so round 2's outer results are as honest as round 1's:

| Change | Inner evidence from round 1 (best inner ROC-AUC, boosters, mean over folds; folds choosing) |
|---|---|
| Low-variance filter dropped | Task A off {evidence('A', 'variance', 'False')} vs on {evidence('A', 'variance', 'True')}; Task B off {evidence('B', 'variance', 'False')} vs on {evidence('B', 'variance', 'True')} |
| PCA dropped | Task A none {evidence('A', 'pca', 'none')} vs 99 % {evidence('A', 'pca', '0.99')}; Task B none {evidence('B', 'pca', 'none')} vs 99 % {evidence('B', 'pca', '0.99')} |
| No scaler for boosters | A tree is unchanged by any monotone rescaling once PCA is gone |
| Wider ranges | Chosen values at or near a bound: CatBoost `l2_leaf_reg` up to 19.5 of 20, `reg_alpha` up to 9.7 of 10, LightGBM `min_child_samples` 186 of 200, XGBoost `gamma` 4.9 of 5 ([`results/search_choices.md`](results/search_choices.md)) |
| More trials | {tuning.N_TRIALS_R2['lgbm']} for LightGBM and XGBoost, {tuning.N_TRIALS_R2['catboost']} for CatBoost, {tuning.N_TRIALS_R2['logistic']} for logistic regression, per outer fold (round 1: {tuning.N_TRIALS['lgbm']}, {tuning.N_TRIALS['catboost']}, {tuning.N_TRIALS['logistic']}) |
| Seed bagging | Each outer refit is the mean of {len(tuning.BAG_SEEDS[2])} pipelines with seeds {', '.join(map(str, tuning.BAG_SEEDS[2]))} |
| Stacked ensemble | A logistic regression on the three boosters' inner out-of-fold logits, beside the equal and tuned-weight soft votes |

The engineered terms, the correlation filter and SMOTE stay as options. The nesting, the
inner splits, the objective (pooled inner ROC-AUC) and the in-fold thresholds are as in round 1.

## Results

Leave-one-city-out (estimate [95 % UA cluster-bootstrap CI]); accuracy and ROC-AUC at τ_acc,
F1 at τ_F1; Δ is paired, on the same tracts and bootstrap draws
([`results/round2_deltas.md`](results/round2_deltas.md)):

| Task | Round-2 model | Accuracy | F1 | ROC-AUC | Δ accuracy vs round 1 | Δ accuracy vs fixed LightGBM |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

**Targets.**

{chr(10).join(target_lines)}

Round-2 models better than their round-1 counterpart on any of accuracy, F1 or ROC-AUC (95 %
interval above zero): Task A, {names(better['A'])}; Task B, {names(better['B'])}. Worse on any:
Task A, {names(worse['A'])}; Task B, {names(worse['B'])}. What inner CV chose is in
[`results/round2_choices.md`](results/round2_choices.md).
"""
    report.write_text(text, HERE / "README_round2.md")


if __name__ == "__main__":
    main()
