"""Section 5.1: which model the attributions explain, and on what scale.

The explained model is the evaluated LightGBM (`models.py`, fixed settings), one
fit per training fold of the UA-grouped 5-fold split, explained on its held-out
fold, on the input columns themselves, so every attribution is on a named input
column. Writes the additivity check against the cached out-of-fold scores, the
attribution groups with the identities each one contains, and the README.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from uhi_poverty import features, interpret, models, paths, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
ERRORS = ["max_abs_error_probability", "max_abs_error_logit"]
DELTAS = paths.ROOT / "4_Model_Specifications" / "4.5_Baselines" / "results" / "paired_deltas.csv"


def additivity(res: dict) -> pd.DataFrame:
    rows = []
    for task, r in res.items():
        m = r["meta"]
        imp = r["importance"]
        rows.append({"task": task, "config": m["config"], "n_tracts": m["additivity"]["n"],
                     "n_features": len(m["features"]),
                     "fold_models_of_record": imp.loc[imp.seed == m["record_seed"], "fold"]
                     .nunique(),
                     "max_abs_error_probability": m["additivity"]["max_abs_error_probability"],
                     "max_abs_error_logit": m["additivity"]["max_abs_error_logit"]})
    return pd.DataFrame(rows)


def groups_table() -> pd.DataFrame:
    rows = []
    for task in interpret.TASKS:
        cfg = interpret.config(task)
        for g, members in features.attribution_groups(cfg).items():
            inside = [d for d, u, r in features.IDENTITIES if {d, u, r} <= set(members)]
            rows.append({"task": task, "config": cfg, "group": g, "n_features": len(members),
                         "identities_inside": len(inside), "features": ", ".join(members)})
    return pd.DataFrame(rows)


def main() -> None:
    df = validation.load_analysis()
    res = {t: interpret.compute(t, df=df) for t in interpret.TASKS}
    a = additivity(res)
    OUT.mkdir(parents=True, exist_ok=True)
    a.to_csv(OUT / "additivity.csv", index=False)
    show = a.assign(**{c: a[c].map(lambda v: f"{v:.1e}") for c in ERRORS})
    report.write_text("# Contributions against the cached out-of-fold scores\n\n"
                      + report.md_table(show)
                      + "\nError = |sigmoid(bias + sum of SHAP) − OOF score|, and the same on the "
                        "logit scale, over every tract.\n", OUT / "additivity.md")
    report.write_table(groups_table(), OUT / "attribution_groups", digits=0,
                       title="Attribution groups (closed under the exact identities of 3.1)")
    write_readme()


def write_readme() -> None:
    a = pd.read_csv(OUT / "additivity.csv").set_index("task")
    g = pd.read_csv(OUT / "attribution_groups.csv")
    d = pd.read_csv(DELTAS)
    auc = d[(d.comparison == "LightGBM − logistic") & (d.metric == "roc_auc")]
    auc = auc.set_index(["task", "scheme"])["a"]
    p = models.LGBM_PARAMS
    # The prose below states this; fail rather than publish it if it changes.
    report.require(all(abs(auc.loc[(t, "ua5")] - auc.loc[(t, "loco")]) < 0.01
                       for t in interpret.TASKS), "UA-grouped AUC no longer within 0.01 of LOCO")
    lines = "\n".join(
        f"| {t} | `{a.loc[t, 'config']}` | {a.loc[t, 'n_features']} | {a.loc[t, 'n_tracts']:,} | "
        f"{a.loc[t, 'max_abs_error_probability']:.0e} | {a.loc[t, 'max_abs_error_logit']:.0e} |"
        for t in interpret.TASKS)
    glines = "\n".join(
        f"| {r.task} | {r.group} | {r.n_features} | {r.identities_inside} |"
        for r in g.itertuples())
    text = f"""# 5.1 The explained model

## Purpose

States which model the SHAP values explain, on what scale they are reported, and how each
attribution maps back to a named input column.

## Method

- The model explained is the model evaluated in sections 3 and 4: LightGBM with fixed settings
  (`n_estimators={p['n_estimators']}`, `learning_rate={p['learning_rate']}`,
  `num_leaves={p['num_leaves']}`, seed {p['random_state']}), fitted directly on the input
  columns, so every attribution belongs to a named input column.
- `src/uhi_poverty/interpret.py` refits that model on each training fold of the UA-grouped
  5-fold split and computes LightGBM's exact TreeSHAP values (`pred_contrib=True`) for the
  held-out fold only. No tract is explained by a model that saw its urbanized area.
- The values are in **log-odds** of the positive class. For each tract, the bias term plus the
  sum of its SHAP values equals the logit of its out-of-fold score. `explained_model.py` checks
  this against the out-of-fold predictions cached by `validation.oof()`, the same predictions
  behind the UA-grouped results in section 4
  ([`results/additivity.md`](results/additivity.md)).
- Columns linked by an exact identity (section 3.1: every `UHI_*` is an urban-minus-rural LST
  difference, every `DelNDVI_*` an NDVI difference, `DelDEM` a DEM difference) split credit
  arbitrarily among themselves. Attributions are therefore also reported for groups that keep
  each identity whole ([`results/attribution_groups.md`](results/attribution_groups.md)). The
  tract and city SUHI variants share their rural reference, so heat separates by season and
  time of day but not into tract versus city, or SUHI versus LST.

## Results

| Task | Configuration | Predictors | Tracts explained | Max error, probability | Max error, logit |
|---|---|---:|---:|---:|---:|
{lines}

The attributions reproduce the evaluated model's predictions to floating-point precision.
UA-grouped 5-fold is used here because the stability checks in 5.2 need several fold models
for every tract; its LightGBM ROC-AUC ({auc.loc[('A', 'ua5')]:.3f} on Task A,
{auc.loc[('B', 'ua5')]:.3f} on Task B) is within 0.01 of leave-one-city-out
({auc.loc[('A', 'loco')]:.3f} and {auc.loc[('B', 'loco')]:.3f}; section 4.5).

Attribution groups:

| Task | Group | Columns | Exact identities inside |
|---|---|---:|---:|
{glines}
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
