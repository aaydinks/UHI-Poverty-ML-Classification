"""Section 5.1: which model the attributions explain, and on what scale.

The explained model is the evaluated LightGBM (`models.py`, fixed settings), one
fit per training fold of the UA-grouped 5-fold split, explained on its held-out
fold, on the input columns themselves, so every attribution is on a named input
column. Writes the additivity check against the cached out-of-fold scores and the
attribution groups with the identities each one contains.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from uhi_poverty import features, interpret, paths, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
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
    report.write_table(groups_table(), OUT / "attribution_groups")


if __name__ == "__main__":
    main()
