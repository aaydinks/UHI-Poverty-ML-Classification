"""Section 4.6: fit and save the final tuned model for each task.

For each task the model is chosen by its mean inner ROC-AUC across the five nested
outer folds, which uses training cities only, not by its leave-one-city-out
score. The chosen procedure is then applied once to every tract
(`tuning.fit_final`) and saved with joblib, beside a model card that records its
settings, thresholds and nested leave-one-city-out performance.

Loading a model needs this repository's package importable (`uv sync`), because
the pipelines use its transformers.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from uhi_poverty import report, tuning, validation

HERE = Path(__file__).resolve().parent
OUT = HERE / "models"
FILES = {"A": "uhi_high_suhi", "B": "poverty_high_pov"}
SHOW = ["accuracy", "f1", "roc_auc", "mcc", "balanced_accuracy"]


def choose(task: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rows = [{"model": m, "mean_inner_roc_auc": float(np.mean(
        [s["inner_auc"] for s in tuning.tuned_settings(task, m, "ua5", df)]))}
        for m in tuning.ALL_MODELS]
    t = pd.DataFrame(rows).sort_values("mean_inner_roc_auc", ascending=False)
    return str(t["model"].iloc[0]), t


def nested_scores(task: str, model: str) -> dict:
    t = pd.read_csv(HERE / "results" / "tuned_panel.csv", dtype={"threshold": str})
    out = {}
    for scheme in ("loco", "ua5"):
        for threshold in ("tau_acc", "tau_f1"):
            sub = t[(t.task == task) & (t.entry == model) & (t.scheme == scheme)
                    & (t.threshold == threshold) & t.metric.isin(SHOW)]
            out[f"{scheme}_{threshold}"] = {r.metric: [r.estimate, r.ci_low, r.ci_high]
                                            for r in sub.itertuples()}
    return out


def main() -> None:
    df = validation.frame("primary")
    OUT.mkdir(exist_ok=True)
    cards = {}
    for task, stem in FILES.items():
        model, ranking = choose(task, df)
        final = tuning.fit_final(task, model, df)
        final.card["selection"] = ranking.to_dict(orient="records")
        final.card["nested_performance"] = nested_scores(task, model)
        final.card["inputs"] = final.columns
        joblib.dump(final, OUT / f"{stem}.joblib", compress=3)
        report.write_json(final.card, OUT / f"{stem}.json")
        # the saved file must reproduce the in-memory model
        reloaded = joblib.load(OUT / f"{stem}.joblib")
        X = df[final.columns].head(500)
        if not np.allclose(reloaded.predict_proba(X), final.predict_proba(X)):
            raise AssertionError(f"{stem}: reloaded model disagrees")
        cards[task] = final.card
    write_readme(cards)


def write_readme(cards: dict) -> None:
    f = report.fmt_num
    rows = []
    for task, card in cards.items():
        perf = card["nested_performance"]["loco_tau_acc"]
        f1 = card["nested_performance"]["loco_tau_f1"]["f1"]
        rows.append(f"| `{FILES[task]}.joblib` | {card['target']} | {card['label']} | "
                    f"{len(card['inputs'])} | {f(card['tau_acc'], 2)} / {f(card['tau_f1'], 2)} | "
                    f"{f(perf['accuracy'][0])} [{f(perf['accuracy'][1])}, {f(perf['accuracy'][2])}] | "
                    f"{f(f1[0])} | {f(perf['roc_auc'][0])} |")
    text = f"""# Final tuned models

Each file is the tuned procedure of section 4.6 applied once to all 54,243 tracts: inner
UA-grouped 5-fold cross-validation over every city chose the preprocessing options, the
parameters and the thresholds, and the pipeline was refitted on all tracts. The model type was
chosen by mean inner ROC-AUC in the nested runs, which uses training cities only. These models
have no held-out score of their own; their expected performance in a city they have not seen is
the nested leave-one-city-out estimate below.

| File | Target | Model | Inputs | τ_acc / τ_F1 | Accuracy, leave-one-city-out [95 % CI] | F1 (τ_F1) | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The `.json` beside each file is its model card: the input columns in order, every chosen
setting, the thresholds, the selection table and the nested performance with intervals.

## Use

```python
import joblib
import pandas as pd

model = joblib.load("uhi_high_suhi.joblib")  # needs the repository's package: `uv sync`
X = pd.read_parquet("tracts.parquet")       # one row per tract, with model.columns present
probability = model.predict_proba(X)        # P(class 1)
label = model.predict(X)                    # τ_acc; model.predict(X, threshold="f1") for τ_F1
```

The UHI model takes no temperature input. Its inputs are vegetation, elevation, morphology and
tract geometry from the SUHI database and the 2019 Gazetteer, on 2010 tract boundaries. The
poverty model takes every SUHI-database column plus the same geometry. Both are cross-sectional
classifiers of the median split of the 2015–2019 sample; they are not forecasts and are not
validated for planning decisions (see the paper's limitations).
"""
    report.write_text(text, OUT / "README.md")
    print(json.dumps({t: c["model"] for t, c in cards.items()}))


if __name__ == "__main__":
    main()
