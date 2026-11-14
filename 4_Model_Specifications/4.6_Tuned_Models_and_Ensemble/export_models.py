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


if __name__ == "__main__":
    main()
