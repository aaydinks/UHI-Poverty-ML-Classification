"""Algorithm comparison: eight algorithms under leave-one-city-out.

The algorithm comparison shows eight algorithms by precision, recall, F1 and accuracy on the national
sample (54,243 tracts, 486 urbanized areas) under leave-one-city-out validation: panel (a)
is High_UHI on `A-noLST`, panel (b) is High_Pov on `B-env`. The folds are
`validation.splits(..., "loco")` on `validation.frame("primary")`.

- Logistic regression and LightGBM are read from the repository's cached out-of-fold
  engine (`validation.oof`, fixed settings) and checked against section 4.4's
  `metric_panel.csv`. The cache must already hold them; nothing is recomputed into it.
- The other six use library defaults with `random_state = 37` and no tuning. Every learned
  step is inside the pipeline refitted per fold: median imputation, then for KNN and SVM
  `log1p` on the skewed counts and standardisation.
- SVM: an exact RBF SVM refitted 486 times on ~54k rows is impractical, so the kernel is
  approximated with `Nystroem(n_components=500)` + `LinearSVC`. Its score is the decision
  function (used only for ROC-AUC); its class is decision ≥ 0.
- Bars are aggregated (pooled) over all out-of-fold predictions at τ = 0.5; error bars
  are ±1 SD of 1,000 UA cluster-bootstrap replicates. ROC-AUC carries a 95 % UA
  cluster-bootstrap CI.

Out-of-fold predictions of the six other models are cached under
`data/interim/figures/algorithms/` with a fingerprint sidecar. `--fit dt,knn` fills the cache for
those models only and stops (for running the slow ones in the background).

Writes `results/algorithms_bars.csv`, `results/algorithms_auc.csv`, `results/algorithms_repo_check.csv`,
and `results/algorithms.png`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from uhi_poverty import features, metrics, models, paths, plotting, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
CACHE = paths.INTERIM / "figures" / "algorithms"
METRIC_PANEL = (paths.ROOT / "4_Model_Specifications" / "4.4_Metric_Panel_with_Uncertainty"
                / "results" / "metric_panel.csv")

SEED = 37
N_JOBS = 6
SCHEME = "loco"
PANELS = {"A": "A-noLST", "B": "B-env"}
PANEL_TITLES = {"A": "(a) UHI, A-noLST", "B": "(b) Poverty, B-env"}

ALGORITHMS = ("logistic", "dt", "rf", "svm", "knn", "catboost", "xgboost", "lgbm")
ALGORITHM_LABELS = {"logistic": "Logistic Regression", "dt": "Decision Tree",
                    "rf": "Random Forest", "svm": "SVM", "knn": "KNN", "catboost": "CatBoost",
                    "xgboost": "XGBoost", "lgbm": "LGBM"}
REPO_MODELS = ("logistic", "lgbm")  # from the repository's out-of-fold cache
FIT_ORDER = ("dt", "knn", "xgboost", "svm", "rf", "catboost")  # cheapest first

METRICS = ("precision", "recall", "f1", "accuracy")  # bar order
METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1 Score",
                 "accuracy": "Accuracy"}
# An ordered ramp from the sequential blue scale: four hues would exceed the three
# categorical slots that pass the colour-vision checks.
METRIC_COLORS = dict(zip(METRICS, [plotting.SEQ_BLUE[i] for i in (1, 3, 5, 7)], strict=True))
NOTE = ("Leave-one-city-out, 486 urbanized areas, τ = 0.5. Bars: scores aggregated over all "
        "held-out tracts; error bars: ±1 SD of 1,000 UA cluster-bootstrap replicates. "
        "Logistic regression and LGBM use "
        "the repository's fixed settings; the other six use library defaults (random_state = 37). "
        "SVM: RBF kernel approximated by Nystroem (500 components) + LinearSVC.")

# --------------------------------------------------------------------------- models


def _scaled(model) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", models.Log1pColumns(features.SKEWED)),
        ("scale", StandardScaler()),
        ("model", model),
    ]).set_output(transform="pandas")


def _imputed(model) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])


def make_model(name: str) -> Pipeline:
    if name == "dt":
        return _imputed(DecisionTreeClassifier(random_state=SEED))
    if name == "rf":
        return _imputed(RandomForestClassifier(random_state=SEED, n_jobs=1))
    if name == "svm":
        pipe = _scaled(LinearSVC(random_state=SEED))
        pipe.steps.insert(3, ("nystroem", Nystroem(n_components=500, random_state=SEED)))
        return pipe
    if name == "knn":
        return _scaled(KNeighborsClassifier(n_jobs=1))
    if name == "catboost":
        return _imputed(CatBoostClassifier(random_state=SEED, thread_count=1, verbose=0,
                                           allow_writing_files=False))
    if name == "xgboost":
        return _imputed(XGBClassifier(random_state=SEED, n_jobs=1))
    raise KeyError(name)


def _fit_fold(name: str, X: np.ndarray, y: np.ndarray, columns: list[str], train_idx,
              test_idx) -> tuple[np.ndarray, np.ndarray]:
    """(score, class) for the test rows."""
    Xtr = pd.DataFrame(X[train_idx], columns=columns)
    Xte = pd.DataFrame(X[test_idx], columns=columns)
    pipe = make_model(name)
    pipe.fit(Xtr, y[train_idx])
    if name == "svm":
        score = pipe.decision_function(Xte)
        return score, (score >= 0).astype(int)
    score = pipe.predict_proba(Xte)[:, list(pipe.classes_).index(1)]
    return score, (score >= models.THRESHOLD).astype(int)


# --------------------------------------------------------------------------- out-of-fold


def _design(task: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = features.config(PANELS[task])
    y = df[features.TASKS[task]["target"]].astype(int).to_numpy()
    return df[cols].to_numpy(dtype=np.float64), y, cols


def _fingerprint(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, name: str) -> str:
    h = hashlib.sha256()
    h.update("\n".join(df["GEOID"]).encode())
    h.update("\n".join(df["ua"].astype(str)).encode())
    h.update(np.ascontiguousarray(y, dtype=np.int8).tobytes())
    h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    settings = {"model": name, "pipeline": repr(make_model(name)), "scheme": SCHEME,
                "seed": SEED, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                "catboost": catboost.__version__}
    h.update(json.dumps(settings, sort_keys=True).encode())
    return h.hexdigest()


def fitted_oof(task: str, name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-city-out predictions of one of the six other models, cached."""
    X, y, cols = _design(task, df)
    key = f"{task}__{PANELS[task]}__{name}__{SCHEME}"
    out_path = CACHE / f"{key}.parquet"
    meta_path = out_path.with_suffix(".json")
    fp = _fingerprint(df, X, y, name)
    if out_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()).get("fingerprint") == fp:
            return pd.read_parquet(out_path)
        print(f"{key}: cache fingerprint changed, recomputing")

    fold_list = validation.splits(df, y, SCHEME)
    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_fold)(name, X, y, cols, tr, te) for _, tr, te in fold_list)
    score = np.full(len(df), np.nan)
    pred = np.full(len(df), -1)
    fold = np.empty(len(df), dtype=object)
    for (label, _, te), (s, p) in zip(fold_list, results, strict=True):
        score[te], pred[te], fold[te] = s, p, str(label)
    if np.isnan(score).any() or (pred < 0).any():
        raise AssertionError(f"{key}: a tract has no out-of-fold prediction")
    out = pd.DataFrame({"GEOID": df["GEOID"], "ua": df["ua"], "fold": fold, "y": y,
                        "score": score, "pred": pred})
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    meta = {"key": key, "fingerprint": fp, "n_rows": len(out), "n_folds": len(fold_list),
            "features": cols, "seconds": round(time.time() - t0, 1)}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{key}: {len(fold_list)} folds, {meta['seconds']:.0f} s")
    return out


def repo_oof(task: str, name: str, df: pd.DataFrame) -> pd.DataFrame:
    """The repository's cached predictions; refuses to recompute into the shared cache."""
    X, y, _ = _design(task, df)
    key = validation.run_key(task, PANELS[task], name, SCHEME)
    meta_path = paths.OOF_DIR / f"{key}.json"
    fp = validation._fingerprint(df, X, y, name, SCHEME)
    if not meta_path.exists() or json.loads(meta_path.read_text()).get("fingerprint") != fp:
        raise RuntimeError(f"{key}: not in the repository's OOF cache (run `make engine`)")
    return validation.oof(task, PANELS[task], name, SCHEME, df=df, n_jobs=N_JOBS)


def predictions(task: str, name: str, df: pd.DataFrame) -> pd.DataFrame:
    return repo_oof(task, name, df) if name in REPO_MODELS else fitted_oof(task, name, df)


# --------------------------------------------------------------------------- tables


def score_all(df: pd.DataFrame) -> pd.DataFrame:
    """Long table: task, model, metric, estimate and UA cluster-bootstrap CI."""
    frames = []
    for task in PANELS:
        for name in ALGORITHMS:
            o = predictions(task, name, df)
            s = metrics.Scored(o["y"], o["score"], o["pred"])
            t = metrics.bootstrap_panel(s, o["ua"].to_numpy(), metrics=[*METRICS, "roc_auc"],
                                        sd=True)
            t.insert(0, "model", name)
            t.insert(0, "config", PANELS[task])
            t.insert(0, "task", task)
            t["n"] = len(o)
            t["n_folds"] = o["fold"].nunique()
            frames.append(t)
    return pd.concat(frames, ignore_index=True)


def repo_check(t: pd.DataFrame) -> pd.DataFrame:
    """LR and LightGBM against the committed 4.4 metric panel (must be identical)."""
    ref = pd.read_csv(METRIC_PANEL)
    ref = ref[ref["scheme"] == SCHEME]
    rows = []
    for _, r in t[t["model"].isin(REPO_MODELS)].iterrows():
        m = ref[(ref.task == r.task) & (ref.config == r.config) & (ref.model == r.model)
                & (ref.metric == r.metric)].iloc[0]
        diff = max(abs(r[c] - m[c]) for c in ("estimate", "ci_low", "ci_high"))
        rows.append({"task": r.task, "model": r.model, "metric": r.metric,
                     "estimate": r.estimate, "repo_estimate": m.estimate,
                     "max_abs_diff": diff})
    check = pd.DataFrame(rows)
    if check["max_abs_diff"].max() > 1e-12:
        raise AssertionError("LR/LightGBM differ from 4.4's metric panel")
    return check


def bars_table(t: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, model): the four bar heights and their bootstrap SDs."""
    w = t[t["metric"].isin(METRICS)].pivot_table(index=["task", "config", "model"],
                                                  columns="metric", values=["estimate", "boot_sd"])
    w.columns = [m if v == "estimate" else f"{m}_sd" for v, m in w.columns]
    w = w.reset_index()[["task", "config", "model", *METRICS, *(f"{m}_sd" for m in METRICS)]]
    w["order"] = w["model"].map(ALGORITHMS.index)
    w = w.sort_values(["task", "order"]).drop(columns="order").reset_index(drop=True)
    w.insert(3, "algorithm", w["model"].map(ALGORITHM_LABELS))
    return w


# --------------------------------------------------------------------------- figure


def legend_handles() -> list[Patch]:
    return [Patch(facecolor=METRIC_COLORS[m], label=METRIC_LABELS[m]) for m in METRICS]


def draw_bars(ax, bars: pd.DataFrame, title: str, ylabel: bool = True) -> None:
    """Grouped bars: one group per algorithm, four metrics in order."""
    bars = bars.set_index("model").loc[list(ALGORITHMS)]
    x = np.arange(len(ALGORITHMS))
    width = 0.2
    for j, m in enumerate(METRICS):
        ax.bar(x + (j - 1.5) * width, bars[m], width, color=METRIC_COLORS[m],
               edgecolor=plotting.SURFACE, linewidth=0.4, label=METRIC_LABELS[m],
               yerr=bars[f"{m}_sd"], capsize=1.6,
               error_kw={"ecolor": plotting.INK, "elinewidth": 0.7, "capthick": 0.7})
    ax.axhline(0.5, color=plotting.INK_2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=3)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticks(x, [ALGORITHM_LABELS[a] for a in ALGORITHMS], rotation=45, ha="right",
                  rotation_mode="anchor")
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Machine learning models")
    if ylabel:
        ax.set_ylabel("Score (aggregated)")
    ax.set_title(title)


def plot(bars: pd.DataFrame) -> None:
    plotting.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for k, (ax, task) in enumerate(zip(axes, PANELS, strict=True)):
        draw_bars(ax, bars[bars["task"] == task], PANEL_TITLES[task], ylabel=k == 0)
    fig.tight_layout()
    fig.legend(handles=legend_handles(), loc="lower left", bbox_to_anchor=(0.01, 0.98),
               ncol=4)
    plotting.note(fig, NOTE + " Dashed line: 0.5.", y=-0.01)
    plotting.savefig(fig, OUT / "algorithms.png")




def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit", default=None,
                        help="comma-separated models: fill their cache for both tasks and stop")
    args = parser.parse_args(argv)
    df = validation.frame("primary")
    if args.fit:
        for name in args.fit.split(","):
            for task in PANELS:
                fitted_oof(task, name, df)
        return
    t = score_all(df)
    t.to_csv(OUT / "algorithms_long.csv", index=False)
    repo_check(t).to_csv(OUT / "algorithms_repo_check.csv", index=False)
    bars = bars_table(t)
    bars.to_csv(OUT / "algorithms_bars.csv", index=False)
    auc = t[t["metric"] == "roc_auc"].drop(columns="metric").reset_index(drop=True)
    auc.to_csv(OUT / "algorithms_auc.csv", index=False)
    plot(bars)
    print(bars.to_string(index=False))


if __name__ == "__main__":
    main()
