"""Model factories: the fixed settings, and the tuned pipelines of section 4.6."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier

from . import features
from .features import SKEWED

SEED = 37
THRESHOLD = 0.5

LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": 1,
    "verbose": -1,
}
LOGISTIC_PARAMS = {"C": 1.0, "max_iter": 5000}

MODELS = ("majority", "logistic", "lgbm")
MODEL_LABELS = {"majority": "Majority class", "logistic": "Logistic regression",
                "lgbm": "LightGBM"}


class Log1pColumns(BaseEstimator, TransformerMixin):
    """`log1p` on the named columns of a DataFrame; other columns pass through."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns or []:
            if c in X.columns:
                X[c] = np.log1p(X[c].clip(lower=0))
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features,
                                                                                dtype=object)


def make_model(name: str) -> Pipeline:
    if name == "majority":
        return Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
    if name == "logistic":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", Log1pColumns(SKEWED)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**LOGISTIC_PARAMS)),
        ])
        return pipe.set_output(transform="pandas")
    if name == "lgbm":
        # LightGBM handles missing values natively; no imputer is needed.
        return Pipeline([("model", LGBMClassifier(**LGBM_PARAMS))])
    raise KeyError(name)


def fit_predict(name: str, X_train: pd.DataFrame, y_train: np.ndarray,
                X_test: pd.DataFrame) -> np.ndarray:
    """Positive-class probability for the test rows."""
    return _positive_proba(make_model(name).fit(X_train, y_train), X_test)


def _positive_proba(model, X_test) -> np.ndarray:
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X_test))
    return proba[:, classes.index(1)]


# --------------------------------------------------------------------------- tuned (section 4.6)
#
# The tuned models assemble a full preprocessing pipeline from its parts. Every
# optional step is a search option chosen by inner cross-validation, and every
# fitted step (imputer, filters, scaler, PCA, SMOTE) is fitted inside the
# training fold. The fixed models above are untouched.

TUNED_MODELS = ("logistic", "lgbm", "xgb", "catboost")
BOOSTERS = ("lgbm", "xgb", "catboost")
ENSEMBLES = ("ens_equal", "ens_weighted")
TUNED_LABELS = {"logistic": "Logistic regression, tuned", "lgbm": "LightGBM, tuned",
                "xgb": "XGBoost, tuned", "catboost": "CatBoost, tuned",
                "ens_equal": "Soft-voting ensemble, equal weights",
                "ens_weighted": "Soft-voting ensemble, tuned weights"}
VARIANCE_THRESHOLD = 0.1  # low-variance filter threshold
CORRELATION_THRESHOLD = 0.9  # multicollinearity filter threshold (|r|)


class Engineer(BaseEstimator, TransformerMixin):
    """Appends the engineered terms for one task (`features.engineer`)."""

    def __init__(self, task: str = "B"):
        self.task = task

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return features.engineer(X, self.task)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Keeps a column only if its |r| with every column already kept is at most `threshold`.

    Columns are visited in order, so of each correlated pair the earlier one stays.
    """

    def __init__(self, threshold: float = CORRELATION_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(np.corrcoef(X, rowvar=False))
        r = np.nan_to_num(np.atleast_2d(r), nan=0.0)
        keep: list[int] = []
        for j in range(X.shape[1]):
            if all(r[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.keep_ = np.asarray(keep, dtype=int)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def make_estimator(name: str, params: dict, seed: int = SEED):
    if name == "logistic":
        return LogisticRegression(max_iter=5000, **params)
    if name == "lgbm":
        return LGBMClassifier(subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1,
                              **params)
    if name == "xgb":
        return XGBClassifier(tree_method="hist", random_state=seed, n_jobs=1,
                             eval_metric="logloss", **params)
    if name == "catboost":
        return CatBoostClassifier(random_seed=seed, thread_count=1, verbose=0,
                                  allow_writing_files=False, **params)
    raise KeyError(name)


def make_tuned_pipeline(task: str, name: str, prep: dict, params: dict,
                        seed: int = SEED) -> ImbPipeline:
    """engineer → log1p (logistic) → impute → variance → correlation → scale → PCA → SMOTE → model."""
    scalers = {"none": "passthrough", "minmax": MinMaxScaler(), "standard": StandardScaler()}
    pca = prep["pca"]
    steps = [
        ("engineer", Engineer(task) if prep["engineer"] else "passthrough"),
        ("log1p", Log1pColumns(SKEWED) if name == "logistic" else "passthrough"),
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(VARIANCE_THRESHOLD) if prep["variance"]
         else "passthrough"),
        ("correlation", CorrelationFilter() if prep["corr"] else "passthrough"),
        ("scale", scalers[prep["scaler"]]),
        ("pca", "passthrough" if pca == "none"
         else PCA(n_components=float(pca), svd_solver="full", random_state=seed)),
        ("smote", SMOTE(random_state=seed) if prep["smote"] else "passthrough"),
        ("model", make_estimator(name, params, seed)),
    ]
    return ImbPipeline(steps)


class SoftVote:
    """Weighted mean of fitted members' positive-class probabilities."""

    classes_ = np.array([0, 1])

    def __init__(self, members: dict, weights: dict):
        self.members = members
        self.weights = weights

    def predict_proba(self, X) -> np.ndarray:
        p = sum(self.weights[name] * _positive_proba(m, X) for name, m in self.members.items())
        return np.column_stack([1 - p, p])


class FinalModel:
    """A fitted model packaged with its inputs, its target and its decision thresholds.

    `predict` uses τ_acc (the threshold that maximised accuracy in cross-validation on
    the training cities); `predict(X, threshold="f1")` uses τ_F1.
    """

    def __init__(self, task: str, name: str, estimator, columns: list[str], tau_acc: float,
                 tau_f1: float, card: dict):
        self.task, self.name, self.estimator = task, name, estimator
        self.columns, self.tau_acc, self.tau_f1, self.card = columns, tau_acc, tau_f1, card

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"missing input columns: {missing}")
        return _positive_proba(self.estimator, X[self.columns])

    def predict(self, X: pd.DataFrame, threshold: str = "accuracy") -> np.ndarray:
        tau = self.tau_acc if threshold == "accuracy" else self.tau_f1
        return (self.predict_proba(X) >= tau).astype(int)


def fit_predict_tuned(task: str, name: str, prep: dict, params: dict, X_train: pd.DataFrame,
                      y_train: np.ndarray, X_test: pd.DataFrame,
                      seeds: tuple[int, ...] = (SEED,)) -> np.ndarray:
    """Positive-class probability for the test rows, averaged over one pipeline per seed."""
    p = [_positive_proba(make_tuned_pipeline(task, name, prep, params, s).fit(X_train, y_train),
                         X_test) for s in seeds]
    return np.mean(p, axis=0)
