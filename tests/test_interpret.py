"""Attribution groups respect the exact identities; contributions and ALE behave."""

import numpy as np
import pandas as pd
import pytest

from uhi_poverty import features, interpret, models, paths

IDENTITY_CHECKS = (paths.ROOT / "3_UHI_Model_Predictors" / "3.1_UHI_Formula" / "results"
                   / "identity_checks.csv")


def test_identities_are_the_ones_verified_in_4_1():
    t = pd.read_csv(IDENTITY_CHECKS)
    verified = {(r.target, *(s.strip() for s in r.identity.split("−")))
                for r in t[t.holds].itertuples()}
    assert set(features.IDENTITIES) == verified


def test_no_identity_straddles_two_groups():
    group_of = {c: g for g, cols in features.ATTRIBUTION_GROUPS.items() for c in cols}
    for diff, urban, rural in features.IDENTITIES:
        assert group_of[diff] == group_of[urban] == group_of[rural], diff


def test_groups_partition_every_feature_once():
    grouped = [c for cols in features.ATTRIBUTION_GROUPS.values() for c in cols]
    every = [c for cols in features.FAMILIES.values() for c in cols]
    assert sorted(grouped) == sorted(every)


@pytest.mark.parametrize("task", interpret.TASKS)
def test_explained_configurations_are_fully_grouped(task):
    cfg = interpret.config(task)
    grouped = [c for cols in features.attribution_groups(cfg).values() for c in cols]
    assert sorted(grouped) == sorted(features.CONFIGS[cfg])


def test_contributions_sum_to_the_predicted_logit():
    rng = np.random.default_rng(37)
    X = pd.DataFrame(rng.normal(size=(400, 5)), columns=list("abcde"))
    y = (X.a - X.b + rng.normal(size=400) > 0).astype(int).to_numpy()
    pipe = models.make_model("lgbm").fit(X, y)
    c = interpret.contributions(pipe.named_steps["model"], X)
    assert c.shape == (400, 6)
    np.testing.assert_allclose(interpret.sigmoid(c.sum(axis=1)), pipe.predict_proba(X)[:, 1],
                               atol=1e-12)


def test_linear_contributions_sum_to_the_predicted_logit():
    rng = np.random.default_rng(37)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=["Area", "b", "c", "d"])
    X["Area"] = np.exp(X["Area"])  # a log1p column
    X.loc[::17, "b"] = np.nan  # an imputed value
    y = (X.c - X.d + rng.normal(size=400) > 0).astype(int).to_numpy()
    pipe = models.make_model("logistic").fit(X, y)
    c = interpret.linear_contributions(pipe, X)
    assert c.shape == (400, 5)
    np.testing.assert_allclose(c.sum(axis=1), pipe.decision_function(X), atol=1e-10)
    # centred on the training sample: each feature's mean contribution is zero
    np.testing.assert_allclose(c[:, :-1].mean(axis=0), 0, atol=1e-10)


def test_group_sums():
    phi = pd.DataFrame({"a": [1.0, -2.0], "b": [0.5, 0.5], "c": [3.0, 0.0]})
    gs = interpret.group_sums(phi, {"g1": ["a", "b"], "g2": ["c"]})
    assert gs["g1"].tolist() == [1.5, -1.5]
    assert gs["g2"].tolist() == [3.0, 0.0]


def test_ale_accumulates_and_centres():
    # a constant local effect of 1 per bin, equal counts: ALE 1, 2, 3, 4 centred on 2
    ale = pd.DataFrame({"seed": 37, "fold": "0", "feature": "x", "bin": [1, 2, 3, 4],
                        "lo": [0, 1, 2, 3], "hi": [1, 2, 3, 4], "n": 10, "sum_diff": 10.0})
    b = interpret.pooled_ale(ale, "x")
    np.testing.assert_allclose(b["ale"], [-1.0, 0.0, 1.0, 2.0])
