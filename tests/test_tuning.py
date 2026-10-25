import numpy as np
import pandas as pd
import pytest

from uhi_poverty import features, models, tuning, validation


def test_inner_splits_stay_inside_outer_training_rows(synthetic_frame):
    df, y = synthetic_frame, synthetic_frame["y"].to_numpy()
    for scheme in ("ua5", "random5"):
        for _, train, test in validation.splits(df, y, scheme):
            inner = tuning.inner_splits(df, y, train, scheme)
            covered = np.concatenate([va for _, va in inner])
            assert np.array_equal(np.sort(covered), np.sort(train))
            for itr, iva in inner:
                assert not np.intersect1d(itr, test).size
                assert not np.intersect1d(iva, test).size
                assert not np.intersect1d(itr, iva).size


def test_inner_ua_splits_keep_cities_together(synthetic_frame):
    df, y = synthetic_frame, synthetic_frame["y"].to_numpy()
    ua = df["ua"].to_numpy()
    for _, train, _ in validation.splits(df, y, "ua5"):
        for itr, iva in tuning.inner_splits(df, y, train, "ua5"):
            assert not set(ua[itr]) & set(ua[iva])


def test_each_city_uses_the_settings_of_the_fold_that_held_it_out(synthetic_frame):
    df, y = synthetic_frame, synthetic_frame["y"].to_numpy()
    outer = validation.splits(df, y, "ua5")
    held_out_by = {u: k for k, (_, _, te) in enumerate(outer) for u in df["ua"].iloc[te]}
    folds = tuning.evaluation_folds(df, y, "loco")
    assert len(folds) == df["ua"].nunique()
    for label, _, test, k in folds:
        assert set(df["ua"].iloc[test]) == {label}
        assert held_out_by[label] == k
        # the settings of fold k were tuned on its training rows, which exclude this city
        assert label not in set(df["ua"].iloc[outer[k][1]])


def test_tau_is_chosen_on_the_scores_it_is_given():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.35, 0.8, 0.9])
    assert tuning.choose_tau(y, score, "accuracy") == pytest.approx(0.31)
    # predicting everything positive gives F1 0.667; the best split gives 1.0
    assert tuning.choose_tau(y, score, "f1") == pytest.approx(0.31)
    assert tuning.choose_tau(y, np.zeros(6) + 0.5, "f1") == pytest.approx(0.01)


def test_tuning_never_sees_outer_test_rows(synthetic_frame, monkeypatch):
    df = synthetic_frame.copy()
    df["High_Pov"] = df["y"]
    seen: list[np.ndarray] = []

    def fake_design(frame, task):
        X = frame[["x1", "x2"]].to_numpy(dtype=float)
        return X, frame["y"].to_numpy(), ["x1", "x2"]

    def fake_fit(task, model, prep, params, X, y, cols, train_idx, test_idx):
        seen.append(np.concatenate([train_idx, test_idx]))
        return X[test_idx, 0]

    monkeypatch.setattr(tuning, "_design", fake_design)
    monkeypatch.setattr(tuning, "_fit_score", fake_fit)
    settings = tuning.tune_folds("B", "logistic", df, "ua5", n_trials=2, n_jobs=1)
    outer = validation.splits(df, df["y"].to_numpy(), "ua5")
    assert len(seen) == len(outer) * 2 * tuning.INNER_FOLDS
    for s, (_, train, test) in zip(settings, outer, strict=True):
        assert set(s["inner_rows"]) == set(train)
        assert not set(s["inner_rows"]) & set(test)
        assert 0 < s["tau_acc"] < 1 and 0 < s["tau_f1"] < 1
    all_test = [set(te) for _, _, te in outer]
    # every fit call drew its rows from a single outer fold's training rows
    for rows in seen:
        assert any(not set(rows) & te for te in all_test)


def test_settings_hash_changes_with_params_and_tau():
    s = [{"prep": {"pca": "none"}, "params": {"C": 1.0}, "tau_acc": 0.5, "tau_f1": 0.4}]
    h = tuning.settings_hash(s)
    assert tuning.settings_hash([{**s[0], "params": {"C": 2.0}}]) != h
    assert tuning.settings_hash([{**s[0], "tau_acc": 0.51}]) != h
    assert tuning.settings_hash([{**s[0], "inner_score": [0.1]}]) == h


def test_ensemble_weights_sum_to_one_and_equal_is_equal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    rows = np.arange(200)
    members = {m: [{"inner_rows": rows.tolist(),
                    "inner_score": (y * 0.3 + rng.random(200) * (i + 1)).tolist()}]
               for i, m in enumerate(models.BOOSTERS)}
    ens = tuning.ensemble_settings(members, y)
    for name in models.ENSEMBLES:
        w = ens[name][0]["weights"]
        assert sum(w.values()) == pytest.approx(1.0)
    assert list(ens["ens_equal"][0]["weights"].values()) == pytest.approx([1 / 3] * 3)
    # the least noisy member should carry the most weight
    w = ens["ens_weighted"][0]["weights"]
    assert w["lgbm"] == max(w.values())


def test_tuned_pipeline_runs_with_every_option():
    rng = np.random.default_rng(1)
    cols = features.config("A-noLST")
    X = pd.DataFrame(rng.random((120, len(cols))), columns=cols)
    y = (X["DelNDVI_annual"] + 0.2 * rng.random(120) > 0.6).astype(int).to_numpy()
    prep = {"engineer": True, "variance": False, "corr": True, "scaler": "minmax",
            "pca": "0.95", "smote": True}
    for name in models.TUNED_MODELS:
        params = {"n_estimators": 20} if name in ("lgbm", "xgb") else (
            {"iterations": 20} if name == "catboost" else {})
        p = models.fit_predict_tuned("A", name, prep, params, X.iloc[:90], y[:90], X.iloc[90:])
        assert p.shape == (30,) and np.all((p >= 0) & (p <= 1))


def test_correlation_filter_drops_the_later_of_a_correlated_pair():
    rng = np.random.default_rng(2)
    a = rng.normal(size=500)
    X = np.column_stack([a, a + 0.01 * rng.normal(size=500), rng.normal(size=500)])
    f = models.CorrelationFilter(0.9).fit(X)
    assert f.keep_.tolist() == [0, 2]


def test_round_two_fixes_the_options_inner_cv_never_favoured():
    import optuna

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for model in models.TUNED_MODELS:
        for _ in range(5):
            prep, _ = tuning.suggest(study.ask(), model, rnd=2)
            assert prep["variance"] is False and prep["pca"] == "none"
            assert (prep["scaler"] != "none") == (model == "logistic")


def test_round_one_paths_and_fingerprints_are_unchanged_by_round_two():
    assert tuning._settings_path("A", "lgbm", "ua5").name == "A__lgbm__tune-ua5.json"
    assert tuning._pred_path("A", "lgbm", "loco", 2).name == "A__lgbm__r2__loco.parquet"


def test_stacked_ensemble_is_a_logistic_regression_on_member_logits():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    rows = np.arange(400)
    members = {m: [{"inner_rows": rows.tolist(),
                    "inner_score": np.clip(y * 0.4 + rng.random(400) * (i + 1) * 0.5, 0.01,
                                           0.99).tolist()}]
               for i, m in enumerate(models.BOOSTERS)}
    ens = tuning.ensemble_settings(members, y, stacked=True)
    s = ens["ens_stacked"][0]
    assert set(s["weights"]) == set(models.BOOSTERS) and "intercept" in s
    preds = {m: pd.DataFrame({"GEOID": [str(i) for i in rows], "ua": "u", "state_fips": "01",
                              "region": "South", "fold": "0", "settings_fold": 0, "y": y,
                              "score": members[m][0]["inner_score"]})
             for m in models.BOOSTERS}
    out = tuning.combine(preds, ens["ens_stacked"])
    assert out["score"].between(0, 1).all()
    # the least noisy member gets the largest coefficient
    assert max(s["weights"], key=s["weights"].get) == "lgbm"


def test_seed_bagging_averages_one_fit_per_seed():
    rng = np.random.default_rng(4)
    cols = features.config("A-noLST")
    X = pd.DataFrame(rng.random((150, len(cols))), columns=cols)
    y = (X["DelNDVI_annual"] > 0.5).astype(int).to_numpy()
    prep = {"engineer": False, "variance": False, "corr": False, "scaler": "none",
            "pca": "none", "smote": False}
    args = ("A", "lgbm", prep, {"n_estimators": 20, "subsample": 0.7},
            X.iloc[:100], y[:100], X.iloc[100:])
    one = [models.fit_predict_tuned(*args, seeds=(s,)) for s in (1, 2)]
    both = models.fit_predict_tuned(*args, seeds=(1, 2))
    assert np.allclose(both, (one[0] + one[1]) / 2)
