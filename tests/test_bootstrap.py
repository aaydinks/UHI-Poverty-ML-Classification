import numpy as np

from uhi_poverty import metrics


def test_weights_resample_uas_not_rows():
    clusters = np.repeat(["a", "b", "c", "d"], [5, 3, 7, 1])
    for w in metrics.cluster_weights(clusters, B=50, seed=37):
        # every tract of a UA gets the same weight: the UA's draw count
        for ua in np.unique(clusters):
            assert len(set(w[clusters == ua])) == 1
        # UA draw counts sum to the number of UAs
        per_ua = [w[clusters == ua][0] for ua in np.unique(clusters)]
        assert sum(per_ua) == 4


def test_draws_are_reproducible_and_shared():
    clusters = np.repeat(np.arange(30), 4)
    a = np.array(list(metrics.cluster_weights(clusters, B=20, seed=37)))
    b = np.array(list(metrics.cluster_weights(clusters[::-1], B=20, seed=37)))
    # same UA set -> same draws per UA regardless of row order
    assert np.array_equal(a[:, 0], b[:, -1])


def test_paired_delta_of_identical_models_is_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    score = rng.uniform(size=200)
    s = metrics.Scored(y, score, (score >= 0.5).astype(int))
    clusters = np.repeat(np.arange(40), 5)
    d = metrics.paired_delta(s, s, clusters, B=100)
    assert np.allclose(d["delta"], 0) and np.allclose(d[["ci_low", "ci_high"]], 0)


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    score = np.clip(0.3 * y + rng.uniform(size=600) * 0.7, 0, 1)
    s = metrics.Scored(y, score, (score >= 0.5).astype(int))
    clusters = np.repeat(np.arange(60), 10)
    t = metrics.bootstrap_panel(s, clusters, B=200, metrics=["roc_auc", "f1"])
    assert ((t["ci_low"] <= t["estimate"]) & (t["estimate"] <= t["ci_high"])).all()
