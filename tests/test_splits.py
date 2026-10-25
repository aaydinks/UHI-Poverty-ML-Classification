import numpy as np
import pytest

from tests.conftest import needs_build
from uhi_poverty import validation

PARTITIONS = ("random5", "county5", "ua5", "loco", "state", "region")
UA_DISJOINT = ("ua5", "loco", "state", "region", "major_ua")


def _check(df, scheme):
    y = df["y"].to_numpy()
    folds = validation.splits(df, y, scheme)
    tested = np.zeros(len(df), dtype=int)
    ua = df["ua"].to_numpy()
    for _, train, test in folds:
        assert len(np.intersect1d(train, test)) == 0
        tested[test] += 1
        if scheme in UA_DISJOINT:
            assert not set(ua[train]) & set(ua[test]), f"{scheme}: UA in train and test"
    return folds, tested


@pytest.mark.parametrize("scheme", PARTITIONS)
def test_every_tract_tested_exactly_once(synthetic_frame, scheme):
    _, tested = _check(synthetic_frame, scheme)
    assert (tested == 1).all()


@pytest.mark.parametrize("scheme", UA_DISJOINT)
def test_no_ua_in_both_train_and_test(synthetic_frame, scheme):
    _check(synthetic_frame, scheme)


def test_major_ua_tests_each_large_ua_once(synthetic_frame):
    folds, tested = _check(synthetic_frame, "major_ua")
    assert tested.max() == 1
    held = [label for label, _, _ in folds]
    assert len(held) == min(validation.N_MAJOR_UA, synthetic_frame["ua"].nunique())


def test_state_holdout_is_ua_buffered(synthetic_frame):
    folds = validation.splits(synthetic_frame, synthetic_frame["y"].to_numpy(), "state")
    by_state = {label: (train, test) for label, train, test in folds}
    train, test = by_state["32"]  # Nevada: its only UA also spans California
    assert set(synthetic_frame["state_fips"].iloc[test]) == {"32"}
    # the cross-state UA is out of training entirely, including its California tracts
    assert "00003" not in set(synthetic_frame["ua"].iloc[train])


def test_county_folds_keep_counties_together(synthetic_frame):
    folds = validation.splits(synthetic_frame, synthetic_frame["y"].to_numpy(), "county5")
    county = synthetic_frame["county_fips"].to_numpy()
    for _, train, test in folds:
        assert not set(county[train]) & set(county[test])


@needs_build
@pytest.mark.parametrize("scheme", PARTITIONS + ("major_ua",))
def test_real_sample_splits(scheme):
    df = validation.frame("primary")
    df = df.assign(y=df["High_UHI"].astype(int))
    _, tested = _check(df, scheme)
    if scheme in PARTITIONS:
        assert (tested == 1).all()
