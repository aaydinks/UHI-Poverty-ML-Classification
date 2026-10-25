import numpy as np
import pandas as pd
import pytest

from tests.conftest import needs_build
from uhi_poverty import labels


def _frame():
    rng = np.random.default_rng(1)
    n = 400
    rate = rng.uniform(0, 0.6, n)
    return pd.DataFrame({
        "UHI_annual_day": rng.normal(1.5, 2, n),
        "pov_rate": rate,
        # HUD's index falls as poverty rises
        "LPI": np.clip(99 - rate * 150 + rng.normal(0, 5, n), 0, 99).round(),
    })


def test_median_split_and_orientation():
    df = _frame()
    everyone = pd.Series(True, index=df.index)
    cuts = labels.medians(df, everyone, everyone)
    out = labels.add_labels(df, cuts)
    assert cuts["UHI_annual_day"] == pytest.approx(df["UHI_annual_day"].median())
    assert out["High_UHI"].mean() == pytest.approx(0.5, abs=0.01)
    assert out["High_Pov"].mean() == pytest.approx(0.5, abs=0.01)
    # positives are the poorer tracts for both poverty labels
    assert (out.loc[out["High_Pov"] == 1, "pov_rate"].mean()
            > out.loc[out["High_Pov"] == 0, "pov_rate"].mean())
    assert (out.loc[out["High_Pov_LPI"] == 1, "pov_rate"].mean()
            > out.loc[out["High_Pov_LPI"] == 0, "pov_rate"].mean())
    labels.assert_orientation(out)


def test_lpi_label_is_low_index():
    lpi = pd.Series([10, 53, 54, 90, np.nan])
    assert labels.high_pov_lpi(lpi, 54).tolist()[:4] == [1, 1, 0, 0]
    assert pd.isna(labels.high_pov_lpi(lpi, 54).iloc[4])


def test_flipped_label_is_caught():
    df = labels.add_labels(_frame(), {"UHI_annual_day": 1.5, "pov_rate": 0.3, "LPI": 54})
    df["High_Pov"] = 1 - df["High_Pov"]
    with pytest.raises(AssertionError):
        labels.assert_orientation(df)


@needs_build
def test_built_labels_orientation():
    from uhi_poverty import validation

    df = validation.frame("primary")
    labels.assert_orientation(df)
    assert df["High_UHI"].mean() == pytest.approx(0.5, abs=0.001)
    assert df["High_Pov"].mean() == pytest.approx(0.5, abs=0.001)
