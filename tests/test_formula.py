import numpy as np
import pandas as pd

from tests.conftest import needs_data
from uhi_poverty import paths


@needs_data
def test_uhi_identity_on_real_data():
    df = pd.read_csv(paths.SUHI_CSV, usecols=["UHI_annual_day", "LST_urb_day_CT", "LST_rur_day"])
    df = df.dropna()
    err = (df["UHI_annual_day"] - (df["LST_urb_day_CT"] - df["LST_rur_day"])).abs()
    assert len(df) > 50_000
    assert err.max() < 1e-6
    assert np.corrcoef(df["UHI_annual_day"], df["LST_urb_day_CT"] - df["LST_rur_day"])[0, 1] > 0.999999
