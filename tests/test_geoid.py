import pandas as pd
import pytest

from uhi_poverty import geo


def test_zero_padding_from_int_float_and_str():
    out = geo.normalize_geoid(pd.Series([1001020100, 6037101110]))
    assert out.tolist() == ["01001020100", "06037101110"]
    assert geo.normalize_geoid(pd.Series([1001020100.0])).tolist() == ["01001020100"]
    assert geo.normalize_geoid(pd.Series([" 1001020100 "])).tolist() == ["01001020100"]
    assert geo.normalize_geoid(pd.Series([280]), width=5).tolist() == ["00280"]


def test_rejects_bad_geoids():
    with pytest.raises(ValueError):
        geo.normalize_geoid(pd.Series(["12A"]))
    with pytest.raises(ValueError):
        geo.normalize_geoid(pd.Series(["123456789012"]))


def test_reno_and_yuma_substring_trap():
    """`"CA" in Urban_name` pulls in Nevada and Arizona tracts; the FIPS prefix does not."""
    df = pd.DataFrame({
        "Urban_name": ["Reno, NV--CA", "Yuma, AZ--CA", "Fresno, CA", "Chicago, IL--IN"],
        "GEOID": geo.normalize_geoid(pd.Series([32031000100, 4027000100, 6019000100,
                                                17031010100])),
    })
    by_substring = df[df["Urban_name"].str.contains("CA")]
    by_fips = df[geo.state_of(df["GEOID"]) == "06"]
    assert len(by_substring) == 3
    assert set(geo.lookup(geo.state_of(by_substring["GEOID"]), "abbr")) == {"NV", "AZ", "CA"}
    assert by_fips["Urban_name"].tolist() == ["Fresno, CA"]
