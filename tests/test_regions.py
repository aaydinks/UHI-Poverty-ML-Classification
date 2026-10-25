import pandas as pd

from uhi_poverty import geo


def test_all_51_jurisdictions_mapped():
    table = geo.state_table()
    assert len(table) == 51
    assert table["state_fips"].is_unique and table["state_abbr"].is_unique
    assert set(table["region"]) == set(geo.REGIONS)
    assert table["region"].value_counts().to_dict() == {
        "South": 17, "West": 13, "Midwest": 12, "Northeast": 9}
    assert table.loc[table["state_abbr"] == "DC", "region"].item() == "South"


def test_puerto_rico_flagged_as_territory():
    assert geo.PUERTO_RICO not in geo.STATES
    assert geo.is_territory(["72", "06"]).tolist() == [True, False]
    assert geo.lookup(pd.Series(["72"]), "region").item() == "Territory"


def test_ua_primary_state_and_span():
    ua = pd.Series(["A", "A", "A", "B", "B"])
    st = pd.Series(["32", "06", "32", "06", "32"])
    assert geo.ua_primary_state(ua, st).to_dict() == {"A": "32", "B": "06"}  # tie -> lower FIPS
    assert geo.ua_states_spanned(ua, st).to_dict() == {"A": "CA,NV", "B": "CA,NV"}
