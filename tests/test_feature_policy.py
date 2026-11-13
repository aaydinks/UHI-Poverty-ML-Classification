from uhi_poverty import features


def test_family_sizes():
    assert len(features.LST) == 24
    assert len(features.SUHI) == 12
    assert len(features.NDVI) == 12
    assert len(features.DELTA_NDVI) == 6
    assert len(features.DEM) == 5
    assert len(features.MORPHOLOGY) == 3
    assert len(features.SUHI_FEATURES) == 62
    assert len(set(features.SUHI_FEATURES)) == 62


def test_a_nolst_has_no_lst_or_uhi_columns():
    cols = features.config("A-noLST")
    assert len(cols) == 28
    assert not [c for c in cols if c.startswith(("LST_", "UHI_"))]


def test_target_never_in_task_a_configs():
    for name, cols in features.CONFIGS.items():
        if name.startswith("A-"):
            assert features.TARGET_UHI not in cols, name


def test_no_identifiers_in_any_config():
    for name, cols in features.CONFIGS.items():
        assert not set(cols) & set(features.IDENTIFIERS), name


def test_task_a_engineered_terms_use_no_temperature():
    sources = features.engineered_sources("A")
    assert not [c for c in sources if c.startswith(("LST_", "UHI_"))]
    assert sources <= set(features.config("A-noLST"))
    names = [name for name, *_ in features.ENGINEERED_A]
    assert not [n for n in names if n.startswith(("LST_", "UHI_", "temp_"))]


def test_engineered_terms_use_only_the_task_configuration():
    assert features.engineered_sources("B") <= set(features.config("B-env"))
    assert not features.engineered_sources("B") & set(features.IDENTIFIERS)
    assert not features.engineered_sources("A") & set(features.IDENTIFIERS)


def test_config_sizes():
    sizes = {k: len(v) for k, v in features.CONFIGS.items()}
    assert sizes == {"A-full": 63, "A-narrow": 61, "A-noLST-keepUHI": 39, "A-noLST": 28,
                     "A-noLST+pov": 29, "A-lst-only": 1, "A-lst": 24, "B-env": 64, "B-nonthermal": 28,
                     "geo-only": 6}
    # every rung of the ladder differs from the previous by only the named columns
    full, narrow = set(features.CONFIGS["A-full"]), set(features.CONFIGS["A-narrow"])
    assert full - narrow == {"LST_urb_day_CT", "LST_rur_day"}
    keep_uhi, no_lst = set(features.CONFIGS["A-noLST-keepUHI"]), set(features.CONFIGS["A-noLST"])
    assert keep_uhi - no_lst == {c for c in features.SUHI if c != features.TARGET_UHI}
    assert no_lst <= keep_uhi
    # the LST reference is every LST column and nothing else, so it holds both target terms
    assert features.CONFIGS["A-lst"] == features.LST
    assert {"LST_urb_day_CT", "LST_rur_day"} <= set(features.CONFIGS["A-lst"])
