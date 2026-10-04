"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten, Formatierer."""

import pytest

import jn_constants as C
import jn_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 5
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert C.GAMMA_MIN <= preset["gamma"] <= C.GAMMA_MAX and P.snap_gamma(preset["gamma"]) == preset["gamma"]
        assert C.R_PCT_MIN <= preset["r_pct"] <= C.R_PCT_MAX and P.snap_r(preset["r_pct"]) == preset["r_pct"]
        assert preset["cs2"] in C.CS2_OPTIONS
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_every_preset_is_a_stable_network():
    """Kein Preset soll in einem Netz landen, das überläuft."""
    import jn_evaluation as E
    import jn_formulas as F
    for preset in C.PRESETS.values():
        gamma, p = E.model(preset["gamma"], preset["r_pct"])
        assert F.jackson(gamma, p, C.SERVERS, C.MEANS)["stable"]


def test_preset_names_state_the_values_they_set():
    assert C.PRESETS["Kran am Limit"]["gamma"] == 36 and C.PRESETS["Viele Rückläufer"]["r_pct"] == 40
    assert C.PRESETS["Feste Dauer"]["cs2"] == 0.0 and C.PRESETS["Stark streuende Dauer"]["cs2"] == 4.0


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Ausgangslage"]
    assert (p["gamma"], p["r_pct"], p["cs2"], p["seed"]) == (C.DEFAULT_GAMMA, C.DEFAULT_R_PCT, C.DEFAULT_CS2, C.DEFAULT_SEED)


def test_bounds_and_url_params():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("gamma_slider") == (12, 42) and P.bounds("r_slider") == (0, 40)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


@pytest.mark.parametrize("value,expected", [(0, 12), (12, 12), (13, 12), (14, 15), (34, 33), (35, 36), (99, 42)])
def test_arrivals_snap_to_the_step_from_the_lower_bound(value, expected):
    assert P.snap_gamma(value) == expected


@pytest.mark.parametrize("value,expected", [(0, 0), (4, 0), (6, 10), (14, 10), (26, 30), (99, 40)])
def test_feedback_share_snaps_to_the_step_inside_the_bounds(value, expected):
    assert P.snap_r(value) == expected


@pytest.mark.parametrize("value,expected", [(0.1, 0.0), (0.2, 0.25), (0.6, 0.25), (0.7, 1.0), (2.0, 1.0), (2.6, 4.0), (9.0, 4.0)])
def test_streuung_snaps_to_the_nearest_option(value, expected):
    assert P.snap_to_option("cs2_select", value) == expected


def test_formatters():
    assert C.fmt_int(150000) == "150.000" and C.fmt_pct(0.2) == "20 %" and C.fmt_pct(0.0898, 1) == "9.0 %"
    assert C.fmt_signed_pct(0.46) == "+46 %" and C.fmt_signed_pct(-0.163, 1) == "−16.3 %" and C.fmt_signed_pct(0.0) == "+0 %"
