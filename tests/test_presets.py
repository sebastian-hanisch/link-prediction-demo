"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen."""

import lkp_constants as C
import lkp_evaluation as ev
import lkp_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert p["netztyp"] in C.ALL_TYPES and C.N_MIN <= p["n"] <= C.N_MAX and C.C_MIN <= p["c"] <= C.C_MAX, name
        assert 0 <= p["seed"] <= C.SEED_MAX and C.FRAC_MIN <= p["frac"] <= C.FRAC_MAX and p["method"] in C.METHODS and p["nav"] in C.NAV_CHOICES and p["leak"] in C.LEAK_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[C.DEFAULT_PRESET_NAME]) == ev.Settings()
    assert next(iter(C.PRESETS)) == C.DEFAULT_PRESET_NAME


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_cover_the_teaching_cases():
    ps = C.PRESETS
    assert any(p["netztyp"] == C.T_RANDOM for p in ps.values()) and any(p["netztyp"] == C.T_RING for p in ps.values())
    leak = [p for p in ps.values() if p["leak"] == C.LEAK_ON]
    assert leak and all(p["method"] == C.M_PA for p in leak)                                  # das Leck betrifft nur die bevorzugte Anbindung
    assert {p["nav"] for p in ps.values() if p["netztyp"] == C.T_LINK} == set(C.NAV_CHOICES)
    grid = next(p for p in ps.values() if p["netztyp"] == C.T_GRID)
    assert grid["c"] < 6 and grid["method"] == C.M_KATZ                                       # ohne Diagonalen keine Dreiecke


def test_every_preset_runs_and_the_katz_preset_has_no_common_neighbours():
    for name, p in C.PRESETS.items():
        an = ev.analyse(ev.Settings(**p))
        assert an.k >= 1, name
    an = ev.analyse(ev.Settings(**C.PRESETS["Gitter ohne Diagonalen: Katz"]))
    assert an.proto["coverage"] == 0.0 and an.proto["scores"][C.M_KATZ]["auc_easy"] > 0.8
