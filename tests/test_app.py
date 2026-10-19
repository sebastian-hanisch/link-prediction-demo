"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Slider an seine Grenzen,
prüft Permalink, Netztypen, Maße und alle sechs Experimente; es darf kein Python-Fehler auftreten."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import lkp_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
EXPERIMENTS = ("types", "precision", "katz", "community", "time", "leak")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_app_loads_without_exception():
    at = _fresh_app()
    assert _metric(at, "Knoten / Kanten") == "150 / 450" and _metric(at, "Versteckte Kanten (k)") == "45" and _metric(at, "AUC leichte Negative") == "0,73"
    assert _metric(at, "Treffer unter den k besten") == "2 von 45" and _metric(at, "Lift gegen Raten") == "10"
    assert not at.error


def test_every_preset_click_does_not_raise():
    for name in C.PRESETS:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            slider.set_value(int(value) if isinstance(slider.value, int) else value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_all_sliders_at_their_minimum_and_all_at_their_maximum_do_not_raise():
    for edge in ("min", "max"):
        at = _fresh_app()
        for index in range(len(at.slider)):
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            slider.set_value(int(value) if isinstance(slider.value, int) else value)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("typ", C.ALL_TYPES)
def test_every_network_type_renders_with_every_method(typ):
    at = _fresh_app()
    at.selectbox(key="netztyp_select").set_value(typ).run(timeout=TIMEOUT)
    assert not at.exception
    for method in C.METHODS:
        at.selectbox(key="method_select").set_value(method).run(timeout=TIMEOUT)
        assert not at.exception, (typ, method, [str(e) for e in at.exception])


def test_the_link_network_is_fixed_and_the_navigation_switch_works():
    at = _fresh_app()
    at.selectbox(key="netztyp_select").set_value(C.T_LINK).run(timeout=TIMEOUT)
    assert at.slider(key="n_slider").disabled and at.slider(key="c_slider").disabled and _metric(at, "Knoten / Kanten") == "350 / 2374"
    assert any("Schnappschuss" in i.value for i in at.info)
    at.selectbox(key="nav_select").set_value(C.NAV_DROP).run(timeout=TIMEOUT)
    assert _metric(at, "Knoten / Kanten") == "348 / 1679"


def test_the_leak_switch_raises_the_preferential_attachment_and_flags_it():
    at = _fresh_app()
    at.selectbox(key="netztyp_select").set_value(C.T_HUBS).run(timeout=TIMEOUT)
    at.selectbox(key="method_select").set_value(C.M_PA).run(timeout=TIMEOUT)
    before = float(_metric(at, "AUC leichte Negative").replace(",", "."))
    at.selectbox(key="leak_select").set_value(C.LEAK_ON).run(timeout=TIMEOUT)
    assert not at.exception and float(_metric(at, "AUC leichte Negative").replace(",", ".")) > before + 0.1
    assert any("Datenleck" in w.value for w in at.warning)


def test_permalink_values_are_clamped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH)
    at.query_params["n"] = "5"
    at.query_params["c"] = "99"
    at.query_params["frac"] = "-3"
    at.query_params["typ"] = "gibt es nicht"
    at.query_params["mass"] = "Unsinn"
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="n_slider").value == C.N_MIN and at.slider(key="c_slider").value == C.C_MAX and at.slider(key="frac_slider").value == C.FRAC_MIN
    assert at.selectbox(key="netztyp_select").value == C.T_COMMUNITIES and at.selectbox(key="method_select").value == C.M_CN


def test_permalink_selects_a_network_type_and_a_method():
    at = AppTest.from_file(APP_PATH)
    at.query_params["typ"] = C.T_RING
    at.query_params["mass"] = C.M_KATZ
    at.run(timeout=TIMEOUT)
    assert not at.exception and at.selectbox(key="netztyp_select").value == C.T_RING and at.selectbox(key="method_select").value == C.M_KATZ


def test_every_experiment_runs_and_is_invalidated_by_a_setting_change():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    at.slider(key="n_slider").set_value(C.N_MIN).run(timeout=TIMEOUT)         # kleines Netz: schnell
    for name in EXPERIMENTS:
        at.button(key=f"run_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
    texts = " ".join(x.value for x in list(at.info) + list(at.warning) + list(at.success))
    for fragment in ("Gemeinsame Nachbarn erreichen mit leichten Negativen", "Der größte Lift ist", "Katz liegt in", "Ohne Struktur (Anteil innen", "überschätzt hier um", "Mit den Graden aus dem vollen Netz"):
        assert fragment in texts, fragment
    at.slider(key="n_slider").set_value(C.N_MIN + 10).run(timeout=TIMEOUT)
    assert not at.exception
    assert sum("neu starten" in i.value for i in at.info) >= len(EXPERIMENTS)           # alle von n abhängigen Ergebnisse sind ungültig
