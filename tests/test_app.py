"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Randwerte, überlaufendes Netz, lokale Regler, Würfel-Knopf, Permalink-Grenzen, Abschnitte, Footer."""

import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import jn_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def _table(at, start):
    return next(m.value for m in at.markdown if m.value.startswith(start))


def test_default_run_has_no_exception_and_shows_the_reference_values():
    at = _run()
    _ok(at)
    assert _metric(at, "Gesamtzeit (Formel, Jackson)") == "19.7 min" and _metric(at, "Gesamtzeit (Zerlegung)") == "19.7 min"             # exponentiell: beide gleich
    assert _metric(at, "Engpass") == "Kran (82 %)"
    assert abs(float(_metric(at, "Gesamtzeit (Simulation)").split()[0]) / 19.68 - 1) < 0.08
    assert len(at.get("plotly_chart")) == 8


def test_all_sections_are_present():
    at = _run()
    heads = [s.value for s in at.subheader]
    assert heads == ["📐 Der Engpass wandert", "🔬 Produktform: Kette gegen Produkt", "🔬 Burke und die Weitergabe der Streuung", "🔬 Nicht exponentielle Dauer: Jackson, Zerlegung, Simulation",
                     "🔬 Stapel voll: Blockieren", "🚧 Wo die Annahmen enden"]
    assert any(m.value.startswith("## 🔗 Vom Gate zum Netz") for m in at.markdown)
    assert any("Diese Demo ist Teil des Portfolios" in c.value for c in at.caption)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert (at.session_state["gamma_slider"], at.session_state["r_slider"], at.session_state["cs2_select"]) == (p["gamma"], p["r_pct"], p["cs2"])
    assert at.metric


@pytest.mark.parametrize("kw", [dict(gamma_slider=12, r_slider=0, cs2_select=0.0), dict(gamma_slider=42, r_slider=0, cs2_select=4.0), dict(gamma_slider=30, r_slider=40, cs2_select=0.25),
                                 dict(gamma_slider=36, r_slider=20, cs2_select=4.0)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_an_overloaded_network_shows_a_warning_instead_of_a_simulation():
    at = _run(gamma_slider=42, r_slider=40)
    _ok(at)
    assert any("Das Netz läuft über" in w.value and "Kran" in w.value for w in at.warning)
    assert not any(m.label.startswith("Gesamtzeit") for m in at.metric)
    assert any("Kran oder Stapel überlaufen" in w.value or "läuft Kran oder Stapel über" in w.value for w in at.warning)


def test_streuung_other_than_one_switches_the_explanation():
    exp = _run(cs2_select=1.0)
    assert any("gilt Jackson" in i.value for i in exp.info)
    other = _run(cs2_select=4.0)
    assert any("gilt die Produktform nicht mehr" in i.value for i in other.info)
    assert _metric(other, "Gesamtzeit (Zerlegung)") != _metric(other, "Gesamtzeit (Formel, Jackson)") and _metric(other, "Gesamtzeit (Formel, Jackson)") == "19.7 min"


def test_traffic_equations_are_written_with_the_current_numbers():
    at = _run(gamma_slider=33, r_slider=20)
    text = next(m.value for m in at.markdown if m.value.startswith("**Verkehrsgleichungen**"))
    assert "λ_Gate = γ = 33.0" in text and "λ_Kran = λ_Gate + 0.2·λ_Stapel = 41.2" in text and "41.2" in text


def test_the_crane_regulator_moves_the_bottleneck_to_the_stack():
    two = _run()
    assert "| 10 % | 45.0 | Kran |" in _table(two, "| Rückläufer |")
    three = _run(crane_select=3)
    _ok(three)
    table = _table(three, "| Rückläufer |")
    assert "| 10 % | 54.0 | Stapel |" in table and "| 0 % | 60.0 | Gate |" in table
    assert any("zuerst läuft der Stapel über" in i.value for i in three.info)


def test_product_form_section_reports_a_tiny_difference():
    at = _run(gamma_slider=24, r_slider=20)
    _ok(at)
    diff = float(_metric(at, "Größter Unterschied"))
    assert diff < 1e-6
    assert any("unterscheiden sich um höchstens" in i.value for i in at.info)


def test_burke_table_has_one_row_per_streuung():
    at = _run()
    table = _table(at, "| Dauer | Streuung der Abgangsabstände")
    assert table.count("| cs² = ") == 3 and "| cs² = 1 | 0.99" in table


@pytest.mark.parametrize("key,value", [("study_gamma", 24), ("study_gamma", 36), ("study_r", 0), ("block_gamma", 30), ("block_gamma", 42)])
def test_local_regulators_run(key, value):
    at = _run(**{key: value})
    _ok(at)
    assert len(at.get("plotly_chart")) == 8


def test_study_table_lists_all_four_streuungen_and_the_blocking_table_all_caps():
    at = _run()
    assert _table(at, "| Streuung cs² |").count("\n| ") == 4
    blocks = _table(at, "| Plätze im Stapel |")
    assert blocks.count("\n| ") == 6 and "| unbegrenzt |" in blocks and "| 4 |" in blocks


def test_dice_button_changes_the_seed_and_the_simulated_run(monkeypatch):
    """Der Würfel zieht sonst einen unseeded Zufalls-Seed; deshalb ist der gewürfelte Seed im Test fest."""
    monkeypatch.setattr(random, "randint", lambda a, b: 508145)
    at = _run()
    old_seed, old = at.session_state["seed_input"], _metric(at, "Gesamtzeit (Simulation)")
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] == 508145 != old_seed and _metric(at, "Gesamtzeit (Simulation)") != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["g"] = "34"
    at.query_params["r"] = "26"
    at.query_params["cs2"] = "2.6"
    at.run()
    _ok(at)
    assert at.session_state["gamma_slider"] == 33 and at.session_state["r_slider"] == 30 and at.session_state["cs2_select"] == 4.0


def test_permalink_ignores_garbage():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["g"] = "viele"
    at.query_params["seed"] = "nan"
    at.run()
    _ok(at)
    assert at.session_state["gamma_slider"] == C.DEFAULT_GAMMA and at.session_state["seed_input"] == C.DEFAULT_SEED


def test_no_sentence_wide_comma_replacement_in_the_app_source():
    """Regressionsschutz: `.replace(",", ".")` auf einem ganzen (verketteten) Satz macht aus Kommas im Fließtext Punkte; Tausender nur über `fmt_int`."""
    source = Path(APP).read_text(encoding="utf-8")
    assert '.replace(",", ".")' not in source
