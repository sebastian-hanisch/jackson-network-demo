"""Auswertung: Modell, Live-Bericht, Sweeps, Produktform, Burke, Blockieren, vorgerechnete Studie."""

import math

import numpy as np
import pytest

import jn_constants as C
import jn_evaluation as E
import jn_formulas as F

PRE = E.load_precomputed()


def test_model_converts_per_hour_to_per_minute():
    gamma, p = E.model(33, 20)
    assert gamma == pytest.approx([0.55, 0, 0]) and p[2][1] == pytest.approx(0.2)


def test_live_report_for_a_stable_and_an_overloaded_network():
    ok = E.live_report(24, 20, 1.0, 1, trucks=3_000)
    assert ok["stable"] and ok["sim"] is not None and ok["sim"].n == 2_850
    bad = E.live_report(42, 40, 1.0, 1, trucks=3_000)
    assert not bad["stable"] and bad["sim"] is None and math.isinf(bad["jackson"]["total"]) and not bad["qna"]["stable"]


def test_sweep_marks_overload_with_infinity():
    gammas = [12, 30, 45, 60]
    rho, total = E.sweep_gamma(20, gammas)
    assert rho.shape == (4, 3) and rho[:, 1].tolist() == pytest.approx([g / 60 / 0.8 * 1.2 for g in gammas])
    assert math.isfinite(total[0]) and math.isfinite(total[1]) and math.isinf(total[2]) and math.isinf(total[3])


def test_saturation_per_hour_with_two_and_three_crane_devices():
    """Zwei Geräte am Kran: 50, 45, 40, 35, 30 Lkw/h (Kran). Drei Geräte: Gate 60 (ohne Rückläufer, gleichauf mit dem Stapel), dann Stapel 54, 48, 42, 36."""
    two = [E.saturation_per_hour(r, E.crane_servers(2)) for r in range(0, 41, 10)]
    assert [round(x[0], 6) for x in two] == [50.0, 45.0, 40.0, 35.0, 30.0] and {x[1] for x in two} == {1}
    three = [E.saturation_per_hour(r, E.crane_servers(3)) for r in range(0, 41, 10)]
    assert [round(x[0], 6) for x in three] == [60.0, 54.0, 48.0, 42.0, 36.0] and [x[1] for x in three] == [0, 2, 2, 2, 2]


def test_product_form_report_for_a_stable_and_an_overloaded_network():
    rep = E.product_form_report(24, 20)
    assert rep["max_diff"] < 1e-6 and rep["L1"] == pytest.approx(rep["jackson_L1"], rel=1e-4) and rep["L2"] == pytest.approx(rep["jackson_L2"], rel=1e-4)
    assert rep["size"] ** 2 == rep["pi"].size and rep["tail"] <= 1.1e-8
    assert E.product_form_report(42, 40) is None


def test_burke_report_matches_the_linking_equation():
    for cs2, linking in ((0.0, 0.36), (1.0, 1.0), (4.0, 2.92)):
        rep = E.burke_report(cs2, 35, trucks=40_000)
        assert rep["linking"] == pytest.approx(linking) and rep["scv"] == pytest.approx(linking, rel=0.12)


def test_blocking_report_without_a_limit_is_jackson_and_with_a_limit_is_longer():
    free = E.blocking_report(36, None)
    assert free["chain_total"] == free["jackson_total"] and free["edge"] == 0.0
    tight = E.blocking_report(36, 5)
    assert tight["chain_total"] > free["jackson_total"] and tight["L1"] > free["L1"] and tight["edge"] < 1e-9
    big = E.blocking_report(36, 40)
    assert big["chain_total"] == pytest.approx(free["jackson_total"], rel=1e-3)


def test_gate_time_is_the_mm3_time():
    assert E.gate_time(36) == pytest.approx(F.station_metrics(3, 3.0, 0.6)["w"])


def test_study_row_gives_the_formula_sides():
    jack, qna = E.study_row(33, 20, 0.0)
    assert jack == pytest.approx(19.68, abs=0.01) and qna < jack and E.study_row(33, 20, 1.0)[0] == pytest.approx(E.study_row(33, 20, 1.0)[1])


def test_precomputed_study_covers_every_cell_with_all_runs():
    assert len(PRE["study"]) == len(C.STUDY_GAMMA) * len(C.STUDY_R_PCT) * len(C.STUDY_CS2) == 24
    assert len(PRE["block"]) == len(C.BLOCK_GAMMA) * (len(C.BLOCK_CAPS) + 1) == 18
    for x in PRE["study"]:
        assert len(x["runs"]) == C.STUDY_REPS and x["total"] == pytest.approx(np.mean(x["runs"])) and x["total_se"] > 0 and len(x["L"]) == 3
        assert E.study_cell(PRE, x["gamma"], x["r_pct"], x["cs2"]) is x
    for x in PRE["block"]:
        assert len(x["runs"]) == C.BLOCK_REPS and E.block_cell(PRE, x["gamma"], x["cap"]) is x
    assert PRE["study_trucks"] == C.STUDY_TRUCKS and PRE["block_trucks"] == C.BLOCK_TRUCKS


def test_precomputed_formula_columns_equal_the_formulas():
    for x in PRE["study"]:
        assert (x["jackson"], x["qna"]) == pytest.approx(E.study_row(x["gamma"], x["r_pct"], x["cs2"]))


def test_summarize_runs_by_hand():
    class R:
        def __init__(self, t, L):
            self.mean_sojourn, self.L = t, L

    s = E.summarize_runs([R(2.0, [1, 2, 3]), R(4.0, [3, 2, 1])])
    assert s["total"] == 3.0 and s["L"] == [2.0, 2.0, 2.0] and s["total_se"] == pytest.approx(1.0)
    assert E.summarize_runs([R(5.0, [1, 1, 1])])["total_se"] == 0.0
