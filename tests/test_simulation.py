"""Simulation: Sampler, von Hand verfolgte Mini-Instanzen (Tandem, Blockieren, Rückläufer, Abweisung), Reproduzierbarkeit, Simulation gegen Formel, Little, Burke."""

import numpy as np
import pytest

import jn_constants as C
import jn_formulas as F
import jn_simulation as S
from conftest import Route, Seq


@pytest.mark.parametrize("scv", [0.0, 0.25, 1.0, 4.0])
def test_samplers_have_the_requested_mean_and_variation(scv):
    draw = S.make_sampler(2.0, scv, S.SplitMix64(3))
    x = np.array([draw() for _ in range(200_000)])
    assert x.mean() == pytest.approx(2.0, rel=0.015)
    assert x.var() / x.mean() ** 2 == pytest.approx(scv, abs=0.05 * max(scv, 0.2))


def test_balanced_h2_probability_by_hand():
    """scv = 1 → p₁ = 0.5; scv = 4: ½(1 + √(3/5))."""
    assert S.balanced_h2_probability(1.0) == pytest.approx(0.5) and S.balanced_h2_probability(4.0) == pytest.approx(0.5 * (1 + (3 / 5) ** 0.5))


def test_tandem_by_hand(tandem_script):
    """Siehe conftest: mittlere Gesamtzeit 3.0, Ende 6.5, L = (6.0/6.5, 3.0/6.5)."""
    res = S.simulate(1.0, [1, 1], [1.5, 1.0], [0, 0], [[0, 1], [0, 0]], 3, 1, **tandem_script)
    assert res.mean_sojourn == pytest.approx(3.0) and res.end_time == pytest.approx(6.5) and res.n == 3 and res.rejected == 0
    assert res.L == pytest.approx([6.0 / 6.5, 3.0 / 6.5])


def test_blocking_by_hand(blocking_script):
    """Siehe conftest: Gesamtzeiten 4.0 und 5.8 (Lkw 2 steht von 3.2 bis 5.0 blockiert am Gerät 0), Mittel 4.9, Ende 8.0, L = (3.8/8, 6/8)."""
    res = S.simulate(1.0, [1, 1], [1.0, 3.0], [0, 0], [[0, 1], [0, 0]], 2, 1, cap=[None, 1], **blocking_script)
    assert res.mean_sojourn == pytest.approx(4.9) and res.end_time == pytest.approx(8.0)
    assert res.L == pytest.approx([3.8 / 8, 6.0 / 8])


def test_without_the_platzgrenze_the_waiting_moves_but_the_total_stays(blocking_script):
    """Gegenprobe zum Blockieren: ohne Grenze geht Lkw 2 bei 3.2 an Station 1 und wartet dort in der Schlange (Station 1: Lkw 1 von 2 bis 5, Lkw 2 von 3.2 bis 8). Gesamtzeiten wie vorher (Mittel 4.9),
    aber L = (2.0/8, 7.8/8): das Warten liegt an Station 1 statt am Gerät 0; die Summe 1.225 ist in beiden Fällen gleich."""
    res = S.simulate(1.0, [1, 1], [1.0, 3.0], [0, 0], [[0, 1], [0, 0]], 2, 1, **blocking_script)
    assert res.mean_sojourn == pytest.approx(4.9) and res.L == pytest.approx([2.0 / 8, 7.8 / 8]) and sum(res.L) == pytest.approx(3.8 / 8 + 6.0 / 8)


def test_feedback_by_hand():
    """Ein Lkw, Station A → B, von B mit u < 0.5 zurück zu A: Routing-Zufall 0.0 (A → B), 0.4 (B → A), 0.0 (A → B), 0.9 (B verlässt). Alle Dauern 1, Ankunft bei 1: A 1–2, B 2–3, A 3–4, B 4–5.
    Gesamtzeit 4.0, Ende 5.0, L_A = L_B = 2/5."""
    res = S.simulate(1.0, [1, 1], [1.0, 1.0], [0, 0], [[0, 1], [0.5, 0]], 1, 1, arrival=Seq([1.0]), services=[lambda: 1.0, lambda: 1.0], rng_route=Route([0.0, 0.4, 0.0, 0.9]))
    assert res.mean_sojourn == pytest.approx(4.0) and res.end_time == pytest.approx(5.0) and res.L == pytest.approx([0.4, 0.4])


def test_abweisung_an_station_0_by_hand():
    """Ein Gerät, Platzgrenze 1, Dauer 3, Lkw bei t = 1 und 1.5: der zweite wird abgewiesen. Gesamtzeit 3, Ende 4, L = 3/4, abgewiesen 1."""
    res = S.simulate(1.0, [1], [3.0], [0], [[0.0]], 2, 1, cap=[1], arrival=Seq([1.0, 0.5]), services=[lambda: 3.0])
    assert res.rejected == 1 and res.n == 1 and res.mean_sojourn == pytest.approx(3.0) and res.end_time == pytest.approx(4.0) and res.L == pytest.approx([0.75])


def test_platzgrenzen_with_feedback_are_rejected():
    with pytest.raises(ValueError):
        S.simulate(0.5, C.SERVERS, C.MEANS, 1.0, F.routing_matrix(0.2), 100, 1, cap=[None, None, 6])


def test_exit_probability_follows_the_routing_row():
    """Bei P[1][0] = 0.3 gehen im Mittel 30 % der Lkw von Station 1 zurück: die Rate an Station 0 ist γ/(1 − 0.3) (Verkehrsgleichung), gemessen über die Zahl der Abgänge."""
    gamma, trucks = 0.2, 40_000
    res = S.simulate(gamma, [4, 4], [1.0, 1.0], 1.0, [[0, 1], [0.3, 0]], trucks, 5, record_station=0)
    assert len(res.departures) / res.end_time == pytest.approx(gamma / 0.7, rel=0.03)


def test_same_seed_same_result_and_other_seed_differs():
    args = (0.4, C.SERVERS, C.MEANS, 1.0, F.routing_matrix(0.2), 5_000)
    a, b, c = S.simulate(*args, seed=7), S.simulate(*args, seed=7), S.simulate(*args, seed=8)
    assert a.mean_sojourn == b.mean_sojourn and a.L == b.L and a.mean_sojourn != c.mean_sojourn


def test_streams_are_separate():
    """Die Dauer an Station 1 hängt nicht davon ab, wie viele Zufallszahlen die anderen Ströme verbraucht haben."""
    a = S.streams(5, 3)
    b = S.streams(5, 3)
    for _ in range(100):
        a[0].next()
    assert a[1][1].next() == b[1][1].next() and a[2].next() == b[2].next()


@pytest.mark.parametrize("gamma_ph,r_pct", [(24, 20), (33, 0), (30, 10)])
def test_simulation_matches_jackson_for_exponential_service(gamma_ph, r_pct):
    gamma, p = gamma_ph / 60.0, F.routing_matrix(r_pct / 100)
    j = F.jackson([gamma, 0, 0], p, C.SERVERS, C.MEANS)
    res = S.simulate(gamma, C.SERVERS, C.MEANS, 1.0, p, 150_000, 11)
    assert res.mean_sojourn == pytest.approx(j["total"], rel=0.04)
    assert res.L == pytest.approx(list(j["L"]), rel=0.08)


def test_littles_law_holds_on_the_simulated_path():
    """Σ L ≈ γ · Gesamtzeit (Little über das ganze Netz; hier keine Abweisung)."""
    gamma = 0.4
    res = S.simulate(gamma, C.SERVERS, C.MEANS, 1.0, F.routing_matrix(0.2), 150_000, 3)
    assert sum(res.L) == pytest.approx(gamma * res.mean_sojourn, rel=0.03)


def test_burke_departures_of_an_mm1_station_are_poisson_and_of_an_md1_station_are_regular():
    """M/M/1, ρ = 0.8: Streuung der Abgangsabstände 1, keine Korrelation; M/D/1: 0.36 = Whitt, leicht positive Korrelation."""
    for scv, expected, tol in ((1.0, 1.0, 0.04), (0.0, 0.36, 0.03)):
        res = S.simulate(0.8 / 3.0, [1], [3.0], scv, [[0.0]], 120_000, 4, record_station=0)
        d = np.diff(np.array(res.departures))
        assert d.var() / d.mean() ** 2 == pytest.approx(expected, abs=tol)
        if scv == 1.0:
            assert abs(np.corrcoef(d[:-1], d[1:])[0, 1]) < 0.02
