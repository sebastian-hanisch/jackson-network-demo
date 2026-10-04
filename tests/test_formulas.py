"""Formeln: Erlang C, Verteilung, Verkehrsgleichungen, Jackson-Netz, Engpass, Whitt-Gleichung, Allen-Cunneen, Zerlegung (alles von Hand nachgerechnet)."""

import math

import numpy as np
import pytest

import jn_constants as C
import jn_formulas as F


def test_erlang_c_wait_by_hand():
    """M/M/1, a = 0.5: Wq = ρ/(1 − ρ) = 1. M/M/2, a = 1: Erlang B = 0.2, P(warten) = 0.2/(1 − 0.5·0.8) = 1/3, Wq = (1/3)/(2·0.5) = 1/3."""
    assert F.erlang_c_wait(1, 0.5) == pytest.approx(1.0)
    assert F.erlang_c_wait(2, 1.0) == pytest.approx(1 / 3)


def test_erlang_c_wait_is_infinite_without_a_steady_state():
    assert F.erlang_c_wait(2, 2.0) == math.inf and F.erlang_c_wait(1, 3.0) == math.inf


def test_mmc_pmf_by_hand_and_normalised():
    """c = 2, a = 1: p₀ = 1/(1 + 1 + 1/(2·0.5)) = 1/3, p₁ = 1/3, danach mit ρ = 0.5 halbiert: p₂ = 1/6."""
    pmf = F.mmc_pmf(2, 1.0, 5)
    assert pmf[:3] == pytest.approx([1 / 3, 1 / 3, 1 / 6])
    assert F.mmc_pmf(3, 1.8, 400).sum() == pytest.approx(1.0, abs=1e-12)
    geo = F.mmc_pmf(1, 0.6, 5)
    assert geo == pytest.approx([0.4 * 0.6 ** n for n in range(5)])


def test_traffic_rates_by_hand():
    """Zwei Stationen, γ = (1, 0), von 2 geht die Hälfte zurück zu 1: λ₁ = 1 + 0.5·λ₂, λ₂ = λ₁ → λ = (2, 2)."""
    assert F.traffic_rates([1.0, 0.0], [[0, 1], [0.5, 0]]) == pytest.approx([2.0, 2.0])


def test_traffic_rates_of_the_terminal_follow_the_closed_form():
    """Gate γ, Kran und Stapel γ/(1 − r)."""
    for r in (0.0, 0.1, 0.4):
        lam = F.traffic_rates([0.55, 0, 0], F.routing_matrix(r))
        assert lam == pytest.approx([0.55, 0.55 / (1 - r), 0.55 / (1 - r)])


def test_jackson_by_hand_for_one_and_two_stations():
    """Eine M/M/1-Station, γ = 0.5, Dauer 1: L = 1, Gesamtzeit 1/(1 − 0.5) = 2. Zwei M/M/1 hintereinander: 4."""
    one = F.jackson([0.5], [[0.0]], [1], [1.0])
    assert one["L"][0] == pytest.approx(1.0) and one["total"] == pytest.approx(2.0) and one["stable"]
    two = F.jackson([0.5, 0.0], [[0, 1], [0, 0]], [1, 1], [1.0, 1.0])
    assert two["total"] == pytest.approx(4.0) and two["rho"] == pytest.approx([0.5, 0.5])


def test_jackson_gives_little_exactly():
    j = F.jackson([0.55, 0, 0], F.routing_matrix(0.2), C.SERVERS, C.MEANS)
    assert j["total"] == pytest.approx(j["L"].sum() / 0.55)
    assert j["L"] == pytest.approx(j["lam"] * j["w"])


def test_jackson_flags_an_overloaded_network():
    j = F.jackson([0.7, 0, 0], F.routing_matrix(0.4), C.SERVERS, C.MEANS)                  # Kran: 0.7/0.6·2.4/2 = 1.4
    assert not j["stable"] and j["rho"][1] == pytest.approx(1.4) and math.isinf(j["total"]) and math.isinf(j["L"][1]) and math.isfinite(j["L"][0])


def test_bottleneck_and_saturation_of_the_terminal():
    """r = 0: Kran 2/2.4 = 50 Lkw/h; mit r Rückläufern 50·(1 − r). Das Gate kann 60/h."""
    for r, expected in ((0.0, 50.0), (0.2, 40.0), (0.4, 30.0)):
        gamma, p = [1.0, 0.0, 0.0], F.routing_matrix(r)
        assert 60 * F.saturation_gamma(gamma, p, C.SERVERS, C.MEANS) == pytest.approx(expected)
        lam = F.traffic_rates(gamma, p)
        assert F.bottleneck([lam[i] * C.MEANS[i] / C.SERVERS[i] for i in range(3)]) == 1
    assert F.bottleneck([0.2, 0.9, 0.5]) == 1


def test_departure_scv_by_hand():
    """ρ = 0.5, ein Gerät, ca² = 1: cs² = 0 → 1 + 0.25·(−1) = 0.75; cs² = 1 → 1; cs² = 4 → 1.75. Vier Geräte: ρ²(cs² − 1)/2."""
    assert F.departure_scv(0.5, 1, 1.0, 0.0) == pytest.approx(0.75) and F.departure_scv(0.5, 1, 1.0, 1.0) == pytest.approx(1.0) and F.departure_scv(0.5, 1, 1.0, 4.0) == pytest.approx(1.75)
    assert F.departure_scv(0.5, 4, 1.0, 5.0) == pytest.approx(1 + 0.25 * 4 / 2)
    assert F.departure_scv(0.8, 1, 2.0, 1.0) == pytest.approx(1 + 0.36)                     # (1 − ρ²)(ca² − 1)


def test_allen_cunneen_matches_exact_special_cases():
    """ca² = cs² = 1 ist M/M/c; bei einem Gerät und Poisson-Ankünften ist es Pollaczek-Khinchine: ρ(1 + cs²)/(2(1 − ρ)) · Dauer."""
    assert F.allen_cunneen_wait(3, 3.0, 0.5, 1.0, 1.0) == pytest.approx(F.erlang_c_wait(3, 1.5) * 3.0)
    rho, m = 0.5, 2.0
    for cs2 in (0.0, 1.0, 4.0):
        assert F.allen_cunneen_wait(1, m, rho / m, 1.0, cs2) == pytest.approx(rho * (1 + cs2) / (2 * (1 - rho)) * m)


def test_qna_equals_jackson_for_exponential_service():
    j = F.jackson([0.55, 0, 0], F.routing_matrix(0.2), C.SERVERS, C.MEANS)
    q = F.qna([0.55, 0, 0], F.routing_matrix(0.2), C.SERVERS, C.MEANS, 1.0)
    assert q["total"] == pytest.approx(j["total"]) and q["L"] == pytest.approx(j["L"]) and q["ca2"] == pytest.approx([1, 1, 1])


def test_qna_of_two_md1_in_tandem_by_hand():
    """Zwei Stationen, ein Gerät, Dauer 1 (fest), γ = 0.5: Station 1 ist M/D/1 (Wartezeit 0.5, cd² = 0.75). Station 2 sieht ca² = 0.75: Wartezeit 1·0.75/2 = 0.375. Gesamtzeit (0.5 + 1) + (0.375 + 1) = 2.875."""
    q = F.qna([0.5, 0.0], [[0, 1], [0, 0]], [1, 1], [1.0, 1.0], 0.0)
    assert q["wq"] == pytest.approx([0.5, 0.375]) and q["cd2"][0] == pytest.approx(0.75) and q["total"] == pytest.approx(2.875)


def test_qna_with_feedback_solves_the_merge_equations_by_hand():
    """Zwei Stationen (ein Gerät, Dauer 1, fest), γ = (0.25, 0), von 2 geht die Hälfte zurück zu 1: λ = (0.5, 0.5), ρ = 0.5. Mit x = ca₁², y = ca₂²: y = cd₁ = 0.75x, x = 0.75 + 0.25·cd₂ mit cd₂ = 0.75y
    → x = 0.75/0.859375 = 0.872727…, y = 0.654545…; Wartezeiten x/2 und y/2; Gesamtzeit (L₁ + L₂)/γ = 5.5273."""
    q = F.qna([0.25, 0.0], [[0, 1], [0.5, 0]], [1, 1], [1.0, 1.0], 0.0)
    x = 0.75 / 0.859375
    assert q["ca2"] == pytest.approx([x, 0.75 * x], abs=1e-9)
    assert q["wq"] == pytest.approx([x / 2, 0.75 * x / 2])
    assert q["total"] == pytest.approx((0.5 * (x / 2 + 1) + 0.5 * (0.75 * x / 2 + 1)) / 0.25)


def test_qna_marks_an_overloaded_network_and_accepts_per_station_streuung():
    q = F.qna([0.7, 0, 0], F.routing_matrix(0.4), C.SERVERS, C.MEANS, 1.0)
    assert not q["stable"] and math.isinf(q["total"])
    mixed = F.qna([0.4, 0, 0], F.routing_matrix(0.2), C.SERVERS, C.MEANS, [1.0, 4.0, 1.0])
    same = F.qna([0.4, 0, 0], F.routing_matrix(0.2), C.SERVERS, C.MEANS, 1.0)
    assert mixed["total"] > same["total"]


def test_routing_matrix_rows_describe_the_terminal():
    p = np.array(F.routing_matrix(0.3))
    assert p[0].tolist() == [0, 1, 0] and p[1].tolist() == [0, 0, 1] and p[2].tolist() == [0, 0.3, 0] and (p.sum(axis=1) <= 1).all()
