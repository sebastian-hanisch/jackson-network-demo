"""Ketten: Zweistationen-Kette von Hand, Produktform (Jackson) auch mit Rückläufern, Blockier-Kette von Hand und gegen Jackson im Grenzfall."""

import numpy as np
import pytest

import jn_chain as CH
import jn_constants as C
import jn_formulas as F


def test_truncation_size_follows_the_load_and_is_bounded():
    assert CH.truncation_size(0.5) < CH.truncation_size(0.8) < CH.truncation_size(0.9)
    assert CH.truncation_size(0.99) == C.CHAIN_MAX_SIZE and CH.truncation_size(0.0) == 10 and CH.truncation_size(0.6) ** 1 >= 10
    assert 0.6 ** CH.truncation_size(0.6) <= C.CHAIN_TAIL * 1.0001


def test_two_station_chain_by_hand():
    """Je ein Gerät, Dauer 1, γ = 1, ohne Rückläufer, Kantenlänge 2 (n₁, n₂ ∈ {0, 1}): Gleichgewicht p₀₀ = p₀₁ = p₁₁ = 0.2, p₁₀ = 0.4 (Ankunft nur bei n₁ = 0, Übergabe nur bei n₂ = 0)."""
    pi = CH.two_station_chain(1.0, 0.0, 1, 1.0, 1, 1.0, 2)
    assert pi == pytest.approx(np.array([[0.2, 0.2], [0.4, 0.2]]))


def test_product_form_holds_without_feedback():
    """Zwei M/M/1 hintereinander, ρ = 0.5 und 0.4: die Kette ist das Produkt der Einzelverteilungen auf 1e-10."""
    pi = CH.two_station_chain(0.5, 0.0, 1, 1.0, 1, 0.8, 60)
    prod = CH.product_form(0.5, 1, 1.0, 0.5, 1, 0.8, 60)
    assert np.abs(pi - prod).max() < 1e-10 and pi.sum() == pytest.approx(1.0)


def test_product_form_holds_with_feedback_for_the_terminal():
    """Kran und Stapel mit 20 % Rückläufern (γ = 0.4/min): λ = γ/(1 − r) = 0.5; Kette gegen Produkt auf 1e-6 (Abschneiden)."""
    gamma, r = 0.4, 0.2
    j = F.jackson([gamma, 0, 0], F.routing_matrix(r), C.SERVERS, C.MEANS)
    size = CH.truncation_size(float(max(j["rho"][1], j["rho"][2])))
    pi = CH.two_station_chain(gamma, r, C.SERVERS[1], C.MEANS[1], C.SERVERS[2], C.MEANS[2], size)
    prod = CH.product_form(j["lam"][1], C.SERVERS[1], C.MEANS[1], j["lam"][2], C.SERVERS[2], C.MEANS[2], size)
    assert np.abs(pi - prod).max() < 1e-6
    n = np.arange(size)
    assert (pi.sum(axis=1) * n).sum() == pytest.approx(j["L"][1], rel=1e-5) and (pi.sum(axis=0) * n).sum() == pytest.approx(j["L"][2], rel=1e-5)


def test_chain_reproduces_the_traffic_equation_flows():
    """Durchsatz des Stapels in der Kette = γ/(1 − r) (alle Ankünfte plus alle Rückläufer)."""
    gamma, r = 0.4, 0.2
    pi = CH.two_station_chain(gamma, r, 2, 2.4, 4, 4.0, 60)
    n2 = np.arange(60)
    throughput = float((pi.sum(axis=0) * np.minimum(n2, 4) / 4.0).sum())
    assert throughput == pytest.approx(gamma / (1 - r), rel=1e-6)


def test_the_product_form_breaks_when_the_pair_is_not_a_jackson_network():
    """Gegenprobe der Gegenprobe: mit Platzgrenze 6 am zweiten Ort (Blockier-Kette) ist die Verteilung KEIN Produkt: Kran und Stapel sind abhängig (Zahl am Kran steigt mit der Zahl im Stapel)."""
    dist, _ = CH.blocking_chain(0.6, 2, 2.4, 4, 4.0, 5, 400)
    n1 = np.zeros(401)
    n2 = np.zeros(6)
    joint = np.zeros((401, 6))
    for (a, b, _), p in dist.items():
        joint[a, b] += p
    n1, n2 = joint.sum(axis=1), joint.sum(axis=0)
    assert np.abs(joint - np.outer(n1, n2)).max() > 1e-3
    cond_full = float((joint[:, 5] * np.arange(401)).sum() / n2[5])
    cond_empty = float((joint[:, 0] * np.arange(401)).sum() / n2[0])
    assert cond_full > cond_empty


def test_blocking_chain_by_hand():
    """Je ein Gerät, Dauer 1, γ = 1, Platzgrenze 1, Kran-Zählgrenze 1: Zustände (n₁, n₂, b): (0,0,0) = 2/9, (1,0,0) = 3/9, (0,1,0) = 2/9, (1,1,0) = 1/9, (1,1,1) = 1/9
    (Herleitung: p(0,0,0) = p(0,1,0), 2p(0,1,0) = p(1,0,0) + p(1,1,1), p(1,0,0) = p(0,0,0) + p(1,1,0), 2p(1,1,0) = p(0,1,0), p(1,1,1) = p(1,1,0)).
    Kennzahlen: L₁ = 5/9, L₂ = 4/9, Durchsatz 4/9, Zeit (5/9 + 4/9)/(4/9) = 2.25, Rand n₁ = 1: 5/9."""
    dist, m = CH.blocking_chain(1.0, 1, 1.0, 1, 1.0, 1, 1)
    assert dist == pytest.approx({(0, 0, 0): 2 / 9, (1, 0, 0): 3 / 9, (0, 1, 0): 2 / 9, (1, 1, 0): 1 / 9, (1, 1, 1): 1 / 9})
    assert m["L1"] == pytest.approx(5 / 9) and m["L2"] == pytest.approx(4 / 9) and m["throughput"] == pytest.approx(4 / 9) and m["time"] == pytest.approx(2.25) and m["edge"] == pytest.approx(5 / 9)


def test_blocking_chain_state_space_invariants():
    dist, _ = CH.blocking_chain(0.5, 2, 2.4, 3, 4.0, 5, 40)
    assert sum(dist.values()) == pytest.approx(1.0) and all(p > 0 for p in dist.values())
    assert all(b == 0 or n2 == 5 for (n1, n2, b) in dist) and all(b <= min(n1, 2) for (n1, n2, b) in dist) and all(n2 <= 5 for (_, n2, _) in dist)


def test_blocking_chain_becomes_the_jackson_tandem_for_a_huge_stack():
    """Platzgrenze 60 (praktisch unbegrenzt): Kran und Stapel wie M/M/c hintereinander."""
    gamma = 0.5
    _, m = CH.blocking_chain(gamma, 2, 2.4, 4, 4.0, 60, 300)
    j = F.jackson([gamma, 0, 0], F.routing_matrix(0.0), C.SERVERS, C.MEANS)
    assert m["L1"] == pytest.approx(j["L"][1], rel=1e-4) and m["L2"] == pytest.approx(j["L"][2], rel=1e-4) and m["throughput"] == pytest.approx(gamma, rel=1e-6)


def test_blocking_makes_the_time_longer_and_the_crane_queue_longer():
    times, l1 = [], []
    for k2 in (12, 8, 6, 5, 4):
        _, m = CH.blocking_chain(0.6, 2, 2.4, 4, 4.0, k2, 400)
        times.append(m["time"])
        l1.append(m["L1"])
    assert all(a < b for a, b in zip(times, times[1:])) and all(a < b for a, b in zip(l1, l1[1:]))


def test_blocking_chain_rejects_invalid_input():
    for args in ((0.5, 2, 2.4, 4, 4.0, 3, 40), (0.5, 0, 2.4, 4, 4.0, 5, 40), (0.5, 2, 2.4, 0, 4.0, 5, 40), (0.5, 2, 2.4, 4, 4.0, 5, 0)):
        with pytest.raises(ValueError):
            CH.blocking_chain(*args)
