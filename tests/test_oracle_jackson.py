"""Orakel-Tests (unabhängiger Rechenweg): Verkehrsgleichungen als Neumann-Reihe, Jackson-Formeln und Blockier-Kette gegen eine eigene generische Markov-Kette über (wartend, in
Arbeit, blockiert) je Station, die Zerlegung (QNA) gegen ein von Hand aufgestelltes lineares System, die Simulation gegen eine Neuimplementierung (Zeitgeber-Scan statt Heap)
mit denselben Zufallszahlen."""

import math
from collections import deque

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")
spla = pytest.importorskip("scipy.sparse.linalg")

import jn_chain as CH  # noqa: E402
import jn_formulas as F  # noqa: E402
import jn_simulation as S  # noqa: E402


def _stationary(start, trans):
    idx, states, rows, cols, vals = {start: 0}, [start], [], [], []
    queue = deque([start])
    while queue:
        s = queue.popleft()
        for t, r in trans(s):
            if r > 0:
                if t not in idx:
                    idx[t] = len(states)
                    states.append(t)
                    queue.append(t)
                rows.append(idx[s]); cols.append(idx[t]); vals.append(r)
    n = len(states)
    q = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    q = q - sp.diags(np.asarray(q.sum(axis=1)).ravel())
    a = sp.vstack([sp.csr_matrix(np.ones((1, n))), q.T.tocsr()[1:]]).tocsc()
    b = np.zeros(n)
    b[0] = 1.0
    return states, spla.spsolve(a, b)


def _network_chain(gamma, c, mean, P, cap=None, n_total=30):
    """Exponentielle Dauer, Poisson-Ankünfte an Station 0. Zustand je Station (w, s, b) = wartend, in Arbeit, blockiert. Mit `cap` (nur Linie i → i + 1 oder Ausgang):
    Blockieren nach Abfertigung und Abweisung an Station 0. Gibt (mittlere Zahl je Station, angenommene Ankunftsrate) zurück."""
    n = len(c)
    cap = cap or [None] * n

    def room(st, j):
        return cap[j] is None or sum(st[j]) < cap[j]

    def settle(st):
        st = [list(x) for x in st]
        changed = True
        while changed:
            changed = False
            for i in range(n):
                while st[i][0] > 0 and st[i][1] + st[i][2] < c[i]:
                    st[i][0] -= 1; st[i][1] += 1; changed = True
            for i in range(n - 1):
                if st[i][2] > 0 and (cap[i + 1] is None or sum(st[i + 1]) < cap[i + 1]):
                    st[i][2] -= 1; st[i + 1][0] += 1; changed = True
        return tuple(tuple(x) for x in st)

    def trans(state):
        out = []
        if sum(sum(x) for x in state) < n_total and room(state, 0):
            st = [list(x) for x in state]
            st[0][0] += 1
            out.append((settle(st), gamma))
        for i in range(n):
            s = state[i][1]
            if s == 0:
                continue
            rate = s / mean[i]
            st = [list(x) for x in state]
            st[i][1] -= 1
            if 1 - sum(P[i]) > 1e-15:
                out.append((settle(st), rate * (1 - sum(P[i]))))
            for j in range(n):
                if P[i][j] > 0:
                    st = [list(x) for x in state]
                    st[i][1] -= 1
                    if room(state, j):
                        st[j][0] += 1
                    else:
                        st[i][2] += 1
                    out.append((settle(st), rate * P[i][j]))
        return out

    states, pi = _stationary(settle(tuple((0, 0, 0) for _ in range(n))), trans)
    length = [sum(pi[k] * sum(states[k][i]) for k in range(len(states))) for i in range(n)]
    accepted = sum(pi[k] * gamma for k, s in enumerate(states) if sum(sum(x) for x in s) < n_total and room(s, 0))
    return length, accepted


def test_oracle_itself_by_hand():
    """Tandem zweier M/M/1 mit ρ = 0.5 je Station: L = 1 je Station, Gesamtzeit nach Little 4."""
    length, accepted = _network_chain(0.5, [1, 1], [1.0, 1.0], [[0, 1], [0, 0]], n_total=40)
    assert length == pytest.approx([1.0, 1.0], abs=1e-6) and sum(length) / accepted == pytest.approx(4.0, abs=1e-6)


def test_traffic_equations_against_the_neumann_series():
    P = np.array([[0, 1, 0], [0, 0, 1], [0, 0.3, 0.0]])
    gamma = np.array([0.5, 0.0, 0.0])
    total, term = np.zeros(3), gamma.copy()
    for _ in range(500):
        total += term
        term = P.T @ term
    assert F.traffic_rates(gamma, P) == pytest.approx(total, rel=1e-12)


@pytest.mark.parametrize("c,mean,P,gamma", [([1, 2, 1], [1.0, 1.0, 0.5], [[0, 1, 0], [0, 0, 1], [0, 0.2, 0]], 0.5),
                                              ([2, 1], [1.0, 0.8], [[0, 1], [0.25, 0]], 0.6),
                                              ([1, 1, 2], [0.5, 1.0, 1.0], [[0.1, 0.5, 0], [0, 0, 0.6], [0.2, 0, 0]], 0.7)])
def test_jackson_against_the_generic_markov_chain(c, mean, P, gamma):
    n = len(c)
    j = F.jackson([gamma] + [0.0] * (n - 1), P, c, mean)
    length, accepted = _network_chain(gamma, c, mean, P, n_total=16 if n == 3 else 24)
    assert j["L"] == pytest.approx(length, rel=2e-3) and j["total"] == pytest.approx(sum(length) / accepted, rel=2e-3)


@pytest.mark.parametrize("gamma_ph,k2", [(30, 4), (36, 6), (42, 5)])
def test_blocking_chain_against_the_generic_markov_chain(gamma_ph, k2):
    """Kran (2 Geräte, 2.4 min) → Stapel (4 Geräte, 4 min, Platzgrenze k2): die Kette der Demo gegen die generische mit Zustand (wartend, in Arbeit, blockiert)."""
    g = gamma_ph / 60.0
    _, m = CH.blocking_chain(g, 2, 2.4, 4, 4.0, k2, 40)
    length, accepted = _network_chain(g, [2, 4], [2.4, 4.0], [[0, 1], [0, 0]], cap=[40, k2], n_total=40 + k2)
    assert (m["L1"], m["L2"]) == pytest.approx(tuple(length), rel=1e-7) and m["throughput"] == pytest.approx(accepted, rel=1e-7)
    assert m["time"] == pytest.approx(sum(length) / accepted, rel=1e-7)


def test_two_station_chain_with_feedback_against_the_generic_markov_chain():
    g, r, size = 0.4, 0.3, 30
    pi = CH.two_station_chain(g, r, 2, 2.4, 4, 4.0, size)
    n = np.arange(size)
    length, _ = _network_chain(g, [2, 4], [2.4, 4.0], [[0, 1], [r, 0]], n_total=40)
    assert float((pi.sum(axis=1) * n).sum()) == pytest.approx(length[0], rel=2e-3) and float((pi.sum(axis=0) * n).sum()) == pytest.approx(length[1], rel=2e-3)


def test_decomposition_against_a_hand_set_linear_system():
    """ca²_j = (γ_j + Σ_i λ_i p_ij (1 + p_ij (cd²_i − 1)))/λ_j mit cd²_i − 1 = (1 − ρ_i²)(ca²_i − 1) + ρ_i²(cs²_i − 1)/√c_i ist linear in ca²; Wartezeit Allen-Cunneen mit M/M/c aus Geburts-Todes-Gleichungen."""
    c, mean, cs2, r, g = [3, 2, 4], [3.0, 2.4, 4.0], [0.0, 4.0, 0.25], 0.2, 0.5
    P = F.routing_matrix(r)
    gv = [g, 0.0, 0.0]
    lam = np.linalg.solve(np.eye(3) - np.array(P).T, gv)
    rho = lam * np.array(mean) / np.array(c)
    A, b = np.eye(3), np.zeros(3)
    for j in range(3):
        b[j] += gv[j] / lam[j]
        for i in range(3):
            if P[i][j] > 0:
                coef = lam[i] * P[i][j] ** 2 / lam[j]
                A[j, i] -= coef * (1 - rho[i] ** 2)
                b[j] += lam[i] * P[i][j] / lam[j] + coef * (-(1 - rho[i] ** 2) + rho[i] ** 2 * (cs2[i] - 1) / math.sqrt(c[i]))
    ca2 = np.linalg.solve(A, b)

    def mmc_wq(cc, a, size=2000):
        w = np.ones(size)
        for k in range(1, size):
            w[k] = w[k - 1] * a / min(k, cc)
        pi = w / w.sum()
        return (np.arange(size) - cc).clip(0) @ pi / a

    wq = [mmc_wq(c[i], lam[i] * mean[i]) * mean[i] * (ca2[i] + cs2[i]) / 2 for i in range(3)]
    q = F.qna(gv, P, c, mean, cs2)
    assert q["ca2"] == pytest.approx(ca2, abs=1e-9) and q["total"] == pytest.approx(sum(lam[i] * (wq[i] + mean[i]) for i in range(3)) / g, rel=1e-9)
    assert F.qna(gv, P, c, mean, 1.0)["total"] == pytest.approx(F.jackson(gv, P, c, mean)["total"], rel=1e-9)


def _des(gaps, c, services, P, routes, trucks, cap, warm_fraction=0.05):
    """Neuimplementierung: laufende Abfertigungen je Station als Liste [Ende, Startnummer, Job, Ziel bei Blockade, Blockiernummer]; das nächste Ereignis per min()."""
    n = len(c)
    cap = cap or [None] * n
    warm = int(warm_fraction * trucks)
    queues, running, count, area = [deque() for _ in range(n)], [[] for _ in range(n)], [0] * n, [0.0] * n
    order, t_prev, t_start, entry, soj, finished, rejected, n_arr, jid, now = 0, 0.0, 0.0, {}, [], 0, 0, 0, 0, 0.0
    next_arrival = next(gaps)
    room = lambda j: cap[j] is None or count[j] < cap[j]

    def start_jobs(i):
        nonlocal order
        while len(running[i]) < c[i] and queues[i]:
            order += 1
            running[i].append([now + next(services[i]), order, queues[i].popleft(), None, 0])

    def route(i):
        u, acc = next(routes), 0.0
        for j in range(n):
            acc += P[i][j]
            if u < acc:
                return j
        return None

    while finished + rejected < trucks:
        cand = [(next_arrival, 10 ** 9, "arr", None, None)] if n_arr < trucks else []
        cand += [(r[0], r[1], "done", i, r) for i in range(n) for r in running[i] if r[3] is None]
        ev = min(cand, key=lambda x: (x[0], x[1]))
        now = ev[0]
        for k in range(n):
            area[k] += count[k] * (now - t_prev)
        t_prev = now
        if ev[2] == "arr":
            n_arr += 1
            job, jid = jid, jid + 1
            if room(0):
                entry[job] = now
                count[0] += 1
                queues[0].append(job)
                start_jobs(0)
            else:
                rejected += 1
            if n_arr < trucks:
                next_arrival = now + next(gaps)
        else:
            i, r = ev[3], ev[4]
            dest = route(i)
            if dest is None:
                running[i].remove(r)
                count[i] -= 1
                soj.append(now - entry.pop(r[2]))
                finished += 1
                if finished == warm and warm > 0:
                    area, t_start = [0.0] * n, now
                start_jobs(i)
            elif room(dest):
                running[i].remove(r)
                count[i] -= 1
                count[dest] += 1
                queues[dest].append(r[2])
                start_jobs(dest)
                start_jobs(i)
            else:
                order += 1
                r[3], r[4] = dest, order
        progress = True
        while progress:
            progress = False
            for i in range(n):
                blocked = sorted([r for r in running[i] if r[3] is not None], key=lambda r: r[4])
                if blocked and room(blocked[0][3]):
                    r = blocked[0]
                    running[i].remove(r)
                    count[i] -= 1
                    count[r[3]] += 1
                    queues[r[3]].append(r[2])
                    start_jobs(r[3])
                    start_jobs(i)
                    progress = True
    span = t_prev - t_start
    return [a / span for a in area], float(np.mean(soj[warm:])), len(soj[warm:]), rejected, t_prev


class _Route:
    def __init__(self, values):
        self.it = iter(values)

    def uniform(self):
        return next(self.it)


@pytest.mark.parametrize("name,c,mean,P,cap,trucks", [
    ("Tandem mit Blockieren", [2, 1, 2], [1.0, 2.0, 1.5], [[0, 1, 0], [0, 0, 1], [0, 0, 0]], [None, 2, 3], 250),
    ("Abweisung und Verzweigung", [1, 2], [1.0, 1.5], [[0, 0.7], [0, 0]], [3, 2], 250),
    ("Rückläufer ohne Grenzen", [2, 2, 3], [1.0, 2.4, 4.0], [[0, 1, 0], [0, 0, 1], [0, 0.3, 0]], None, 300),
])
@pytest.mark.parametrize("scv", [0.0, 1.0, 4.0])
def test_simulation_against_independent_event_reimplementation(name, c, mean, P, cap, trucks, scv):
    rng = np.random.default_rng(len(name) + int(scv))
    n, m = len(c), 4000
    gamma = 0.9 * c[0] / mean[0]
    gaps = rng.exponential(1.0 / gamma, trucks + 1)
    p1 = 0.5 * (1 + ((scv - 1) / (scv + 1)) ** 0.5) if scv > 1 else 1.0
    svc = [np.full(m, mean[i]) if scv == 0 else rng.exponential(mean[i], m) if scv == 1 else
           np.where(rng.random(m) < p1, rng.exponential(mean[i] / (2 * p1), m), rng.exponential(mean[i] / (2 * (1 - p1)), m)) for i in range(n)]
    routes = rng.random(m)
    nxt = lambda it: (lambda: next(it))
    res = S.simulate(gamma, c, mean, scv, P, trucks, 0, cap=cap, arrival=nxt(iter(gaps)), services=[nxt(iter(s)) for s in svc], rng_route=_Route(routes))
    L, sojourn, kept, rejected, end = _des(iter(gaps), c, [iter(s) for s in svc], P, iter(routes), trucks, cap)
    assert (res.n, res.rejected) == (kept, rejected)
    assert res.mean_sojourn == pytest.approx(sojourn, abs=1e-9) and res.end_time == pytest.approx(end, abs=1e-9) and res.L == pytest.approx(L, abs=1e-9)
