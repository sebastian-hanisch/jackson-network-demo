"""Exakte Markov-Ketten der Jackson-Demo (Gegenproben und Referenz, wo keine Formel existiert).

**Zweistationen-Kette** (Kran, Stapel): Zustand (n₁, n₂), Poisson-Ankünfte γ am Kran, Kran fertig → Stapel, Stapel fertig → mit Wahrscheinlichkeit r zurück zum Kran, sonst Ausgang. Das Gleichgewicht πQ = 0
wird als dünn besetztes lineares System über alle (n₁, n₂) ≤ N gelöst; es ist das **Produkt** zweier M/M/c-Verteilungen (Jackson), bis auf den abgeschnittenen Rest ≈ ρ^N.

**Blockier-Kette** (Kran → Stapel mit Platzgrenze K₂): Wer am Kran fertig ist und den Stapel voll findet, bleibt auf dem Kran stehen und hält ihn besetzt (Blockieren nach Abfertigung). Zustand
(n₁, n₂, b): n₁ Lkw am Kran (wartend, in Arbeit, blockiert), n₂ im Stapel (≤ K₂), b blockierte Lkw am Kran (nur b > 0, wenn n₂ = K₂). Hier gibt es keine Produktform und keine Formel."""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

from jn_constants import CHAIN_MAX_SIZE, CHAIN_TAIL


def _stationary(q):
    """Gleichgewicht πQ = 0, Σπ = 1 für einen dünn besetzten Generator: Qᵀπ = 0, erste Gleichung durch die Normierung ersetzt."""
    n = q.shape[0]
    a = csr_matrix(q).T.tolil()
    a[0, :] = 1.0
    b = np.zeros(n)
    b[0] = 1.0
    return spsolve(a.tocsr(), b)


def _generator(rows, cols, vals, n):
    q = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    out = np.asarray(q.sum(axis=1)).ravel()
    return (q - diags(out)).tocsr()


def truncation_size(rho_max):
    """Kantenlänge N der abgeschnittenen Kette, so dass der Rest ρ^N höchstens `CHAIN_TAIL` beträgt (höchstens `CHAIN_MAX_SIZE`)."""
    if rho_max <= 0:
        return 10
    return int(min(CHAIN_MAX_SIZE, max(10, np.ceil(np.log(CHAIN_TAIL) / np.log(rho_max)))))


def two_station_chain(gamma, r, c1, m1, c2, m2, size):
    """Gleichgewicht der Kette (Kran, Stapel) als Feld pi[n1, n2] mit n1, n2 < size; Raten: Ankunft γ, Kran min(n₁, c₁)/m₁, Stapel min(n₂, c₂)/m₂, davon Anteil r zurück zum Kran.
    Übergänge, die über den Rand führen würden, entfallen (Abschneiden)."""
    idx = lambda a, b: a * size + b
    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i)
        cols.append(j)
        vals.append(v)

    for a in range(size):
        for b in range(size):
            i = idx(a, b)
            if a + 1 < size:
                add(i, idx(a + 1, b), gamma)
            if a > 0 and b + 1 < size:
                add(i, idx(a - 1, b + 1), min(a, c1) / m1)
            if b > 0:
                rate = min(b, c2) / m2
                add(i, idx(a, b - 1), rate * (1.0 - r))
                if r > 0 and a + 1 < size:
                    add(i, idx(a + 1, b - 1), rate * r)
    pi = _stationary(_generator(rows, cols, vals, size * size))
    return pi.reshape(size, size)


def product_form(lam1, c1, m1, lam2, c2, m2, size):
    """Produkt der beiden M/M/c-Verteilungen (Jackson) als Feld [n1, n2]."""
    from jn_formulas import mmc_pmf
    return np.outer(mmc_pmf(c1, lam1 * m1, size), mmc_pmf(c2, lam2 * m2, size))


def blocking_chain(gamma, c1, m1, c2, m2, k2, n1_max):
    """Gleichgewicht der Blockier-Kette (Kran → Stapel mit Platzgrenze k2 ≥ c2, Kran-Zählgrenze n1_max; Ankünfte bei n₁ = n1_max gehen verloren).
    Gibt ein Dict {(n1, n2, b): Wahrscheinlichkeit} und die Kennzahlen zurück: mittlere Zahl am Kran / im Stapel, Durchsatz, Gesamtzeit Kran + Stapel (Little), Verlustanteil am Rand."""
    if k2 < c2 or c1 < 1 or c2 < 1 or n1_max < 1:
        raise ValueError("k2 ≥ c2, c1 ≥ 1, c2 ≥ 1, n1_max ≥ 1 erwartet")
    states = {}
    for n1 in range(n1_max + 1):
        for n2 in range(k2 + 1):
            for b in range(min(n1, c1) + 1):
                if b == 0 or n2 == k2:
                    states[(n1, n2, b)] = len(states)
    mu1, mu2 = 1.0 / m1, 1.0 / m2
    rows, cols, vals = [], [], []

    def add(s, t, v):
        rows.append(states[s])
        cols.append(states[t])
        vals.append(v)

    for (n1, n2, b) in states:
        s = (n1, n2, b)
        if n1 < n1_max:
            add(s, (n1 + 1, n2, b), gamma)
        working = min(n1 - b, c1 - b)
        if working > 0:
            if n2 < k2:
                add(s, (n1 - 1, n2 + 1, b), working * mu1)
            else:
                add(s, (n1, n2, b + 1), working * mu1)
        if n2 > 0:
            rate = min(n2, c2) * mu2
            if b > 0:
                add(s, (n1 - 1, n2, b - 1), rate)
            else:
                add(s, (n1, n2 - 1, b), rate)
    pi = _stationary(_generator(rows, cols, vals, len(states)))
    dist = {s: float(pi[i]) for s, i in states.items()}
    l1 = sum(p * s[0] for s, p in dist.items())
    l2 = sum(p * s[1] for s, p in dist.items())
    thr = sum(p * min(s[1], c2) * mu2 for s, p in dist.items())
    loss = sum(p for s, p in dist.items() if s[0] == n1_max)
    return dist, {"L1": l1, "L2": l2, "throughput": thr, "time": (l1 + l2) / thr, "edge": loss}
