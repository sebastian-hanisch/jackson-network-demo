"""Auswertung der Jackson-Demo: Live-Bericht (Formel, Zerlegung, Simulation), Produktform-Bericht, Burke-Bericht, Blockier-Bericht und Zugriff auf die vorgerechnete Studie.
Zeiten in Minuten, Raten je Minute; die Regler führen in Lkw je Stunde."""

import json
import math
from pathlib import Path

import numpy as np

import jn_chain as CH
import jn_constants as C
import jn_formulas as F
import jn_simulation as S

PRECOMPUTED_PATH = Path(__file__).resolve().parent / "precomputed_sweep.json"


def gamma_per_minute(gamma_ph):
    return gamma_ph / 60.0


def model(gamma_ph, r_pct):
    """(Ankunftsvektor je Minute, Routing-Matrix) des Terminals."""
    return [gamma_per_minute(gamma_ph), 0.0, 0.0], F.routing_matrix(r_pct / 100.0)


def summarize_runs(runs):
    """Mittel und Standardfehler über Läufe: Gesamtzeit, mittlere Zahl je Station."""
    tot = [r.mean_sojourn for r in runs]
    L = np.array([r.L for r in runs])
    se = (float(np.std(tot, ddof=1)) / math.sqrt(len(tot))) if len(tot) > 1 else 0.0
    return {"total": float(np.mean(tot)), "total_se": se, "L": [float(x) for x in L.mean(axis=0)], "runs": [float(x) for x in tot]}


def live_report(gamma_ph, r_pct, cs2, seed, trucks=C.LIVE_TRUCKS):
    """Formel (Jackson), Zerlegung (QNA) und ein Simulationslauf für die gewählte Einstellung; bei instabilem Netz entfällt die Simulation."""
    gamma, p = model(gamma_ph, r_pct)
    jackson = F.jackson(gamma, p, C.SERVERS, C.MEANS)
    qna = F.qna(gamma, p, C.SERVERS, C.MEANS, cs2)
    sim = S.simulate(gamma[0], C.SERVERS, C.MEANS, cs2, p, trucks, seed) if jackson["stable"] else None
    return {"jackson": jackson, "qna": qna, "sim": sim, "stable": jackson["stable"]}


def crane_servers(n_crane):
    """Geräte je Station mit `n_crane` Geräten am Kran (Gate und Stapel unverändert)."""
    return (C.SERVERS[0], n_crane, C.SERVERS[2])


def sweep_gamma(r_pct, gammas, servers=C.SERVERS):
    """Auslastung je Station und Jackson-Gesamtzeit über die Ankünfte (Lkw je Stunde); Gesamtzeit unendlich, wo eine Station überläuft."""
    rho, total = [], []
    for g in gammas:
        gamma, p = model(g, r_pct)
        j = F.jackson(gamma, p, servers, C.MEANS)
        rho.append(j["rho"])
        total.append(j["total"] if j["stable"] else math.inf)
    return np.array(rho), total


def saturation_per_hour(r_pct, servers=C.SERVERS):
    """Ankünfte je Stunde, bei denen die erste Station voll ausgelastet ist, und der Index dieser Station."""
    gamma, p = model(60.0, r_pct)
    lam = F.traffic_rates(gamma, p)
    rho = [lam[i] * C.MEANS[i] / servers[i] for i in range(3)]
    return 60.0 * F.saturation_gamma(gamma, p, servers, C.MEANS), F.bottleneck(rho)


def product_form_report(gamma_ph, r_pct):
    """Zweistationen-Kette (Kran, Stapel) mit Rückläufern gegen das Produkt der M/M/c-Verteilungen; None, wenn Kran oder Stapel überlaufen."""
    gamma, p = model(gamma_ph, r_pct)
    j = F.jackson(gamma, p, C.SERVERS, C.MEANS)
    if not (j["rho"][1] < 1 and j["rho"][2] < 1):
        return None
    size = CH.truncation_size(float(max(j["rho"][1], j["rho"][2])))
    pi = CH.two_station_chain(gamma[0], r_pct / 100.0, C.SERVERS[1], C.MEANS[1], C.SERVERS[2], C.MEANS[2], size)
    prod = CH.product_form(j["lam"][1], C.SERVERS[1], C.MEANS[1], j["lam"][2], C.SERVERS[2], C.MEANS[2], size)
    n = np.arange(size)
    return {"size": size, "pi": pi, "prod": prod, "max_diff": float(np.abs(pi - prod).max()), "L1": float((pi.sum(axis=1) * n).sum()), "L2": float((pi.sum(axis=0) * n).sum()),
            "jackson_L1": float(j["L"][1]), "jackson_L2": float(j["L"][2]), "tail": float(max(j["rho"][1], j["rho"][2]) ** size)}


def burke_report(cs2, seed, trucks=C.BURKE_TRUCKS):
    """Abgangsprozess einer Station mit einem Gerät bei Auslastung `BURKE_RHO` und Poisson-Ankünften: gemessene Streuung und Korrelation der Zeitabstände gegen die Whitt-Gleichung."""
    mean = 3.0
    res = S.simulate(C.BURKE_RHO / mean, [1], [mean], cs2, [[0.0]], trucks, seed, record_station=0)
    d = np.diff(np.array(res.departures))
    return {"cs2": cs2, "scv": float(d.var() / d.mean() ** 2), "lag1": float(np.corrcoef(d[:-1], d[1:])[0, 1]), "linking": F.departure_scv(C.BURKE_RHO, 1, 1.0, cs2)}


def gate_time(gamma_ph):
    """Mittlere Zeit am Gate (M/M/3) in Minuten."""
    return F.station_metrics(C.SERVERS[0], C.MEANS[0], gamma_per_minute(gamma_ph))["w"]


def blocking_report(gamma_ph, k2):
    """Blockier-Kette (Kran → Stapel mit Platzgrenze k2; None = unbegrenzt, dann Jackson) und die Jackson-Gesamtzeit; Gesamtzeit = Gate (M/M/3) + Kette."""
    gamma, p = model(gamma_ph, 0)
    j = F.jackson(gamma, p, C.SERVERS, C.MEANS)
    out = {"jackson_total": j["total"], "gate": gate_time(gamma_ph)}
    if k2 is None:
        out.update({"chain_total": j["total"], "L1": float(j["L"][1]), "L2": float(j["L"][2]), "edge": 0.0})
        return out
    _, m = CH.blocking_chain(gamma[0], C.SERVERS[1], C.MEANS[1], C.SERVERS[2], C.MEANS[2], k2, C.BLOCK_N1_MAX)
    out.update({"chain_total": out["gate"] + m["time"], "L1": m["L1"], "L2": m["L2"], "edge": m["edge"]})
    return out


def load_precomputed():
    return json.loads(PRECOMPUTED_PATH.read_text(encoding="utf-8"))


def study_cell(pre, gamma_ph, r_pct, cs2):
    return next(x for x in pre["study"] if x["gamma"] == gamma_ph and x["r_pct"] == r_pct and x["cs2"] == cs2)


def block_cell(pre, gamma_ph, cap):
    return next(x for x in pre["block"] if x["gamma"] == gamma_ph and x["cap"] == cap)


def study_row(gamma_ph, r_pct, cs2):
    """Formel, Zerlegung und Simulationsläufe einer Studienzelle (wird in der Studie berechnet; hier nur die formelseitigen Werte)."""
    gamma, p = model(gamma_ph, r_pct)
    return F.jackson(gamma, p, C.SERVERS, C.MEANS)["total"], F.qna(gamma, p, C.SERVERS, C.MEANS, cs2)["total"]
