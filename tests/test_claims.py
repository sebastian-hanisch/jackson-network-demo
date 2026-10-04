"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet: Formelwerte exakt, Studien-Zahlen aus der vorgerechneten Datei (je 4 Läufe à 300 000 bzw. 150 000 Lkw), Kettenwerte aus der Kette.
Die Datei ist fest; ändern sich die Zahlen nach einer neuen Rechnung, müssen README und Hilfetexte nachgezogen werden. Zeiten in Minuten."""

import numpy as np
import pytest

import jn_constants as C
import jn_evaluation as E
import jn_formulas as F

PRE = E.load_precomputed()
STUDY = PRE["study"]
NON_EXP = [x for x in STUDY if x["cs2"] != 1.0]


def dev(x, key):
    return x[key] / x["total"] - 1


def pct(x, digits=1):
    return round(100 * x, digits)


def test_preset_numbers():
    """PRESET_HELP und README: Ausgangslage 19.7 min, Kran 82 % (λ 41 statt 33 Lkw/h); Kran am Limit 27.2 min, 90 % (+7.5 min); Viele Rückläufer 22.9 min, Kran 80 % bei 40/h;
    Feste Dauer: Zerlegung 14.4 gegen Jackson 19.7; cs² = 4: Zerlegung 35.6 gegen 19.7."""
    def run(name):
        p = C.PRESETS[name]
        gamma, pm = E.model(p["gamma"], p["r_pct"])
        return F.jackson(gamma, pm, C.SERVERS, C.MEANS), F.qna(gamma, pm, C.SERVERS, C.MEANS, p["cs2"])
    base, _ = run("Ausgangslage")
    assert f"{base['total']:.1f}" == "19.7" and C.fmt_pct(base["rho"][1]) == "82 %" and round(base["lam"][1] * 60) == 41
    limit, _ = run("Kran am Limit")
    assert f"{limit['total']:.1f}" == "27.2" and C.fmt_pct(limit["rho"][1]) == "90 %" and f"{limit['total'] - base['total']:.1f}" == "7.5"
    many, _ = run("Viele Rückläufer")
    assert f"{many['total']:.1f}" == "22.9" and C.fmt_pct(many["rho"][1]) == "80 %" and round(many["lam"][1] * 60) == 40
    fixed_j, fixed_q = run("Feste Dauer")
    assert f"{fixed_q['total']:.1f}" == "14.4" and f"{fixed_j['total']:.1f}" == "19.7"
    wide_j, wide_q = run("Stark streuende Dauer")
    assert f"{wide_q['total']:.1f}" == "35.6" and f"{wide_j['total']:.1f}" == "19.7"


def test_app_default_numbers():
    """README: Standardlauf der App (33 Lkw/h, 20 % Rückläufer, exponentiell, Seed 35, 60 000 Lkw): Formel 19.7 min, Simulation 19.4 min (−1.7 %)."""
    rep = E.live_report(33, 20, 1.0, 35)
    assert f"{rep['jackson']['total']:.1f}" == "19.7" and f"{rep['sim'].mean_sojourn:.1f}" == "19.4" and pct(rep["sim"].mean_sojourn / rep["jackson"]["total"] - 1) == -1.7


def test_the_bottleneck_table():
    """README: Zwei Geräte am Kran: Netz voll bei 50 / 45 / 40 / 35 / 30 Lkw/h (Rückläufer 0 … 40 %), immer der Kran; drei Geräte: 60 (Gate, gleichauf mit dem Stapel) / 54 / 48 / 42 / 36, ab 10 % der Stapel."""
    two = [E.saturation_per_hour(r, E.crane_servers(2)) for r in range(0, 41, 10)]
    three = [E.saturation_per_hour(r, E.crane_servers(3)) for r in range(0, 41, 10)]
    assert [round(s, 1) for s, _ in two] == [50.0, 45.0, 40.0, 35.0, 30.0] and [C.NAMES[i] for _, i in two] == ["Kran"] * 5
    assert [round(s, 1) for s, _ in three] == [60.0, 54.0, 48.0, 42.0, 36.0] and [C.NAMES[i] for _, i in three] == ["Gate", "Stapel", "Stapel", "Stapel", "Stapel"]


def test_jackson_matches_the_simulation_for_exponential_service():
    """README: bei exponentieller Dauer weicht die Simulation in allen 6 Zellen höchstens 0.8 % von der Jackson-Formel ab, im Mittel 0.2 %."""
    devs = [abs(dev(x, "jackson")) for x in STUDY if x["cs2"] == 1.0]
    assert len(devs) == 6 and pct(max(devs)) == 0.8 and pct(sum(devs) / len(devs)) == 0.2


def test_jackson_fails_for_other_streuung_and_the_decomposition_does_not():
    """README: Jackson bei fester Dauer +7.2 … +46.2 %, bei cs² = 0.25 +4.9 … +31.1 %, bei cs² = 4 −46.4 … −11.3 % neben der Simulation; Zerlegung −8.0 … +2.9, −5.3 … +1.2 und +4.0 … +13.3 %."""
    def span(cs2, key):
        d = [dev(x, key) for x in STUDY if x["cs2"] == cs2]
        return pct(min(d)), pct(max(d))
    assert span(0.0, "jackson") == (7.2, 46.2) and span(0.25, "jackson") == (4.9, 31.1) and span(4.0, "jackson") == (-46.4, -11.3)
    assert span(0.0, "qna") == (-8.0, 2.9) and span(0.25, "qna") == (-5.3, 1.2) and span(4.0, "qna") == (4.0, 13.3)


def test_decomposition_is_closer_in_every_cell_but_not_exact():
    """README: In allen 18 Zellen mit cs² ≠ 1 liegt die Zerlegung näher an der Simulation als Jackson (mindestens 2.0-fach); größter Fehler 13.3 %, im Mittel über alle 24 Zellen 3.1 %;
    ohne Rückläufer höchstens 4.9 %."""
    assert len(NON_EXP) == 18 and all(abs(dev(x, "qna")) < abs(dev(x, "jackson")) for x in NON_EXP)
    assert round(min(abs(dev(x, "jackson")) / abs(dev(x, "qna")) for x in NON_EXP), 1) == 2.1
    assert pct(max(abs(dev(x, "qna")) for x in STUDY)) == 13.3 and pct(np.mean([abs(dev(x, "qna")) for x in STUDY])) == 3.1
    assert pct(max(abs(dev(x, "qna")) for x in STUDY if x["r_pct"] == 0)) == 4.9
    assert pct(max(abs(dev(x, "jackson")) for x in STUDY)) == 46.4


def test_headline_cells_of_the_streuung_study():
    """README: 33 Lkw/h, 20 % Rückläufer: fest Jackson 19.68 / Zerlegung 14.39 / Simulation 14.72 (+33.7 / −2.3 %); cs² = 4: 19.68 / 35.55 / 32.29 (−39.0 / +10.1 %).
    36 Lkw/h: fest +46.2 / −8.0 %, cs² = 4 −46.4 / +13.3 %."""
    def row(g, cs2):
        x = E.study_cell(PRE, g, 20, cs2)
        return round(x["jackson"], 2), round(x["qna"], 2), round(x["total"], 2), pct(-dev(x, "jackson") / (1 + dev(x, "jackson"))) * -1, pct(dev(x, "qna"))
    assert row(33, 0.0)[:3] == (19.68, 14.39, 14.72) and row(33, 4.0)[:3] == (19.68, 35.55, 32.29)
    d = lambda g, cs2, key: pct(dev(E.study_cell(PRE, g, 20, cs2), key))
    assert (d(33, 0.0, "jackson"), d(33, 0.0, "qna")) == (33.7, -2.3) and (d(33, 4.0, "jackson"), d(33, 4.0, "qna")) == (-39.0, 10.1)
    assert (d(36, 0.0, "jackson"), d(36, 0.0, "qna")) == (46.2, -8.0) and (d(36, 4.0, "jackson"), d(36, 4.0, "qna")) == (-46.4, 13.3)


def test_burke_numbers():
    """README (Seed 35, 120 000 Lkw, ein Gerät, Auslastung 80 %): Streuung der Abgangsabstände fest 0.355 (Whitt 0.360), exponentiell 0.998 (1.000), cs² = 4: 2.908 (2.920);
    Korrelation benachbarter Abstände +0.141 / −0.000 / −0.004."""
    reps = [E.burke_report(c, 35) for c in C.BURKE_CS2]
    assert [round(r["scv"], 3) for r in reps] == [0.355, 0.998, 2.908] and [round(r["linking"], 3) for r in reps] == [0.36, 1.0, 2.92]
    assert 0.10 < reps[0]["lag1"] < 0.18 and abs(reps[1]["lag1"]) < 0.02 and abs(reps[2]["lag1"]) < 0.02


def test_product_form_numbers():
    """README (33 Lkw/h, 20 % Rückläufer): Kette über 96 × 96 = 9 216 Zustände, größter Unterschied zum Produkt unter 1e-8, mittlere Zahl Kran 5.166, Stapel 3.651, gleich den Formelwerten."""
    rep = E.product_form_report(33, 20)
    assert rep["size"] == 96 and C.fmt_int(rep["size"] ** 2) == "9.216" and rep["max_diff"] < 1e-8
    assert round(rep["L1"], 3) == 5.166 and round(rep["L2"], 3) == 3.651 and round(rep["jackson_L1"], 3) == 5.166 and round(rep["jackson_L2"], 3) == 3.651


def test_blocking_numbers_against_jackson():
    """README: Gesamtzeit der exakten Kette gegen Jackson bei Platzgrenze 4 / 5 / 6 / 8 / 12: 30 Lkw/h +3.0 / +1.1 / +0.5 / +0.1 / +0.0 %; 36 Lkw/h +14.4 / +6.1 / +2.9 / +0.8 / +0.1 %;
    42 Lkw/h +278 / +53 / +24 / +7 / +1 %."""
    def row(g):
        out = []
        for cap in C.BLOCK_CAPS:
            b = E.blocking_report(g, cap)
            out.append(pct(b["chain_total"] / b["jackson_total"] - 1))
        return out
    assert row(30) == [3.0, 1.1, 0.5, 0.1, 0.0] and row(36) == [14.4, 6.1, 2.9, 0.8, 0.1]
    assert [round(x) for x in row(42)] == [278, 53, 24, 7, 1]


def test_blocking_chain_agrees_with_the_simulation():
    """README: Kette und Simulation stimmen bei 30 und 36 Lkw/h in allen Zellen auf höchstens 1.3 % überein, bei 42 Lkw/h auf höchstens 5.9 % (dort streuen die Läufe stark: Platz 4 hat ±5.6 min Standardfehler)."""
    worst = {30: 0.0, 36: 0.0, 42: 0.0}
    for x in PRE["block"]:
        b = E.blocking_report(x["gamma"], x["cap"])
        worst[x["gamma"]] = max(worst[x["gamma"]], abs(b["chain_total"] / x["total"] - 1))
    assert pct(worst[30]) == 0.4 and pct(worst[36]) == 1.3 and pct(worst[42]) == 5.9
    assert round(E.block_cell(PRE, 42, 4)["total_se"], 1) == 5.6


def test_blocking_example_numbers_in_the_app_text_and_readme():
    """README und App: 36 Lkw/h, Platzgrenze 4: Kette 15.55 min statt 13.59 nach Jackson (+14 %), am Kran 4.6 statt 3.0 Lkw; 42 Lkw/h: 68.9 statt 18.2 min (+278 %), am Kran 42.2 statt 5.7."""
    b = E.blocking_report(36, 4)
    free = E.blocking_report(36, None)
    assert round(b["chain_total"], 2) == 15.55 and round(b["jackson_total"], 2) == 13.59 and round(b["L1"], 1) == 4.6 and round(free["L1"], 1) == 3.0
    b42, free42 = E.blocking_report(42, 4), E.blocking_report(42, None)
    assert round(b42["chain_total"], 1) == 68.9 and round(b42["jackson_total"], 1) == 18.2 and round(b42["L1"], 1) == 42.2 and round(free42["L1"], 1) == 5.7


def test_app_texts_quote_the_measured_extremes():
    """App-Text: „um fast die Hälfte“ (größte Jackson-Abweichung 46.4 %)."""
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert "um fast die Hälfte daneben" in source and pct(max(abs(dev(x, "jackson")) for x in STUDY)) == 46.4


def test_the_truncation_of_the_blocking_chain_does_not_matter():
    """README: Wahrscheinlichkeit am Rand (400 Lkw am Kran): bei 30 Lkw/h unter 1e-75, bei 36 Lkw/h unter 1e-38, bei 42 Lkw/h höchstens 1.5e-6 (Platz 4)."""
    edge = {g: [E.blocking_report(g, cap)["edge"] for cap in C.BLOCK_CAPS] for g in C.BLOCK_GAMMA}
    assert max(edge[30]) < 1e-75 and max(edge[36]) < 1e-38 and f"{max(edge[42]):.1e}" == "1.5e-06" and max(edge[42]) == edge[42][0]


def test_the_unlimited_simulation_at_42_per_hour_marks_the_noise_floor():
    """README: Die Simulation ohne Grenze liegt bei 42 Lkw/h um 1.5 % neben Jackson (Rauschgrenze der Studie)."""
    x = E.block_cell(PRE, 42, None)
    assert pct(x["total"] / x["jackson"] - 1) == 1.5
