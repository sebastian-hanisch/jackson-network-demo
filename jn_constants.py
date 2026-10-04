"""Konstanten der Jackson-Netz-Demo: Terminal-Gate als Netz aus Gate, Kran und Stapel, Regler, Voreinstellungen, Studien-Achsen. Zeiten in Minuten, Raten in Lkw je Minute (Regler in Lkw je Stunde)."""


def fmt_int(n):
    """Ganzzahl mit Punkt als Tausendertrenner (10000 -> 10.000)."""
    return f"{n:,}".replace(",", ".")


def fmt_pct(x, digits=0):
    """Anteil als Prozent mit Leerzeichen (0.086 -> "9 %", mit digits=1 "8.6 %")."""
    return f"{x:.{digits}%}".replace("%", " %")


def fmt_signed_pct(x, digits=0):
    """Vorzeichenbehafteter Prozentwert mit echtem Minus (−0.16 -> "−16 %", 0.46 -> "+46 %")."""
    s = f"{abs(x):.{digits}%}".replace("%", " %")
    return ("−" if x < 0 else "+") + s


# Stationen des Terminals: (Name, Spuren/Geräte c, mittlere Dauer in Minuten)
STATIONS = (("Gate", 3, 3.0), ("Kran", 2, 2.4), ("Stapel", 4, 4.0))
NAMES = tuple(s[0] for s in STATIONS)
SERVERS = tuple(s[1] for s in STATIONS)
MEANS = tuple(s[2] for s in STATIONS)

GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 12, 42, 3, 33       # Ankünfte in Lkw je Stunde (33/h = 0.55 je Minute)
R_PCT_MIN, R_PCT_MAX, R_PCT_STEP, DEFAULT_R_PCT = 0, 40, 10, 20       # Anteil der Lkw, die vom Stapel zum Kran zurückmüssen (Umstapeln)
CS2_OPTIONS = (0.0, 0.25, 1.0, 4.0)                                   # Streuung der Dauer (Variationskoeffizient²) an allen Stationen
DEFAULT_CS2 = 1.0
SEED_MAX = 999999
DEFAULT_SEED = 35

LIVE_TRUCKS = 60_000                                                  # Lkw im Live-Lauf der App
CHAIN_MAX_SIZE = 150                                                  # größte Kantenlänge der abgeschnittenen Zweistationen-Kette
CHAIN_TAIL = 1e-8                                                     # Ziel für den abgeschnittenen Rest ρ^N der Kette

BURKE_RHO = 0.8                                                       # Abschnitt Burke: eine Station, ein Gerät, Auslastung
BURKE_TRUCKS = 120_000
BURKE_CS2 = (0.0, 1.0, 4.0)

STUDY_GAMMA = (24, 33, 36)                                            # Studie Streuung: Lkw je Stunde
STUDY_R_PCT = (0, 20)
STUDY_CS2 = CS2_OPTIONS
STUDY_TRUCKS = 300_000
STUDY_REPS = 4

BLOCK_GAMMA = (30, 36, 42)                                            # Studie Blockieren: Lkw je Stunde, ohne Rückläufer, exponentiell
BLOCK_CAPS = (4, 5, 6, 8, 12)                                         # Platzgrenze des Stapels (Warte- und Bedienplätze); 4 = nur Geräte, kein Warteplatz
BLOCK_TRUCKS = 150_000
BLOCK_REPS = 4
BLOCK_N1_MAX = 400                                                    # Abschneiden der Kran-Warteschlange in der Blockier-Kette

PRESET_ORDER = ("Ausgangslage", "Kran am Limit", "Viele Rückläufer", "Feste Dauer", "Stark streuende Dauer")


def _preset(gamma=DEFAULT_GAMMA, r_pct=DEFAULT_R_PCT, cs2=DEFAULT_CS2):
    return {"gamma": gamma, "r_pct": r_pct, "cs2": cs2, "seed": DEFAULT_SEED}


PRESETS = {
    "Ausgangslage": _preset(),
    "Kran am Limit": _preset(gamma=36),
    "Viele Rückläufer": _preset(gamma=24, r_pct=40),
    "Feste Dauer": _preset(cs2=0.0),
    "Stark streuende Dauer": _preset(cs2=4.0),
}
# Zahlen aus den Formeln (Jackson, Zerlegung); tests/test_claims.py rechnet jede nach
PRESET_HELP = {
    "Ausgangslage": "33 Lkw/h, 20 % Rückläufer, exponentielle Dauer: Gesamtzeit nach Jackson 19.7 min; der Kran ist zu 82 % ausgelastet, er arbeitet für 41 statt 33 Lkw/h.",
    "Kran am Limit": "36 Lkw/h, 20 % Rückläufer: Gesamtzeit 27.2 min, der Kran ist zu 90 % ausgelastet (3 Lkw/h mehr als in der Ausgangslage kosten 7.5 min).",
    "Viele Rückläufer": "24 Lkw/h, 40 % Rückläufer: nur 24 Lkw/h kommen an, aber Kran und Stapel arbeiten für 40/h; Gesamtzeit 22.9 min, der Kran ist zu 80 % ausgelastet.",
    "Feste Dauer": "Wie die Ausgangslage, aber feste Dauer an allen Stationen: die Zerlegung rechnet 14.4 min, die Jackson-Formel (nur für exponentielle Dauer) würde 19.7 min sagen.",
    "Stark streuende Dauer": "Wie die Ausgangslage, aber cs² = 4: die Zerlegung rechnet 35.6 min, die Jackson-Formel 19.7 min.",
}
