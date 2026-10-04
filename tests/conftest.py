import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Seq:
    """Liefert vorgegebene Werte der Reihe nach (Zwischenankunftszeiten, Dauern) statt Zufall: Mini-Instanzen von Hand gerechnet."""

    def __init__(self, values):
        self.values = list(values)
        self.i = 0

    def __call__(self):
        v = self.values[self.i]
        self.i += 1
        return v


class Route:
    """Routing-Zufall mit festen Werten: uniform() liefert der Reihe nach die vorgegebenen Zahlen."""

    def __init__(self, values):
        self.values = list(values)
        self.i = 0

    def uniform(self):
        v = self.values[self.i]
        self.i += 1
        return v


@pytest.fixture
def tandem_script():
    """Zwei Stationen mit je einem Gerät, Lkw 1 / 2 / 3 kommen bei t = 1 / 2 / 3. Dauer Station 0: 1.5, Station 1: 1.0.
    Von Hand: Lkw 1: Station 0 von 1 bis 2.5, Station 1 von 2.5 bis 3.5 (Gesamtzeit 2.5). Lkw 2: Station 0 von 2.5 bis 4.0, Station 1 von 4.0 bis 5.0 (3.0). Lkw 3: Station 0 von 4.0 bis 5.5,
    Station 1 von 5.5 bis 6.5 (3.5). Mittel 3.0, Ende 6.5. Zahl an Station 0: 1 auf [1, 2], 2 auf [2, 2.5], 1 auf [2.5, 3], 2 auf [3, 4], 1 auf [4, 5.5]: Fläche 6.0; Station 1: Lkw 1, 2, 3 je eine Zeiteinheit: Fläche 3.0."""
    return dict(arrival=Seq([1, 1, 1]), services=[lambda: 1.5, lambda: 1.0])


@pytest.fixture
def blocking_script():
    """Station 0 mit einem Gerät, Station 1 mit einem Gerät und Platzgrenze 1, Lkw 1 / 2 kommen bei t = 1 / 2.2. Dauer Station 0: 1.0, Station 1: 3.0.
    Von Hand: Lkw 1: Station 0 von 1 bis 2, Station 1 von 2 bis 5. Lkw 2: Station 0 von 2.2 bis 3.2, Station 1 ist bis 5 voll: Lkw 2 bleibt am Gerät 0 stehen (blockiert), geht bei 5 in Station 1 und ist dort bei 8.
    Gesamtzeiten 4.0 und 5.8, Mittel 4.9, Ende 8.0. Zahl an Station 0: 1 auf [1, 2], 0 auf [2, 2.2], 1 auf [2.2, 5]: Fläche 3.8; Station 1: 1 auf [2, 8]: Fläche 6.0."""
    return dict(arrival=Seq([1, 1.2]), services=[lambda: 1.0, lambda: 3.0])
