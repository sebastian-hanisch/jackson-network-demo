# Jackson-Netze – Gate, Kran, Stapel (Streamlit-Demo)

Interaktive Demo zu **Bediennetzen** am Containerterminal. **Zwölftes Stück der Konzepte-Linie „Warteschlangentheorie und Simulation“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein wachsendes Beispiel, jedes Folgestück hebt genau eine Annahme auf.

Bisher war das Terminal **eine** Station. Hier durchläuft ein Lkw **Gate, Kran und Stapel**, und ein Teil muss vom Stapel zurück zum Kran (Umstapeln). Der **Jackson-Satz** macht das einfach: bei Poisson-Ankünften und
exponentieller Dauer rechnet jede Station für sich wie ein M/M/c, und die gemeinsame Verteilung ist das Produkt der Einzelverteilungen. Die Demo prüft das an einer exakten Kette, zeigt, **wo der Engpass liegt und wie er
wandert**, und misst, **wo der Satz endet**: bei nicht exponentieller Dauer (die Zerlegungsnäherung nach Whitt hilft) und bei vollem Stapel (dort gibt es keine Formel, aber eine exakte Kette).

## Kernfrage

Wie weit trägt die Produktform von Jackson für ein Terminal, wo liegt der Engpass, und was passiert bei anderer Streuung und bei vollem Stapel?

## Modell und Methodik

- **Netz** (`jn_formulas.py`): Gate (3 Spuren, 3 min), Kran (2 Geräte, 2.4 min), Stapel (4 Geräte, 4 min). Poisson-Ankünfte nur am Gate; vom Stapel gehen r der Lkw zurück zum Kran, die übrigen verlassen das Terminal.
  Die **Verkehrsgleichungen** λ = γ + Pᵀλ liefern die Raten (Gate γ, Kran und Stapel γ/(1 − r)); jede Station ist ein M/M/c (Erlang C), die Gesamtzeit folgt aus Little: Summe der mittleren Zahlen geteilt durch γ.
- **Zerlegung** (`qna` in `jn_formulas.py`): für beliebige Streuung der Dauer jede Station als G/G/c nach Allen-Cunneen (Erlang-C-Wartezeit mal (ca² + cs²)/2), die Streuung des Abgangsprozesses nach Whitt
  (cd² = 1 + (1 − ρ²)(ca² − 1) + ρ²(cs² − 1)/√c), Verzweigung c² = 1 + p(cd² − 1), Zusammenfluss mit den Raten gewichtet; die Streuungen der Ankunftsströme werden per Fixpunktiteration gelöst (mit Rückläufern ein Gleichungssystem).
- **Ketten** (`jn_chain.py`): (1) Kran und Stapel mit Rückläufern als Markov-Kette über alle (n₁, n₂), dünn besetztes Gleichgewicht πQ = 0, abgeschnitten bei einer Kantenlänge N mit ρ^N ≤ 10⁻⁸; (2) **Blockier-Kette**
  mit Zustand (n₁, n₂, b): b fertige Lkw bleiben am Kran stehen und halten ihn besetzt, wenn der Stapel seine Platzgrenze erreicht hat; ein Stapelabgang lässt einen blockierten Lkw nachrücken.
- **Simulation** (`jn_simulation.py`): Ereignisse über alle Stationen, FIFO, Dauer fest / Erlang-k / exponentiell / hyperexponentiell (gleiche Phasenanteile am Mittel), Routing nach der Matrix, optional Platzgrenzen mit
  **Blockieren nach Abfertigung** (nur ohne Rückläufer, sonst droht eine Verklemmung); SplitMix64 mit getrennten Strömen (Ankunft, Dauer je Station, Routing); die ersten 5 % der Abgänge und die zugehörige Zeit werden nicht gewertet.
- **Gegenproben:** (1) Verkehrsgleichungen, Jackson, Zerlegung (auch die Rückkopplungsgleichungen) und Ketten von Hand; (2) die Zweistationen-Kette ist das **Produkt** der M/M/c-Verteilungen, auch mit Rückläufern;
  (3) die Blockier-Kette von Hand (Dreizustands-Herleitung für fünf Zustände) und im Grenzfall eines riesigen Stapels gleich Jackson; (4) die Simulation von Hand verfolgt: Tandem, Blockieren, Rückläufer, Abweisung;
  (5) Simulation gegen Formel und gegen Kette; (6) Little über den Simulationspfad; (7) Burke gemessen gegen die Whitt-Gleichung; (8) die Gegenprobe der Gegenprobe: mit Platzgrenze ist die Verteilung **kein** Produkt mehr.
- **Vorgerechnete Studien** (`generate_precomputed.py` → `precomputed_sweep.json`, rund zweieinhalb Minuten parallel): Streuung der Dauer (24 / 33 / 36 Lkw/h × 0 / 20 % Rückläufer × cs² = 0, 0.25, 1, 4 an allen Stationen, je 4 Läufe à 300 000 Lkw)
  und Blockieren (30 / 36 / 42 Lkw/h, ohne Rückläufer, exponentiell × Platzgrenze 4, 5, 6, 8, 12, unbegrenzt, je 4 Läufe à 150 000 Lkw).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Zeiten in Minuten, Raten in Lkw je Stunde.

| Frage | Befund |
|---|---|
| Voreinstellungen? | Ausgangslage (33 Lkw/h, 20 % Rückläufer): Gesamtzeit **19.7 min** nach Jackson, der Kran ist zu 82 % ausgelastet und arbeitet für 41 statt 33 Lkw/h. Kran am Limit (36/h): 27.2 min bei 90 % (drei Lkw mehr je Stunde kosten **7.5 min**). Viele Rückläufer (24/h, 40 %): 22.9 min, Kran 80 % bei 40/h. Standardlauf der App (Seed 35, 60 000 Lkw): Simulation 19.4 min, −1.7 % gegenüber der Formel. |
| Wo liegt der Engpass, und wandert er? | Mit **zwei** Geräten am Kran ist der Kran immer der Engpass: das Netz ist bei **50 / 45 / 40 / 35 / 30 Lkw/h** voll (Rückläufer 0 / 10 / 20 / 30 / 40 %). Mit **drei** Geräten wandert er: bei 60 Lkw/h voll ohne Rückläufer (Gate gleichauf mit dem Stapel), mit Rückläufern der **Stapel** bei **54 / 48 / 42 / 36**. |
| Stimmt Jackson bei exponentieller Dauer? | Ja. In allen 6 Zellen der Streuungsstudie höchstens **0.8 %** Abweichung der Simulation von der Formel, im Mittel 0.2 %. Die Kette für Kran und Stapel (33 Lkw/h, 20 % Rückläufer, 96 × 96 = 9 216 Zustände) unterscheidet sich vom **Produkt** der Einzelverteilungen um unter 10⁻⁸; mittlere Zahl am Kran 5.166, im Stapel 3.651, gleich den Formelwerten. |
| Burke? | Eine Station mit einem Gerät bei 80 % Auslastung (Seed 35, 120 000 Lkw): Streuung der Abgangsabstände bei exponentieller Dauer **0.998** (Poisson = 1), keine Korrelation (−0.000). Bei fester Dauer **0.355** (Whitt-Gleichung 0.360, Korrelation +0.141), bei cs² = 4 **2.908** (2.920). Die Abgänge sind nur bei exponentieller Dauer Poisson. |
| Was kostet andere Streuung? | Die Jackson-Formel liegt neben der Simulation: bei fester Dauer **+7.2 bis +46.2 %**, bei cs² = 0.25 +4.9 bis +31.1 %, bei cs² = 4 **−46.4 bis −11.3 %** (24 Zellen mit je 4 Läufen). Beispiel 36 Lkw/h, 20 % Rückläufer: fest +46.2 %, cs² = 4 −46.4 %. |
| Hilft die Zerlegung? | Ja, aber sie ist nur eine Näherung. In allen **18** Zellen mit cs² ≠ 1 liegt sie näher an der Simulation als die Formel (mindestens 2.1-fach). Abweichung bei fester Dauer −8.0 bis +2.9 %, bei cs² = 4 **+4.0 bis +13.3 %**; über alle 24 Zellen im Mittel 3.1 %, ohne Rückläufer höchstens 4.9 %. Beispiel 33 Lkw/h, 20 % Rückläufer: fest Jackson 19.68 / Zerlegung 14.39 / Simulation 14.72 min (+33.7 / −2.3 %), cs² = 4: 19.68 / 35.55 / 32.29 (−39.0 / +10.1 %). |
| Was passiert, wenn der Stapel voll ist? | Es gibt keine Formel. Die **exakte Blockier-Kette** gegen Jackson, Platzgrenze 4 / 5 / 6 / 8 / 12 (Geräte und Warteplätze): bei 30 Lkw/h **+3.0 / +1.1 / +0.5 / +0.1 / +0.0 %**, bei 36 Lkw/h **+14.4 / +6.1 / +2.9 / +0.8 / +0.1 %**, bei 42 Lkw/h **+278 / +53 / +24 / +7 / +1 %**. Beispiel 36 Lkw/h, Platzgrenze 4: 15.55 statt 13.59 min, am Kran im Mittel 4.6 statt 3.0 Lkw. Bei 42 Lkw/h: 68.9 statt 18.2 min, am Kran 42.2 statt 5.7 Lkw: der Stau läuft in den Kran zurück. |
| Stimmt die Blockier-Kette? | Kette und Simulation weichen bei 30 und 36 Lkw/h in allen Zellen um höchstens **1.3 %** voneinander ab, bei 42 Lkw/h um höchstens 5.9 % (Platz 4: Standardfehler ±5.6 min, die Läufe streuen dort stark; die Simulation ohne Grenze liegt bei 42/h um 1.5 % neben Jackson, das zeigt die Rauschgrenze). |

## Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Der Engpass wandert nur, wenn man baut.** Die Vorab-Messreihe sprach von „Engpass wandert“; mit den festen Stationen bleibt der Kran bei jedem Rückläuferanteil der Engpass, nur die Reserve schrumpft (50 bis 30 Lkw/h). Die Demo
  zeigt das Wandern darum mit einem eigenen Regler (zwei oder drei Geräte am Kran).
- **Die Zerlegung ist mit Rückläufern schlechter als im Tandem.** Die Vorab-Messreihe (Tandem, 33 Lkw/h) zeigte höchstens 2.4 % Fehler; die Studie mit Rückläufern und höherer Last zeigt bis **13.3 %** (cs² = 4, 36 Lkw/h, 20 %).
  Die README nennt deshalb die Studienwerte, nicht die der Messreihe.
- **Blockieren ist erst bei höherer Last ein Befund.** Bei 33 Lkw/h (Vorab-Messreihe, 0.55 je Minute) kostete eine Platzgrenze von 6 nur 1 %; die Studie nimmt 30 / 36 / 42 Lkw/h und findet bei 42/h bis zu +278 %.
- **Die Standardfehler der Studie sind klein, aber nicht der ganze Fehler.** Vier Läufe schätzen die Streuung der Mittel nur grob, bei fester Dauer ist sie wegen der geringen Streuung kaum sichtbar; die Abweichungen im Text sind deshalb Mittelwerte
  über Läufe, keine Konfidenzaussagen.

## Ehrliche Grenzen

- **Poisson-Ankünfte von außen und feste Wege nach festen Wahrscheinlichkeiten.** Bei Schüben oder zustandsabhängigem Routing gilt weder Burke noch die Produktform (zeitvariable Ankünfte: Stück 6).
- **Eine Klasse von Lkw.** Mit Prioritäten gibt es je Klasse eigene Verkehrsgleichungen (Stück 11).
- **Die Zerlegung ist eine Näherung** ohne Fehlerschranke: hier bis 13.3 % daneben, bei höherer Last und stärkerer Streuung wohl mehr, nicht gemessen.
- **Blockieren nur für Kran und Stapel ohne Rückläufer.** Platzgrenzen mit Rückläufern können verklemmen (alle Plätze voll, der blockierte Lkw braucht den Kran, den er selbst besetzt); die Simulation lehnt das ab.
  Das Gate hat unbegrenzten Warteraum; Abweisung statt Blockieren ist Erlang B (Stück 8).
- **Die Blockier-Kette** bildet nur das Paar Kran und Stapel ab, mit Poisson-Zufluss (nach Burke gilt das, weil das Gate vorgelagert ist und nicht zurückwirkt); sie ist bei 400 Lkw am Kran abgeschnitten (Wahrscheinlichkeit am Rand bei 36 Lkw/h unter 10⁻³⁸, bei 30 Lkw/h unter 10⁻⁷⁵, bei 42 Lkw/h höchstens 1.5·10⁻⁶ bei Platz 4: das Abschneiden spielt keine Rolle).
- Ein Live-Lauf hat 60 000 Lkw und streut bei hoher Auslastung um mehrere Prozent. Die Studien nutzen vier Läufe.
- Dauer an allen Stationen mit gleicher Streuung cs²; unterschiedliche Streuung je Station kann `qna` (Test), die App bietet sie nicht an.

## Verwandte Demos im Portfolio

- [`mmc-queue-demo`](https://github.com/sebastian-hanisch/mmc-queue-demo) (Stück 3): jede Station des Netzes ist ein M/M/c.
- [`mg1-kingman-demo`](https://github.com/sebastian-hanisch/mg1-kingman-demo) (Stück 10): Kingman und Allen-Cunneen, auf denen die Zerlegung beruht.
- [`markov-queue-demo`](https://github.com/sebastian-hanisch/markov-queue-demo) (Zusatzstück): die Kette hinter den Formeln; hier wird sie zweidimensional und mit Blockieren dreidimensional.
- [`erlang-b-demo`](https://github.com/sebastian-hanisch/erlang-b-demo) (Stück 8): Verlust statt Blockieren.
- [`priority-queue-demo`](https://github.com/sebastian-hanisch/priority-queue-demo) (Stück 11): zwei Klassen, ebenfalls eine exakte Kette als Referenz.
- [`truck-appointment-demo`](https://github.com/sebastian-hanisch/truck-appointment-demo) und [`berth-allocation-demo`](https://github.com/sebastian-hanisch/berth-allocation-demo): Terminvergabe und Liegeplätze am Hafen.

## Bewusst nicht umgesetzt

Jede dieser Annahmen hebt ein Folgestück der Linie auf:

| Annahme | Folgestück |
|---|---|
| Exponentielle Dauer an allen Stationen | [M/G/1, Kingman-Näherung](https://github.com/sebastian-hanisch/mg1-kingman-demo) |
| Unbegrenzte Warteräume | [Erlang B](https://github.com/sebastian-hanisch/erlang-b-demo) |
| Poisson-Ankünfte von außen | [Zeitvariable Ankünfte](https://github.com/sebastian-hanisch/time-varying-arrivals-demo) |
| Eine Klasse von Lkw | [Prioritätsklassen](https://github.com/sebastian-hanisch/priority-queue-demo) |
| Rechenaufwand der Simulation | Surrogat-Modelle (Folgestück) |

Kein Folgestück: zustandsabhängiges Routing, Blockieren mit Rückläufern, mehr als drei Stationen.

## Tests

139 Tests, rund 75 Sekunden: Verkehrsgleichungen, Jackson, Zerlegung (Whitt, Allen-Cunneen, Rückkopplung von Hand) und Fälle ohne Gleichgewicht, die Zweistationen-Kette von Hand und als Produkt, die Blockier-Kette (von Hand, Grenzfall, Monotonie, Invarianten),
die Simulation (Mini-Instanzen von Hand für Tandem, Blockieren, Rückläufer und Abweisung, getrennte Ströme, Reproduzierbarkeit, Simulation gegen Formel, Little, Burke), Auswertung und Vollständigkeit der vorgerechneten Datei, Presets und Permalink,
Diagramme (gesperrte Achsen), AppTest-Rauchtests mit festem Würfel-Seed, der Smoke-Test der Portfolio-Vorlage, ein Quelltext-Test gegen Satz-Komma-Fehler und `test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `jn_formulas.py` | Verkehrsgleichungen, Erlang C, Jackson, Whitt-Gleichung, Zerlegung |
| `jn_chain.py` | Zweistationen-Kette (Produktform), Blockier-Kette |
| `jn_simulation.py` | Netz-Simulation mit Routing und Blockieren |
| `jn_evaluation.py` | Live-Bericht, Sweeps, Berichte, Zugriff auf die vorgerechnete Studie |
| `generate_precomputed.py` | rechnet die Studien vor → `precomputed_sweep.json` |
| `jn_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `jn_presets.py`, `jn_constants.py` | Presets, Permalink, Grenzen |
| `tests/` | siehe oben |

## Literatur

- Jackson, J. R. (1957): Networks of waiting lines. *Operations Research* 5(4), 518–521 (Produktform offener Netze).
- Burke, P. J. (1956): The output of a queuing system. *Operations Research* 4(6), 699–704 (Abgänge einer M/M/c-Station sind Poisson).
- Whitt, W. (1983): The queueing network analyzer. *The Bell System Technical Journal* 62(9), 2779–2815 (Zerlegungsnäherung für Netze mit beliebiger Streuung).
- Die Näherung von Allen und Cunneen für G/G/c ist hier als Standardform verwendet; eine Quelle ist nicht einzeln belegt, die Näherung ist für c = 1 durch die Pollaczek-Khinchine-Formel im Test geprüft.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`. Studien neu rechnen: `python generate_precomputed.py`.

Gebaut mit Streamlit und Plotly (die Ketten löst scipy).
