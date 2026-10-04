"""Jackson-Netze – Gate, Kran, Stapel - Stück 12 der Konzepte-Linie "Warteschlangentheorie und Simulation"
Sebastian Hanisch - Operations Research und Machine Learning

Ein Lkw durchläuft Gate, Kran und Stapel, ein Teil muss vom Stapel zurück zum Kran (Umstapeln). Die Demo rechnet das Netz nach Jackson (Verkehrsgleichungen, Produktform), zeigt wo der Engpass liegt und wandert,
prüft die Produktform an einer exakten Kette, zeigt Burke und die Weitergabe der Streuung, die Zerlegungsnäherung bei nicht exponentieller Dauer und das Blockieren bei vollem Stapel. Siehe README.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import jn_constants as C
import jn_evaluation as E
import jn_formulas as F
from jn_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params)
from jn_visualization import (build_blocking_chart, build_burke_chart, build_cs2_chart, build_network_diagram, build_product_chart, build_rho_sweep_chart, build_station_chart,
                              build_total_sweep_chart, cap_label)

st.set_page_config(page_title="Jackson-Netze – Sebastian Hanisch", layout="wide")

PRE = E.load_precomputed()
SWEEP_GAMMAS = list(range(12, 61, 3))


@st.cache_data(show_spinner=False)
def _live(gamma_ph, r_pct, cs2, seed):
    return E.live_report(gamma_ph, r_pct, cs2, seed)


@st.cache_data(show_spinner=False)
def _sweep(r_pct, n_crane):
    return E.sweep_gamma(r_pct, SWEEP_GAMMAS, E.crane_servers(n_crane))


@st.cache_data(show_spinner=False)
def _saturation_table(n_crane):
    rows = []
    for r in range(C.R_PCT_MIN, C.R_PCT_MAX + 1, C.R_PCT_STEP):
        sat, idx = E.saturation_per_hour(r, E.crane_servers(n_crane))
        rows.append((r, sat, C.NAMES[idx]))
    return rows


@st.cache_data(show_spinner=False)
def _product(gamma_ph, r_pct):
    return E.product_form_report(gamma_ph, r_pct)


@st.cache_data(show_spinner=False)
def _burke(seed):
    return [E.burke_report(cs2, seed) for cs2 in C.BURKE_CS2]


@st.cache_data(show_spinner=False)
def _blocking(gamma_ph):
    return {cap: E.blocking_report(gamma_ph, cap) for cap in (*C.BLOCK_CAPS, None)}


st.title("🏗️ Jackson-Netze – Gate, Kran, Stapel")
st.markdown(
    """
Bisher war das Terminal **eine** Station. In Wirklichkeit durchläuft ein Lkw das **Gate**, dann den **Kran**, dann den **Stapel**, und ein Teil muss vom Stapel zurück zum Kran (Umstapeln). Hier steckt ein
Satz, der Netze überraschend einfach macht: Sind die Ankünfte Poisson und die Dauern exponentiell, dann **rechnet jede Station für sich wie ein M/M/c** (mit der Rate aus den Verkehrsgleichungen), und die gemeinsame
Verteilung ist das **Produkt** der Einzelverteilungen (Jackson). Diese Demo zeigt, wie weit das trägt: wo der **Engpass** liegt und wandert, was **Burke** über die Abgänge sagt, wie die **Streuung** durch das Netz
wandert und was passiert, wenn der **Stapel voll** ist und es keine Formel mehr gibt.
"""
)
st.caption(
    "Stück 12 der Linie „Warteschlangentheorie und Simulation“, baut auf [mmc-queue-demo](https://sebastianhanisch-mmc-queue-demo.streamlit.app/) (Stück 3: eine Station), "
    "[mg1-kingman-demo](https://sebastianhanisch-mg1-kingman-demo.streamlit.app/) (Stück 10: Streuung) und [markov-queue-demo](https://sebastianhanisch-markov-queue-demo.streamlit.app/) (Ketten) auf. "
    "Jedes Folgestück hebt eine der Annahmen unter „Wo die Annahmen enden“ auf."
)

with st.expander("So wird aus drei Stationen ein Netz", expanded=True):
    st.markdown(
        """
- **Die Stationen:** **Gate** (3 Spuren, 3 min), **Kran** (2 Geräte, 2.4 min), **Stapel** (4 Geräte, 4 min). Ankünfte von außen nur am Gate. Vom Stapel gehen r der Lkw zurück zum Kran, die übrigen verlassen das Terminal.
- **Verkehrsgleichungen:** Die Rate λ an jeder Station ist die Rate von außen plus alles, was von anderen Stationen kommt: λ_Gate = γ, λ_Kran = λ_Gate + r·λ_Stapel, λ_Stapel = λ_Kran. Mit Rückläufern
  arbeitet der Kran also nicht für γ, sondern für γ/(1 − r) Lkw.
- **Jackson-Satz:** Ist jede Station stabil (λ·Dauer/Geräte < 1), dann ist die Zahl der Lkw an den Stationen **unabhängig**, jede wie ein M/M/c. Die Gesamtzeit eines Lkw folgt aus Little: Summe der mittleren Zahlen geteilt durch γ.
- **Engpass:** Alle Raten wachsen mit γ im gleichen Verhältnis; die Station mit der höchsten Auslastung läuft zuerst über, und welche das ist, hängt vom Rücklaufanteil ab.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    gamma_ph = st.slider("Ankünfte (Lkw je Stunde)", *bounds("gamma_slider"), step=C.GAMMA_STEP, key="gamma_slider", help="Poisson-Ankünfte von außen am Gate.")
    r_pct = st.slider("Rückläufer vom Stapel zum Kran", *bounds("r_slider"), step=C.R_PCT_STEP, key="r_slider", format="%d %%", help="Anteil der Lkw, die nach dem Stapel noch einmal zum Kran müssen (Umstapeln).")
    cs2 = st.select_slider("Streuung der Dauer (cs²)", options=C.CS2_OPTIONS, key="cs2_select", format_func=lambda v: f"{v:g}",
                           help="Variationskoeffizient² der Dauer an allen Stationen: 0 fest, 1 exponentiell (Jackson), 4 stark streuend.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1, key="seed_input", help="Bestimmt alle Zufallszahlen der Simulationen.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

gamma_ph, r_pct, cs2, seed = int(gamma_ph), int(r_pct), float(cs2), int(seed)
r = r_pct / 100.0
sync_query_params({"gamma_slider": gamma_ph, "r_slider": r_pct, "cs2_select": cs2, "seed_input": seed})

with st.spinner(f"Simuliere {C.fmt_int(C.LIVE_TRUCKS)} Lkw durch das Netz …"):
    rep = _live(gamma_ph, r_pct, cs2, seed)
jk, qn, sim = rep["jackson"], rep["qna"], rep["sim"]
worst = F.bottleneck(jk["rho"])

st.markdown("---")
st.markdown("## 🔗 Vom Gate zum Netz")
st.caption(f"{gamma_ph} Lkw je Stunde, {r_pct} % Rückläufer, Streuung der Dauer cs² = {cs2:g}. Farbe der Kästen: Auslastung (grün unter 70 %, gelb bis 90 %, rot darüber).")
st.plotly_chart(build_network_diagram(jk["lam"], jk["rho"], r), width="stretch", key=f"diagram_{gamma_ph}_{r_pct}")
st.markdown(
    "**Verkehrsgleichungen** (je Stunde): "
    f"λ_Gate = γ = {jk['lam'][0] * 60:.1f}; λ_Kran = λ_Gate + {r:g}·λ_Stapel = {jk['lam'][1] * 60:.1f} (= γ/(1 − {r:g}) = {gamma_ph / (1 - r):.1f}); λ_Stapel = λ_Kran = {jk['lam'][2] * 60:.1f}."
)
if not rep["stable"]:
    over = [f"{C.NAMES[i]} (Auslastung {C.fmt_pct(jk['rho'][i])})" for i in range(3) if jk["rho"][i] >= 1]
    st.warning(
        f"**Das Netz läuft über:** {', '.join(over)}. Keine Station darf eine Auslastung von 100 % erreichen; ohne Gleichgewicht gibt es weder eine Formel noch einen sinnvollen Simulationslauf. "
        "Weniger Ankünfte oder weniger Rückläufer wählen."
    )
else:
    c1 = st.columns(4)
    c1[0].metric("Gesamtzeit (Formel, Jackson)", f"{jk['total']:.1f} min", help="Summe der mittleren Zahlen an den Stationen geteilt durch γ (Little); gilt bei exponentieller Dauer.")
    c1[1].metric("Gesamtzeit (Zerlegung)", f"{qn['total']:.1f} min", help="Näherung nach Whitt für beliebige Streuung der Dauer.")
    c1[2].metric("Gesamtzeit (Simulation)", f"{sim.mean_sojourn:.1f} min", help=f"Ein Lauf mit {C.fmt_int(C.LIVE_TRUCKS)} Lkw, die ersten 5 % nicht gewertet.")
    c1[3].metric("Engpass", f"{C.NAMES[worst]} ({C.fmt_pct(jk['rho'][worst])})", help="Station mit der höchsten Auslastung.")
    rows = []
    for i in range(3):
        rows.append(f"| {C.NAMES[i]} | {C.SERVERS[i]} | {C.MEANS[i]:g} | {jk['lam'][i] * 60:.1f} | {C.fmt_pct(jk['rho'][i])} | {jk['wq'][i]:.1f} | {jk['L'][i]:.2f} | {qn['L'][i]:.2f} | {sim.L[i]:.2f} |")
    st.markdown("| Station | Geräte | Dauer (min) | Rate λ (Lkw/h) | Auslastung | Wartezeit (min) | L Formel | L Zerlegung | L Simulation |\n|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    st.plotly_chart(build_station_chart(jk["L"], qn["L"], sim.L), width="stretch", key=f"stations_{gamma_ph}_{r_pct}_{cs2}_{seed}")
    dev = sim.mean_sojourn / jk["total"] - 1
    dev_q = sim.mean_sojourn / qn["total"] - 1
    if cs2 == 1.0:
        st.info(f"Bei exponentieller Dauer gilt Jackson: Formel {jk['total']:.1f} min gegen Simulation {sim.mean_sojourn:.1f} min ({C.fmt_signed_pct(dev, 1)}, der Rest ist Rauschen eines Laufs). Der Kran ist mit {C.fmt_pct(jk['rho'][1])} ausgelastet.")
    else:
        st.info(f"Bei Streuung cs² = {cs2:g} gilt die Produktform nicht mehr: die Jackson-Formel ({jk['total']:.1f} min) liegt {C.fmt_signed_pct(-dev / (1 + dev), 0)} neben der Simulation ({sim.mean_sojourn:.1f} min); "
                f"die Zerlegung ({qn['total']:.1f} min) weicht um {C.fmt_signed_pct(-dev_q / (1 + dev_q), 1)} ab (ein Lauf, rauscht bei hoher Auslastung um mehrere Prozent).")

st.markdown("---")
st.subheader("📐 Der Engpass wandert")
n_crane = st.select_slider("Geräte am Kran", options=(2, 3), value=2, key="crane_select", help="Nur für diesen Abschnitt: Ausbau des Krans von 2 auf 3 Geräte.")
st.markdown(
    "Die Raten wachsen alle mit γ, aber nicht alle gleich schnell: Rückläufer erhöhen die Last von Kran **und** Stapel. Die Kurven zeigen die Auslastung je Station über die Ankünfte; wo die erste Linie 100 % erreicht, "
    "ist das Netz voll, und die Gesamtzeit läuft gegen unendlich. Mit zwei Geräten am Kran bleibt der Kran der Engpass; wird er auf drei Geräte ausgebaut, **wandert** der Engpass (zum Stapel oder zum Gate)."
)
sat, sat_idx = E.saturation_per_hour(r_pct, E.crane_servers(n_crane))
rho_sweep, total_sweep = _sweep(r_pct, n_crane)
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(build_rho_sweep_chart(SWEEP_GAMMAS, rho_sweep, gamma_ph, sat), width="stretch", key=f"rho_sweep_{r_pct}_{gamma_ph}_{n_crane}")
with col_b:
    st.plotly_chart(build_total_sweep_chart(SWEEP_GAMMAS, total_sweep, gamma_ph), width="stretch", key=f"total_sweep_{r_pct}_{gamma_ph}_{n_crane}")
st.markdown(
    "| Rückläufer | Netz voll bei (Lkw/h) | Engpass-Station |\n|---|---|---|\n" + "\n".join(f"| {rr} % | {s_:.1f} | {name} |" for rr, s_, name in _saturation_table(n_crane))
)
st.info(
    f"Mit {n_crane} Geräten am Kran und {r_pct} % Rückläufern ist das Netz bei {sat:.1f} Lkw je Stunde voll; zuerst läuft der {C.NAMES[sat_idx]} über. Jeder Prozentpunkt Rückläufer schiebt die Grenze nach unten, "
    "weil Kran und Stapel mehr Arbeit je Ankunft haben."
)

st.markdown("---")
st.subheader("🔬 Produktform: Kette gegen Produkt")
prod = _product(gamma_ph, r_pct)
st.markdown(
    "Der Jackson-Satz behauptet, dass die gemeinsame Verteilung von Kran und Stapel das **Produkt** der beiden M/M/c-Verteilungen ist, **auch mit Rückläufern**. Hier wird das geprüft: links die exakte Markov-Kette über alle "
    "Zustände (n₁, n₂) (Gleichgewicht πQ = 0 als dünn besetztes lineares System, abgeschnitten bei einer Kantenlänge N), rechts das Produkt. Die Kette weiß nichts von Jackson."
)
if prod is None:
    st.warning("Bei dieser Einstellung läuft Kran oder Stapel über; ohne Gleichgewicht gibt es keine Verteilung zum Vergleichen.")
else:
    st.plotly_chart(build_product_chart(prod["pi"], prod["prod"]), width="stretch", key=f"product_{gamma_ph}_{r_pct}")
    p1 = st.columns(4)
    p1[0].metric("Zustände der Kette", C.fmt_int(prod["size"] ** 2), help=f"{prod['size']} × {prod['size']}")
    p1[1].metric("Größter Unterschied", f"{prod['max_diff']:.0e}", help="größter Betrag der Differenz zwischen Kette und Produkt über alle Zustände")
    p1[2].metric("Kran: mittlere Zahl", f"{prod['L1']:.3f}", delta=f"Formel {prod['jackson_L1']:.3f}", delta_color="off")
    p1[3].metric("Stapel: mittlere Zahl", f"{prod['L2']:.3f}", delta=f"Formel {prod['jackson_L2']:.3f}", delta_color="off")
    st.info(
        f"Die Kette ({C.fmt_int(prod['size'] ** 2)} Zustände) und das Produkt unterscheiden sich um höchstens {prod['max_diff']:.0e}; der Rest kommt vom Abschneiden bei N = {prod['size']} (Rest etwa {prod['tail']:.0e}). "
        "Die Kette musste dafür den ganzen Zustandsraum lösen, das Produkt kostet nichts: bei drei Stationen mit je 100 Zuständen wären es eine Million."
    )

st.markdown("---")
st.subheader("🔬 Burke und die Weitergabe der Streuung")
st.markdown(
    "Warum geht das Produkt auf? Nach **Burke** sind die Abgänge einer M/M/c-Station wieder ein Poisson-Strom: die nächste Station sieht dieselbe Art von Ankünften wie die erste. Gilt das auch bei anderer Dauer? "
    f"Eine Station mit einem Gerät bei {C.fmt_pct(C.BURKE_RHO)} Auslastung, {C.fmt_int(C.BURKE_TRUCKS)} Lkw: gemessen wird die Streuung der Zeitabstände zwischen den Abgängen (1 = Poisson) und die Korrelation "
    "aufeinanderfolgender Abstände."
)
with st.spinner("Simuliere die Abgangsprozesse …"):
    burke = _burke(seed)
st.plotly_chart(build_burke_chart(burke), width="stretch", key=f"burke_{seed}")
st.markdown(
    "| Dauer | Streuung der Abgangsabstände (gemessen) | Whitt-Gleichung | Korrelation benachbarter Abstände |\n|---|---|---|---|\n"
    + "\n".join(f"| cs² = {b['cs2']:g} | {b['scv']:.3f} | {b['linking']:.3f} | {b['lag1']:+.3f} |" for b in burke)
)
st.info(
    "Bei exponentieller Dauer ist der Abgangsprozess Poisson (Streuung 1, keine Korrelation). Bei fester Dauer sind die Abgänge **regelmäßiger** als Poisson, bei stark streuender Dauer **unregelmäßiger**: die Streuung "
    "wandert durch das Netz. Die Gleichung von Whitt, cd² = 1 + (1 − ρ²)(ca² − 1) + ρ²(cs² − 1)/√c, beschreibt das; sie ist Grundlage der Zerlegungsnäherung unten."
)

st.markdown("---")
st.subheader("🔬 Nicht exponentielle Dauer: Jackson, Zerlegung, Simulation")
col_s1, col_s2 = st.columns(2)
with col_s1:
    st_gamma = st.select_slider("Ankünfte (Studie)", options=C.STUDY_GAMMA, value=33, key="study_gamma", format_func=lambda v: f"{v} Lkw/h")
with col_s2:
    st_r = st.select_slider("Rückläufer (Studie)", options=C.STUDY_R_PCT, value=20, key="study_r", format_func=lambda v: f"{v} %")
cells = [E.study_cell(PRE, st_gamma, st_r, x) for x in C.STUDY_CS2]
st.markdown(
    f"Alle Stationen mit derselben Streuung cs² der Dauer. Die Produktform gilt nur bei cs² = 1; die **Zerlegung** rechnet jede Station als G/G/c (Allen-Cunneen, Kingman aus Stück 10) und gibt die Streuung der Abgänge "
    f"nach Whitt weiter. Vorgerechnete Studie: je {PRE['study_reps']} Läufe à {C.fmt_int(PRE['study_trucks'])} Lkw."
)
st.plotly_chart(build_cs2_chart(cells), width="stretch", key=f"cs2_{st_gamma}_{st_r}")
st.markdown(
    "| Streuung cs² | Formel (Jackson) | Zerlegung | Simulation (± Standardfehler) | Jackson gegen Simulation | Zerlegung gegen Simulation |\n|---|---|---|---|---|---|\n"
    + "\n".join(f"| {x['cs2']:g} | {x['jackson']:.2f} | {x['qna']:.2f} | {x['total']:.2f} ± {x['total_se']:.2f} | {C.fmt_signed_pct(x['jackson'] / x['total'] - 1, 1)} | {C.fmt_signed_pct(x['qna'] / x['total'] - 1, 1)} |" for x in cells)
)
worst_j = max(cells, key=lambda x: abs(x["jackson"] / x["total"] - 1))
worst_q = max(cells, key=lambda x: abs(x["qna"] / x["total"] - 1))
st.info(
    f"Bei dieser Einstellung liegt die Jackson-Formel um bis zu {C.fmt_pct(abs(worst_j['jackson'] / worst_j['total'] - 1))} daneben (cs² = {worst_j['cs2']:g}), die Zerlegung um höchstens "
    f"{C.fmt_pct(abs(worst_q['qna'] / worst_q['total'] - 1), 1)}. Die Zerlegung ist eine Näherung ohne Garantie, liegt hier aber deutlich näher an der Simulation als die Formel, die ihre Voraussetzung verletzt sieht."
)

st.markdown("---")
st.subheader("🔬 Stapel voll: Blockieren")
bl_gamma = st.select_slider("Ankünfte (Blockieren)", options=C.BLOCK_GAMMA, value=36, key="block_gamma", format_func=lambda v: f"{v} Lkw/h")
with st.spinner("Löse die Blockier-Ketten …"):
    blocks = _blocking(bl_gamma)
caps = [*C.BLOCK_CAPS, None]
sim_cells = [E.block_cell(PRE, bl_gamma, cap) for cap in caps]
chain_totals = [blocks[cap]["chain_total"] for cap in caps]
sim_totals = [x["total"] for x in sim_cells]
st.markdown(
    "Der Stapel hat nur begrenzt Platz (Geräte und Warteplätze zusammen). Ist er voll, bleibt ein fertiger Lkw **auf dem Kran stehen und hält ihn besetzt**, bis ein Platz frei wird: der Stau läuft in den Kran zurück. "
    "Dafür gibt es **keine Formel und keine Produktform**; die Referenz ist eine exakte Kette mit Zustand (am Kran, im Stapel, blockiert), die Simulation prüft sie. Ohne Rückläufer (sie könnten das Netz verklemmen), "
    f"exponentielle Dauer, {bl_gamma} Lkw je Stunde."
)
st.plotly_chart(build_blocking_chart(caps, blocks[None]["jackson_total"], chain_totals, sim_totals), width="stretch", key=f"blocking_{bl_gamma}")
st.markdown(
    "| Plätze im Stapel | Jackson (ohne Grenze) | exakte Kette | Simulation (± Standardfehler) | Kette gegen Jackson | am Kran im Mittel |\n|---|---|---|---|---|---|\n"
    + "\n".join(f"| {cap_label(cap)} | {blocks[cap]['jackson_total']:.1f} | {blocks[cap]['chain_total']:.1f} | {x['total']:.1f} ± {x['total_se']:.1f} | {C.fmt_signed_pct(blocks[cap]['chain_total'] / blocks[cap]['jackson_total'] - 1, 0)} | {blocks[cap]['L1']:.1f} |"
               for cap, x in zip(caps, sim_cells))
)
tight = blocks[C.BLOCK_CAPS[0]]
st.info(
    f"Bei nur {C.BLOCK_CAPS[0]} Plätzen (keine Warteplätze) dauert ein Lkw laut Kette {tight['chain_total']:.1f} min statt {tight['jackson_total']:.1f} min nach Jackson ({C.fmt_signed_pct(tight['chain_total'] / tight['jackson_total'] - 1, 0)}); "
    f"am Kran stehen im Mittel {tight['L1']:.1f} Lkw statt {blocks[None]['L1']:.1f}. Die Kette und die Simulation stimmen überein, die Formel liegt daneben: sie kennt die Rückwirkung des Stapels auf den Kran nicht."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Exponentielle Dauer an allen Stationen** | Die Produktform ist weg; die Formel kann um fast die Hälfte daneben liegen. Die Zerlegung nach Whitt rechnet jede Station als G/G/c, ist aber nur eine Näherung. | **[M/G/1, Kingman-Näherung](https://sebastianhanisch-mg1-kingman-demo.streamlit.app/)** (eine Station) |
| **Unbegrenzte Warteräume** | Ist ein Platz knapp, stauen sich Lkw rückwärts durch das Netz (Blockieren); es gibt keine Formel, nur eine exakte Kette für das Paar Kran und Stapel. Mit Abweisung statt Blockieren gilt Erlang B. | **[M/M/c/c (Erlang B)](https://sebastianhanisch-erlang-b-demo.streamlit.app/)** |
| **Poisson-Ankünfte von außen** | Bei Schüben und Tagesgang ist der Zustrom nicht Poisson, und die Raten ändern sich mit der Zeit. | **[zeitvariable Ankünfte](https://sebastianhanisch-time-varying-arrivals-demo.streamlit.app/)** |
| **Eine Klasse von Lkw** | Eilige Lkw überholen an jeder Station; dann gibt es je Klasse eine eigene Verkehrsgleichung und die Produktform gilt nicht mehr. | **[Prioritätsklassen](https://sebastianhanisch-priority-queue-demo.streamlit.app/)** (eine Station) |
| **Feste Wege nach festen Wahrscheinlichkeiten** | Hier entscheidet jeder Lkw am Stapel unabhängig, ob er zurück muss; wer seine Route aus dem Zustand ableitet (kürzeste Schlange), bricht die Voraussetzung. | **[Power-of-d-Choices](https://sebastianhanisch-power-of-d-demo.streamlit.app/)** (Wahl zwischen Spuren) |
| **Rechenaufwand der Simulation** | Jede Variante kostet einen eigenen Lauf; für Entwurfsfragen über viele Netze lohnt ein schnelles Ersatzmodell. | **Surrogat-Modelle** (Folgestück) |
"""
)
st.caption(
    "Verwandt im Portfolio: [markov-queue-demo](https://sebastianhanisch-markov-queue-demo.streamlit.app/) (Zusatzstück: die Kette hinter den Formeln), [mmc-queue-demo](https://sebastianhanisch-mmc-queue-demo.streamlit.app/) "
    "(Stück 3: jede Station des Netzes), [mg1-kingman-demo](https://sebastianhanisch-mg1-kingman-demo.streamlit.app/) (Stück 10: Kingman, auf dem die Zerlegung beruht), "
    "[erlang-b-demo](https://sebastianhanisch-erlang-b-demo.streamlit.app/) (Stück 8: Verlust statt Blockieren) und die Hafen-Fall-Demos "
    "[berth-allocation-demo](https://sebastianhanisch-berth-allocation-demo.streamlit.app/) und [truck-appointment-demo](https://sebastianhanisch-truck-appointment-demo.streamlit.app/)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Verkehrsgleichungen.** $\lambda_j = \gamma_j + \sum_i \lambda_i P_{ij}$, also $\lambda = (I - P^\top)^{-1}\gamma$. Terminal: $\lambda_{\text{Gate}} = \gamma$, $\lambda_{\text{Kran}} = \gamma/(1-r)$, $\lambda_{\text{Stapel}} = \lambda_{\text{Kran}}$.

**Jackson.** Ist $\rho_j = \lambda_j m_j / c_j < 1$ für alle $j$, dann $\pi(n_1,\dots,n_J) = \prod_j \pi_j(n_j)$ mit $\pi_j$ der M/M/$c_j$-Verteilung der Rate $\lambda_j$. Gesamtzeit nach Little: $W = \sum_j L_j / \sum_j \gamma_j$.

**Burke.** Der Abgangsprozess einer stationären M/M/$c$-Station ist ein Poisson-Strom der Rate $\lambda$ und unabhängig vom späteren Zustand der Station.

**Zerlegung (Whitt).** Je Station G/G/$c$ mit $W_q \approx W_q^{\text{M/M/}c}\,(c_a^2 + c_s^2)/2$ (Allen-Cunneen); Abgangsstreuung $c_d^2 = 1 + (1-\rho^2)(c_a^2 - 1) + \rho^2 (c_s^2 - 1)/\sqrt{c}$;
Verzweigung $c^2 = 1 + p\,(c_d^2 - 1)$, Zusammenfluss gewichtet mit den Raten.

**Blockier-Kette.** Zustand $(n_1, n_2, b)$, $b$ blockierte Lkw am Kran ($b > 0$ nur bei $n_2 = K_2$); Kranabgang mit Rate $\min(n_1 - b, c_1 - b)\mu_1$ führt bei $n_2 < K_2$ in den Stapel, sonst zu $b + 1$; ein Stapelabgang lässt einen
blockierten Lkw nachrücken.

Implementiert in `jn_formulas.py` (Gleichungen, Zerlegung), `jn_chain.py` (Ketten), `jn_simulation.py` (Netz mit Blockieren).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html))."
)
