"""Plotly-Abbildungen der Jackson-Demo: Netzplan mit Auslastung, mittlere Zahl je Station, Engpass über die Ankünfte, Produktform, Burke, Streuung, Blockieren. Achsen sind gesperrt
(fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import jn_constants as C

JACKSON_COLOR = "#4c78a8"
QNA_COLOR = "#54a24b"
SIM_COLOR = "#f58518"
CHAIN_COLOR = "#b279a2"
BASE_COLOR = "#9d9d9d"
LOW_COLOR, MID_COLOR, HIGH_COLOR = "#7fbf7b", "#f0c05a", "#e45756"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def rho_color(rho):
    """Ampel der Auslastung: unter 70 % grün, bis 90 % gelb, darüber rot."""
    return LOW_COLOR if rho < 0.7 else (MID_COLOR if rho < 0.9 else HIGH_COLOR)


def build_network_diagram(lam, rho, r):
    """Netzplan Gate → Kran → Stapel: Kästen nach Auslastung eingefärbt, Pfeile mit Raten, Rücklauf vom Stapel zum Kran (Anteil r)."""
    fig = go.Figure()
    xs = [0.0, 1.0, 2.0]
    texts = [f"<b>{C.NAMES[i]}</b><br>{C.SERVERS[i]} Geräte<br>λ = {lam[i] * 60:.1f}/h<br>ρ = {C.fmt_pct(rho[i])}" for i in range(3)]
    fig.add_trace(go.Scatter(x=xs, y=[0, 0, 0], mode="markers+text", marker=dict(symbol="square", size=96, color=[rho_color(x) for x in rho], line=dict(color="white", width=2)),
                             text=texts, textfont=dict(size=11, color="black"), hoverinfo="skip", name="Station"))
    arrow = dict(xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=1.8)
    fig.add_annotation(x=-0.28, y=0, ax=-0.7, ay=0, arrowcolor=JACKSON_COLOR, **arrow)
    fig.add_annotation(x=-0.5, y=0.1, text=f"{lam[0] * 60:.1f}/h", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.5, y=0.1, text=f"{lam[0] * 60:.1f}/h", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.72, y=0, ax=0.28, ay=0, arrowcolor=JACKSON_COLOR, **arrow)
    fig.add_annotation(x=1.5, y=0.1, text=f"{lam[1] * 60:.1f}/h", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=1.72, y=0, ax=1.28, ay=0, arrowcolor=JACKSON_COLOR, **arrow)
    fig.add_annotation(x=2.7, y=0, ax=2.28, ay=0, arrowcolor=BASE_COLOR, **arrow)
    fig.add_annotation(x=2.5, y=0.1, text=f"fertig {(1 - r) * lam[2] * 60:.1f}/h", showarrow=False, font=dict(size=10))
    if r > 0:
        fig.add_annotation(x=1.0, y=0.33, ax=2.0, ay=0.33, arrowcolor=HIGH_COLOR, **arrow)
        fig.add_annotation(x=1.5, y=0.42, text=f"Umstapeln: {r * lam[2] * 60:.1f}/h", showarrow=False, font=dict(size=10, color=HIGH_COLOR))
    fig.update_xaxes(visible=False, range=[-0.8, 3.0])
    fig.update_yaxes(visible=False, range=[-0.3, 0.55])
    fig.update_layout(showlegend=False)
    return _base(fig, 240, top=0)


def build_station_chart(jackson_l, qna_l, sim_l):
    """Mittlere Zahl im System je Station: Formel (Jackson), Zerlegung und Simulation."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(C.NAMES), y=list(jackson_l), marker_color=JACKSON_COLOR, name="Formel (Jackson)"))
    fig.add_trace(go.Bar(x=list(C.NAMES), y=list(qna_l), marker_color=QNA_COLOR, name="Zerlegung (Whitt)"))
    fig.add_trace(go.Bar(x=list(C.NAMES), y=list(sim_l), marker_color=SIM_COLOR, name="Simulation"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="mittlere Zahl Lkw im System", rangemode="tozero")
    return _base(fig, 320)


def build_rho_sweep_chart(gammas, rho, gamma_now, sat):
    """Auslastung je Station über die Ankünfte (Lkw/h): wo die Linie 100 % erreicht, läuft das Netz über; senkrecht: gewählter Wert."""
    fig = go.Figure()
    colors = [JACKSON_COLOR, SIM_COLOR, QNA_COLOR]
    for i in range(3):
        fig.add_trace(go.Scatter(x=list(gammas), y=[float(v) for v in rho[:, i]], mode="lines", line=dict(color=colors[i], width=2.5), name=C.NAMES[i]))
    fig.add_hline(y=1.0, line=dict(color=HIGH_COLOR, dash="dash"), annotation_text="voll ausgelastet")
    fig.add_vline(x=gamma_now, line=dict(color=BASE_COLOR, dash="dot"), annotation_text="Auswahl")
    fig.add_vline(x=sat, line=dict(color=HIGH_COLOR, dash="dot"), annotation_text=f"Netz voll bei {sat:.1f}/h", annotation_position="top left")
    fig.update_xaxes(title_text="Ankünfte (Lkw je Stunde)")
    fig.update_yaxes(title_text="Auslastung der Station", tickformat=".0%", rangemode="tozero")
    return _base(fig, 320)


def build_total_sweep_chart(gammas, totals, gamma_now):
    """Gesamtzeit eines Lkw nach Jackson über die Ankünfte; wo das Netz überläuft, endet die Kurve."""
    ys = [t if math.isfinite(t) else None for t in totals]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(gammas), y=ys, mode="lines+markers", line=dict(color=JACKSON_COLOR, width=2.5), name="Gesamtzeit (Formel)"))
    fig.add_vline(x=gamma_now, line=dict(color=BASE_COLOR, dash="dot"), annotation_text="Auswahl")
    fig.update_xaxes(title_text="Ankünfte (Lkw je Stunde)")
    fig.update_yaxes(title_text="Gesamtzeit eines Lkw (min)", rangemode="tozero")
    return _base(fig, 300)


def build_product_chart(pi, prod, nmax=22):
    """Gemeinsame Verteilung (Kran n₁, Stapel n₂) aus der exakten Kette (links) und als Produkt der Einzelverteilungen (rechts), gleiche Farbskala."""
    nmax = min(nmax, pi.shape[0])
    zmax = float(max(pi[:nmax, :nmax].max(), prod[:nmax, :nmax].max()))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("exakte Kette", "Produkt der Einzelverteilungen"), horizontal_spacing=0.08)
    for col, z in ((1, pi), (2, prod)):
        fig.add_trace(go.Heatmap(z=z[:nmax, :nmax].T, zmin=0, zmax=zmax, colorscale="Blues", showscale=(col == 2), colorbar=dict(title="Wahrsch.")), row=1, col=col)
    fig.update_xaxes(title_text="Lkw am Kran n₁")
    fig.update_yaxes(title_text="Lkw im Stapel n₂", col=1)
    return _base(fig, 330, legend_y=-0.3, top=30)


def build_burke_chart(reports):
    """Streuung der Zeitabstände zwischen Abgängen einer Station (Variationskoeffizient²): gemessen und nach der Whitt-Gleichung; 1 = Poisson-Strom."""
    labels = [f"Dauer cs² = {r['cs2']:g}" for r in reports]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["scv"] for r in reports], marker_color=SIM_COLOR, name="gemessen"))
    fig.add_trace(go.Scatter(x=labels, y=[r["linking"] for r in reports], mode="markers", marker=dict(color=JACKSON_COLOR, size=12, symbol="diamond"), name="Whitt-Gleichung"))
    fig.add_hline(y=1.0, line=dict(color=BASE_COLOR, dash="dash"), annotation_text="Poisson (Burke)")
    fig.update_yaxes(title_text="Streuung der Abgangsabstände (cd²)", rangemode="tozero")
    return _base(fig, 320)


def build_cs2_chart(cells):
    """Gesamtzeit je Streuung der Dauer: Formel (Jackson), Zerlegung und Simulation (mit Standardfehler)."""
    labels = [f"cs² = {c['cs2']:g}" for c in cells]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[c["jackson"] for c in cells], marker_color=JACKSON_COLOR, name="Formel (Jackson)"))
    fig.add_trace(go.Bar(x=labels, y=[c["qna"] for c in cells], marker_color=QNA_COLOR, name="Zerlegung (Whitt)"))
    fig.add_trace(go.Bar(x=labels, y=[c["total"] for c in cells], error_y=dict(type="data", array=[c["total_se"] for c in cells]), marker_color=SIM_COLOR, name="Simulation"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Gesamtzeit eines Lkw (min)", rangemode="tozero")
    return _base(fig, 320)


def cap_label(cap):
    return "unbegrenzt" if cap is None else str(cap)


def build_blocking_chart(caps, jackson_total, chain_totals, sim_totals):
    """Gesamtzeit über die Platzgrenze des Stapels: Jackson-Formel (waagerecht, gilt nur ohne Grenze), exakte Kette und Simulation."""
    labels = [cap_label(c) for c in caps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=[jackson_total] * len(caps), mode="lines", line=dict(color=JACKSON_COLOR, dash="dash", width=2), name="Formel (Jackson, ohne Grenze)"))
    fig.add_trace(go.Scatter(x=labels, y=chain_totals, mode="lines+markers", line=dict(color=CHAIN_COLOR, width=2.5), name="exakte Kette"))
    fig.add_trace(go.Scatter(x=labels, y=sim_totals, mode="markers", marker=dict(color=SIM_COLOR, size=11, symbol="diamond"), name="Simulation"))
    fig.update_xaxes(title_text="Plätze im Stapel (Geräte + Warteplätze)", type="category")
    fig.update_yaxes(title_text="Gesamtzeit eines Lkw (min)", rangemode="tozero")
    return _base(fig, 320)
