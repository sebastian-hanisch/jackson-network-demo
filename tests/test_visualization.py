"""Abbildungen: gesperrte Achsen, Zahl der Linien, Beschriftungen."""

import numpy as np

import jn_constants as C
import jn_evaluation as E
import jn_formulas as F
import jn_visualization as V

PRE = E.load_precomputed()


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def test_all_charts_lock_their_axes():
    j = F.jackson([0.4, 0, 0], F.routing_matrix(0.2), C.SERVERS, C.MEANS)
    rho, total = E.sweep_gamma(20, [12, 30, 60])
    prod = E.product_form_report(24, 20)
    cells = [E.study_cell(PRE, 33, 20, x) for x in C.STUDY_CS2]
    burke = [{"cs2": 1.0, "scv": 1.0, "linking": 1.0, "lag1": 0.0}]
    figs = [V.build_network_diagram(j["lam"], j["rho"], 0.2), V.build_station_chart(j["L"], j["L"], j["L"]), V.build_rho_sweep_chart([12, 30, 60], rho, 33, 40.0),
            V.build_total_sweep_chart([12, 30, 60], total, 33), V.build_product_chart(prod["pi"], prod["prod"]), V.build_burke_chart(burke), V.build_cs2_chart(cells),
            V.build_blocking_chart([4, 6, None], 13.6, [15.0, 14.0, 13.6], [15.1, 14.0, 13.6])]
    assert all(_locked(f) for f in figs)


def test_rho_color_traffic_light():
    assert V.rho_color(0.5) == V.LOW_COLOR and V.rho_color(0.7) == V.MID_COLOR and V.rho_color(0.89) == V.MID_COLOR and V.rho_color(0.9) == V.HIGH_COLOR


def test_network_diagram_shows_the_feedback_arrow_only_with_feedback():
    j0 = F.jackson([0.4, 0, 0], F.routing_matrix(0.0), C.SERVERS, C.MEANS)
    j2 = F.jackson([0.4, 0, 0], F.routing_matrix(0.2), C.SERVERS, C.MEANS)
    texts0 = [a.text for a in V.build_network_diagram(j0["lam"], j0["rho"], 0.0).layout.annotations if a.text]
    texts2 = [a.text for a in V.build_network_diagram(j2["lam"], j2["rho"], 0.2).layout.annotations if a.text]
    assert not any("Umstapeln" in t for t in texts0) and any("Umstapeln: 6.0/h" in t for t in texts2)
    assert "24.0/h" in texts2 and "30.0/h" in texts2                                          # Gate-Rate und Rate am Kran (24/0.8)


def test_network_diagram_colours_the_boxes_by_load():
    j = F.jackson([0.62, 0, 0], F.routing_matrix(0.2), C.SERVERS, C.MEANS)                 # ρ = 0.62 / 0.93 / 0.775
    colors = V.build_network_diagram(j["lam"], j["rho"], 0.2).data[0].marker.color
    assert list(colors) == [V.LOW_COLOR, V.HIGH_COLOR, V.MID_COLOR]


def test_station_chart_has_three_bar_groups():
    assert [t.name for t in V.build_station_chart([1, 2, 3], [1, 2, 3], [1, 2, 3]).data] == ["Formel (Jackson)", "Zerlegung (Whitt)", "Simulation"]


def test_total_sweep_chart_leaves_a_gap_where_the_network_overflows():
    fig = V.build_total_sweep_chart([12, 30, 60], [10.0, 20.0, float("inf")], 30)
    assert fig.data[0].y == (10.0, 20.0, None)


def test_product_chart_uses_one_colour_scale_for_both_heatmaps():
    pi = np.full((30, 30), 0.001)
    fig = V.build_product_chart(pi, pi * 0.5)
    assert fig.data[0].zmax == fig.data[1].zmax == 0.001 and fig.data[0].z.shape == (22, 22)


def test_blocking_chart_labels_the_unlimited_stack():
    fig = V.build_blocking_chart([4, 12, None], 13.6, [15, 13.7, 13.6], [15, 13.7, 13.6])
    assert list(fig.data[1].x) == ["4", "12", "unbegrenzt"] and V.cap_label(None) == "unbegrenzt" and V.cap_label(7) == "7"


def test_cs2_chart_has_error_bars_on_the_simulation_only():
    fig = V.build_cs2_chart([E.study_cell(PRE, 33, 20, x) for x in C.STUDY_CS2])
    assert [t.name for t in fig.data] == ["Formel (Jackson)", "Zerlegung (Whitt)", "Simulation"] and fig.data[2].error_y.array is not None and fig.data[0].error_y.array is None
