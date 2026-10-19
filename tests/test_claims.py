"""Jede Zahl aus der README wird hier über dieselben Auswertungsfunktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Netze der Messreihen: die acht festen Netze `SWEEP_SEEDS` je Zelle, n = 250, c = 6, 10 % der Kanten versteckt, wo nichts anderes steht; Mediane. AUC mit leichten Negativen
(zufällige Nichtkanten), schweren Negativen (Abstand 2), Precision@k als erwartete Precision bei zufälliger Auflösung der Gleichstände. Zufall läuft über seedbare Generatoren
und ist plattformstabil; Kennzahlen werden mit Band geprüft."""

from functools import lru_cache

import networkx as nx
import numpy as np
import pytest

import lkp_algorithm as A
import lkp_constants as C
import lkp_evaluation as ev

SW = C.SWEEP_SEEDS
ORDER = (C.M_CN, C.M_JACCARD, C.M_AA, C.M_RA, C.M_PA, C.M_KATZ)


@lru_cache(maxsize=None)
def _types(c=6):
    return {r["typ"]: r for r in ev.types_rows(250, c, 10, seeds=SW)}


@lru_cache(maxsize=None)
def _link(nav):
    return ev.types_rows(0, 0, 10, seeds=SW, types=(C.T_LINK,), nav=nav)[0]


@lru_cache(maxsize=None)
def _katz():
    return {(r["typ"], r["c"]): r for r in ev.katz_rows(250, 10, seeds=SW)}


@lru_cache(maxsize=None)
def _community():
    return {(r["c"], r["inside"]): r for r in ev.community_rows(250, 10, seeds=SW)}


@lru_cache(maxsize=None)
def _time():
    return {r["mech"]: r for r in ev.time_rows(250, 6, seeds=SW)}


@lru_cache(maxsize=None)
def _leak():
    return {r["typ"]: r for r in ev.leak_rows(250, 6, 10, seeds=SW)}


def _check(row, key, expect, tol=0.03):
    for name, value in zip(ORDER, expect):
        assert row["scores"][name][key] == pytest.approx(value, abs=tol), (row["typ"], name, key)


def test_auc_with_easy_negatives():
    rows = _types()
    _check(rows[C.T_COMMUNITIES], "auc_easy", (0.62, 0.62, 0.62, 0.62, 0.51, 0.74))
    _check(rows[C.T_RANDOM], "auc_easy", (0.50, 0.50, 0.50, 0.50, 0.50, 0.53))
    _check(rows[C.T_POWER], "auc_easy", (0.67, 0.59, 0.68, 0.67, 0.73, 0.68))
    _check(rows[C.T_BA], "auc_easy", (0.58, 0.56, 0.58, 0.58, 0.66, 0.62))
    _check(rows[C.T_RING], "auc_easy", (0.95, 0.96, 0.95, 0.95, 0.18, 0.97))
    _check(rows[C.T_GRID], "auc_easy", (0.96, 0.97, 0.97, 0.97, 0.27, 0.98))
    _check(rows[C.T_HUBS], "auc_easy", (0.70, 0.68, 0.71, 0.71, 0.63, 0.66))
    for r in rows.values():
        assert r["scores"][C.M_RANDOM]["auc_easy"] == pytest.approx(0.5, abs=0.04)                 # Grundlinie


def test_auc_with_hard_negatives_is_below_chance_by_construction():
    rows = _types()
    _check(rows[C.T_COMMUNITIES], "auc_hard", (0.17, 0.19, 0.18, 0.18, 0.37, 0.20))
    _check(rows[C.T_RANDOM], "auc_hard", (0.05, 0.05, 0.05, 0.05, 0.37, 0.05))
    _check(rows[C.T_POWER], "auc_hard", (0.37, 0.09, 0.40, 0.39, 0.64, 0.42))
    _check(rows[C.T_BA], "auc_hard", (0.20, 0.09, 0.19, 0.19, 0.51, 0.23))
    _check(rows[C.T_RING], "auc_hard", (0.72, 0.82, 0.74, 0.74, 0.13, 0.69))
    _check(rows[C.T_GRID], "auc_hard", (0.60, 0.78, 0.62, 0.62, 0.18, 0.41))
    _check(rows[C.T_HUBS], "auc_hard", (0.41, 0.16, 0.44, 0.44, 0.52, 0.45))


def test_coverage_of_the_hidden_edges():
    rows = _types()
    expect = {C.T_COMMUNITIES: (0.31, 0.095), C.T_RANDOM: (0.11, 0.109), C.T_POWER: (0.53, 0.25), C.T_BA: (0.34, 0.177), C.T_RING: (0.92, 0.035), C.T_GRID: (0.96, 0.036), C.T_HUBS: (0.49, 0.113)}
    for typ, (hidden, nonedge) in expect.items():
        assert rows[typ]["coverage"] == pytest.approx(hidden, abs=0.03) and rows[typ]["nonedge_coverage"] == pytest.approx(nonedge, abs=0.02), typ
    thin = _types(3)
    for typ, value in {C.T_COMMUNITIES: 0.12, C.T_RANDOM: 0.03, C.T_POWER: 0.21, C.T_BA: 0.15, C.T_RING: 0.88, C.T_GRID: 0.0, C.T_HUBS: 0.49}.items():
        assert thin[typ]["coverage"] == pytest.approx(value, abs=0.03), typ


def test_precision_at_k_and_lift():
    rows = _types()
    lift = lambda typ, m: rows[typ]["scores"][m]["lift"]
    prec = lambda typ, m: rows[typ]["scores"][m]["precision"]
    assert lift(C.T_RANDOM, C.M_CN) == pytest.approx(1, abs=1.5) and max(prec(C.T_RANDOM, m) for m in ORDER) < 0.005
    assert lift(C.T_POWER, C.M_CN) == pytest.approx(41, abs=6) and lift(C.T_POWER, C.M_KATZ) == pytest.approx(47, abs=7) and prec(C.T_POWER, C.M_CN) == pytest.approx(0.081, abs=0.015)
    assert lift(C.T_COMMUNITIES, C.M_CN) == pytest.approx(5, abs=2) and prec(C.T_COMMUNITIES, C.M_CN) == pytest.approx(0.012, abs=0.006)
    assert lift(C.T_RING, C.M_CN) == pytest.approx(103, abs=12) and lift(C.T_RING, C.M_JACCARD) == pytest.approx(163, abs=18) and prec(C.T_RING, C.M_JACCARD) == pytest.approx(0.417, abs=0.04)
    assert lift(C.T_GRID, C.M_JACCARD) == pytest.approx(136, abs=15) and prec(C.T_GRID, C.M_JACCARD) == pytest.approx(0.298, abs=0.03)
    assert lift(C.T_HUBS, C.M_CN) == pytest.approx(68, abs=10)
    for typ in (C.T_RANDOM, C.T_POWER, C.T_BA, C.T_HUBS):
        assert lift(typ, C.M_JACCARD) <= 1                                                         # Jaccard bevorzugt Paare mit kleinem Grad: in vier Netztypen kein Lift
    assert lift(C.T_COMMUNITIES, C.M_JACCARD) == pytest.approx(5, abs=2)
    assert all(prec(typ, C.M_PA) < 0.01 for typ in (C.T_COMMUNITIES, C.T_RING, C.T_GRID, C.T_HUBS)) and prec(C.T_GRID, C.M_KATZ) == 0.0


def test_verdicts_of_experiments_1_and_2():
    rows = list(_types().values())
    kind, text = ev.verdict_types(rows)
    assert "mit leichten Negativen 0,50 bis 0,96 (am besten in „Gitter“, dort haben 96 % der versteckten Kanten einen gemeinsamen Nachbarn; am wenigsten in „Zufallsnetz“ mit 11 %)" in text
    assert "In „Zufallsnetz“ ist mit keinem Maß etwas zu finden" in text and "auf 0,05 bis 0,72" in text
    kind, text = ev.verdict_precision(rows)
    assert "Der größte Lift ist 163 („Jaccard“ in „Ring mit Abkürzungen“, Precision 0,417)" in text and "nur 0,002" in text and "In „Zufallsnetz“ liegt keine Precision über 0,5 %." in text


def test_katz_against_common_neighbours():
    k = _katz()
    expect = {(C.T_GRID, 3): (0.49, 0.95), (C.T_GRID, 6): (0.96, 0.98), (C.T_COMMUNITIES, 3): (0.54, 0.61), (C.T_COMMUNITIES, 6): (0.62, 0.74), (C.T_RANDOM, 3): (0.50, 0.50),
              (C.T_RANDOM, 6): (0.50, 0.53), (C.T_POWER, 6): (0.67, 0.68), (C.T_RING, 3): (0.92, 0.93), (C.T_RING, 6): (0.95, 0.97), (C.T_HUBS, 3): (0.70, 0.66), (C.T_HUBS, 6): (0.70, 0.66)}
    for key, (cn, katz) in expect.items():
        assert k[key]["cn"] == pytest.approx(cn, abs=0.03) and k[key]["katz"] == pytest.approx(katz, abs=0.03), key
    kind, text = ev.verdict_katz(list(k.values()))
    assert "in 3 von 14 Zellen mehr als 0,05" in text and "„Gitter“ bei c = 3 mit 0,95 gegen 0,49" in text


def test_katz_in_a_grid_has_no_precision_because_distance_two_pairs_come_first():
    for seed in SW[:3]:
        for c, hidden_distance in ((4, 3),):
            a = ev.build_matrix(C.T_GRID, 250, c, seed)
            t, hidden = A.split_random(a, 0.1, seed)
            g = nx.from_numpy_array(t)
            assert min(nx.shortest_path_length(g, int(x), int(y)) for x, y in hidden) >= hidden_distance             # ohne Diagonalen: versteckte Kanten im Abstand 3 oder mehr
            cand = A.candidate_pairs(t)
            top = A.top_pairs(A.katz(t, C.KATZ_BETA_FACTOR)[cand[:, 0], cand[:, 1]], cand, len(hidden))
            assert {nx.shortest_path_length(g, int(x), int(y)) for x, y in top} == {2}                                    # alle besten Vorhersagen im Abstand 2
            assert 700 < len(A.hard_pairs(t, A.non_edges(t))) < 900                                                      # rund 800 Paare im Abstand 2 konkurrieren


def test_fraction_of_hidden_edges_changes_little():
    expect = {5: (0.65, 0.78), 10: (0.62, 0.74), 20: (0.59, 0.72), 30: (0.57, 0.68)}
    for frac, (cn, katz) in expect.items():
        r = ev.types_rows(250, 6, frac, seeds=SW, types=(C.T_COMMUNITIES,))[0]["scores"]
        assert r[C.M_CN]["auc_easy"] == pytest.approx(cn, abs=0.03) and r[C.M_KATZ]["auc_easy"] == pytest.approx(katz, abs=0.03), frac


def test_community_strength_and_the_group_ceiling():
    r = _community()
    expect = {(6, 0.17): (0.48, 0.48, 0.49, 0.50), (6, 0.4): (0.49, 0.48, 0.48, 0.63), (6, 0.6): (0.55, 0.59, 0.50, 0.72), (6, 0.8): (0.60, 0.74, 0.49, 0.81), (6, 0.95): (0.71, 0.90, 0.49, 0.91),
              (12, 0.17): (0.49, 0.50, 0.50, 0.49), (12, 0.4): (0.52, 0.51, 0.49, 0.63), (12, 0.6): (0.59, 0.62, 0.49, 0.73), (12, 0.8): (0.77, 0.82, 0.49, 0.83), (12, 0.95): (0.88, 0.91, 0.47, 0.91)}
    for key, (cn, katz, pa, group) in expect.items():
        assert r[key]["cn"] == pytest.approx(cn, abs=0.03) and r[key]["katz"] == pytest.approx(katz, abs=0.03) and r[key]["pa"] == pytest.approx(pa, abs=0.04) and r[key]["group"] == pytest.approx(group, abs=0.03), key
    kind, text = ev.verdict_community(list(r.values()))
    assert "Ohne Struktur (Anteil innen 0,17) liegen gemeinsame Nachbarn bei 0,48" in text and "bei c = 6 bei 0,71 und Katz bei 0,90; die Obergrenze „gleiche Gruppe“ ist 0,91" in text
    assert "Bei c = 12 und Anteil 0,95 liegen gemeinsame Nachbarn bei 0,88, Katz bei 0,91, die Obergrenze bei 0,91" in text


def test_time_against_random_split():
    r = _time()
    same, pa, rnd = r[C.MECH_SAME], r[C.MECH_PA], r[C.MECH_RANDOM]
    assert same["time"][C.M_KATZ] == pytest.approx(0.74, abs=0.04) and same["random"][C.M_KATZ] == pytest.approx(0.76, abs=0.04)
    assert same["time"][C.M_CN] == pytest.approx(0.61, abs=0.04) and same["random"][C.M_CN] == pytest.approx(0.62, abs=0.04)
    assert pa["time"][C.M_CN] == pytest.approx(0.51, abs=0.04) and pa["random"][C.M_CN] == pytest.approx(0.60, abs=0.04)
    assert pa["time"][C.M_PA] == pytest.approx(0.55, abs=0.04) and pa["random"][C.M_PA] == pytest.approx(0.52, abs=0.04)
    assert pa["time"][C.M_KATZ] == pytest.approx(0.54, abs=0.05) and pa["random"][C.M_KATZ] == pytest.approx(0.70, abs=0.04)
    assert rnd["time"][C.M_KATZ] == pytest.approx(0.47, abs=0.05) and rnd["random"][C.M_KATZ] == pytest.approx(0.70, abs=0.04)
    assert rnd["time"][C.M_CN] == pytest.approx(0.50, abs=0.03) and rnd["random"][C.M_CN] == pytest.approx(0.59, abs=0.04)
    kind, text = ev.verdict_time(list(r.values()))
    assert kind == "warning" and "Entstehen die neuen Kanten zufällig, erreicht Katz mit Zeitaufteilung 0,47, mit Zufallsaufteilung 0,70" in text and "überschätzt hier um 0,24" in text
    assert "stimmen beide Aufteilungen überein (Katz 0,74 gegen 0,76)" in text


def test_degree_leak_of_the_preferential_attachment():
    r = _leak()
    expect = {C.T_COMMUNITIES: (0.51, 0.68), C.T_RANDOM: (0.50, 0.67), C.T_POWER: (0.73, 0.85), C.T_BA: (0.66, 0.78), C.T_RING: (0.18, 0.53), C.T_GRID: (0.27, 0.58), C.T_HUBS: (0.63, 0.95)}
    for typ, (clean, leak) in expect.items():
        assert r[typ]["clean"] == pytest.approx(clean, abs=0.03) and r[typ]["leak"] == pytest.approx(leak, abs=0.03), typ
    kind, text = ev.verdict_leak(list(r.values()))
    assert "in „Ring mit Abkürzungen“ von 0,18 auf 0,53" in text and "unter dem Zufall" in text


def test_the_link_network_with_and_without_the_navigation_pages():
    full, drop = _link(C.NAV_COUNT), _link(C.NAV_DROP)
    assert full["coverage"] == pytest.approx(1.0, abs=0.01) and full["nonedge_coverage"] == pytest.approx(0.97, abs=0.01)
    assert drop["coverage"] == pytest.approx(0.89, abs=0.02) and drop["nonedge_coverage"] == pytest.approx(0.26, abs=0.02)
    _check(full, "auc_easy", (0.83, 0.49, 0.87, 0.89, 0.96, 0.93))
    _check(drop, "auc_easy", (0.90, 0.89, 0.91, 0.91, 0.87, 0.88))
    assert full["scores"][C.M_PA]["lift"] == pytest.approx(155, abs=15) and full["scores"][C.M_PA]["precision"] == pytest.approx(0.624, abs=0.04)
    assert full["scores"][C.M_JACCARD]["precision"] == 0.0 and drop["scores"][C.M_JACCARD]["lift"] <= 3
    assert drop["scores"][C.M_RA]["lift"] == pytest.approx(235, abs=25) and drop["scores"][C.M_CN]["lift"] == pytest.approx(220, abs=25) and drop["scores"][C.M_RA]["precision"] == pytest.approx(0.670, abs=0.04)
    assert drop["scores"][C.M_CN]["auc_hard"] == pytest.approx(0.78, abs=0.03)


def test_default_network_numbers():
    """Standardnetz (Verbünde, n = 150, c = 6, Seed 35, 10 % versteckt, gemeinsame Nachbarn): die Zahlen der Voreinstellung."""
    an = ev.analyse(ev.Settings())
    assert (an.inst.m, an.k, int(an.top_hit.sum())) == (450, 45, 2)
    r = an.result
    assert r["auc_easy"] == pytest.approx(0.73, abs=0.01) and r["auc_hard"] == pytest.approx(0.36, abs=0.01) and r["precision"] == pytest.approx(0.040, abs=0.003) and r["lift"] == pytest.approx(10, abs=1)
    assert an.proto["coverage"] == pytest.approx(0.62, abs=0.02) and an.proto["nonedge_coverage"] == pytest.approx(0.14, abs=0.01)
    assert ev.verdict_main(an)[0] == "info"


def test_katz_is_insensitive_to_beta():
    expect = {(C.T_COMMUNITIES, 6): (0.748, 0.747, 0.744, 0.729), (C.T_GRID, 3): (0.944, 0.947, 0.945, 0.927)}
    for (typ, c), values in expect.items():
        for factor, value in zip((0.05, 0.2, 0.5, 0.8), values):
            aucs = []
            for seed in SW:
                a = ev.build_matrix(typ, 250, c, seed)
                t, hidden = A.split_random(a, 0.1, seed)
                easy = A.sample_pairs(A.non_edges(a), max(C.NEG_FACTOR * len(hidden), C.NEG_MIN), seed + 1)
                m = A.katz(t, factor)
                aucs.append(A.auc(m[hidden[:, 0], hidden[:, 1]], m[easy[:, 0], easy[:, 1]]))
            assert float(np.median(aucs)) == pytest.approx(value, abs=0.02), (typ, factor)


def test_more_precision_cells():
    rows = _types()
    assert rows[C.T_BA]["scores"][C.M_CN]["lift"] == pytest.approx(12, abs=3) and rows[C.T_BA]["scores"][C.M_CN]["precision"] == pytest.approx(0.029, abs=0.008)
    assert rows[C.T_HUBS]["scores"][C.M_CN]["precision"] == pytest.approx(0.105, abs=0.02) and rows[C.T_RING]["scores"][C.M_CN]["precision"] == pytest.approx(0.266, abs=0.03)
    assert _link(C.NAV_COUNT)["scores"][C.M_CN]["lift"] == pytest.approx(126, abs=15)
