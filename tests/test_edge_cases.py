"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): wenige Kanten, keine schweren Negative, Gleichstände, Grenzen der Regler."""

import math

import numpy as np
import pytest

import lkp_algorithm as A
import lkp_constants as C
import lkp_evaluation as ev
from conftest import make_matrix


def test_protocol_on_a_tiny_network_has_defined_values():
    a = make_matrix(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    t, hidden = A.split_random(a, 0.2, 1)
    r = ev.evaluate_protocol(a, t, hidden, 1)
    assert r["k"] == 1 and r["n_cand"] == len(A.candidate_pairs(t))
    for name in C.SCORE_NAMES:
        assert 0.0 <= r["scores"][name]["auc_easy"] <= 1.0
        assert math.isnan(r["scores"][name]["auc_hard"]) or 0.0 <= r["scores"][name]["auc_hard"] <= 1.0
    assert math.isnan(r["scores"][C.M_CN]["auc_hard"]) == (r["hard_n"] < C.HARD_MIN)                       # zu wenige schwere Negative: nicht definiert


def test_protocol_with_few_hard_negatives_reports_nan_for_the_hard_auc():
    a = make_matrix(8, [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2)])
    t, hidden = A.split_random(a, 0.2, 3)
    r = ev.evaluate_protocol(a, t, hidden, 3)
    assert r["hard_n"] < C.HARD_MIN and all(math.isnan(v["auc_hard"]) for v in r["scores"].values())


def test_protocol_without_hidden_edges_is_never_called_by_the_app_but_the_split_is_defined():
    t, hidden = A.split_random(np.zeros((5, 5)), 0.1, 1)
    assert len(hidden) == 0 and t.sum() == 0 and math.isnan(A.expected_precision_at_k([0.0, 0.0], [0, 0], 0))


def test_all_equal_scores_give_the_base_rate_as_precision():
    cand = A.candidate_pairs(np.zeros((6, 6)))
    hit = np.zeros(len(cand), dtype=bool)
    hit[:3] = True
    assert A.expected_precision_at_k(np.zeros(len(cand)), hit, 3) == pytest.approx(3 / len(cand))                # alle Scores gleich: Zufallsniveau
    assert A.lift(3 / len(cand), 3, len(cand)) == pytest.approx(1.0)


def test_analysis_of_every_type_and_method_at_the_slider_corners_does_not_fail():
    for typ in C.GENERATED_TYPES:
        for n, c, frac in ((C.N_MIN, C.C_MIN, C.FRAC_MAX), (C.N_MAX, C.C_MAX, C.FRAC_MIN)):
            an = ev.analyse(ev.Settings(typ, n, c, 1, frac, C.M_KATZ))
            assert an.k >= 1 and len(an.top) == an.k and an.top_hit.dtype == bool
            assert ev.verdict_main(an)[0] in ("success", "info", "warning")
    for method in C.METHODS:
        an = ev.analyse(ev.Settings(C.T_COMMUNITIES, 80, 6, 2, 10, method))
        assert an.method == method and 0.0 <= an.result["auc_easy"] <= 1.0


def test_leak_switch_changes_only_the_preferential_attachment():
    clean = ev.analyse(ev.Settings(C.T_HUBS, 100, 6, 4, 10, C.M_PA, leak=C.LEAK_OFF))
    leaked = ev.analyse(ev.Settings(C.T_HUBS, 100, 6, 4, 10, C.M_PA, leak=C.LEAK_ON))
    assert leaked.result["auc_easy"] > clean.result["auc_easy"] + 0.1
    for name in C.SCORE_NAMES:
        if name != C.M_PA:
            assert clean.proto["scores"][name] == leaked.proto["scores"][name]


def test_top_pairs_are_candidates_and_hits_are_hidden_edges():
    an = ev.analyse(ev.Settings(C.T_RING, 80, 6, 3, 10, C.M_CN))
    cand = {tuple(e) for e in A.candidate_pairs(an.t).tolist()}
    hidden = {tuple(e) for e in an.hidden.tolist()}
    assert all(tuple(e) in cand for e in an.top.tolist())
    assert [tuple(e) in hidden for e in an.top.tolist()] == an.top_hit.tolist()
    assert all(an.t[x, y] == 0 for x, y in an.top.tolist())                  # nie eine Trainingskante vorhersagen


def test_experiments_survive_tiny_networks():
    n = C.N_MIN
    assert len(ev.types_rows(n, C.C_MIN, C.FRAC_MAX, seeds=(1,))) == len(C.GENERATED_TYPES)
    assert len(ev.katz_rows(n, C.FRAC_MAX, seeds=(1,))) == len(C.GENERATED_TYPES) * len(C.KATZ_DEGREES)
    assert len(ev.community_rows(n, C.FRAC_MAX, seeds=(1,))) == len(C.COMM_DEGREES) * len(C.INSIDE_SHARES)
    assert len(ev.time_rows(n, C.C_MIN, seeds=(1,))) == len(C.MECHANISMS)
    assert len(ev.leak_rows(n, C.C_MIN, C.FRAC_MAX, seeds=(1,))) == len(C.GENERATED_TYPES)


def test_grid_rounds_to_a_square():
    an = ev.analyse(ev.Settings(C.T_GRID, 100, 4, 1))
    assert an.inst.n == 100 and an.proto["coverage"] == 0.0                   # 10 mal 10, keine Dreiecke: keine gemeinsamen Nachbarn nach dem Verstecken
