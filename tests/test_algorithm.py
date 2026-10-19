"""Einheiten des Kerns an Mini-Instanzen mit von Hand gerechneten Werten (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np
import pytest

import lkp_algorithm as A
import lkp_constants as C
from conftest import make_matrix


def test_matrix_helpers_and_edge_lists(mini):
    assert A.edge_list(mini).tolist() == [[0, 1], [0, 2], [1, 2], [2, 3]]
    assert A.non_edges(mini).tolist() == [[0, 3], [1, 3]]
    assert A.remove_edges(mini, np.array([[0, 2]]))[0, 2] == 0 and A.remove_edges(mini, np.array([[0, 2]]))[2, 0] == 0 and mini[0, 2] == 1
    adj = ((1, 2), (0, 2), (0, 1, 3), (2,))
    assert A.to_matrix(adj).tolist() == [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]


def test_split_random_is_disjoint_complete_and_deterministic(mini):
    t, hidden = A.split_random(mini, 0.5, 3)
    assert len(hidden) == 2 and t.sum() / 2 == 2
    assert np.array_equal(A.remove_edges(mini, np.zeros((0, 2), dtype=int)), mini)                  # nichts entfernen ändert nichts
    both = {tuple(e) for e in A.edge_list(t).tolist()} | {tuple(e) for e in hidden.tolist()}
    assert both == {tuple(e) for e in A.edge_list(mini).tolist()} and not ({tuple(e) for e in A.edge_list(t).tolist()} & {tuple(e) for e in hidden.tolist()})
    t2, hidden2 = A.split_random(mini, 0.5, 3)
    assert np.array_equal(t, t2) and np.array_equal(hidden, hidden2)


def test_split_random_hides_at_least_one_edge_and_handles_edgeless_networks(mini):
    assert len(A.split_random(mini, 0.01, 1)[1]) == 1
    assert len(A.split_random(mini, 1.0, 1)[1]) == 4
    t, hidden = A.split_random(np.zeros((3, 3)), 0.1, 1)
    assert len(hidden) == 0 and t.sum() == 0


def test_measures_hand_values(mini):
    cn, jac, aa, ra, pa = A.common_neighbors(mini), A.jaccard(mini), A.adamic_adar(mini), A.resource_allocation(mini), A.preferential_attachment(mini)
    for x, y in ((0, 3), (1, 3)):
        assert cn[x, y] == 1 and jac[x, y] == pytest.approx(0.5) and aa[x, y] == pytest.approx(1 / math.log(3)) and ra[x, y] == pytest.approx(1 / 3) and pa[x, y] == pytest.approx(2.0)
    assert cn[0, 1] == 1 and cn[0, 0] == 2                                  # CN einer Kante: gemeinsamer Nachbar 2; Diagonale = Grad (wird nie bewertet)
    for m in (cn, jac, aa, ra, pa):
        assert np.allclose(m, m.T)


def test_measures_are_defined_for_isolated_nodes_and_degree_one_nodes(path_with_isolate):
    t = path_with_isolate
    assert A.jaccard(t)[0, 3] == 0.0 and A.jaccard(t)[0, 2] == pytest.approx(1.0) and A.adamic_adar(t)[0, 3] == 0.0
    assert A.adamic_adar(t)[0, 2] == pytest.approx(1 / math.log(2)) and A.resource_allocation(t)[0, 2] == pytest.approx(1 / 2)
    assert A.preferential_attachment(t)[0, 3] == 0.0
    star = make_matrix(3, [(0, 1)])                                           # Blätter tragen nie als gemeinsamer Nachbar bei
    assert A.adamic_adar(star).sum() == 0 or np.isfinite(A.adamic_adar(star)).all()


def test_preferential_attachment_degree_override(mini):
    leak = A.preferential_attachment(mini, deg=[5, 1, 1, 7])
    assert leak[0, 3] == 35.0 and leak[1, 2] == 1.0


def test_katz_matches_the_walk_series_and_is_zero_without_edges(mini):
    k = A.katz(mini, 0.2)
    lam = float(np.linalg.eigvalsh(mini)[-1])
    beta = 0.2 / lam
    series = sum(beta ** l * np.linalg.matrix_power(mini, l) for l in range(1, 80))
    assert np.allclose(k, series, atol=1e-9)
    assert A.katz(np.zeros((3, 3)), 0.5).tolist() == np.zeros((3, 3)).tolist()
    assert k[0, 3] > beta ** 2 * 1.0                                       # mindestens die Wege der Länge 2 und 3: 1 und 1 (Handrechnung)
    assert k[0, 3] == pytest.approx(beta ** 2 * 1 + beta ** 3 * 1, rel=0.6)


def test_random_score_is_symmetric_deterministic_and_uniform():
    r = A.random_score(30, 5)
    assert np.allclose(r, r.T) and np.array_equal(r, A.random_score(30, 5)) and not np.array_equal(r, A.random_score(30, 6))
    assert 0.4 < r.mean() < 0.6


def test_score_all_returns_every_requested_measure(mini):
    S = A.score_all(mini, C.SCORE_NAMES, C.KATZ_BETA_FACTOR, 1)
    assert list(S) == list(C.SCORE_NAMES) and all(m.shape == (4, 4) for m in S.values())
    assert A.score_all(mini, (C.M_PA,), 0.5, 1, deg=[1, 1, 1, 1])[C.M_PA].sum() == 16


def test_average_ranks_and_auc_hand_values():
    assert A.average_ranks([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert A.auc([3, 4], [1, 3]) == pytest.approx(0.875)                    # Paare: 3>1, 3=3 (1/2), 4>1, 4>3
    assert A.auc([1, 1], [1, 1]) == pytest.approx(0.5) and A.auc([5], [1, 2]) == 1.0 and A.auc([1], [5, 6]) == 0.0
    assert math.isnan(A.auc([], [1])) and math.isnan(A.auc([1], []))


def test_float_noise_does_not_break_ties():
    assert A.average_ranks([0.3, 0.3 + 3e-16, 0.1]).tolist() == [2.5, 2.5, 1.0]
    assert A.auc([0.3], [0.3 + 3e-16]) == pytest.approx(0.5)


def test_hard_pairs_and_coverage(mini, path_with_isolate):
    assert A.hard_pairs(mini, A.non_edges(mini)).tolist() == [[0, 3], [1, 3]] and A.coverage(mini, A.non_edges(mini)) == 1.0
    t = path_with_isolate
    ne = A.non_edges(t)
    assert A.hard_pairs(t, ne).tolist() == [[0, 2]] and A.coverage(t, ne) == pytest.approx(1 / 4)
    assert math.isnan(A.coverage(t, np.zeros((0, 2), dtype=int))) and len(A.hard_pairs(t, np.zeros((0, 2), dtype=int))) == 0


def test_candidate_pairs_and_hidden_mask(mini):
    t = A.remove_edges(mini, np.array([[0, 2]]))
    cand = A.candidate_pairs(t)
    assert cand.tolist() == [[0, 2], [0, 3], [1, 3]]
    assert A.hidden_mask(cand, np.array([[0, 2]]), 4).tolist() == [True, False, False]
    assert A.hidden_mask(cand, np.zeros((0, 2), dtype=int), 4).tolist() == [False] * 3


def test_expected_precision_hand_values():
    scores, hits = [3, 2, 2, 1], [0, 1, 0, 1]
    assert A.expected_precision_at_k(scores, hits, 1) == 0.0
    assert A.expected_precision_at_k(scores, hits, 2) == pytest.approx(0.25)          # Platz 2 aus der Gruppe {2, 2}: halbe Trefferchance
    assert A.expected_precision_at_k(scores, hits, 3) == pytest.approx(1 / 3)         # ganze Gruppe: ein Treffer
    assert A.expected_precision_at_k(scores, hits, 4) == pytest.approx(0.5)
    assert A.expected_precision_at_k(scores, hits, 99) == pytest.approx(0.5)          # k auf die Zahl der Paare begrenzt
    assert math.isnan(A.expected_precision_at_k(scores, hits, 0))
    assert A.expected_precision_at_k([1, 1, 1, 1], [1, 0, 0, 0], 2) == pytest.approx(0.25)    # alles gleich: Basisrate


def test_lift_and_top_pairs():
    assert A.lift(0.5, 2, 10) == pytest.approx(2.5) and math.isnan(A.lift(0.5, 0, 10)) and math.isnan(A.lift(0.5, 2, 0)) and math.isnan(A.lift(float("nan"), 2, 10))
    cand = np.array([[0, 1], [0, 2], [1, 2]])
    assert A.top_pairs([1.0, 3.0, 3.0], cand, 2).tolist() == [[0, 2], [1, 2]]          # Gleichstand: frühere Position zuerst


def test_sample_pairs_is_deterministic_and_complete_when_small():
    pairs = np.array([[0, i] for i in range(1, 30)])
    a, b = A.sample_pairs(pairs, 5, 1), A.sample_pairs(pairs, 5, 1)
    assert len(a) == 5 and np.array_equal(a, b) and A.sample_pairs(pairs, 100, 1).shape == pairs.shape


def test_sbm_matrix_shape_degree_and_structure():
    a, lab = A.sbm_matrix(300, 6, 8, 0.8, 4)
    assert np.allclose(a, a.T) and np.trace(a) == 0 and a.sum() / 300 == pytest.approx(8, rel=0.15)
    inside = np.sum(a * (lab[:, None] == lab[None, :])) / a.sum()
    assert inside == pytest.approx(0.8, abs=0.04)
    assert np.array_equal(a, A.sbm_matrix(300, 6, 8, 0.8, 4)[0])
    flat, lab = A.sbm_matrix(300, 6, 8, 1 / 6, 4)
    assert np.sum(flat * (lab[:, None] == lab[None, :])) / flat.sum() == pytest.approx(1 / 6, abs=0.04)


def test_future_edges_are_new_deterministic_and_follow_their_mechanism():
    a, lab = A.sbm_matrix(200, 6, 6, 0.8, 2)
    k = 60
    for mech in C.MECHANISMS:
        fut = A.future_edges(a, mech, k, 5, lab)
        assert len(fut) == k and np.all(fut[:, 0] < fut[:, 1]) and all(a[x, y] == 0 for x, y in fut.tolist())
        assert np.array_equal(fut, A.future_edges(a, mech, k, 5, lab)) and len({tuple(e) for e in fut.tolist()}) == k
    same = A.future_edges(a, C.MECH_SAME, 200, 1, lab)
    pa = A.future_edges(a, C.MECH_PA, 200, 1, lab)
    rnd = A.future_edges(a, C.MECH_RANDOM, 200, 1, lab)
    inside = lambda f: float(np.mean(lab[f[:, 0]] == lab[f[:, 1]]))
    assert inside(same) > 0.7 and inside(rnd) < 0.35
    deg = a.sum(1)
    assert np.mean(deg[pa].max(1)) > np.mean(deg[rnd].max(1))                 # bevorzugte Anbindung trifft die Knoten mit hohem Grad


def test_future_edges_stop_in_a_complete_network():
    full = np.ones((5, 5)) - np.eye(5)
    assert len(A.future_edges(full, C.MECH_RANDOM, 3, 1)) == 0


def test_hard_pairs_of_a_single_pair_depend_on_its_common_neighbours(path_with_isolate):
    assert A.hard_pairs(path_with_isolate, np.array([[0, 3]])).tolist() == []                         # ein Paar ohne gemeinsamen Nachbarn ist kein schweres Negativ
    assert A.hard_pairs(path_with_isolate, np.array([[0, 2]])).tolist() == [[0, 2]]


def test_sbm_with_one_node_per_group_has_only_edges_between_groups():
    a, lab = A.sbm_matrix(6, 6, 2, 0.5, 1)                                                         # keine Paare innerhalb einer Gruppe: p_in wird nicht berechnet
    assert np.trace(a) == 0 and len(set(lab.tolist())) == 6
    b, _ = A.sbm_matrix(8, 1, 4, 0.5, 1)                                                           # eine Gruppe: keine Paare zwischen den Gruppen
    assert np.allclose(b, b.T)


def test_resource_allocation_and_adamic_adar_weights_of_leaves(path_with_isolate):
    star = make_matrix(4, [(0, 1), (0, 2), (0, 3)])
    assert A.resource_allocation(star)[1, 2] == pytest.approx(1 / 3) and A.adamic_adar(star)[1, 2] == pytest.approx(1 / math.log(3))


def test_single_candidate_pair_is_handled():
    cand = np.array([[0, 2]])
    assert A.hidden_mask(cand, np.array([[0, 2]]), 4).tolist() == [True] and A.hidden_mask(cand, np.array([[1, 3]]), 4).tolist() == [False]
    assert A.lift(1.0, 1, 1) == pytest.approx(1.0) and A.lift(0.5, 1, 1) == pytest.approx(0.5)


def test_sbm_probabilities_by_hand():
    # n = 6, q = 2: Gruppen {0,1,2} und {3,4,5}; Paare innerhalb 3 + 3 = 6, zwischen 15 - 6 = 9; c = 2 gibt m = 6 Kanten
    p_in, p_out = A.sbm_probabilities(6, 2, 2, 0.5)
    assert p_in == pytest.approx(0.5 * 6 / 6) and p_out == pytest.approx(0.5 * 6 / 9)
    p_in, p_out = A.sbm_probabilities(6, 2, 10, 0.9)                                         # m = 30: p_in = 4,5 wird auf 1 begrenzt, p_out = 0,1 · 30 / 9
    assert p_in == 1.0 and p_out == pytest.approx(1 / 3)
    assert A.sbm_probabilities(6, 6, 2, 0.5) == (0.0, pytest.approx(0.5 * 6 / 15))           # eine Gruppe je Knoten: keine Paare innerhalb
    p_in, p_out = A.sbm_probabilities(6, 1, 2, 0.5)                                          # eine Gruppe: keine Paare zwischen den Gruppen
    assert p_out == 0.0 and p_in == pytest.approx(0.5 * 6 / 15)
    assert A.sbm_probabilities(12, 3, 4, 0.25) == (pytest.approx(0.25 * 24 / 18), pytest.approx(0.75 * 24 / 48))      # 3 · 6 = 18 Paare innerhalb, 66 - 18 = 48 zwischen, m = 24


def test_empty_inputs_of_the_precision_and_the_coverage():
    import warnings

    assert math.isnan(A.expected_precision_at_k([], [], 3)) and math.isnan(A.expected_precision_at_k([1.0], [1], 0))
    path = make_matrix(4, [(0, 1), (1, 2)])
    assert A.coverage(path, np.array([[0, 2]])) == 1.0 and A.coverage(path, np.array([[0, 3]])) == 0.0              # ein einzelnes Paar
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(A.coverage(path, np.zeros((0, 2), dtype=int)))


def test_future_edges_in_an_edgeless_network_and_their_order():
    empty = np.zeros((10, 10))
    fut = A.future_edges(empty, C.MECH_PA, 5, 1)                                              # ohne Grade fällt die Wahl auf gleichverteilt zurück
    assert len(fut) == 5 and fut.tolist() == sorted(fut.tolist())
    one = make_matrix(10, [(0, 1)])
    assert A.future_edges(one, C.MECH_PA, 4, 2).tolist() == sorted(A.future_edges(one, C.MECH_PA, 4, 2).tolist())


def test_sbm_probabilities_with_a_single_pair_in_a_class():
    p_in, p_out = A.sbm_probabilities(3, 2, 2, 0.5)             # Gruppen {0,1} und {2}: genau ein Paar innerhalb, zwei zwischen; m = 3
    assert p_in == 1.0 and p_out == pytest.approx(0.5 * 3 / 2)
    assert A.sbm_probabilities(2, 2, 1, 0.0) == (0.0, 1.0)      # zwei Gruppen mit je einem Knoten: genau ein Paar zwischen den Gruppen
