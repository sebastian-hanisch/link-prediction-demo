"""Orakel-Test link-prediction-demo: die Eigenbauten gegen einen anderen Rechenweg (networkx, Mengenformel, Reihe und Wegezählung, Aufzählung).

Orakel je Einheit (ORACLE-PLAYBOOK §2): Jaccard, Adamic–Adar, Ressourcenzuteilung und bevorzugte Anbindung gegen `networkx` auf 100 festen Zufallsnetzen (auch mit isolierten
Knoten); gemeinsame Nachbarn gegen den Mengenschnitt; Katz gegen die abgebrochene Reihe und gegen die Aufzählung aller Wege auf Mini-Netzen; AUC gegen die Aufzählung aller Paare;
erwartete Precision@k gegen die Aufzählung aller Gleichstands-Reihenfolgen bei kleinen Gruppen und gegen Mittelung über viele zufällige Auflösungen."""

import itertools
import math

import numpy as np
import pytest

import lkp_algorithm as A
import lkp_constants as C
from conftest import make_matrix

nx = pytest.importorskip("networkx")


def _random_matrix(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 24))
    p = float(rng.uniform(0.05, 0.5))
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < p]
    return make_matrix(n, pairs)


def _graph(a):
    g = nx.Graph()
    g.add_nodes_from(range(len(a)))
    g.add_edges_from(A.edge_list(a).tolist())
    return g


@pytest.mark.parametrize("seed", range(100))
def test_measures_match_networkx(seed):
    a = _random_matrix(seed)
    g = _graph(a)
    pairs = [tuple(p) for p in A.non_edges(a).tolist()]
    if not pairs:
        pytest.skip("vollständiges Netz")
    expect = {C.M_JACCARD: nx.jaccard_coefficient(g, pairs), C.M_AA: nx.adamic_adar_index(g, pairs), C.M_RA: nx.resource_allocation_index(g, pairs),
              C.M_PA: nx.preferential_attachment(g, pairs)}
    mine = A.score_all(a, tuple(expect), 0.5, 0)
    for name, gen in expect.items():
        for x, y, value in gen:
            assert mine[name][x, y] == pytest.approx(value, abs=1e-9), (name, x, y)
    cn = A.common_neighbors(a)
    for x, y in pairs:
        assert cn[x, y] == len(set(g[x]) & set(g[y]))


def test_katz_matches_the_series_and_the_walk_enumeration():
    for seed in range(20):
        a = _random_matrix(seed)
        n = len(a)
        lam = float(np.linalg.eigvalsh(a)[-1])
        if lam == 0:
            continue
        beta = 0.4 / lam
        k = A.katz(a, 0.4)
        series = sum(beta ** l * np.linalg.matrix_power(a, l) for l in range(1, 120))
        assert np.allclose(k, series, atol=1e-8)
    tiny = make_matrix(5, [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)])
    lam = float(np.linalg.eigvalsh(tiny)[-1])
    beta = 0.3 / lam
    walks = np.zeros((5, 5))                                              # Wege der Länge l zählen: Aufzählung aller Folgen, bis Länge 5, Rest als Reihe
    for length in range(1, 6):
        for seq in itertools.product(range(5), repeat=length + 1):
            if all(tiny[seq[i], seq[i + 1]] for i in range(length)):
                walks[seq[0], seq[-1]] += beta ** length
    rest = sum(beta ** l * np.linalg.matrix_power(tiny, l) for l in range(6, 150))
    assert np.allclose(A.katz(tiny, 0.3), walks + rest, atol=1e-9)


def test_auc_matches_the_enumeration_of_all_pairs():
    rng = np.random.default_rng(6)
    for _ in range(60):
        pos = rng.integers(0, 8, size=int(rng.integers(1, 12))).tolist()
        neg = rng.integers(0, 8, size=int(rng.integers(1, 12))).tolist()
        wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p, q in itertools.product(pos, neg))
        assert A.auc(pos, neg) == pytest.approx(wins / (len(pos) * len(neg)))


def test_expected_precision_matches_the_enumeration_of_all_tie_orders():
    rng = np.random.default_rng(7)
    for _ in range(40):
        n = int(rng.integers(3, 8))
        scores = rng.integers(0, 3, size=n).astype(float)
        hits = rng.integers(0, 2, size=n)
        k = int(rng.integers(1, n + 1))
        total, count = 0.0, 0
        for perm in itertools.permutations(range(n)):                      # alle Reihenfolgen; die Sortierung nach Score ist stabil, innerhalb der Gruppen zählt die Reihenfolge
            order = sorted(perm, key=lambda i: -scores[i])
            total += sum(hits[i] for i in order[:k]) / k
            count += 1
        assert A.expected_precision_at_k(scores, hits, k) == pytest.approx(total / count), (scores, hits, k)


def test_expected_precision_matches_a_random_tie_resolution_on_a_real_split():
    a, lab = A.sbm_matrix(80, 4, 6, 0.8, 1)
    t, hidden = A.split_random(a, 0.1, 1)
    cand = A.candidate_pairs(t)
    is_hid = A.hidden_mask(cand, hidden, 80)
    cn = A.common_neighbors(t)[cand[:, 0], cand[:, 1]]
    rng = np.random.default_rng(2)
    runs = []
    for _ in range(400):
        order = np.lexsort((rng.random(len(cand)), -cn))
        runs.append(is_hid[order[: len(hidden)]].mean())
    assert A.expected_precision_at_k(cn, is_hid, len(hidden)) == pytest.approx(float(np.mean(runs)), abs=0.02)


def test_split_never_leaks_hidden_edges_into_the_training_network():
    for seed in range(30):
        a = _random_matrix(seed)
        t, hidden = A.split_random(a, 0.2, seed)
        for x, y in hidden.tolist():
            assert t[x, y] == 0 and a[x, y] == 1
        assert (t.sum() + 2 * len(hidden)) == a.sum()
