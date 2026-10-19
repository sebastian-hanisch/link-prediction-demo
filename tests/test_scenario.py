"""Netze: gültige Netze, Determinismus (auch über Prozesse), Positionen, Link-Netz mit und ohne Navigationsseiten, Hashseed-Unabhängigkeit."""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import lkp_algorithm as A
import lkp_constants as C
import lkp_scenario as S

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("typ", C.GENERATED_TYPES)
@pytest.mark.parametrize("n,c", [(C.N_MIN, C.C_MIN), (C.N_MIN, C.C_MAX), (C.DEFAULT_N, C.DEFAULT_C), (C.N_MAX, C.C_MAX)])
def test_every_corner_gives_a_valid_network(typ, n, c):
    inst = S.make_instance(typ, n, c, 35)
    assert inst.n >= 2 and len(inst.pos) == inst.n
    for x, y in inst.pos:
        assert math.isfinite(x) and math.isfinite(y) and -1e-9 <= x <= 1 + 1e-9 and -1e-9 <= y <= 1 + 1e-9
    for a, nb in enumerate(inst.adj):
        assert list(nb) == sorted(set(nb)) and a not in nb
        for b in nb:
            assert a in inst.adj[b]


def test_instances_are_deterministic_per_seed_and_differ_between_seeds():
    a = S.make_instance(C.T_COMMUNITIES, 80, 6, 5)
    assert a.adj == S.make_instance(C.T_COMMUNITIES, 80, 6, 5).adj and a.pos == S.make_instance(C.T_COMMUNITIES, 80, 6, 5).pos
    assert S.make_instance(C.T_COMMUNITIES, 80, 6, 6, layout=False).adj != a.adj


def test_layout_free_instances_have_the_same_network():
    for typ in (C.T_RANDOM, C.T_LINK):
        assert S.make_instance(typ, 60, 6, 3, layout=False).adj == S.make_instance(typ, 60, 6, 3).adj


def test_the_link_network_with_and_without_the_navigation_pages():
    full = S.make_instance(C.T_LINK, 0, 0, 0, layout=False)
    drop = S.make_instance(C.T_LINK, 0, 0, 0, nav=C.NAV_DROP, layout=False)
    assert full.n == 350 and drop.n == 348 and "index.html" in full.labels and "index.html" not in drop.labels and "kontakt.html" not in drop.labels
    assert full.m == 2374 and drop.m == 1679 and S.link_stand() == "2026-09-30"


def test_the_grid_without_diagonals_has_no_triangles():
    inst = S.make_instance(C.T_GRID, 100, 4, 1, layout=False)
    a = A.to_matrix(inst.adj)
    assert (a @ a @ a).trace() == 0                                      # keine Dreiecke
    assert (A.to_matrix(S.make_instance(C.T_GRID, 100, 6, 1, layout=False).adj) @ a).sum() > 0


def test_unknown_network_type_raises():
    with pytest.raises(ValueError):
        S.make_instance("gibt es nicht", 50, 6, 1)


def test_the_force_layout_of_a_single_node():
    assert S.layout_force(((),), 1) == ((0.5, 0.5),)


def test_results_do_not_depend_on_the_hash_seed():
    """Dasselbe Szenario in zwei Prozessen mit verschiedenem PYTHONHASHSEED rechnet dasselbe."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import lkp_evaluation as ev, lkp_constants as C\n"
        "out = []\n"
        "for typ in (C.T_COMMUNITIES, C.T_POWER, C.T_HUBS):\n"
        "    an = ev.analyse(ev.Settings(typ, 90, 6, 7))\n"
        "    r = an.proto['scores'][C.M_KATZ]\n"
        "    out.append((an.inst.m, an.k, round(r['auc_easy'], 9), round(r['precision'], 9), int(an.top_hit.sum())))\n"
        "print(repr(out))\n"
    ) % str(ROOT)
    outputs = []
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONIOENCODING="utf-8")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180)
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1]
