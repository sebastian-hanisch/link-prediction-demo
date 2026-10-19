"""Gemeinsame Fixtures und Hilfen (DEMO-PLAYBOOK §1b): Mini-Instanzen, an denen jede Einheit von Hand geprüft wird."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_matrix(n, pairs):
    """Symmetrische 0/1-Matrix aus einer Kantenliste."""
    a = np.zeros((n, n))
    for x, y in pairs:
        if x != y:
            a[x, y] = a[y, x] = 1.0
    return a


@pytest.fixture
def mini():
    """Vier Knoten, Kanten 0-1, 1-2, 2-3, 0-2; Nichtkanten (0,3) und (1,3).
    Handrechnung für (0,3): gemeinsame Nachbarn {2} = 1; Jaccard 1 / |{1,2} ∪ {2}| = 1/2; Adamic–Adar 1/ln 3 (Grad von Knoten 2 ist 3); Ressourcenzuteilung 1/3;
    bevorzugte Anbindung Grad 2 mal Grad 1 = 2. Für (1,3): gemeinsame Nachbarn {2} = 1, Jaccard 1 / |{0,2} ∪ {2}| = 1/2, sonst wie (0,3)."""
    return make_matrix(4, [(0, 1), (1, 2), (2, 3), (0, 2)])


@pytest.fixture
def path_with_isolate():
    """Pfad 0 - 1 - 2 und ein isolierter Knoten 3: Nichtkanten (0,2) mit einem gemeinsamen Nachbarn, (0,3), (1,3), (2,3) ohne."""
    return make_matrix(4, [(0, 1), (1, 2)])
