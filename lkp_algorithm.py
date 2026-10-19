"""Kern: Aufteilung eines Netzes in Trainingsnetz und versteckte Kanten, die Nähemaße der Link-Prediction, AUC, Negativbeispiele, Abdeckung und Precision@k mit erwarteter Gleichstandsauflösung.

Netze sind dichte 0/1-Matrizen `A` (symmetrisch, Diagonale 0, ungewichtet, ungerichtet); Kantenlisten sind Arrays `(k, 2)` mit a < b. Zufall nur über `numpy.random.default_rng(Seed)`."""

import numpy as np

import lkp_constants as C

ROUND = 12   # Scores werden vor jedem Vergleich auf 12 Stellen gerundet: Gleitkomma-Summen mathematisch gleicher Paare unterscheiden sich um ~1e-16


# --- Netze als Matrix ---------------------------------------------------------------------------------------------------------------------------------


def to_matrix(adj) -> np.ndarray:
    n = len(adj)
    a = np.zeros((n, n))
    for i, nb in enumerate(adj):
        a[i, list(nb)] = 1.0
    return a


def edge_list(a) -> np.ndarray:
    """Alle Kanten (a < b) als Array (m, 2), nach Zeile sortiert."""
    return np.argwhere(np.triu(a, 1) > 0)


def non_edges(a) -> np.ndarray:
    """Alle Paare a < b ohne Kante."""
    n = len(a)
    return np.argwhere(np.triu(1.0 - a - np.eye(n), 0) > 0)


def remove_edges(a, edges) -> np.ndarray:
    t = a.copy()
    if len(edges):
        t[edges[:, 0], edges[:, 1]] = 0.0
        t[edges[:, 1], edges[:, 0]] = 0.0
    return t


def split_random(a, frac, seed):
    """Versteckt `frac` der Kanten zufällig (mindestens eine, wenn es Kanten gibt). Rückgabe (Trainingsnetz, versteckte Kanten); beide zusammen sind das Gesamtnetz."""
    edges = edge_list(a)
    if len(edges) == 0:
        return a.copy(), edges
    k = min(len(edges), max(1, int(round(frac * len(edges)))))
    idx = np.random.default_rng(int(seed)).choice(len(edges), size=k, replace=False)
    hidden = edges[np.sort(idx)]
    return remove_edges(a, hidden), hidden


# --- Nähemaße (Matrizen über alle Paare) --------------------------------------------------------------------------------------------------------------


def common_neighbors(t) -> np.ndarray:
    return t @ t


def jaccard(t) -> np.ndarray:
    deg = t.sum(1)
    cn = t @ t
    union = deg[:, None] + deg[None, :] - cn
    return np.divide(cn, union, out=np.zeros_like(cn), where=union > 0)


def adamic_adar(t) -> np.ndarray:
    """Summe über gemeinsame Nachbarn z von 1 / ln(Grad von z); ein gemeinsamer Nachbar hat Grad >= 2, Knoten mit Grad 0 oder 1 tragen nichts bei."""
    deg = t.sum(1)
    w = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2.0)), 0.0)
    return (t * w[None, :]) @ t


def resource_allocation(t) -> np.ndarray:
    deg = t.sum(1)
    w = np.where(deg > 0, 1.0 / np.maximum(deg, 1.0), 0.0)
    return (t * w[None, :]) @ t


def preferential_attachment(t, deg=None) -> np.ndarray:
    """Produkt der Grade; `deg` überschreibt die Grade des Trainingsnetzes (das Datenleck der Demo: Grade aus dem vollen Netz)."""
    d = t.sum(1) if deg is None else np.asarray(deg, dtype=float)
    return np.outer(d, d)


def katz(t, beta_factor) -> np.ndarray:
    """Katz-Index (I - beta A)^-1 - I mit beta = beta_factor / größter Eigenwert (gewichtete Zahl aller Wege, kürzere zählen mehr); beta = 0 ohne Kanten."""
    n = len(t)
    lam = float(np.linalg.eigvalsh(t)[-1]) if n else 0.0
    beta = beta_factor / lam if lam > 0 else 0.0
    return np.linalg.inv(np.eye(n) - beta * t) - np.eye(n)


def random_score(n, seed) -> np.ndarray:
    r = np.random.default_rng(int(seed)).random((n, n))
    return (r + r.T) / 2.0


def score_all(t, names, beta_factor, seed, deg=None) -> dict:
    """Alle Maße der Liste `names` (Namen aus lkp_constants) als Matrizen über alle Paare; `deg` überschreibt die Grade der bevorzugten Anbindung."""
    funcs = {
        C.M_CN: lambda: common_neighbors(t), C.M_JACCARD: lambda: jaccard(t), C.M_AA: lambda: adamic_adar(t), C.M_RA: lambda: resource_allocation(t),
        C.M_PA: lambda: preferential_attachment(t, deg), C.M_KATZ: lambda: katz(t, beta_factor), C.M_RANDOM: lambda: random_score(len(t), seed),
    }
    return {name: funcs[name]() for name in names}


# --- Auswertung -----------------------------------------------------------------------------------------------------------------------------------


def average_ranks(values) -> np.ndarray:
    v = np.round(np.asarray(values, dtype=float), ROUND)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auc(pos, neg) -> float:
    """Wahrscheinlichkeit, dass einer versteckten Kante ein größerer Score gehört als einer Nichtkante (Gleichstand zählt halb); `nan` ohne Paare."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = average_ranks(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def sample_pairs(pairs, k, seed) -> np.ndarray:
    """k Paare ohne Zurücklegen (alle, wenn es weniger gibt), in der Reihenfolge der Liste."""
    if len(pairs) <= k:
        return pairs
    idx = np.random.default_rng(int(seed)).choice(len(pairs), size=k, replace=False)
    return pairs[np.sort(idx)]


def hard_pairs(t, pairs) -> np.ndarray:
    """Schwere Negative: Paare ohne Kante, die im Trainingsnetz mindestens einen gemeinsamen Nachbarn haben (Abstand 2)."""
    if len(pairs) == 0:
        return pairs
    cn = t @ t
    return pairs[cn[pairs[:, 0], pairs[:, 1]] >= 1]


def coverage(t, pairs) -> float:
    """Anteil der Paare mit mindestens einem gemeinsamen Nachbarn im Trainingsnetz; `nan` ohne Paare."""
    if len(pairs) == 0:
        return float("nan")
    cn = t @ t
    return float((cn[pairs[:, 0], pairs[:, 1]] >= 1).mean())


def candidate_pairs(t) -> np.ndarray:
    """Alle Paare ohne Trainingskante: unter ihnen die versteckten Kanten und die echten Nichtkanten."""
    return non_edges(t)


def hidden_mask(cand, hidden, n) -> np.ndarray:
    """Welche Kandidatenpaare sind versteckte Kanten?"""
    if len(hidden) == 0 or len(cand) == 0:
        return np.zeros(len(cand), dtype=bool)
    return np.isin(cand[:, 0] * n + cand[:, 1], hidden[:, 0] * n + hidden[:, 1])


def expected_precision_at_k(scores, is_hit, k) -> float:
    """Erwartete Precision@k, wenn Gleichstände zufällig aufgelöst werden (exakt): ganze Gruppen gleicher Scores zählen voll, die Gruppe am Rand anteilig
    (hits in der Gruppe / Gruppengröße mal die noch fehlenden Plätze). k = 0 gibt `nan`; k größer als die Zahl der Paare wird auf diese Zahl begrenzt."""
    s = np.round(np.asarray(scores, dtype=float), ROUND)
    h = np.asarray(is_hit, dtype=float)
    k = min(int(k), len(s))
    if k <= 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, h = s[order], h[order]
    values, start = np.unique(-s, return_index=True)                 # Gruppen gleicher Scores in absteigender Reihenfolge
    ends = np.append(start[1:], len(s))
    hits = 0.0
    taken = 0
    for a, b in zip(start, ends):
        size = b - a
        if taken + size <= k:
            hits += h[a:b].sum()
            taken += size
            if taken == k:
                break
        else:
            hits += (k - taken) * h[a:b].sum() / size
            break
    return float(hits / k)


def lift(precision, k, n_candidates) -> float:
    """Precision@k geteilt durch den Anteil versteckter Kanten unter den Kandidatenpaaren (= Precision@k bei Zufall)."""
    if n_candidates <= 0 or k <= 0 or precision != precision:
        return float("nan")
    return float(precision / (min(k, n_candidates) / n_candidates))


def top_pairs(scores, cand, k) -> np.ndarray:
    """Die k Kandidatenpaare mit den größten Scores (für die Anzeige; Gleichstand: kleinere Position in der Kandidatenliste, die Kennzahl rechnet mit der erwarteten Auflösung)."""
    s = np.round(np.asarray(scores, dtype=float), ROUND)
    order = np.argsort(-s, kind="mergesort")[:k]
    return cand[order]


# --- Blockmodell und Wachstum (Experimente 4 und 5) -------------------------------------------------------------------------------------------------


def sbm_probabilities(n, q, c, inside):
    """Kantenwahrscheinlichkeit innerhalb (p_in) und zwischen (p_out) den Gruppen für mittleren Grad c und den Anteil `inside` der Kanten innerhalb; auf höchstens 1 begrenzt.
    Gruppen gleich groß (Knoten i gehört zu Gruppe i * q // n). Gibt es keine Paare innerhalb oder zwischen den Gruppen, ist die jeweilige Wahrscheinlichkeit 0."""
    lab = np.arange(n) * q // n
    same = lab[:, None] == lab[None, :]
    n_in = (same.sum() - n) / 2.0
    n_out = n * (n - 1) / 2.0 - n_in
    m = c * n / 2.0
    p_in = min(1.0, inside * m / n_in) if n_in > 0 else 0.0
    p_out = min(1.0, (1.0 - inside) * m / n_out) if n_out > 0 else 0.0
    return p_in, p_out


def sbm_matrix(n, q, c, inside, seed):
    """Blockmodell: q gleich große Gruppen, mittlerer Grad c, Anteil `inside` der Kanten innerhalb der Gruppen. Rückgabe (Matrix, Gruppe je Knoten)."""
    lab = np.arange(n) * q // n
    p_in, p_out = sbm_probabilities(n, q, c, inside)
    prob = np.where(lab[:, None] == lab[None, :], p_in, p_out)
    u = np.random.default_rng(int(seed)).random((n, n))
    upper = np.triu((u < prob).astype(float), 1)
    return upper + upper.T, lab


def future_edges(a, mech, k, seed, labels=None) -> np.ndarray:
    """k neue Kanten (a < b, noch nicht im Netz) nach einem Mechanismus: `wie bisher` (80 % innerhalb der Gruppe des ersten Endes), `bevorzugte Anbindung` (zweites Ende proportional zum Grad)
    oder `zufällig` (beide Enden gleichverteilt). Bricht nach 200 000 Versuchen ab (dichte Netze)."""
    rng = np.random.default_rng(int(seed))
    n = len(a)
    deg = a.sum(1)
    new = set()
    tries = 0
    while len(new) < k and tries < 200_000:
        tries += 1
        u = int(rng.integers(n))
        if mech == C.MECH_SAME:
            same = np.flatnonzero(labels == labels[u])
            v = int(rng.choice(same)) if rng.random() < 0.8 else int(rng.integers(n))
        elif mech == C.MECH_PA:
            v = int(rng.choice(n, p=deg / deg.sum())) if deg.sum() > 0 else int(rng.integers(n))
        else:
            v = int(rng.integers(n))
        lo, hi = min(u, v), max(u, v)
        if lo != hi and a[lo, hi] == 0:
            new.add((lo, hi))
    return np.array(sorted(new), dtype=int).reshape(-1, 2)
