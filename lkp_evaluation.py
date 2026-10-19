"""Kennzahlen, Sweeps und Verdikte der Demo „Link-Prediction“. Verdikte sind reine Funktionen (Art, Text); jede Zahl im Text kommt aus der Rechnung.

Protokoll: ein Teil der Kanten wird zufällig versteckt, die Maße rechnen auf dem Rest. AUC: versteckte Kanten gegen Nichtkanten des vollen Netzes, leichte Negative = zufällige Nichtkanten
(das Fünffache der versteckten Kanten, mindestens 200), schwere Negative = Nichtkanten mit mindestens einem gemeinsamen Nachbarn im Trainingsnetz. Precision@k: alle Paare ohne Trainingskante nach
Score sortiert, k = Zahl der versteckten Kanten, Gleichstände zufällig aufgelöst (erwartete Precision, exakt); Lift = Precision@k geteilt durch die Precision@k bei Zufall.
Jede Messreihe nimmt den Median über feste Netze (`LIVE_SEEDS` in der App, `SWEEP_SEEDS` für die README-Zahlen): nie ein einzelner Seed als Befund."""

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

import lkp_algorithm as A
import lkp_constants as C
from lkp_scenario import Instance, make_instance

NAN = float("nan")


def de(x, digits=1) -> str:
    """Dezimalzahl mit Komma (deutsche Texte); eine nicht definierte Zahl erscheint als Gedankenstrich."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{digits}f}".replace(".", ",")


def _med(values) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(median(vals)) if vals else NAN


# --- Protokoll ----------------------------------------------------------------------------------------------------------------------------------------


def evaluate_protocol(a_full, t, hidden, seed, names=C.SCORE_NAMES, pa_deg=None, keep_scores=False) -> dict:
    """Wertet alle Maße auf einer Aufteilung aus. Rückgabe: je Maß {auc_easy, auc_hard, precision, lift} und die Kennzahlen der Aufteilung
    (`coverage`: Anteil der versteckten Kanten mit gemeinsamem Nachbarn im Trainingsnetz, `nonedge_coverage`: dasselbe für alle Nichtkanten, `hard_n`: Zahl schwerer Negative im Pool,
    `k`, `n_cand`; mit `keep_scores` zusätzlich `matrices`: die Score-Matrizen je Maß)."""
    n = len(a_full)
    ne = A.non_edges(a_full)
    k = len(hidden)
    n_neg = max(C.NEG_FACTOR * k, C.NEG_MIN)
    easy = A.sample_pairs(ne, n_neg, seed + 1)
    pool = A.hard_pairs(t, ne)
    hard = A.sample_pairs(pool, n_neg, seed + 2) if len(pool) >= C.HARD_MIN else None
    cand = A.candidate_pairs(t)
    is_hid = A.hidden_mask(cand, hidden, n)
    scores = A.score_all(t, names, C.KATZ_BETA_FACTOR, seed + 3, deg=pa_deg)
    per = {}
    for name, m in scores.items():
        pos = m[hidden[:, 0], hidden[:, 1]]
        prec = A.expected_precision_at_k(m[cand[:, 0], cand[:, 1]], is_hid, k)
        per[name] = {"auc_easy": A.auc(pos, m[easy[:, 0], easy[:, 1]]) if len(easy) else NAN,
                     "auc_hard": A.auc(pos, m[hard[:, 0], hard[:, 1]]) if hard is not None else NAN,
                     "precision": prec, "lift": A.lift(prec, k, len(cand))}
    out = {"scores": per, "coverage": A.coverage(t, hidden), "nonedge_coverage": A.coverage(t, ne), "hard_n": len(pool), "k": k, "n_cand": len(cand)}
    if keep_scores:
        out["matrices"] = scores
    return out


def build_matrix(typ, n, c, seed, nav=C.NAV_COUNT):
    inst = make_instance(typ, n, c, seed, nav, layout=False)
    return A.to_matrix(inst.adj)


# --- Hauptanalyse -----------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    netztyp: str = C.T_COMMUNITIES
    n: int = C.DEFAULT_N
    c: int = C.DEFAULT_C
    seed: int = C.DEFAULT_SEED
    frac: int = C.DEFAULT_FRAC
    method: str = C.DEFAULT_METHOD
    nav: str = C.NAV_COUNT
    leak: str = C.LEAK_OFF


@dataclass(frozen=True, eq=False)
class Analysis:
    inst: Instance
    a: np.ndarray             # volles Netz
    t: np.ndarray             # Trainingsnetz
    hidden: np.ndarray        # versteckte Kanten
    proto: dict               # evaluate_protocol
    method: str
    leak: bool
    top: np.ndarray           # die k besten Vorhersagen des gewählten Maßes (Anzeige: Gleichstand nach Position)
    top_hit: np.ndarray       # Treffer je Vorhersage

    @property
    def k(self) -> int:
        return self.proto["k"]

    @property
    def result(self) -> dict:
        return self.proto["scores"][self.method]


def analyse(settings: Settings) -> Analysis:
    inst = make_instance(settings.netztyp, settings.n, settings.c, settings.seed, settings.nav, layout=True)
    a = A.to_matrix(inst.adj)
    t, hidden = A.split_random(a, settings.frac / 100.0, settings.seed)
    leak = settings.leak == C.LEAK_ON
    proto = evaluate_protocol(a, t, hidden, settings.seed, pa_deg=a.sum(1) if leak else None, keep_scores=True)
    scores = proto["matrices"][settings.method]
    cand = A.candidate_pairs(t)
    top = A.top_pairs(scores[cand[:, 0], cand[:, 1]], cand, len(hidden)) if len(hidden) else np.zeros((0, 2), dtype=int)
    hid = {tuple(e) for e in hidden.tolist()}
    return Analysis(inst, a, t, hidden, proto, settings.method, leak, top, np.array([tuple(e) in hid for e in top.tolist()], dtype=bool))


def verdict_main(an: Analysis) -> tuple[str, str]:
    """Verdikt der Kernfrage: wie viel verrät die Struktur über die fehlenden Kanten, und was sagt der Wert?"""
    if an.k == 0:
        return "warning", "Das Netz hat keine Kanten, die man verstecken könnte."
    r = an.result
    cov, ncov = an.proto["coverage"], an.proto["nonedge_coverage"]
    hard = (f", mit schweren Negativen (Nichtkanten im Abstand 2) {de(r['auc_hard'], 2)}" if not math.isnan(r["auc_hard"]) else ", schwere Negative gibt es zu wenige")
    base = (f"{an.method}: AUC {de(r['auc_easy'], 2)} mit leichten Negativen{hard}; Precision@{an.k} {de(r['precision'], 3)} (Lift {de(r['lift'], 0)} gegen Raten). "
            f"{de(100 * cov, 0)} % der versteckten Kanten haben im Trainingsnetz einen gemeinsamen Nachbarn, bei beliebigen Nichtkanten sind es {de(100 * ncov, 0)} %.")
    leak = " **Datenleck:** die Grade stammen aus dem vollen Netz, das die versteckten Kanten enthält." if an.leak and an.method == C.M_PA else ""
    if an.method == C.M_RANDOM:
        return "info", "Grundlinie: " + base
    if r["auc_easy"] < 0.45:
        return "warning", (f"Das Maß liegt **unter dem Zufall**. " + base + " Das Verstecken senkt die Grade der Kantenenden, die Maße auf Basis der Grade sehen das." + leak)
    if r["auc_easy"] < 0.55:
        return "warning", "Hier ist **nichts zu finden**: das Maß ist nicht besser als Raten. " + base + leak
    if r["precision"] < 0.02:
        return "warning", ("Die **AUC täuscht**: die besten Vorhersagen enthalten kaum eine versteckte Kante, weil viele Paare im Abstand 2 vor ihnen stehen (und ein Leck verstärkt den Eindruck). "
                           + base + leak)
    if r["auc_easy"] >= 0.8:
        return "success", "Das Maß findet die fehlenden Kanten. " + base + leak
    return "info", "Das Maß findet einen Teil der fehlenden Kanten. " + base + leak


# --- E1 und E2: Netztypen ------------------------------------------------------------------------------------------------------------------------------


def types_rows(n, c, frac, seeds=C.LIVE_SEEDS, types=None, nav=C.NAV_COUNT) -> list[dict]:
    """Alle Maße über die Netztypen: AUC leicht und schwer, Precision@k, Lift und Abdeckung; Median über die festen Netze (beim festen Link-Netz über die Aufteilungen)."""
    rows = []
    for typ in types or C.GENERATED_TYPES:
        recs = []
        for seed in seeds:
            a = build_matrix(typ, n, c, seed, nav)
            t, hidden = A.split_random(a, frac / 100.0, seed)
            if len(hidden):
                recs.append(evaluate_protocol(a, t, hidden, seed))
        row = {"typ": typ, "coverage": _med([r["coverage"] for r in recs]), "nonedge_coverage": _med([r["nonedge_coverage"] for r in recs]), "hard_n": _med([float(r["hard_n"]) for r in recs]), "scores": {}}
        for name in C.SCORE_NAMES:
            row["scores"][name] = {key: _med([r["scores"][name][key] for r in recs]) for key in ("auc_easy", "auc_hard", "precision", "lift")}
        rows.append(row)
    return rows


def verdict_types(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E1: wo findet die Struktur etwas, und was zeigen schwere Negative und Abdeckung?"""
    methods = [m for m in C.SCORE_NAMES if m != C.M_RANDOM]
    best = {r["typ"]: max(methods, key=lambda m: r["scores"][m]["auc_easy"] if not math.isnan(r["scores"][m]["auc_easy"]) else -1) for r in rows}
    none = [r["typ"] for r in rows if max(r["scores"][m]["auc_easy"] for m in methods) < 0.55]
    cn = [r["scores"][C.M_CN]["auc_easy"] for r in rows]
    hard = [r["scores"][C.M_CN]["auc_hard"] for r in rows if not math.isnan(r["scores"][C.M_CN]["auc_hard"])]
    top = max(rows, key=lambda r: r["scores"][C.M_CN]["auc_easy"])
    low = min(rows, key=lambda r: r["coverage"])
    none_txt = f" In „{none[0]}“ ist mit keinem Maß etwas zu finden (alle unter 0,55)." if none else ""
    hard_txt = f" Mit schweren Negativen fallen gemeinsame Nachbarn auf {de(min(hard), 2)} bis {de(max(hard), 2)} (ein Wert unter 0,5 ist Konstruktion: schwere Negative haben per Definition gemeinsame Nachbarn)." if hard else ""
    return "info", (f"Gemeinsame Nachbarn erreichen mit leichten Negativen {de(min(cn), 2)} bis {de(max(cn), 2)} (am besten in „{top['typ']}“, dort haben {de(100 * top['coverage'], 0)} % der versteckten Kanten "
                    f"einen gemeinsamen Nachbarn; am wenigsten in „{low['typ']}“ mit {de(100 * low['coverage'], 0)} %)." + none_txt + hard_txt)


def verdict_precision(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E2: Lift gegen Zufall und absolute Precision."""
    methods = [m for m in C.SCORE_NAMES if m != C.M_RANDOM]
    cells = [(r["typ"], m, r["scores"][m]) for r in rows for m in methods if not math.isnan(r["scores"][m]["lift"])]
    if not cells:
        return "warning", "Keine Precision berechenbar."
    typ, m, cell = max(cells, key=lambda x: x[2]["lift"])
    zero = [r["typ"] for r in rows if all(r["scores"][mm]["precision"] < 0.005 for mm in methods)]
    low = min(rows, key=lambda r: r["scores"][C.M_CN]["precision"])
    zero_txt = f" In „{zero[0]}“ liegt keine Precision über 0,5 %." if zero else ""
    return "info", (f"Der größte Lift ist {de(cell['lift'], 0)} („{m}“ in „{typ}“, Precision {de(cell['precision'], 3)}). Absolut bleibt die Treffsicherheit klein: gemeinsame Nachbarn erreichen in „{low['typ']}“ "
                    f"nur {de(low['scores'][C.M_CN]['precision'], 3)}." + zero_txt)


# --- E3: Katz gegen gemeinsame Nachbarn ------------------------------------------------------------------------------------------------------------------


def katz_rows(n, frac, degrees=C.KATZ_DEGREES, seeds=C.LIVE_SEEDS, types=None) -> list[dict]:
    """AUC leicht von gemeinsamen Nachbarn und Katz je Netztyp bei dünnem und mittlerem Grad (Gitter: unter c = 6 ohne Diagonalen, also ohne Dreiecke)."""
    rows = []
    for typ in types or C.GENERATED_TYPES:
        for c in degrees:
            cn, kz = [], []
            for seed in seeds:
                a = build_matrix(typ, n, c, seed)
                t, hidden = A.split_random(a, frac / 100.0, seed)
                if len(hidden) == 0:
                    continue
                r = evaluate_protocol(a, t, hidden, seed, names=(C.M_CN, C.M_KATZ))["scores"]
                cn.append(r[C.M_CN]["auc_easy"])
                kz.append(r[C.M_KATZ]["auc_easy"])
            rows.append({"typ": typ, "c": c, "cn": _med(cn), "katz": _med(kz)})
    return rows


def verdict_katz(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E3: wann gewinnt Katz?"""
    valid = [r for r in rows if not math.isnan(r["cn"]) and not math.isnan(r["katz"])]
    if not valid:
        return "warning", "Keine Zelle berechenbar."
    wins = [r for r in valid if r["katz"] - r["cn"] > 0.05]
    top = max(valid, key=lambda r: r["katz"] - r["cn"])
    tie = [r for r in valid if abs(r["katz"] - r["cn"]) <= 0.02]
    return "info", (f"Katz liegt in {len(wins)} von {len(valid)} Zellen mehr als 0,05 über den gemeinsamen Nachbarn, gleichauf (bis 0,02) in {len(tie)}. "
                    f"Der größte Abstand: „{top['typ']}“ bei c = {top['c']} mit {de(top['katz'], 2)} gegen {de(top['cn'], 2)}.")


# --- E4: Gemeinschaftsstärke ------------------------------------------------------------------------------------------------------------------------------


def community_rows(n, frac, degrees=C.COMM_DEGREES, insides=C.INSIDE_SHARES, seeds=C.LIVE_SEEDS) -> list[dict]:
    """Blockmodell mit sechs Gruppen: AUC leicht von gemeinsamen Nachbarn, Katz und bevorzugter Anbindung und die Obergrenze „gleiche Gruppe“ (der Score ist 1 für Paare einer Gruppe)
    über den Anteil der Kanten innerhalb der Gruppen."""
    rows = []
    for c in degrees:
        for inside in insides:
            cn, kz, pa, grp = [], [], [], []
            for seed in seeds:
                a, lab = A.sbm_matrix(n, C.BLOCKS, c, inside, seed)
                t, hidden = A.split_random(a, frac / 100.0, seed)
                if len(hidden) == 0:
                    continue
                r = evaluate_protocol(a, t, hidden, seed, names=(C.M_CN, C.M_KATZ, C.M_PA))["scores"]
                cn.append(r[C.M_CN]["auc_easy"])
                kz.append(r[C.M_KATZ]["auc_easy"])
                pa.append(r[C.M_PA]["auc_easy"])
                ne = A.non_edges(a)
                easy = A.sample_pairs(ne, max(C.NEG_FACTOR * len(hidden), C.NEG_MIN), seed + 1)
                same = lambda p: (lab[p[:, 0]] == lab[p[:, 1]]).astype(float)
                grp.append(A.auc(same(hidden), same(easy)))
            rows.append({"c": c, "inside": inside, "cn": _med(cn), "katz": _med(kz), "pa": _med(pa), "group": _med(grp)})
    return rows


def verdict_community(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E4: hilft die Gemeinschaftsstruktur den Maßen, und erreichen sie die Obergrenze?"""
    valid = [r for r in rows if not math.isnan(r["cn"])]
    if not valid:
        return "warning", "Keine Zelle berechenbar."
    lo = min(valid, key=lambda r: r["inside"])
    hi = max(valid, key=lambda r: (r["inside"], -r["c"]))
    dense = max(valid, key=lambda r: (r["c"], r["inside"]))
    return "info", (f"Ohne Struktur (Anteil innen {de(lo['inside'], 2)}) liegen gemeinsame Nachbarn bei {de(lo['cn'], 2)}, mit klaren Gruppen (Anteil innen {de(hi['inside'], 2)}) bei c = {hi['c']} bei {de(hi['cn'], 2)} "
                    f"und Katz bei {de(hi['katz'], 2)}; die Obergrenze „gleiche Gruppe“ ist {de(hi['group'], 2)}. Bei c = {dense['c']} und Anteil {de(dense['inside'], 2)} liegen gemeinsame Nachbarn bei {de(dense['cn'], 2)}, "
                    f"Katz bei {de(dense['katz'], 2)}, die Obergrenze bei {de(dense['group'], 2)}.")


# --- E5: Zeit- gegen Zufallsaufteilung ---------------------------------------------------------------------------------------------------------------


def time_rows(n, c, seeds=C.LIVE_SEEDS) -> list[dict]:
    """Altgraph: Blockmodell (sechs Gruppen, 80 % innen); neue Kanten (10 % der alten) entstehen nach einem Mechanismus. Zeitaufteilung: Training = Altgraph, Test = neue Kanten.
    Zufallsaufteilung: 10 % der Kanten des neuen Gesamtgraphen zufällig versteckt. AUC leicht von gemeinsamen Nachbarn, Adamic–Adar, bevorzugter Anbindung und Katz; Median."""
    names = (C.M_CN, C.M_AA, C.M_PA, C.M_KATZ)
    rows = []
    for mech in C.MECHANISMS:
        time_r = {m: [] for m in names}
        rand_r = {m: [] for m in names}
        for seed in seeds:
            a, lab = A.sbm_matrix(n, C.BLOCKS, c, 0.8, seed)
            k = max(1, int(round(C.FUTURE_FRAC * a.sum() / 2)))
            fut = A.future_edges(a, mech, k, seed + 7, lab)
            if len(fut) == 0:
                continue
            full = a.copy()
            full[fut[:, 0], fut[:, 1]] = 1.0
            full[fut[:, 1], fut[:, 0]] = 1.0
            r_time = evaluate_protocol(full, a, fut, seed, names=names)["scores"]
            t, hid = A.split_random(full, C.FUTURE_FRAC, seed)
            r_rand = evaluate_protocol(full, t, hid, seed, names=names)["scores"]
            for m in names:
                time_r[m].append(r_time[m]["auc_easy"])
                rand_r[m].append(r_rand[m]["auc_easy"])
        rows.append({"mech": mech, "time": {m: _med(time_r[m]) for m in names}, "random": {m: _med(rand_r[m]) for m in names}})
    return rows


def verdict_time(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E5: überschätzt die Zufallsaufteilung die Vorhersage neuer Kanten?"""
    valid = [r for r in rows if not math.isnan(r["time"][C.M_KATZ])]
    if not valid:
        return "warning", "Keine Aufteilung berechenbar."
    gaps = [(r["random"][C.M_KATZ] - r["time"][C.M_KATZ], r) for r in valid]
    gap, worst = max(gaps, key=lambda g: g[0])
    same = next((r for r in valid if r["mech"] == C.MECH_SAME), None)
    same_txt = ""
    if same:
        verb = "stimmen beide Aufteilungen überein" if abs(same["time"][C.M_KATZ] - same["random"][C.M_KATZ]) <= 0.05 else "unterscheiden sich beide Aufteilungen auch dann"
        same_txt = f" Entstehen die neuen Kanten wie die alten, {verb} (Katz {de(same['time'][C.M_KATZ], 2)} gegen {de(same['random'][C.M_KATZ], 2)})."
    kind = "warning" if gap > 0.05 else "info"
    return kind, (f"Entstehen die neuen Kanten {worst['mech']}, erreicht Katz mit Zeitaufteilung {de(worst['time'][C.M_KATZ], 2)}, mit Zufallsaufteilung {de(worst['random'][C.M_KATZ], 2)}: "
                  f"die Zufallsaufteilung misst, wie gut man die **alte** Struktur rekonstruiert, und überschätzt hier um {de(gap, 2)}." + same_txt)


# --- E6: Datenleck über die Grade ---------------------------------------------------------------------------------------------------------------------


def leak_rows(n, c, frac, seeds=C.LIVE_SEEDS, types=None) -> list[dict]:
    """AUC leicht der bevorzugten Anbindung mit den Graden des Trainingsnetzes und mit den Graden des vollen Netzes (das die versteckten Kanten enthält)."""
    rows = []
    for typ in types or C.GENERATED_TYPES:
        clean, leaked = [], []
        for seed in seeds:
            a = build_matrix(typ, n, c, seed)
            t, hidden = A.split_random(a, frac / 100.0, seed)
            if len(hidden) == 0:
                continue
            clean.append(evaluate_protocol(a, t, hidden, seed, names=(C.M_PA,))["scores"][C.M_PA]["auc_easy"])
            leaked.append(evaluate_protocol(a, t, hidden, seed, names=(C.M_PA,), pa_deg=a.sum(1))["scores"][C.M_PA]["auc_easy"])
        rows.append({"typ": typ, "clean": _med(clean), "leak": _med(leaked)})
    return rows


def verdict_leak(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E6: wie stark verändert das Leck das Ergebnis?"""
    valid = [r for r in rows if not math.isnan(r["clean"]) and not math.isnan(r["leak"])]
    if not valid:
        return "warning", "Keine Zelle berechenbar."
    top = max(valid, key=lambda r: r["leak"] - r["clean"])
    low = min(valid, key=lambda r: r["clean"])
    below = (f" In „{low['typ']}“ liegt sie ohne Leck bei {de(low['clean'], 2)}, unter dem Zufall: das Verstecken senkt die Grade der beiden Enden." if low["clean"] < 0.45 else "")
    return "warning", (f"Mit den Graden aus dem vollen Netz steigt die bevorzugte Anbindung in „{top['typ']}“ von {de(top['clean'], 2)} auf {de(top['leak'], 2)}." + below
                       + " Das Leck ist ein Fehler der Auswertung, nicht des Netzes.")
