"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import dataclasses

import lkp_constants as C
import lkp_evaluation as ev

NAN = float("nan")


def _an(method=C.M_CN, auc_easy=0.73, auc_hard=0.36, prec=0.04, lift=10.0, cov=0.62, ncov=0.14, k=45, leak=False):
    base = ev.analyse(ev.Settings(C.T_RANDOM, 60, 4, 3))
    scores = dict(base.proto["scores"])
    scores[method] = {"auc_easy": auc_easy, "auc_hard": auc_hard, "precision": prec, "lift": lift}
    proto = dict(base.proto, k=k, coverage=cov, nonedge_coverage=ncov, scores=scores)
    return dataclasses.replace(base, proto=proto, method=method, leak=leak)


def _row(typ, cov=0.5, easy=0.6, hard=0.3, prec=0.05, lift=10.0, **over):
    scores = {m: {"auc_easy": easy, "auc_hard": hard, "precision": prec, "lift": lift} for m in C.SCORE_NAMES}
    scores[C.M_RANDOM] = {"auc_easy": 0.5, "auc_hard": 0.5, "precision": 0.0, "lift": 0.0}
    for m, vals in over.items():                                                  # Schlüssel sind Namen der Konstanten, z. B. M_JACCARD
        scores[getattr(C, m)] = {**scores[getattr(C, m)], **vals}
    return {"typ": typ, "coverage": cov, "nonedge_coverage": 0.1, "hard_n": 100.0, "scores": scores}


def test_de_uses_a_decimal_comma_and_a_dash_for_undefined_numbers():
    assert ev.de(1.26, 1) == "1,3" and ev.de(0.9487, 3) == "0,949" and ev.de(3, 0) == "3" and ev.de(float("nan"), 2) == "–" and ev.de(None) == "–"


def test_main_verdict_info_for_a_partial_finding():
    kind, text = ev.verdict_main(_an())
    assert kind == "info" and text.startswith("Das Maß findet einen Teil der fehlenden Kanten.")
    assert "Gemeinsame Nachbarn: AUC 0,73 mit leichten Negativen, mit schweren Negativen (Nichtkanten im Abstand 2) 0,36; Precision@45 0,040 (Lift 10 gegen Raten)" in text
    assert "62 % der versteckten Kanten haben im Trainingsnetz einen gemeinsamen Nachbarn, bei beliebigen Nichtkanten sind es 14 %" in text


def test_main_verdict_thresholds():
    assert ev.verdict_main(_an(auc_easy=0.44))[0] == "warning" and "unter dem Zufall" in ev.verdict_main(_an(auc_easy=0.44))[1]
    assert "nichts zu finden" in ev.verdict_main(_an(auc_easy=0.45))[1] and ev.verdict_main(_an(auc_easy=0.45))[0] == "warning"          # genau 0,45: nicht mehr unter dem Zufall
    assert "nichts zu finden" in ev.verdict_main(_an(auc_easy=0.549))[1]
    assert ev.verdict_main(_an(auc_easy=0.55))[0] == "info"                                                                             # genau 0,55: etwas gefunden
    assert ev.verdict_main(_an(auc_easy=0.79))[0] == "info" and ev.verdict_main(_an(auc_easy=0.8))[0] == "success"                        # genau 0,8: Erfolg
    assert "Das Maß findet die fehlenden Kanten." in ev.verdict_main(_an(auc_easy=0.8))[1]


def test_main_verdict_warns_that_the_auc_deceives_when_the_precision_is_tiny():
    kind, text = ev.verdict_main(_an(auc_easy=0.9, prec=0.019, lift=2.0))
    assert kind == "warning" and "**AUC täuscht**" in text
    assert ev.verdict_main(_an(auc_easy=0.9, prec=0.02))[0] == "success"                                              # genau 0,02 zählt nicht mehr als winzig
    assert ev.verdict_main(_an(auc_easy=0.5, prec=0.0))[1].startswith("Hier ist **nichts zu finden**")                 # Vorrang der schwächeren Aussage


def test_main_verdict_random_baseline_hard_negatives_missing_and_leak():
    kind, text = ev.verdict_main(_an(method=C.M_RANDOM, auc_easy=0.5))
    assert kind == "info" and text.startswith("Grundlinie: Zufall: AUC 0,50")
    assert "schwere Negative gibt es zu wenige" in ev.verdict_main(_an(auc_hard=NAN))[1]
    t = ev.verdict_main(_an(method=C.M_PA, auc_easy=0.91, prec=0.009, lift=4.0, leak=True))[1]
    assert "**Datenleck:**" in t and "**AUC täuscht**" in t
    assert "Datenleck" not in ev.verdict_main(_an(method=C.M_PA, auc_easy=0.91, prec=0.5, leak=False))[1]
    assert "Datenleck" not in ev.verdict_main(_an(method=C.M_CN, auc_easy=0.9, prec=0.5, leak=True))[1]                  # das Leck betrifft nur die bevorzugte Anbindung
    assert "Grade der Kantenenden" in ev.verdict_main(_an(method=C.M_PA, auc_easy=0.19))[1]


def test_main_verdict_without_hidden_edges():
    base = ev.analyse(ev.Settings(C.T_RANDOM, 60, 4, 3))
    kind, text = ev.verdict_main(dataclasses.replace(base, proto=dict(base.proto, k=0)))
    assert kind == "warning" and "keine Kanten" in text


def test_types_verdict_names_range_best_worst_and_hard_negatives():
    rows = [_row("A", cov=0.13, easy=0.47, hard=0.07), _row("B", cov=0.94, easy=0.95, hard=0.70)]
    kind, text = ev.verdict_types(rows)
    assert kind == "info"
    assert "mit leichten Negativen 0,47 bis 0,95 (am besten in „B“, dort haben 94 % der versteckten Kanten einen gemeinsamen Nachbarn; am wenigsten in „A“ mit 13 %)" in text
    assert "In „A“ ist mit keinem Maß etwas zu finden (alle unter 0,55)." in text and "auf 0,07 bis 0,70" in text
    assert "ist mit keinem Maß etwas zu finden" not in ev.verdict_types([_row("A", easy=0.6), _row("B", easy=0.9)])[1]


def test_types_verdict_threshold_for_nothing_to_find():
    low = _row("A", easy=0.549)
    assert "In „A“ ist mit keinem Maß" in ev.verdict_types([low, _row("B", easy=0.9)])[1]
    assert "In „A“ ist mit keinem Maß" not in ev.verdict_types([_row("A", easy=0.55), _row("B", easy=0.9)])[1]


def test_precision_verdict_names_the_largest_lift_and_the_weakest_type():
    rows = [_row("Zufallsnetz", prec=0.002, lift=1.0), _row("Gitter", prec=0.1, lift=20.0, M_JACCARD={"precision": 0.335, "lift": 88.0}),
            _row("Verbünde", prec=0.019, lift=5.0)]
    kind, text = ev.verdict_precision(rows)
    assert kind == "info"
    assert "Der größte Lift ist 88 („Jaccard“ in „Gitter“, Precision 0,335)" in text and "gemeinsame Nachbarn erreichen in „Zufallsnetz“ nur 0,002" in text
    assert "In „Zufallsnetz“ liegt keine Precision über 0,5 %." in text
    assert "keine Precision über 0,5 %" not in ev.verdict_precision([_row("A", prec=0.005), _row("B", prec=0.1)])[1]                  # genau 0,005 zählt nicht mehr als „unter“
    assert ev.verdict_precision([_row("A", lift=NAN)])[0] == "warning"


def test_katz_verdict_counts_wins_and_ties():
    rows = [{"typ": "Gitter", "c": 3, "cn": 0.47, "katz": 0.87}, {"typ": "Ring", "c": 3, "cn": 0.92, "katz": 0.93}, {"typ": "Verbünde", "c": 6, "cn": 0.67, "katz": 0.76},
            {"typ": "Zufallsnetz", "c": 6, "cn": 0.47, "katz": 0.47}]
    kind, text = ev.verdict_katz(rows)
    assert kind == "info" and "in 2 von 4 Zellen mehr als 0,05 über den gemeinsamen Nachbarn, gleichauf (bis 0,02) in 2" in text and "„Gitter“ bei c = 3 mit 0,87 gegen 0,47" in text
    edge = [{"typ": "A", "c": 3, "cn": 0.5, "katz": 0.5625}, {"typ": "B", "c": 3, "cn": 0.5, "katz": 0.515625}, {"typ": "C", "c": 3, "cn": 0.5, "katz": 0.53125}]
    assert "in 1 von 3 Zellen mehr als 0,05 über den gemeinsamen Nachbarn, gleichauf (bis 0,02) in 1." in ev.verdict_katz(edge)[1]           # 0,0625 gewinnt, 0,0156 gleichauf, 0,03125 keines
    assert ev.verdict_katz([{"typ": "A", "c": 3, "cn": NAN, "katz": NAN}])[0] == "warning"


def test_community_verdict_names_unstructured_clear_and_dense_cells():
    rows = [{"c": 6, "inside": 0.17, "cn": 0.47, "katz": 0.51, "pa": 0.52, "group": 0.51}, {"c": 6, "inside": 0.95, "cn": 0.74, "katz": 0.90, "pa": 0.46, "group": 0.92},
            {"c": 12, "inside": 0.17, "cn": 0.50, "katz": 0.49, "pa": 0.48, "group": 0.50}, {"c": 12, "inside": 0.95, "cn": 0.93, "katz": 0.94, "pa": 0.43, "group": 0.94}]
    kind, text = ev.verdict_community(rows)
    assert kind == "info"
    assert "Ohne Struktur (Anteil innen 0,17) liegen gemeinsame Nachbarn bei 0,47, mit klaren Gruppen (Anteil innen 0,95) bei c = 6 bei 0,74 und Katz bei 0,90; die Obergrenze „gleiche Gruppe“ ist 0,92." in text
    assert "Bei c = 12 und Anteil 0,95 liegen gemeinsame Nachbarn bei 0,93, Katz bei 0,94, die Obergrenze bei 0,94." in text
    assert ev.verdict_community([{"c": 6, "inside": 0.5, "cn": NAN, "katz": NAN, "pa": NAN, "group": NAN}])[0] == "warning"


def _time(same, pa, rnd):
    names = (C.M_CN, C.M_AA, C.M_PA, C.M_KATZ)
    mk = lambda v: {m: v for m in names}
    return [{"mech": C.MECH_SAME, "time": mk(same[0]), "random": mk(same[1])}, {"mech": C.MECH_PA, "time": mk(pa[0]), "random": mk(pa[1])}, {"mech": C.MECH_RANDOM, "time": mk(rnd[0]), "random": mk(rnd[1])}]


def test_time_verdict_names_the_worst_mechanism_and_the_same_case():
    kind, text = ev.verdict_time(_time((0.82, 0.78), (0.53, 0.67), (0.52, 0.70)))
    assert kind == "warning"
    assert "Entstehen die neuen Kanten zufällig, erreicht Katz mit Zeitaufteilung 0,52, mit Zufallsaufteilung 0,70" in text and "überschätzt hier um 0,18" in text
    assert "Entstehen die neuen Kanten wie die alten, stimmen beide Aufteilungen überein (Katz 0,82 gegen 0,78)." in text


def test_time_verdict_thresholds_and_different_same_case():
    assert ev.verdict_time(_time((0.5, 0.5), (0.5, 0.5), (0.5, 0.5)))[0] == "info"                                      # keine Lücke: keine Warnung
    assert ev.verdict_time(_time((0.5, 0.5), (0.5, 0.5625), (0.5, 0.5)))[0] == "warning"                                  # Lücke 0,0625 über der Schwelle 0,05
    assert "unterscheiden sich beide Aufteilungen auch dann" in ev.verdict_time(_time((0.5, 0.6), (0.5, 0.5), (0.5, 0.5)))[1]
    assert ev.verdict_time(_time((NAN, NAN), (NAN, NAN), (NAN, NAN)))[0] == "warning"


def test_leak_verdict_names_the_largest_gain_and_the_value_below_chance():
    rows = [{"typ": "A", "clean": 0.50, "leak": 0.65}, {"typ": "B", "clean": 0.18, "leak": 0.52}]
    kind, text = ev.verdict_leak(rows)
    assert kind == "warning" and "in „B“ von 0,18 auf 0,52" in text and "In „B“ liegt sie ohne Leck bei 0,18, unter dem Zufall" in text and "Fehler der Auswertung" in text
    assert "unter dem Zufall" not in ev.verdict_leak([{"typ": "A", "clean": 0.45, "leak": 0.6}])[1]                         # genau 0,45 zählt nicht als darunter
    assert ev.verdict_leak([{"typ": "A", "clean": NAN, "leak": NAN}])[0] == "warning"
