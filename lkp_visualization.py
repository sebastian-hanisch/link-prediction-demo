"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import lkp_algorithm as A
import lkp_constants as C
from lkp_evaluation import de

MEASURE_COLORS = {C.M_CN: "#2F6B65", C.M_JACCARD: "#72b7b2", C.M_AA: "#4c78a8", C.M_RA: "#b279a2", C.M_PA: "#c4623b", C.M_KATZ: "#6b5b95", C.M_RANDOM: "#bab0ac"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340, bottom=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=bottom), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _name(inst, i):
    return inst.labels[i] if inst.labels else str(i)


def build_network(inst, t, hidden, top, top_hit, height=430):
    """Netz mit festen Positionen: Trainingskanten grau, versteckte Kanten, die das Maß unter seinen k besten Vorhersagen findet, grün, übersehene orange gestrichelt, falsche Vorhersagen rot gestrichelt."""
    n = inst.n
    fig = go.Figure()
    xs, ys = [], []
    for a, b in A.edge_list(t).tolist():
        xs += [inst.pos[a][0], inst.pos[b][0], None]
        ys += [inst.pos[a][1], inst.pos[b][1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_EDGE, width=0.7), hoverinfo="skip", name="Trainingskante"))

    def lines(pairs, color, name, dash, width):
        px, py = [], []
        for a, b in pairs:
            px += [inst.pos[a][0], inst.pos[b][0], None]
            py += [inst.pos[a][1], inst.pos[b][1], None]
        if px:
            fig.add_trace(go.Scatter(x=px, y=py, mode="lines", line=dict(color=color, width=width, dash=dash), hoverinfo="skip", name=name))

    found = {tuple(e) for e, h in zip(top.tolist(), top_hit.tolist()) if h}
    lines([tuple(e) for e in top.tolist() if tuple(e) in found], C.COLOR_HIT, "versteckt und gefunden", "solid", 2.6)
    lines([tuple(e) for e in hidden.tolist() if tuple(e) not in found], C.COLOR_MISS, "versteckt und übersehen", "dash", 2.0)
    lines([tuple(e) for e, h in zip(top.tolist(), top_hit.tolist()) if not h], C.COLOR_FALSE, "falsche Vorhersage", "dot", 1.6)
    size = 4 if n > 200 else 7
    fig.add_trace(go.Scatter(x=[p[0] for p in inst.pos], y=[p[1] for p in inst.pos], mode="markers", marker=dict(size=size, color="#4c78a8", line=dict(color="rgba(40,40,40,0.45)", width=0.4)),
                             hovertext=[f"Knoten {_name(inst, i)}" for i in range(n)], hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=[-0.04, 1.04], y=[-0.04, 1.04], mode="markers", marker=dict(size=1, color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, autorange=True)
    fig.update_yaxes(visible=False, autorange=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.02, font=dict(size=11)), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_overview(an):
    """AUC leicht und schwer sowie Precision@k aller Maße im gewählten Netz."""
    names = [m for m in C.SCORE_NAMES]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[an.proto["scores"][m]["auc_easy"] for m in names], name="AUC leichte Negative", marker_color=C.COLOR_A,
                         text=[de(an.proto["scores"][m]["auc_easy"], 2) for m in names], textposition="outside"))
    fig.add_trace(go.Bar(x=names, y=[None if math.isnan(an.proto["scores"][m]["auc_hard"]) else an.proto["scores"][m]["auc_hard"] for m in names], name="AUC schwere Negative", marker_color=C.COLOR_B,
                         text=[de(an.proto["scores"][m]["auc_hard"], 2) if not math.isnan(an.proto["scores"][m]["auc_hard"]) else "" for m in names], textposition="outside"))
    fig.add_hline(y=0.5, line=dict(color="rgba(120,120,120,0.9)", dash="dash"), annotation_text="Raten", annotation_position="top left")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="AUC", range=[0, 1.15])
    fig.update_xaxes(tickangle=-20)
    return _base(fig, height=340, bottom=70)


def _grouped(rows, series, label_key, ytitle, rng=(0, 1.15), height=380, hline=0.5, bottom=90):
    fig = go.Figure()
    for key, name, color in series:
        ys = [None if (r[key] is None or (isinstance(r[key], float) and math.isnan(r[key]))) else r[key] for r in rows]
        fig.add_trace(go.Bar(x=[r[label_key] for r in rows], y=ys, name=name, marker_color=color, text=[de(v, 2) if v is not None else "" for v in ys], textposition="outside"))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color="rgba(120,120,120,0.9)", dash="dash"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=ytitle, range=list(rng))
    fig.update_xaxes(tickangle=-25)
    return _base(fig, height=height, bottom=bottom)


def _flat(rows, key):
    """Zeilen mit den Werten eines Maßes als flache Schlüssel (für die Balkendiagramme)."""
    out = []
    for r in rows:
        flat = {"typ": r["typ"]}
        for m in C.SCORE_NAMES:
            flat[m] = r["scores"][m][key]
        out.append(flat)
    return out


def build_auc(rows, key="auc_easy"):
    """AUC (leichte oder schwere Negative) aller Maße außer Zufall je Netztyp."""
    methods = [m for m in C.SCORE_NAMES if m != C.M_RANDOM]
    return _grouped(_flat(rows, key), [(m, m, MEASURE_COLORS[m]) for m in methods], "typ", "AUC", height=400)


def build_coverage(rows):
    """Anteil der Paare mit gemeinsamem Nachbarn im Trainingsnetz: versteckte Kanten gegen alle Nichtkanten."""
    return _grouped(rows, [("coverage", "versteckte Kanten", C.COLOR_A), ("nonedge_coverage", "alle Nichtkanten", C.COLOR_B)], "typ", "Anteil mit gemeinsamem Nachbarn", hline=None, height=340)


def build_lift(rows):
    """Lift von Precision@k gegen Zufall (logarithmische Achse; ein Lift von 0 entfällt)."""
    methods = [m for m in C.SCORE_NAMES if m not in (C.M_RANDOM,)]
    flat = _flat(rows, "lift")
    fig = go.Figure()
    for m in methods:
        ys = [v if (v and not math.isnan(v) and v > 0) else None for v in (r[m] for r in flat)]
        fig.add_trace(go.Bar(x=[r["typ"] for r in flat], y=ys, name=m, marker_color=MEASURE_COLORS[m], text=[de(v, 0) if v else "" for v in ys], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Lift gegen Raten (log)", type="log")
    fig.update_xaxes(tickangle=-25)
    return _base(fig, height=400, bottom=90)


def build_katz(rows):
    """AUC leicht von gemeinsamen Nachbarn und Katz je Netztyp und mittlerem Grad."""
    labelled = [{"label": f"{r['typ']}, c = {r['c']}", "cn": r["cn"], "katz": r["katz"]} for r in rows]
    return _grouped(labelled, [("cn", C.M_CN, MEASURE_COLORS[C.M_CN]), ("katz", C.M_KATZ, MEASURE_COLORS[C.M_KATZ])], "label", "AUC (leichte Negative)", height=420, bottom=130)


def build_community(rows, c):
    """AUC leicht über den Anteil der Kanten innerhalb der Gruppen (Blockmodell mit sechs Gruppen), mit der Obergrenze „gleiche Gruppe“."""
    sel = [r for r in rows if r["c"] == c]
    xs = [de(r["inside"], 2) for r in sel]
    fig = go.Figure()
    for key, name, color, dash in (("cn", C.M_CN, MEASURE_COLORS[C.M_CN], "solid"), ("katz", C.M_KATZ, MEASURE_COLORS[C.M_KATZ], "solid"), ("pa", C.M_PA, MEASURE_COLORS[C.M_PA], "solid"),
                                   ("group", "gleiche Gruppe (Obergrenze)", "#333333", "dot")):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in sel], mode="lines+markers", name=name, line=dict(color=color, width=3, dash=dash)))
    fig.add_hline(y=0.5, line=dict(color="rgba(120,120,120,0.9)", dash="dash"))
    fig.update_xaxes(title_text=f"Anteil der Kanten innerhalb der Gruppen (c = {c})")
    fig.update_yaxes(title_text="AUC (leichte Negative)", range=[0.35, 1.0])
    return _base(fig, height=330, bottom=60)


def build_time(rows):
    """AUC leicht je Mechanismus der neuen Kanten und Maß: Zeitaufteilung gegen Zufallsaufteilung."""
    names = (C.M_CN, C.M_AA, C.M_PA, C.M_KATZ)
    fig = make_subplots(rows=1, cols=len(rows), shared_yaxes=True, subplot_titles=[f"neue Kanten: {r['mech']}" for r in rows])
    for i, r in enumerate(rows, start=1):
        fig.add_trace(go.Bar(x=list(names), y=[r["time"][m] for m in names], name="Zeitaufteilung", marker_color=C.COLOR_A, showlegend=(i == 1),
                             text=[de(r["time"][m], 2) for m in names], textposition="outside"), row=1, col=i)
        fig.add_trace(go.Bar(x=list(names), y=[r["random"][m] for m in names], name="Zufallsaufteilung", marker_color=C.COLOR_B, showlegend=(i == 1),
                             text=[de(r["random"][m], 2) for m in names], textposition="outside"), row=1, col=i)
    fig.update_yaxes(title_text="AUC (leichte Negative)", range=[0, 1.1], row=1, col=1)
    fig.update_xaxes(tickangle=-25)
    fig.update_layout(barmode="group")
    fig.update_annotations(font_size=11)
    return _base(fig, height=360, bottom=90)


def build_leak(rows):
    """AUC leicht der bevorzugten Anbindung mit Trainingsgraden und mit Graden aus dem vollen Netz."""
    return _grouped(rows, [("clean", "Grade aus dem Trainingsnetz", C.COLOR_A), ("leak", "Grade aus dem vollen Netz (Leck)", C.COLOR_B)], "typ", "AUC (leichte Negative)", height=360)
