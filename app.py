"""Link-Prediction: wie viel verrät die Struktur? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Reihe „Netzwerkanalyse“: ein Teil der Kanten eines Netzes wird versteckt, Nähemaße (gemeinsame Nachbarn, Adamic–Adar, Katz ...) sollen sie wiederfinden, und die Demo zeigt, wie stark
das Ergebnis am Protokoll hängt (leichte und schwere Negativbeispiele, Zufalls- und Zeitaufteilung, Datenleck über die Grade).
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import lkp_constants as C
from lkp_evaluation import (
    Settings,
    analyse,
    community_rows,
    de,
    katz_rows,
    leak_rows,
    time_rows,
    types_rows,
    verdict_community,
    verdict_katz,
    verdict_leak,
    verdict_main,
    verdict_precision,
    verdict_time,
    verdict_types,
)
from lkp_layout import dunkles_design, kopf_css
from lkp_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from lkp_scenario import link_stand
from lkp_visualization import (
    build_auc,
    build_community,
    build_coverage,
    build_katz,
    build_leak,
    build_lift,
    build_network,
    build_overview,
    build_time,
)

st.set_page_config(page_title="Link-Prediction – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _types(n, c, frac):
    return types_rows(n, c, frac)


@st.cache_data(show_spinner=False)
def _katz(n, frac):
    return katz_rows(n, frac)


@st.cache_data(show_spinner=False)
def _community(n, frac):
    return community_rows(n, frac)


@st.cache_data(show_spinner=False)
def _time(n, c):
    return time_rows(n, c)


@st.cache_data(show_spinner=False)
def _leak(n, c, frac):
    return leak_rows(n, c, frac)


def experiment(name, label, key, compute, show, hint):
    """Teure Zusatzansicht erst auf Klick; das Ergebnis bleibt nur für genau diese Einstellungen stehen (Session State)."""
    state_key = f"exp_{name}"
    if st.button(label, key=f"run_{name}", help=hint):
        with st.spinner("Rechne …"):
            result = compute()
        st.session_state[state_key] = (key, result)
    state = st.session_state.get(state_key)
    if state is None:
        st.caption(f"Noch nicht gerechnet. {hint}")
        return
    if state[0] != key:
        st.info("Die Einstellungen haben sich seit der letzten Rechnung geändert. Bitte neu starten.")
        return
    show(state[1])


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in lkp_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔮 Link-Prediction: wie viel verrät die Struktur?")
st.markdown(
    """
**Viertes Stück** der Reihe „Netzwerkanalyse“, im Struktur-Arm der Blick auf **fehlende Kanten**: Welche Kooperationen in einem Netz von Subunternehmern und Lieferanten fehlen vermutlich? Die Demo versteckt einen Teil der Kanten und lässt **Nähemaße** (gemeinsame Nachbarn, Adamic–Adar, Katz …) sie unter allen Nichtkanten wiederfinden. Sie nutzt die **gemeinsamen Nachbarn** aus der [Demo zu strukturellen Löchern](https://sebastianhanisch-strukturelle-loecher-demo.streamlit.app/) und die Gruppen aus der [Blockmodell-Demo](https://sebastianhanisch-stochastic-block-demo.streamlit.app/). Anders als die Fall-Demos zeigt sie **ein** Vorgehen an einem wachsenden Beispiel.

Die Kernfrage: **Wie viel verrät die Struktur über fehlende Kanten, und was sagt ein Gütewert wirklich?** Dieselben Maße erreichen je nach Protokoll 0,05 oder 0,96; die Experimente zeigen, woran das liegt.

Weiter unten: sechs Experimente (**🔬**), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Link-Prediction", expanded=True):
    st.markdown(
        """
1. **Verstecken:** ein Anteil der Kanten wird zufällig herausgenommen. Das **Trainingsnetz** ist der Rest; ein Maß darf nur dieses Netz sehen.
2. **Maße:** **gemeinsame Nachbarn** zählen die Bekannten, die zwei Knoten teilen; **Jaccard** teilt durch alle Bekannten; **Adamic–Adar** und **Ressourcenzuteilung** gewichten seltene Bekannte stärker; **bevorzugte Anbindung** multipliziert die Grade; **Katz** zählt alle Wege, kurze stärker.
3. **AUC:** die Wahrscheinlichkeit, dass einer versteckten Kante ein größerer Wert gehört als einer Nichtkante (0,5 = Raten). **Leichte Negative** sind zufällige Nichtkanten, **schwere Negative** Nichtkanten mit mindestens einem gemeinsamen Nachbarn (Abstand 2).
4. **Precision@k:** unter allen Paaren ohne Trainingskante werden die *k* besten Vorhersagen genommen (*k* = Zahl der versteckten Kanten); der Anteil Treffer, die **erwartete** Trefferquote bei zufälliger Auflösung von Gleichständen. **Lift** = Precision@k geteilt durch die Trefferquote beim Raten.
5. **Abdeckung:** der Anteil der versteckten Kanten, die im Trainingsnetz überhaupt einen gemeinsamen Nachbarn haben: ohne gemeinsame Nachbarn kann kein Maß auf Basis gemeinsamer Nachbarn sie finden.
6. **Datenleck:** nimmt ein Maß die Grade aus dem **vollen** Netz, steckt die versteckte Kante schon in den Gradwerten.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Netz zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    typ = st.selectbox("Netztyp", C.ALL_TYPES, key="netztyp_select", help="Welches Netz erzeugt oder geladen wird. Das Link-Netz ist fest.")
    fixed = typ in C.FIXED_TYPES
    n = st.slider("Knoten n", *bounds("n_slider"), key="n_slider", disabled=fixed, help="Zahl der Knoten (beim Gitter auf ein Quadrat gerundet).")
    c = st.slider("Verbindungen je Knoten c", *bounds("c_slider"), key="c_slider", disabled=fixed, help="Was c bedeutet, hängt vom Netztyp ab (siehe darunter).")
    st.caption(C.C_MEANING[typ] if typ in C.C_MEANING else "Dieses Netz ist fest: n, c und Zufalls-Seed des Netzes wirken hier nicht (der Seed bestimmt weiter, welche Kanten versteckt werden).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Bestimmt das Netz und die Auswahl der versteckten Kanten.")
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)
    nav = st.selectbox("Navigationsseiten (wirkt nur beim Link-Netz)", C.NAV_CHOICES, key="nav_select",
                       help="Startseite und Kontaktseite sind mit allen Seiten verlinkt. Mitgezählt wird die Aufgabe zu einer Hub-Aufgabe; weggelassen tragen gemeinsame Nachbarn.")
    st.markdown("**Protokoll**")
    frac = st.slider("Anteil versteckter Kanten (%)", *bounds("frac_slider"), key="frac_slider", help="Wie viele Kanten dem Maß vorenthalten werden.")
    method = st.selectbox("Maß", C.METHODS, key="method_select", help="Welches Maß im Netz als Vorhersage gezeigt wird; die Kennzahlen aller Maße stehen im Diagramm darunter.")
    st.caption(C.METHOD_HELP[method])
    leak = st.selectbox("Grade der bevorzugten Anbindung", C.LEAK_CHOICES, key="leak_select",
                        help="Aus dem vollen Netz ist ein Datenleck: die versteckten Kanten stecken dann in den Graden.")

sync_query_params({"netztyp_select": typ, "n_slider": int(n), "c_slider": int(c), "seed_input": int(seed), "frac_slider": int(frac), "method_select": method, "nav_select": nav, "leak_select": leak})

settings = Settings(typ, int(n), int(c), int(seed), int(frac), method, nav, leak)
an = _analysis(settings)
if typ == C.T_LINK:
    st.info(f"Das echte Link-Netz von sebastianhanisch.net, Schnappschuss vom {link_stand()} (ungerichtet zusammengefasst): 50 Seiten und 300 Demos. Die Zahlen gelten für diesen Stand, nicht für die heutige Website.")

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Versteckte Kanten und Vorhersage")
r = an.result
m1, m2, m3, m4 = st.columns(4)
m1.metric("Knoten / Kanten", f"{an.inst.n} / {an.inst.m}")
m2.metric("Versteckte Kanten (k)", an.k)
m3.metric("Abdeckung", f"{de(100 * an.proto['coverage'], 0)} %", delta=f"Nichtkanten: {de(100 * an.proto['nonedge_coverage'], 0)} %", delta_color="off")
m4.metric("Treffer unter den k besten", f"{int(an.top_hit.sum())} von {an.k}")
k1, k2, k3, k4 = st.columns(4)
k1.metric("AUC leichte Negative", de(r["auc_easy"], 2))
k2.metric("AUC schwere Negative", de(r["auc_hard"], 2))
k3.metric(f"Precision@{an.k} (erwartet)", de(r["precision"], 3))
k4.metric("Lift gegen Raten", de(r["lift"], 0))
st.markdown(f"**{an.method}: Vorhersage im Netz** (grün: versteckt und gefunden, orange gestrichelt: versteckt und übersehen, rot gepunktet: falsche Vorhersage)")
st.plotly_chart(build_network(an.inst, an.t, an.hidden, an.top, an.top_hit), width="stretch", key="net_chart")
st.caption("Die Anzeige löst Gleichstände nach der Position in der Paarliste auf; die Kennzahlen rechnen mit der erwarteten Trefferquote bei zufälliger Auflösung, daher kann die Trefferzahl oben von Precision@k abweichen.")
st.markdown("**Alle Maße im Vergleich** (dieselben versteckten Kanten)")
st.plotly_chart(build_overview(an), width="stretch", key="overview_chart")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viel verrät die Struktur über fehlende Kanten?")
kind, text = verdict_main(an)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

n_exp, c_exp, frac_exp = settings.n, settings.c, settings.frac
st.subheader("🔬 Experiment 1: Wo findet die Struktur etwas, und wie sehr hängt der Wert an den Negativen?")
st.caption(f"Alle Maße über die Netztypen: AUC mit leichten und mit schweren Negativen und die Abdeckung. Median über {len(C.LIVE_SEEDS)} feste Netze je Netztyp, n = {n_exp}, c = {c_exp}, {frac_exp} % versteckt.")


def show_types(rows):
    st.markdown("**AUC mit leichten Negativen** (zufällige Nichtkanten)")
    st.plotly_chart(build_auc(rows, "auc_easy"), width="stretch", key="auc_easy_chart")
    st.markdown("**AUC mit schweren Negativen** (Nichtkanten im Abstand 2; ein Wert unter 0,5 ist Konstruktion, siehe Verdikt)")
    st.plotly_chart(build_auc(rows, "auc_hard"), width="stretch", key="auc_hard_chart")
    st.markdown("**Abdeckung:** Anteil der Paare mit gemeinsamem Nachbarn im Trainingsnetz")
    st.plotly_chart(build_coverage(rows), width="stretch", key="coverage_chart")
    kind_, text_ = verdict_types(rows)
    getattr(st, kind_)(text_)


experiment("types", "Netztypen vergleichen", (n_exp, c_exp, frac_exp), lambda: _types(n_exp, c_exp, frac_exp), show_types, "Rechnet sieben Netztypen mit je vier Netzen und allen Maßen, einige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 2: Wie genau sind die besten Vorhersagen?")
st.caption("Precision@k über alle Paare ohne Trainingskante und der Lift gegen Raten, aus derselben Rechnung wie Experiment 1.")


def show_precision(rows):
    st.plotly_chart(build_lift(rows), width="stretch", key="lift_chart")
    methods = [m for m in C.SCORE_NAMES if m != C.M_RANDOM]
    st.dataframe([{"Netztyp": r_["typ"], **{m: f"{de(r_['scores'][m]['precision'], 3)} (Lift {de(r_['scores'][m]['lift'], 0)})" for m in methods}} for r_ in rows], hide_index=True, width="stretch")
    kind_, text_ = verdict_precision(rows)
    getattr(st, kind_)(text_)


experiment("precision", "Precision@k zeigen", (n_exp, c_exp, frac_exp), lambda: _types(n_exp, c_exp, frac_exp), show_precision, "Nutzt die Rechnung aus Experiment 1; läuft sie noch nicht, rechnet es sie jetzt.")

st.markdown("---")

st.subheader("🔬 Experiment 3: Wann schlägt Katz die gemeinsamen Nachbarn?")
st.caption(f"AUC mit leichten Negativen bei dünnem (c = {C.KATZ_DEGREES[0]}) und mittlerem (c = {C.KATZ_DEGREES[1]}) Grad. Das Gitter hat unter c = 6 keine Diagonalen, also keine Dreiecke und keine gemeinsamen Nachbarn. n = {n_exp}, {frac_exp} % versteckt.")


def show_katz(rows):
    st.plotly_chart(build_katz(rows), width="stretch", key="katz_chart")
    kind_, text_ = verdict_katz(rows)
    getattr(st, kind_)(text_)


experiment("katz", "Katz vergleichen", (n_exp, frac_exp), lambda: _katz(n_exp, frac_exp), show_katz, "Rechnet 14 Zellen mit je vier Netzen, einige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 4: Hilft die Gemeinschaftsstruktur?")
st.caption(f"Blockmodell mit sechs gleich großen Gruppen, n = {n_exp}: AUC mit leichten Negativen über den Anteil der Kanten innerhalb der Gruppen (0,17 = keine Struktur). Die gepunktete Linie ist die Obergrenze „gleiche Gruppe“ (der Score ist 1 für Paare derselben Gruppe).")


def show_community(rows):
    for c_ in C.COMM_DEGREES:
        st.plotly_chart(build_community(rows, c_), width="stretch", key=f"community_chart_{c_}")
    kind_, text_ = verdict_community(rows)
    getattr(st, kind_)(text_)


experiment("community", "Gruppenstärke rechnen", (n_exp, frac_exp), lambda: _community(n_exp, frac_exp), show_community, "Rechnet zehn Zellen mit je vier Netzen, einige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 5: Zeitaufteilung gegen Zufallsaufteilung")
st.caption(f"Ein Altnetz (Blockmodell, 80 % der Kanten innerhalb der Gruppen, n = {n_exp}, c = {c_exp}) bekommt neue Kanten (10 % der alten). Zeitaufteilung: das Maß sieht nur das Altnetz und soll die neuen Kanten finden. Zufallsaufteilung: 10 % der Kanten des neuen Gesamtnetzes werden zufällig versteckt.")


def show_time(rows):
    st.plotly_chart(build_time(rows), width="stretch", key="time_chart")
    kind_, text_ = verdict_time(rows)
    getattr(st, kind_)(text_)


experiment("time", "Aufteilungen vergleichen", (n_exp, c_exp), lambda: _time(n_exp, c_exp), show_time, "Rechnet drei Mechanismen mit je vier Netzen.")

st.markdown("---")

st.subheader("🔬 Experiment 6: Das Datenleck über die Grade")
st.caption(f"Bevorzugte Anbindung mit den Graden des Trainingsnetzes gegen die Grade des vollen Netzes (das die versteckten Kanten enthält); n = {n_exp}, c = {c_exp}, {frac_exp} % versteckt.")


def show_leak(rows):
    st.plotly_chart(build_leak(rows), width="stretch", key="leak_chart")
    kind_, text_ = verdict_leak(rows)
    getattr(st, kind_)(text_)


experiment("leak", "Leck zeigen", (n_exp, c_exp, frac_exp), lambda: _leak(n_exp, c_exp, frac_exp), show_leak, "Rechnet sieben Netztypen mit je vier Netzen.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur die Topologie** | Die Maße sehen weder Eigenschaften der Betriebe noch Zeitpunkte oder Gewichte. Gelernte Verfahren (z. B. Graph-Netze) nutzen mehr; ob sie die einfachen Maße schlagen, ist umstritten (Li u. a. 2023: Basislinien lagen unter ihrem Wert) und hier nicht gemessen. | – |
| **Zufällig versteckte Kanten** | Ein Zufallsversteck misst, wie gut man die **alte** Struktur rekonstruiert. Entstehen neue Kanten anders als die vorhandenen, überschätzt es die Vorhersage (Experiment 5). | – |
| **Leichte Negative** | Zufällige Nichtkanten sind meist weit entfernt: ein hoher AUC verschweigt, ob das Maß auch zwischen nahen Paaren trennt. Schwere Negative (Abstand 2) haben per Definition gemeinsame Nachbarn; ein AUC unter 0,5 ist dann Konstruktion, die Abdeckung gehört daneben. | – |
| **Abdeckung** | Wo die meisten versteckten Kanten keinen gemeinsamen Nachbarn haben (Zufallsnetz, Gitter ohne Diagonalen), kann kein Maß auf Basis gemeinsamer Nachbarn etwas finden. | Stück 3 (Dreiecke) |
| **Grade nach dem Verstecken** | In nahezu regulären Netzen liegt die bevorzugte Anbindung unter 0,5: das Verstecken senkt die Grade der beiden Enden. Mit Graden aus dem vollen Netz (Leck) wird daraus ein Scheinerfolg. | – |
| **Synthetische Netze** | Die Zahlen gelten für diese Netztypen bei diesen Größen (Median über feste Netze). Das Link-Netz ist ein Sonderfall: Startseite und Kontaktseite machen die Aufgabe zu einer Hub-Aufgabe. | Stück 2 (Gruppen) |
| **Wachstumsmodell der Zeitaufteilung** | Es gibt drei erfundene Mechanismen für neue Kanten, keine echten Zeitstempel. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Maße** für ein Paar $(x, y)$ mit Nachbarmengen $N(x)$ und Graden $d_x$:

$$s_\mathrm{CN}(x, y) = |N(x) \cap N(y)|$$

$$s_\mathrm{Jaccard}(x, y) = \frac{|N(x) \cap N(y)|}{|N(x) \cup N(y)|}$$

$$s_\mathrm{AA}(x, y) = \sum_{z \in N(x) \cap N(y)} \frac{1}{\ln d_z}$$

$$s_\mathrm{RA}(x, y) = \sum_{z \in N(x) \cap N(y)} \frac{1}{d_z}$$

$$s_\mathrm{PA}(x, y) = d_x \, d_y$$

**Katz-Index** (gewichtete Zahl aller Wege, $A$ die Adjazenzmatrix, $\beta < 1/\lambda_1$):

$$S_\mathrm{Katz} = \sum_{\ell = 1}^{\infty} \beta^\ell A^\ell = (I - \beta A)^{-1} - I$$

**AUC** (versteckte Kanten $P$, Negativbeispiele $N$, Gleichstand zählt halb):

$$\mathrm{AUC} = P(s_+ > s_-) + \tfrac{1}{2}\, P(s_+ = s_-)$$

**Erwartete Precision@k** bei zufälliger Auflösung von Gleichständen: sind $g$ Paare mit gleichem Score am Rand der Auswahl, davon $h$ versteckte Kanten, und fehlen $r$ Plätze, zählt die Gruppe mit $r h / g$ Treffern. **Lift** gegen Raten (Trefferquote $k / M$ bei $M$ Kandidatenpaaren):

$$\mathrm{Lift} = \frac{\mathrm{Precision@}k}{k / M}$$

**Literatur.**
- Liben-Nowell, D., Kleinberg, J. (2007): *The link-prediction problem for social networks.* JASIST 58(7), 1019–1031.
- Adamic, L. A., Adar, E. (2003): *Friends and neighbors on the Web.* Social Networks 25(3), 211–230.
- Zhou, T., Lü, L., Zhang, Y.-C. (2009): *Predicting missing links via local information.* European Physical Journal B 71, 623–630. (Ressourcenzuteilung)
- Katz, L. (1953): *A new status index derived from sociometric analysis.* Psychometrika 18(1), 39–43.
- Li, J. u. a. (2023): *Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking.* NeurIPS 2023, Datasets and Benchmarks.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
