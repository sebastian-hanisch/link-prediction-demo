# Link-Prediction: wie viel verrät die Struktur? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-link-prediction-demo.streamlit.app/)** -->

Viertes Stück der **Netzwerkanalyse-Reihe** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, im Struktur-Arm der Blick auf **fehlende Kanten**; Zusammenfluss aus der [Blockmodell-Demo](https://sebastianhanisch-stochastic-block-demo.streamlit.app/) (Gemeinschaftsstruktur hilft) und der [Demo zu strukturellen Löchern](https://sebastianhanisch-strukturelle-loecher-demo.streamlit.app/) (gemeinsame Nachbarn sind Dreiecke). Anders als die Fall-Demos zeigt sie **ein** Vorgehen an einem wachsenden Beispiel: einem Netz von Subunternehmern und Lieferanten (synthetisch; dazu das echte Link-Netz von sebastianhanisch.net).

Die Demo versteckt einen Teil der Kanten und lässt **Nähemaße** (gemeinsame Nachbarn, Jaccard, Adamic–Adar, Ressourcenzuteilung, bevorzugte Anbindung, Katz) sie unter allen Nichtkanten wiederfinden.

## Kernfrage

**Wie viel verrät die Struktur über fehlende Kanten, und was sagt ein Gütewert wirklich?** Der Gesamtplan fragte, welches Maß gewinnt; die Vormessung zeigte, dass dieselben Maße je nach Netz und **Protokoll** Werte von 0,05 (schwere Negative im Zufallsnetz) bis 0,96 (leichte Negative im Gitter) erreichen. Die Kernfrage wurde mit Freigabe des Einzelplans umgestellt: nicht „bester Vorhersager“, sondern „was der Wert je nach Protokoll bedeutet“ (leichte und schwere Negativbeispiele, Zufalls- und Zeitaufteilung, Datenleck über die Grade).

## Befunde und Korrekturen gegenüber dem Plan

- **Hook umgestellt** auf das Protokoll (Nutzerentscheidung 2026-10-19).
- **Abdeckung als Schlüsselzahl** neu aufgenommen: der Anteil der versteckten Kanten, die im Trainingsnetz überhaupt einen gemeinsamen Nachbarn haben. Ohne sie ist ein AUC nicht zu lesen.
- **Ein AUC unter 0,5 ist bei schweren Negativen Konstruktion.** Schwere Negative (Nichtkanten im Abstand 2) haben per Definition gemeinsame Nachbarn; die Maße auf Basis gemeinsamer Nachbarn ordnen sie vor die meisten versteckten Kanten. Gemessen: Zufallsnetz 0,05.
- **Die bevorzugte Anbindung liegt in nahezu regulären Netzen unter 0,5** (Ring 0,18, Gitter 0,27): das Verstecken senkt die Grade der beiden Enden. Mit den Graden aus dem vollen Netz (**Datenleck**) wird daraus ein Scheinerfolg.
- **Katz im Gitter ohne Diagonalen: AUC 0,95, aber Precision@k 0,000** (mit Diagonalen AUC 0,98, Precision ebenfalls 0,000). Kein Fehler: Ohne Diagonalen liegen die versteckten Kanten im Abstand 3 oder weiter, rund 800 Nichtkanten im Abstand 2 stehen bei Katz davor (in `tests/test_claims.py` nachgerechnet). Der Vormessung-Verdacht (Wegzahlen der Enden) war damit teils falsch; die Ursache ist die Rangfolge der Abstände.
- **Precision@k mit erwarteter Gleichstandsauflösung** (exakt) statt eines Zufallsrauschens (Nutzerentscheidung); Katz mit festem β = 0,5/λ₁ (in zwei geprüften Zellen unkritisch, siehe unten).
- **GNN-Verfahren** nur als Randnotiz: ob sie die einfachen Maße schlagen, ist umstritten (Li u. a. 2023) und hier nicht gemessen.

## Modell

Ein Netz (ungewichtet, ungerichtet) wird in ein **Trainingsnetz** und **versteckte Kanten** geteilt (zufällig, Anteil 5 bis 30 %; die Vereinigung ist das Gesamtnetz). Jedes Maß rechnet nur auf dem Trainingsnetz und bewertet jedes Paar: gemeinsame Nachbarn |N(x) ∩ N(y)|, Jaccard, Adamic–Adar (Summe 1/ln Grad über gemeinsame Nachbarn), Ressourcenzuteilung (Summe 1/Grad), bevorzugte Anbindung (Grad mal Grad), Katz ((I − βA)⁻¹ − I, β = 0,5/λ₁). **AUC:** die Wahrscheinlichkeit, dass einer versteckten Kante ein größerer Wert gehört als einer Nichtkante (Gleichstand zählt halb); leichte Negative sind zufällige Nichtkanten (das Fünffache der versteckten Kanten, mindestens 200), schwere Negative Nichtkanten mit mindestens einem gemeinsamen Nachbarn im Trainingsnetz. **Precision@k:** alle Paare ohne Trainingskante nach Score, k = Zahl der versteckten Kanten, **erwartete** Trefferquote bei zufälliger Auflösung der Gleichstände; **Lift** = Precision@k geteilt durch die Trefferquote beim Raten. Formeln und Literatur stehen im 📐-Expander der App.

## Methodik

| Baustein | Inhalt |
|---|---|
| **Netze** | Generatoren aus `k-core-demo` (Stück 1) als Kopie: Verbünde (sechs, 80 % der Kanten innen), Zufallsnetz, Potenzgesetz-Netz, skalenfrei, Ring mit Abkürzungen, Gitter, Hubs am Rand; dazu das Link-Netz von sebastianhanisch.net (Schnappschuss vom 2026-09-30, ungerichtet) mit Schalter für die beiden Navigationsseiten |
| **Experiment 1** | alle Maße über die Netztypen: AUC leicht, AUC schwer, Abdeckung |
| **Experiment 2** | Precision@k und Lift |
| **Experiment 3** | Katz gegen gemeinsame Nachbarn bei dünnem (c = 3) und mittlerem (c = 6) Grad |
| **Experiment 4** | Gemeinschaftsstärke: Blockmodell mit sechs Gruppen, AUC über den Anteil der Kanten innerhalb der Gruppen, mit der Obergrenze „gleiche Gruppe“ |
| **Experiment 5** | Zeitaufteilung (Training = Altnetz, Test = neue Kanten) gegen Zufallsaufteilung, drei Mechanismen für die neuen Kanten |
| **Experiment 6** | Datenleck: bevorzugte Anbindung mit Graden aus dem Trainingsnetz gegen die aus dem vollen Netz |

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Netze der Messreihen: die acht festen Netze `SWEEP_SEEDS` je Zelle, n = 250, c = 6, 10 % der Kanten versteckt, wo nichts anderes steht; Median.

| Frage | Befund |
|---|---|
| **Stimmen die Maße?** | Jaccard, Adamic–Adar, Ressourcenzuteilung und bevorzugte Anbindung stimmen auf 100 festen Zufallsnetzen mit `networkx` überein, gemeinsame Nachbarn mit dem Mengenschnitt, Katz mit der abgebrochenen Reihe und der Aufzählung aller Wege, die AUC mit der Aufzählung aller Paare, die erwartete Precision mit der Aufzählung aller Gleichstands-Reihenfolgen (`tests/test_oracle_link_prediction.py`) |
| **Wo findet die Struktur etwas?** (AUC mit leichten Negativen: gemeinsame Nachbarn / Katz) | **Zufallsnetz 0,50 / 0,53** (nichts zu finden), Verbünde 0,62 / **0,74**, skalenfrei 0,58 / 0,62, Potenzgesetz 0,67 / 0,68, Hubs am Rand 0,70 / 0,66, **Ring 0,95 / 0,97, Gitter 0,96 / 0,98**. Zufallsgrundlinie 0,48 bis 0,52 |
| **Wie viele versteckte Kanten haben einen gemeinsamen Nachbarn?** (Abdeckung; daneben beliebige Nichtkanten) | Zufallsnetz **0,11 (0,11)**, Verbünde 0,31 (0,10), skalenfrei 0,34 (0,18), Hubs 0,49 (0,11), Potenzgesetz 0,53 (0,25), **Ring 0,92 (0,04), Gitter 0,96 (0,04)**. Bei c = 3: Zufallsnetz 0,03, Verbünde 0,12, **Gitter 0,00** |
| **Was zeigen schwere Negative?** (AUC, gemeinsame Nachbarn) | **Zufallsnetz 0,05**, Verbünde 0,17, skalenfrei 0,20, Potenzgesetz 0,37, Hubs 0,41, Gitter 0,60, **Ring 0,72**; Jaccard im Ring 0,82 und im Gitter 0,78. Ein Wert unter 0,5 ist Konstruktion (siehe oben) |
| **Wie genau sind die besten Vorhersagen?** (Precision@k, Lift) | Zufallsnetz **0,002 (Lift 1)**; Verbünde 0,012 (Lift 5); skalenfrei 0,029 (12); Potenzgesetz 0,081 (41), Katz Lift 47; Hubs 0,105 (68); **Ring 0,266 (103), Jaccard 0,417 (163)**; Gitter Jaccard 0,298 (136). Jaccard hat in vier Netztypen keinen Lift (es bevorzugt Paare mit kleinem Grad), die bevorzugte Anbindung in vier Netztypen Precision unter 1 % |
| **Wann gewinnt Katz?** (AUC leicht, gemeinsame Nachbarn gegen Katz) | **Gitter ohne Diagonalen (c = 3): 0,49 gegen 0,95** (keine Dreiecke, aber Wege der Länge 3); Verbünde 0,54 gegen 0,61 (c = 3) und **0,62 gegen 0,74** (c = 6); Ring und Gitter mit Diagonalen gleichauf; Hubs am Rand: Katz 0,66 unter gemeinsamen Nachbarn 0,70. In 3 von 14 Zellen liegt Katz mehr als 0,05 vorn. **β ist in den beiden geprüften Zellen unkritisch** (AUC der Verbünde bei β = 0,05 / 0,2 / 0,5 / 0,8 mal 1/λ₁: 0,75 / 0,75 / 0,74 / 0,73; Gitter c = 3: 0,94 / 0,95 / 0,95 / 0,93), der Anteil versteckter Kanten ändert die AUC um 0,08 bis 0,10 zwischen 5 und 30 %: Verbünde bei 5 / 10 / 20 / 30 % versteckt: gemeinsame Nachbarn 0,65 / 0,62 / 0,59 / 0,57, Katz 0,78 / 0,74 / 0,72 / 0,68 |
| **Hilft die Gemeinschaftsstruktur?** (Blockmodell, sechs Gruppen, c = 6; gemeinsame Nachbarn / Katz / Obergrenze „gleiche Gruppe“) | Anteil innen 0,17 (keine Struktur): 0,48 / 0,48 / 0,50; 0,60: 0,55 / 0,59 / 0,72; 0,80: 0,60 / **0,74** / 0,81; **0,95: 0,71 / 0,90 / 0,91**. Bei c = 12 und 0,95: 0,88 / 0,91 / 0,91. Die Nähemaße erreichen die Obergrenze erst bei dichten, klaren Verbünden |
| **Überschätzt die Zufallsaufteilung?** (AUC leicht, Katz; neue Kanten = 10 % der alten) | Entstehen die neuen Kanten **wie bisher**: Zeit 0,74 gegen Zufall 0,76 (gleich). **Nach bevorzugter Anbindung:** Zeit 0,54 gegen Zufall **0,70**, gemeinsame Nachbarn 0,51 gegen 0,60, bevorzugte Anbindung 0,55 gegen 0,52. **Zufällig:** Zeit **0,47** gegen Zufall **0,70** (Überschätzung 0,24) |
| **Wie stark verfälscht ein Leck die bevorzugte Anbindung?** (Grade des Trainingsnetzes → Grade des vollen Netzes) | Hubs am Rand **0,63 → 0,95**, Ring 0,18 → 0,53, Gitter 0,27 → 0,58, Zufallsnetz 0,50 → 0,67, Potenzgesetz 0,73 → 0,85, Verbünde 0,51 → 0,68 |
| **Katz im Gitter ohne Diagonalen** (c = 4, drei Netze) | AUC leicht 0,95, aber **Precision@k 0,000**: alle k besten Vorhersagen liegen im Abstand 2, die versteckten Kanten im Abstand 3 oder weiter; rund 800 Nichtkanten im Abstand 2 stehen davor |
| **Link-Netz mit allen Seiten** (350 Knoten, 2 374 Kanten; Aufteilungen) | Abdeckung **1,00** (Nichtkanten 0,97): die Startseite und die Kontaktseite sind gemeinsame Nachbarn aller Paare. AUC leicht: **bevorzugte Anbindung 0,96**, Katz 0,93, Ressourcenzuteilung 0,89, Adamic–Adar 0,87, gemeinsame Nachbarn 0,83, **Jaccard 0,49** (Precision 0). Lift 126 bis 155 |
| **Link-Netz ohne Startseite und Kontakt** (348 Knoten, 1 679 Kanten) | Abdeckung **0,89** (Nichtkanten 0,26). Alle Maße außer Zufall 0,87 bis 0,91 (bevorzugte Anbindung 0,87, Jaccard 0,89); schwer: gemeinsame Nachbarn 0,78; **Lift 220 (gemeinsame Nachbarn, Precision 0,63) bis 235 (Ressourcenzuteilung, Precision 0,67)**. Jaccard ohne Lift |
| **Standardnetz** (Verbünde, n = 150, c = 6, Seed 35, gemeinsame Nachbarn) | 450 Kanten, 45 versteckt; AUC 0,73 (leicht), 0,36 (schwer); Precision@45 0,040 (Lift 10); Abdeckung 62 % (Nichtkanten 14 %); unter den 45 besten Vorhersagen der Anzeige 2 Treffer |

### Hypothesen des Einzelplans

| | Hypothese | Ergebnis |
|---|---|---|
| H1 | Gemeinsame-Nachbarn-Maße finden fehlende Kanten, wo es Dreiecke gibt | ✅ Ring, Gitter 0,95; ⚠ Verbünde 0,62; ❌ Zufallsnetz 0,50 |
| H2 | Die bevorzugte Anbindung gewinnt in heavy-tailed Netzen | ✅ leicht 0,66 bis 0,73; ❌ schwer 0,51 bis 0,64, und in regulären Netzen 0,18 bis 0,27 |
| H3 | Schwere Negative senken die Güte stark | ✅ (Konstruktion: AUC unter 0,5 möglich, Abdeckung daneben) |
| H4 | Katz schlägt gemeinsame Nachbarn, wo es kaum Dreiecke gibt | ✅ Gitter c = 3: 0,95 gegen 0,49; Verbünde 0,74 gegen 0,62 |
| H5 | Die Zufallsaufteilung überschätzt, wenn die Zukunft anders entsteht | ✅ Katz 0,70 gegen 0,54 / 0,47; gleich, wenn der Mechanismus gleich bleibt |
| H6 | Ein Datenleck über die Grade verändert das Ergebnis stark | ✅ Hubs am Rand 0,63 → 0,95 |
| H7 | Gemeinschaftsstruktur hilft den Nähemaßen | ✅ monoton mit dem Anteil innen; Obergrenze erst bei dichten Verbünden |
| H8 | Precision@k ist gegenüber Zufall hoch, in Zahlen aber klein | ✅ Lift 5 bis 163 bei den besten Maßen in Netzen mit Dreiecken (im Zufallsnetz 1), Precision in Verbünden 1,2 % |
| H9 | Im Link-Netz gewinnt eine Hub-Regel | ✅ mit Navigation bevorzugte Anbindung 0,96; ohne sie alle 0,87 bis 0,91 |

## Ehrliche Grenzen

- **Nur die Topologie.** Weder Eigenschaften der Betriebe noch Zeitpunkte oder Gewichte; gelernte Verfahren (Graph-Netze) sind nicht Teil der Demo, ob sie die einfachen Maße schlagen, ist umstritten (Li u. a. 2023) und hier nicht gemessen.
- **Zufälliges Verstecken** misst die Rekonstruktion der alten Struktur. Entstehen neue Kanten anders, überschätzt es die Vorhersage (Experiment 5, nur mit drei erfundenen Mechanismen, keine echten Zeitstempel).
- **Leichte Negative** sind meist weit entfernt; **schwere** haben per Definition gemeinsame Nachbarn. Beide Zahlen zeigen einen Ausschnitt; die Abdeckung gehört daneben.
- **Synthetische Netze** bei n = 250 (live n ≤ 250), Mediane über acht feste Netze; die Streuung der Werte ist nicht ausgewiesen. Das Link-Netz ist ein Sonderfall (Navigationsseiten machen die Aufgabe zu einer Hub-Aufgabe).
- **Katz mit festem β = 0,5/λ₁** (in einer Messreihe unkritisch, aber nicht für alle Netze geprüft).
- **Literatur:** die bibliografischen Angaben im 📐-Expander (Liben-Nowell und Kleinberg 2007, Adamic und Adar 2003, Zhou u. a. 2009, Katz 1953, Li u. a. 2023) sind am 2026-10-19 nachgeschlagen; die Arbeiten selbst sind nicht gelesen, die Aussagen zu Li u. a. (zu leichte Negative, uneinheitliche Aufteilungen, Basislinien unter ihrem Wert) stammen aus der Zusammenfassung der Proceedings.
- **Precision@k hängt an der Gleichstandsregel:** die Kennzahl rechnet mit der erwarteten Trefferquote bei zufälliger Auflösung; die Anzeige im Netz löst Gleichstände nach der Position auf und kann davon abweichen.

## Verifikation

- **Kern:** Handwerte auf Mini-Instanzen (`tests/conftest.py`, `tests/test_algorithm.py`): alle Maße, Aufteilung, AUC, schwere Negative, Abdeckung, erwartete Precision (Gleichstandsgruppen von Hand), Lift, Blockmodell-Sampler, Wachstumsmodell.
- **Orakel:** `tests/test_oracle_link_prediction.py` – 100 feste Zufallsnetze gegen `networkx`, Mengenschnitt, Katz gegen Reihe und Wegezählung, AUC durch Aufzählung, erwartete Precision gegen alle Gleichstands-Reihenfolgen und gegen 400 zufällige Auflösungen; Lecktest (versteckte Kanten stehen nie im Trainingsnetz).
- **Netze:** gültig in allen Ecken der Regler, deterministisch je Seed (auch über Prozesse mit verschiedenem Hash-Seed), Link-Netz mit und ohne Navigationsseiten, Gitter ohne Dreiecke.
- **Verdikte:** jede Schwelle und jede Zahl im Text an von Hand gesetzten Werten (`tests/test_verdicts.py`), Randfälle in `tests/test_edge_cases.py`.
- **Befunde:** jede Zahl dieser README in `tests/test_claims.py`.
- **Mutationsprüfung des Kerns** (`lkp_algorithm.py`, 141 Stellen, ohne die langsamen Aussagen-Tests in der Suite der Überlebenden): nach Ergänzung von Tests keine Überlebenden; 20 Stellen sind als äquivalent festgehalten (z. B. Schutzgrenzen von Schleifen, Gewichte von Knoten, die nie gemeinsame Nachbarn sind, die Gleichheit zweier Fließkommazahlen bei stetigem Zufall), zwei Mutanten laufen in Endlosschleifen (Zeitüberschreitung). Zustand in `tests/mutation_state.json`.
- **Presets:** vollständig, Werte innerhalb der Reglergrenzen, Standard-Preset gleich Standardeinstellung.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, jeder Netztyp mit jedem Maß, Permalink, Leck-Schalter, alle sechs Experimente) und Footer-Test.
- **Layout:** Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; jede Änderung eines Reglers hebt die Markierung auf (`tests/test_layout.py`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `data/link_netz.json` | Link-Netz von sebastianhanisch.net (Schnappschuss vom 2026-09-30) |
| `requirements-dev.txt` | Laufzeit plus pytest, networkx, scipy (nur für die Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `lkp_algorithm.py` | Kern: Aufteilung, Nähemaße, AUC, Negative, Abdeckung, erwartete Precision, Blockmodell und Wachstum |
| `lkp_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `lkp_evaluation.py` | Protokoll, Analyse, Verdikt-Logik, Messreihen über feste Netze |
| `lkp_layout.py` | Fester Kopf (Stylesheet) |
| `lkp_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `lkp_scenario.py` | Netze: Generatoren (Kopie aus `k-core-demo`), Link-Netz, Anordnung |
| `lkp_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Mini-Instanzen und Handwerte, Orakel, Netze, Verdikte, Randfälle, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe, Befunde |

## Bewusst nicht umgesetzt

- **Gelernte Verfahren und Graph-Netze:** Streit um die Güte (Li u. a. 2023); kein Nachbau.
- **Heuristisch gewählte schwere Negative** (HeaRT): hier einfach Abstand 2; die Abdeckung zeigt, was das für den Wert bedeutet.
- **Gewichtete oder gerichtete Netze, Knotenmerkmale, echte Zeitstempel:** eigene Fragen.
- **Weitere Maße** (Personalized PageRank, SimRank, Blockmodell-Verfahren): würden den Blick vom Protokoll auf die Maße lenken.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
