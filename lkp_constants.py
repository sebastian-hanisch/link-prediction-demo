"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Netztypen -----------------------------------------------------------------------------------------------------------------------------------
T_COMMUNITIES = "Verbünde"
T_RANDOM = "Zufallsnetz"
T_POWER = "Potenzgesetz-Netz"
T_BA = "Skalenfreies Netz (BA)"
T_RING = "Ring mit Abkürzungen"
T_GRID = "Gitter"
T_HUBS = "Hubs am Rand"
T_LINK = "Link-Netz von sebastianhanisch.net"

GENERATED_TYPES = (T_COMMUNITIES, T_RANDOM, T_POWER, T_BA, T_RING, T_GRID, T_HUBS)   # n und c wirken
FIXED_TYPES = (T_LINK,)                                                             # echtes Netz
ALL_TYPES = GENERATED_TYPES + FIXED_TYPES
BLOCKS = 6                                                                         # Zahl der Verbünde

C_MEANING = {
    T_COMMUNITIES: "mittlerer Grad; 80 % der Kanten bleiben im eigenen Verbund (6 Verbünde)",
    T_RANDOM: "mittlerer Grad des Zufallsnetzes",
    T_POWER: "mittlerer Grad (Gradfolge mit Potenzgesetz-Exponent 2,5, Konfigurationsmodell)",
    T_BA: "zwei Kanten je neuem Knoten, also m = c / 2 gerundet (jeder Knoten hat mindestens m Nachbarn)",
    T_RING: "Nachbarn auf dem Ring (c gerade gerundet), dazu Abkürzungen mit 5 % je Ringkante",
    T_GRID: "Nachbarschaft: unter 6 vier Nachbarn (keine Dreiecke), 6 bis 7 mit einer Diagonale, ab 8 mit beiden Diagonalen",
    T_HUBS: "Kernkontakte je Randknoten (c / 3 gerundet, mindestens 2); der Rest sind Blätter an 5 Hubs",
}

N_MIN, N_MAX, DEFAULT_N = 60, 250, 150   # Slider-Maximum: dichte Matrizen, Katz-Inverse O(n^3), Layout O(n^2 * 80)
C_MIN, C_MAX, DEFAULT_C = 3, 12, 6
SEED_MAX, DEFAULT_SEED = 999999, 35
FRAC_MIN, FRAC_MAX, DEFAULT_FRAC = 5, 30, 10     # Anteil der versteckten Kanten in Prozent
SWEEP_SEEDS = tuple(range(100000, 100008))         # feste Netze für alle Sweeps: nie ein einzelner Seed als Befund
LIVE_SEEDS = SWEEP_SEEDS[:4]                       # so viele feste Netze je Zelle rechnet die App live (die README-Zahlen nutzen alle acht)

# --- Maße ------------------------------------------------------------------------------------------------------------------------------------------
M_CN, M_JACCARD, M_AA, M_RA, M_PA, M_KATZ, M_RANDOM = (
    "Gemeinsame Nachbarn", "Jaccard", "Adamic–Adar", "Ressourcenzuteilung", "Bevorzugte Anbindung", "Katz", "Zufall")
SCORE_NAMES = (M_CN, M_JACCARD, M_AA, M_RA, M_PA, M_KATZ, M_RANDOM)
METHODS = SCORE_NAMES
DEFAULT_METHOD = M_CN
METHOD_HELP = {
    M_CN: "Zahl der gemeinsamen Nachbarn zweier Knoten.",
    M_JACCARD: "Gemeinsame Nachbarn geteilt durch alle Nachbarn beider Knoten.",
    M_AA: "Gemeinsame Nachbarn, gewichtet mit 1 / ln(Grad): seltene Bekannte zählen mehr.",
    M_RA: "Gemeinsame Nachbarn, gewichtet mit 1 / Grad.",
    M_PA: "Produkt der Grade: Wer viele Kontakte hat, bekommt noch mehr.",
    M_KATZ: "Gewichtete Zahl aller Wege zwischen zwei Knoten (kurze Wege zählen mehr).",
    M_RANDOM: "Zufällige Rangfolge: die Grundlinie.",
}
KATZ_BETA_FACTOR = 0.5            # Katz mit β = 0,5 / größter Eigenwert (unkritisch, siehe README)
LEAK_OFF, LEAK_ON = "aus dem Trainingsnetz", "aus dem vollen Netz (Leck)"
LEAK_CHOICES = (LEAK_OFF, LEAK_ON)
NAV_COUNT, NAV_DROP = "mitzählen", "weglassen"
NAV_CHOICES = (NAV_COUNT, NAV_DROP)
NAV_PAGES = ("index.html", "kontakt.html")        # Navigationsseiten des Link-Netzes: mit allen Seiten verlinkt

# --- Protokoll und Experimente --------------------------------------------------------------------------------------------------------------------
NEG_FACTOR, NEG_MIN = 5, 200       # Zahl der Negativbeispiele: das Fünffache der versteckten Kanten, mindestens 200
HARD_MIN = 20                      # weniger schwere Negative: AUC schwer ist nicht definiert
KATZ_DEGREES = (3, 6)              # mittlerer Grad für den Vergleich Katz gegen gemeinsame Nachbarn
INSIDE_SHARES = (0.17, 0.4, 0.6, 0.8, 0.95)   # Anteil der Kanten innerhalb der Gruppen (0,17 = 1/6: keine Struktur)
COMM_DEGREES = (6, 12)
MECH_SAME, MECH_PA, MECH_RANDOM = "wie bisher (Verbünde)", "nach bevorzugter Anbindung", "zufällig"
MECHANISMS = (MECH_SAME, MECH_PA, MECH_RANDOM)
FUTURE_FRAC = 0.10                 # neue Kanten: 10 % der alten

# --- Darstellung ---------------------------------------------------------------------------------------------------------------------------------
COLOR_EDGE = "rgba(120,120,120,0.30)"
COLOR_HIT = "#2F6B65"
COLOR_MISS = "#e8a33d"
COLOR_FALSE = "#d62728"
COLOR_LINE = "#4c78a8"
COLOR_A = "#2F6B65"
COLOR_B = "#c4623b"
COLOR_C = "#6b5b95"
COLOR_D = "#9d755d"

PRESETS = {
    "Verbünde (Voreinstellung)": {"netztyp": T_COMMUNITIES, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "frac": DEFAULT_FRAC, "method": M_CN, "nav": NAV_COUNT, "leak": LEAK_OFF},
    "Zufallsnetz (keine Struktur)": {"netztyp": T_RANDOM, "n": DEFAULT_N, "c": 5, "seed": DEFAULT_SEED, "frac": DEFAULT_FRAC, "method": M_CN, "nav": NAV_COUNT, "leak": LEAK_OFF},
    "Ring: Dreiecke finden alles": {"netztyp": T_RING, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "frac": DEFAULT_FRAC, "method": M_CN, "nav": NAV_COUNT, "leak": LEAK_OFF},
    "Hubs am Rand: das Datenleck": {"netztyp": T_HUBS, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "frac": DEFAULT_FRAC, "method": M_PA, "nav": NAV_COUNT, "leak": LEAK_ON},
    "Gitter ohne Diagonalen: Katz": {"netztyp": T_GRID, "n": DEFAULT_N, "c": 4, "seed": DEFAULT_SEED, "frac": DEFAULT_FRAC, "method": M_KATZ, "nav": NAV_COUNT, "leak": LEAK_OFF},
    "Unser Link-Netz (alle Seiten)": {"netztyp": T_LINK, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "frac": DEFAULT_FRAC, "method": M_PA, "nav": NAV_COUNT, "leak": LEAK_OFF},
    "Link-Netz ohne Startseite und Kontakt": {"netztyp": T_LINK, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "frac": DEFAULT_FRAC, "method": M_CN, "nav": NAV_DROP, "leak": LEAK_OFF},
}
PRESET_HELP = {
    "Verbünde (Voreinstellung)": "Sechs Verbünde, 80 % der Kanten innen, dünn vernetzt: gemeinsame Nachbarn finden nur einen Teil der versteckten Kanten.",
    "Zufallsnetz (keine Struktur)": "Ohne jede Struktur: kein Maß ist besser als Raten, und die versteckten Kanten haben so selten gemeinsame Nachbarn wie beliebige Nichtkanten.",
    "Ring: Dreiecke finden alles": "Fast alle versteckten Kanten haben im Trainingsnetz gemeinsame Nachbarn: die Maße auf Basis gemeinsamer Nachbarn erreichen rund 0,96.",
    "Hubs am Rand: das Datenleck": "Bevorzugte Anbindung mit den Graden aus dem vollen Netz: das Maß sieht die versteckte Kante in den Graden und scheint viel besser, als es ist.",
    "Gitter ohne Diagonalen: Katz": "Ohne Dreiecke gibt es keine gemeinsamen Nachbarn; Katz nutzt Wege der Länge 3 und findet die Kanten trotzdem.",
    "Unser Link-Netz (alle Seiten)": "Das echte Link-Netz: Startseite und Kontaktseite machen die Aufgabe zu einer Hub-Aufgabe, die bevorzugte Anbindung gewinnt.",
    "Link-Netz ohne Startseite und Kontakt": "Dasselbe Netz ohne die beiden Navigationsseiten: jetzt tragen gemeinsame Nachbarn.",
}
DEFAULT_PRESET_NAME = "Verbünde (Voreinstellung)"
