"""Wczytanie danych źródłowych: Transfermarkt (mecze, wyceny, transfery) i płace fbref.

Moduł tylko czyta i sprowadza do wspólnego formatu. Agregacja do panelu
klub x sezon jest w `pre_processing`.
"""

from pathlib import Path

import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    """Korzeń projektu wyznaczony przez szukanie w górę pliku src/data_load.py."""
    root = (start or Path.cwd()).resolve()
    while not (root / "src" / "data_load.py").is_file():
        if root == root.parent:
            raise RuntimeError(
                f"Nie znaleziono korzenia projektu startując z {start or Path.cwd()}."
            )
        root = root.parent
    return root


PROJECT_ROOT = find_project_root()
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"

WAGE_FILE = RAW / "place_top4_ligi_2013_2023.csv"

# Zakres: sezony 2013/14 - 2023/24, oznaczane rokiem rozpoczęcia.
SEASON_MIN, SEASON_MAX = 2013, 2023

# Wyceny i stan kadry brane na 1 lipca, czyli początek sezonu - przed rundą
# transferową, która jest już reakcją na wynik poprzedniego sezonu.
SEASON_START_MONTH = 7

# Okno letniego transferowego: czerwiec-sierpień roku rozpoczęcia sezonu.
SUMMER_WINDOW = (6, 8)

# Liga fbref -> competition_id Transfermarkt.
LEAGUES = {
    "Premier_League": "GB1",
    "Bundesliga": "L1",
    "La_Liga": "ES1",
    "Serie_A": "IT1",
}
ID2LEAGUE = {v: k for k, v in LEAGUES.items()}

# Rozgrywki Ligi Mistrzów: faza główna i kwalifikacje.
CL_COMPETITIONS = ["CL", "CLQ"]

# Mapowanie nazw klubów z fbref na club_id Transfermarkt, zweryfikowane ręcznie
# po slugach. Nazwy w obu źródłach różnią się skrótami ("Manchester Utd" vs
# "Manchester United"), więc automatyczne dopasowanie po nazwie zawodzi.
WAGE2ID = {
    "Premier_League": {
        "Arsenal": 11, "Aston Villa": 405, "Bournemouth": 989, "Brentford": 1148,
        "Brighton": 1237, "Burnley": 1132, "Cardiff City": 603, "Chelsea": 631,
        "Crystal Palace": 873, "Everton": 29, "Fulham": 931, "Huddersfield Town": 1110,
        "Hull City": 3008, "Leeds United": 399, "Leicester City": 1003, "Liverpool": 31,
        "Luton Town": 1031, "Manchester City": 281, "Manchester Utd": 985,
        "Middlesbrough": 641, "Newcastle United": 762, "Norwich City": 1123,
        "Nottingham Forest": 703, "QPR": 1039, "Sheffield United": 350,
        "Southampton": 180, "Stoke City": 512, "Sunderland": 289, "Swansea City": 2288,
        "Tottenham Hotspur": 148, "Watford": 1010, "West Brom": 984,
        "West Ham United": 379, "Wolves": 543,
    },
    "Bundesliga": {
        "Arminia": 10, "Augsburg": 167, "BTSV": 23, "Bayern Munich": 27, "Bochum": 80,
        "Darmstadt 98": 105, "Dortmund": 16, "Düsseldorf": 38,
        "Eintracht Frankfurt": 24, "Freiburg": 60, "Gladbach": 18,
        "Greuther Fürth": 65, "Hamburger SV": 41, "Hannover 96": 42,
        "Heidenheim": 2036, "Hertha BSC": 44, "Hoffenheim": 533, "Ingolstadt": 4795,
        "Köln": 3, "Leverkusen": 15, "Mainz 05": 39, "Nürnberg": 4,
        "Paderborn 07": 127, "RB Leipzig": 23826, "Schalke 04": 33, "Stuttgart": 79,
        "Union Berlin": 89, "Werder Bremen": 86, "Wolfsburg": 82,
    },
    "La_Liga": {
        "Alavés": 1108, "Almería": 3302, "Athletic Club": 621, "Atlético Madrid": 13,
        "Barcelona": 131, "Celta Vigo": 940, "Cádiz": 2687, "Córdoba": 993,
        "Dep La Coruña": 897, "Eibar": 1533, "Elche": 1531, "Espanyol": 714,
        "Getafe": 3709, "Girona": 12321, "Granada": 16795, "Huesca": 5358,
        "Las Palmas": 472, "Leganés": 1244, "Levante": 3368, "Mallorca": 237,
        "Málaga": 1084, "Osasuna": 331, "Rayo Vallecano": 367, "Real Betis": 150,
        "Real Madrid": 418, "Real Sociedad": 681, "Sevilla": 368,
        "Sporting Gijón": 2448, "Valencia": 1049, "Valladolid": 366,
        "Villarreal": 1050,
    },
    "Serie_A": {
        "Atalanta": 800, "Benevento": 4171, "Bologna": 1025, "Brescia": 19,
        "Cagliari": 1390, "Carpi": 4102, "Catania": 1627, "Cesena": 1429,
        "Chievo": 862, "Cremonese": 2239, "Crotone": 4083, "Empoli": 749,
        "Fiorentina": 430, "Frosinone": 8970, "Genoa": 252, "Hellas Verona": 276,
        "Inter": 46, "Juventus": 506, "Lazio": 398, "Lecce": 1005, "Livorno": 1210,
        "Milan": 5, "Monza": 2919, "Napoli": 6195, "Palermo": 458, "Parma": 130,
        "Pescara": 2921, "Roma": 12, "SPAL": 2722, "Salernitana": 380,
        "Sampdoria": 1038, "Sassuolo": 6574, "Spezia": 3522, "Torino": 416,
        "Udinese": 410, "Venezia": 607,
    },
}


def load_games() -> pd.DataFrame:
    """Wszystkie mecze - potrzebne także poza czterema ligami, dla LM."""
    return pd.read_csv(RAW / "games.csv")


def load_club_games() -> pd.DataFrame:
    """Mecze z perspektywy klubu: gole własne i przeciwnika, wynik, trener."""
    return pd.read_csv(RAW / "club_games.csv")


def load_valuations() -> pd.DataFrame:
    """Historia wycen rynkowych zawodników (EUR), z datą wyceny."""
    return pd.read_csv(RAW / "player_valuations.csv", parse_dates=["date"])


def load_transfers() -> pd.DataFrame:
    """Transfery z kwotami i datami - podstawa salda transferowego."""
    return pd.read_csv(RAW / "transfers.csv", parse_dates=["transfer_date"])


def load_wages() -> pd.DataFrame:
    """Roczne funduszu płac klubów z fbref, zebrane przez notebooks/01."""
    if not WAGE_FILE.is_file():
        raise FileNotFoundError(
            f"Brak {WAGE_FILE}. Uruchom notebooks/01_pobranie_plac.ipynb."
        )
    return pd.read_csv(WAGE_FILE)


def league_games(games: pd.DataFrame) -> pd.DataFrame:
    """Mecze czterech lig w analizowanym zakresie sezonów."""
    return games[
        games["competition_id"].isin(LEAGUES.values())
        & games["season"].between(SEASON_MIN, SEASON_MAX)
    ].copy()
