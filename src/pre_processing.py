"""Budowa panelu klub x liga x sezon i konstrukcja zmiennych CRE."""

import numpy as np
import pandas as pd

from src.data_load import (
    CL_COMPETITIONS,
    ID2LEAGUE,
    PROCESSED,
    SEASON_MIN,
    SEASON_START_MONTH,
    SUMMER_WINDOW,
    WAGE2ID,
)

# Liczba miejsc dajacych bezposredni awans do Ligi Mistrzow.
QUAL_SPOTS = 4

# Zmienne, ktorych srednie wewnatrzklubowe wchodza do rownania jako urzadzenie
# Mundlaka.
MUNDLAK_VARS = ["log_wage", "log_mv", "net_spend_m"]

# Braki w tych zmiennych usuwaja obserwacje z proby estymacyjnej.
ESTIMATION_VARS = ["Y_lag", "log_wage", "mean_log_wage", "Y_0"]

PANEL_FILE = PROCESSED / "panel_klub_sezon.csv"


def league_table(club_games: pd.DataFrame, lg: pd.DataFrame) -> pd.DataFrame:
    """Tabela koncowa: punkty, bilans, pozycja i Y w obrebie liga x sezon.

    Pozycja i zmienna celu sa wyznaczane **wewnatrz pary liga x sezon**, nie
    globalnie - inaczej kluby slabszej ligi nigdy nie trafilyby do top-4.
    """
    cg = club_games.merge(
        lg[["game_id", "season", "competition_id"]], on="game_id", how="inner"
    )
    cg["draw"] = (cg["own_goals"] == cg["opponent_goals"]).astype(int)
    cg["points"] = cg["is_win"] * 3 + cg["draw"]

    agg = (
        cg.groupby(["competition_id", "club_id", "season"])
        .agg(
            games=("game_id", "count"),
            points=("points", "sum"),
            gf=("own_goals", "sum"),
            ga=("opponent_goals", "sum"),
        )
        .reset_index()
    )
    agg["gd"] = agg["gf"] - agg["ga"]
    agg["rank"] = (
        agg.sort_values(["points", "gd", "gf"], ascending=False)
        .groupby(["competition_id", "season"])
        .cumcount()
        + 1
    )
    agg["Y"] = (agg["rank"] <= QUAL_SPOTS).astype(int)
    agg["league"] = agg["competition_id"].map(ID2LEAGUE)
    return agg, cg


def squad_value(valuations: pd.DataFrame, seasons) -> pd.DataFrame:
    """Suma wycen kadry na 1 lipca kazdego sezonu.

    Dla kazdego zawodnika bierzemy jego **ostatnia wycene nie pozniejsza** niz
    1 lipca, wiec do wartosci kadry na start sezonu nie wchodza wyceny
    powstale juz w trakcie jego trwania.
    """
    val = valuations.dropna(subset=["market_value_in_eur", "current_club_id"])

    frames = []
    for season in seasons:
        cutoff = pd.Timestamp(year=int(season), month=SEASON_START_MONTH, day=1)
        last = (
            val[val["date"] <= cutoff]
            .sort_values("date")
            .groupby("player_id")
            .tail(1)
        )
        out = (
            last.groupby("current_club_id")["market_value_in_eur"]
            .sum()
            .rename("squad_value")
            .reset_index()
            .rename(columns={"current_club_id": "club_id"})
        )
        out["season"] = int(season)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def add_wages(panel: pd.DataFrame, wages: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dolaczenie funduszu plac po mapowaniu nazw fbref na club_id.

    Waluta rozni sie miedzy ligami (GBP, EUR); w logarytmach stala przelicznika
    jest pochlaniana przez efekt staly ligi, wiec nie trzeba jej ujednolicac.

    Zwraca (panel, nazwy niezmapowane) - te drugie warto obejrzec, bo kazda
    niezmapowana nazwa to utracone obserwacje.
    """
    w = wages.copy()
    w["club_id"] = w.apply(
        lambda r: WAGE2ID.get(r["League"], {}).get(r["Squad"]), axis=1
    )
    unmatched = w[w["club_id"].isna()][["League", "Squad"]].drop_duplicates()

    w["season"] = w["Season"].str.slice(0, 4).astype(int)
    w = w.rename(columns={"Annual Wages": "wage_bill"}).dropna(subset=["club_id"])
    w["club_id"] = w["club_id"].astype(int)

    out = panel.merge(
        w[["club_id", "season", "wage_bill"]], on=["club_id", "season"], how="left"
    )
    out["log_wage"] = np.log(out["wage_bill"])
    return out, unmatched


def add_manager_change(panel: pd.DataFrame, cg: pd.DataFrame) -> pd.DataFrame:
    """Zmiana trenera w sezonie: wiecej niz jedno nazwisko w meczach klubu."""
    mgr = (
        cg.groupby(["club_id", "season"])["own_manager_name"]
        .nunique()
        .rename("n_mgr")
        .reset_index()
    )
    mgr["mgr_change"] = (mgr["n_mgr"] > 1).astype(int)
    return panel.merge(
        mgr[["club_id", "season", "mgr_change"]], on=["club_id", "season"], how="left"
    )


def add_cl_participation(
    panel: pd.DataFrame, games: pd.DataFrame, club_games: pd.DataFrame
) -> pd.DataFrame:
    """Udzial w Lidze Mistrzow w danym sezonie.

    Zmienna opisowa: nie wchodzi do zadnego modelu, bo jest niemal tozsama
    z `Y_lag` (awans z poprzedniego sezonu oznacza gre w LM w biezacym).
    Zostaje w panelu, bo jest uzyteczna przy opisie proby.
    """
    cl_games = games[games["competition_id"].isin(CL_COMPETITIONS)][["game_id", "season"]]
    cl_part = (
        club_games.merge(cl_games, on="game_id", how="inner")
        .drop_duplicates(["club_id", "season"])[["club_id", "season"]]
        .assign(cl_part=1)
    )
    out = panel.merge(cl_part, on=["club_id", "season"], how="left")
    out["cl_part"] = out["cl_part"].fillna(0).astype(int)
    return out


def add_promoted(panel: pd.DataFrame) -> pd.DataFrame:
    """Beniaminek: klub nieobecny w **swojej** lidze w sezonie poprzednim.

    Pierwszy sezon panelu dostaje brak, nie zero - nie wiemy, czy klub byl
    wtedy w lidze (cenzura lewostronna).
    """
    present = set(zip(panel["competition_id"], panel["club_id"], panel["season"]))

    def is_promoted(row):
        if row["season"] == SEASON_MIN:
            return np.nan
        return int(
            (row["competition_id"], row["club_id"], row["season"] - 1) not in present
        )

    out = panel.copy()
    out["promoted"] = out.apply(is_promoted, axis=1)
    return out


def add_net_spend(panel: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """Saldo transferowe (mln EUR) w letnim okienku przed sezonem.

    Okno ograniczone do czerwca-sierpnia roku rozpoczecia sezonu, zeby
    uchwycic inwestycje **przed** sezonem, a nie korekte w jego trakcie.
    """
    tr = transfers.copy()
    tr["fee"] = tr["transfer_fee"].fillna(0)
    tr["start_year"] = pd.to_numeric(
        "20" + tr["transfer_season"].str.slice(0, 2), errors="coerce"
    )
    lo, hi = SUMMER_WINDOW
    sw = tr[
        tr["transfer_date"].dt.month.between(lo, hi)
        & (tr["transfer_date"].dt.year == tr["start_year"])
    ]

    buys = (
        sw.groupby(["to_club_id", "start_year"])["fee"]
        .sum()
        .rename("buy")
        .reset_index()
        .rename(columns={"to_club_id": "club_id", "start_year": "season"})
    )
    sells = (
        sw.groupby(["from_club_id", "start_year"])["fee"]
        .sum()
        .rename("sell")
        .reset_index()
        .rename(columns={"from_club_id": "club_id", "start_year": "season"})
    )
    ns = buys.merge(sells, on=["club_id", "season"], how="outer").fillna(0)
    ns["net_spend_m"] = (ns["buy"] - ns["sell"]) / 1e6

    out = panel.merge(
        ns[["club_id", "season", "net_spend_m"]], on=["club_id", "season"], how="left"
    )
    out["net_spend_m"] = out["net_spend_m"].fillna(0.0)
    return out


def add_cre_variables(panel: pd.DataFrame) -> pd.DataFrame:
    """Zmienne CRE Wooldridge'a: `Y_lag`, `Y_0`, srednie Mundlaka, lead-y.

    Wszystkie przesuniecia sa liczone wewnatrz klubu po uporzadkowaniu po
    sezonie, a `Y_0` to stan w pierwszym obserwowanym sezonie klubu, ktory
    w panelu niezbalansowanym nie dla kazdego klubu wypada w tym samym roku.
    Lead-y (`*_lead`) sluza wylacznie testowi scislej egzogenicznosci.
    """
    out = panel.sort_values(["club_id", "season"]).reset_index(drop=True)
    out["Y_lag"] = out.groupby("club_id")["Y"].shift(1)

    first = out.groupby("club_id").first()
    out["Y_0"] = out["club_id"].map(first["Y"])

    for v in MUNDLAK_VARS:
        out[f"mean_{v}"] = out.groupby("club_id")[v].transform("mean")

    out["log_wage_lead"] = out.groupby("club_id")["log_wage"].shift(-1)
    out["net_spend_lead"] = out.groupby("club_id")["net_spend_m"].shift(-1)
    return out


def build_panel(
    games: pd.DataFrame,
    club_games: pd.DataFrame,
    valuations: pd.DataFrame,
    transfers: pd.DataFrame,
    wages: pd.DataFrame,
    lg: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pelny panel klub x sezon ze wszystkimi regresorami.

    Zwraca (panel, nazwy klubow niezmapowane miedzy fbref i Transfermarkt).
    """
    agg, cg = league_table(club_games, lg)

    mv = squad_value(valuations, agg["season"].unique())
    panel = agg.merge(mv, on=["club_id", "season"], how="left")
    panel["log_mv"] = np.log(panel["squad_value"].replace(0, np.nan))

    panel, unmatched = add_wages(panel, wages)
    panel = add_manager_change(panel, cg)
    panel = add_cl_participation(panel, games, club_games)
    panel = add_promoted(panel)
    panel = add_net_spend(panel, transfers)
    panel = add_cre_variables(panel)
    return panel, unmatched


def to_estimation_frame(panel: pd.DataFrame, save: bool = False) -> pd.DataFrame:
    """Proba estymacyjna: obserwacje z pelnym zestawem zmiennych CRE."""
    est = panel.dropna(subset=ESTIMATION_VARS).copy()
    est["season"] = est["season"].astype(int)

    if save:
        PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(PANEL_FILE, index=False, encoding="utf-8")
        print(f"Zapisano panel: {PANEL_FILE.name} ({len(panel)} obserwacji)")
    return est


def load_panel() -> pd.DataFrame:
    """Gotowy panel z data/processed - bez przeliczania od danych surowych."""
    if not PANEL_FILE.is_file():
        raise FileNotFoundError(
            f"Brak {PANEL_FILE}. Uruchom notebooks/02_panel_cre_probit.ipynb."
        )
    return pd.read_csv(PANEL_FILE)


def balanced_clubs(panel: pd.DataFrame) -> pd.Index:
    """Kluby obecne we wszystkich sezonach panelu - podproba zbalansowana."""
    counts = panel.groupby("club_id")["season"].nunique()
    return counts[counts == panel["season"].nunique()].index


def describe_panel(est: pd.DataFrame) -> pd.DataFrame:
    """Liczebnosc, liczba klubow i udzial kwalifikacji w podziale na ligi."""
    return (
        est.groupby("league")
        .agg(obs=("Y", "size"), klubow=("club_id", "nunique"), kwal_rate=("Y", "mean"))
        .round(3)
    )
