"""Specyfikacje probitowe: CRE Wooldridge'a, model referencyjny i rozszerzenia.

Moduł **tworzy** oszacowania. Efekty cząstkowe i testy diagnostyczne są
w `inference`.
"""

import pandas as pd
import statsmodels.formula.api as smf

CLUSTER = "club_id"

# Efekty stałe ligi pochłaniają różnice walut i siły rozgrywek, efekty sezonu -
# wspólne szoki (zmiany w podziale środków z praw telewizyjnych, pandemia).
FE = "C(season) + C(league)"

# Model główny: dynamiczny probit CRE. `Y_lag` mierzy zależność stanu,
# `Y_0` jest warunkiem początkowym, `mean_log_wage` średnią Mundlaka.
CRE = f"Y ~ Y_lag + log_wage + Y_0 + mean_log_wage + {FE}"

# Punkt odniesienia bez zależności stanu - pokazuje, ile z efektu płac
# w statycznym modelu jest w rzeczywistości trwałością pozycji.
STATIC = f"Y ~ log_wage + {FE}"

# Rozszerzenie o saldo transferowe z dekompozycją na część within i between.
NET_SPEND = (
    "Y ~ Y_lag + log_wage + net_spend_m + Y_0 + mean_log_wage + mean_net_spend_m "
    f"+ {FE}"
)

# Odporność: wartość rynkowa kadry (EUR, spójna między ligami) zamiast płac.
MARKET_VALUE = f"Y ~ Y_lag + log_mv + Y_0 + mean_log_mv + {FE}"

# Model per liga: bez efektu stałego ligi, bo próba jest już jednoligowa.
PER_LEAGUE = "Y ~ Y_lag + log_wage + Y_0 + mean_log_wage + C(season)"

# Test różnic zależności stanu między ligami.
INTERACTION = (
    "Y ~ Y_lag * C(league) + log_wage + Y_0 + mean_log_wage + C(season)"
)

LEAGUE_ORDER = ["Premier_League", "Bundesliga", "La_Liga", "Serie_A"]
LEAGUE_LABELS = {
    "Premier_League": "Premier League",
    "Bundesliga": "Bundesliga",
    "La_Liga": "La Liga",
    "Serie_A": "Serie A",
}


def probit(formula: str, data: pd.DataFrame, **kwargs):
    """Probit z błędami standardowymi klastrowanymi po klubie.

    Klastrowanie po klubie, bo obserwacje tego samego klubu w kolejnych
    sezonach nie są niezależne.
    """
    return smf.probit(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[CLUSTER]}, disp=False, **kwargs
    )


def fit_main(est: pd.DataFrame):
    """Model główny CRE."""
    return probit(CRE, est)


def fit_static(est: pd.DataFrame):
    """Statyczny probit bez `Y_lag`."""
    return probit(STATIC, est)


def fit_extensions(est: pd.DataFrame, panel: pd.DataFrame, balanced) -> dict:
    """Rozszerzenia i sprawdzenia odporności modelu głównego.

    Każde rozszerzenie szacujemy na podpróbie o pełnych danych dla dodanej
    zmiennej, więc liczebności różnią się między kolumnami tabeli.
    """
    out = {}

    e_mgr = est.dropna(subset=["mgr_change"])
    out["mgr"] = probit(CRE.replace("Y ~ ", "Y ~ mgr_change + "), e_mgr)

    out["net_spend"] = probit(NET_SPEND, est)

    e_lead = est.dropna(subset=["log_wage_lead"])
    out["wage_lead"] = probit(
        CRE.replace("log_wage +", "log_wage + log_wage_lead +"), e_lead
    )

    e_nslead = est.dropna(subset=["net_spend_lead"])
    out["net_spend_lead"] = probit(
        NET_SPEND.replace("net_spend_m +", "net_spend_m + net_spend_lead +"), e_nslead
    )

    est_mv = panel.dropna(subset=["Y_lag", "log_mv", "mean_log_mv", "Y_0"]).copy()
    est_mv["season"] = est_mv["season"].astype(int)
    out["market_value"] = probit(MARKET_VALUE, est_mv)

    bal = est[est[CLUSTER].isin(balanced)]
    if bal[CLUSTER].nunique() >= 5:
        out["balanced"] = probit(CRE, bal)

    return out


def fit_per_league(est: pd.DataFrame) -> dict:
    """Osobny model CRE dla każdej ligi.

    `warn_convergence=False`, bo na podpróbie jednej ligi (około 200
    obserwacji) optymalizacja bywa na granicy tolerancji, co nie unieważnia
    oszacowania punktowego.
    """
    return {
        lg: probit(PER_LEAGUE, est[est["league"] == lg], warn_convergence=False)
        for lg in LEAGUE_ORDER
        if (est["league"] == lg).any()
    }


def fit_interaction(est: pd.DataFrame):
    """Model z interakcją `Y_lag` x liga - podstawa testu różnic."""
    return probit(INTERACTION, est)
