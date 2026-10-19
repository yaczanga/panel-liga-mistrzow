"""Specyfikacje probitowe: CRE Wooldridge'a, model referencyjny i rozszerzenia.

Modul tworzy oszacowania; efekty czastkowe i testy sa w `inference`.
"""

import pandas as pd
import statsmodels.formula.api as smf

CLUSTER = "club_id"

# Efekty stale ligi pochlaniaja roznice walut i sily rozgrywek, sezonu - szoki wspolne.
FE = "C(season) + C(league)"

# Model glowny: `Y_lag` to zaleznosc stanu, `Y_0` warunek poczatkowy, `mean_log_wage` srednia Mundlaka.
CRE = f"Y ~ Y_lag + log_wage + Y_0 + mean_log_wage + {FE}"

# Punkt odniesienia bez zaleznosci stanu.
STATIC = f"Y ~ log_wage + {FE}"

# Rozszerzenie o saldo transferowe z dekompozycja na czesc within i between.
NET_SPEND = (
    "Y ~ Y_lag + log_wage + net_spend_m + Y_0 + mean_log_wage + mean_net_spend_m "
    f"+ {FE}"
)

# Odpornosc: wartosc rynkowa kadry (EUR, spojna miedzy ligami) zamiast plac.
MARKET_VALUE = f"Y ~ Y_lag + log_mv + Y_0 + mean_log_mv + {FE}"

# Model per liga: bez efektu stalego ligi, bo proba jest juz jednoligowa.
PER_LEAGUE = "Y ~ Y_lag + log_wage + Y_0 + mean_log_wage + C(season)"

# Test roznic zaleznosci stanu miedzy ligami.
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
    """Probit z bledami standardowymi klastrowanymi po klubie.

    Obserwacje tego samego klubu w kolejnych sezonach nie sa niezalezne.
    """
    return smf.probit(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[CLUSTER]}, disp=False, **kwargs
    )


def fit_main(est: pd.DataFrame):
    """Model glowny CRE."""
    return probit(CRE, est)


def fit_static(est: pd.DataFrame):
    """Statyczny probit bez `Y_lag`."""
    return probit(STATIC, est)


def fit_extensions(est: pd.DataFrame, panel: pd.DataFrame, balanced) -> dict:
    """Rozszerzenia i sprawdzenia odpornosci modelu glownego.

    Kazde rozszerzenie szacujemy na podprobie o pelnych danych dla dodanej
    zmiennej, wiec liczebnosci roznia sie miedzy kolumnami tabeli.
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
    """Osobny model CRE dla kazdej ligi.

    `warn_convergence=False`, bo na podprobie jednej ligi (okolo 200
    obserwacji) optymalizacja bywa na granicy tolerancji, co nie uniewaznia
    oszacowania punktowego.
    """
    return {
        lg: probit(PER_LEAGUE, est[est["league"] == lg], warn_convergence=False)
        for lg in LEAGUE_ORDER
        if (est["league"] == lg).any()
    }


def fit_interaction(est: pd.DataFrame):
    """Model z interakcja `Y_lag` x liga - podstawa testu roznic."""
    return probit(INTERACTION, est)
