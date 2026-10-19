"""Efekty czastkowe, testy diagnostyczne i dekompozycja Mundlaka.

W probicie wspolczynniki nie sa prawdopodobienstwami, wiec interpretowalna
wielkoscia jest przecietny efekt czastkowy (AME), nie `beta`.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.model import CLUSTER

# Nazwa kolumn zwracanych przez statsmodels w summary_frame efektow czastkowych.
AME_COLUMNS = ["AME", "se", "z", "p", "ci_lo", "ci_hi"]

LABELS = {
    "Y_lag": "Y_lag (kwalifikacja t–1)",
    "log_wage": "log_wage (fundusz płac)",
    "Y_0": "Y₀ (warunek początkowy)",
    "mean_log_wage": "x̄ mean_log_wage (Mundlak)",
    "log_mv": "log_mv (wartość rynkowa)",
    "mean_log_mv": "x̄ mean_log_mv (Mundlak)",
    "mgr_change": "mgr_change (zmiana trenera)",
    "net_spend_m": "net_spend_m (within)",
    "mean_net_spend_m": "mean_net_spend_m (between)",
    "log_wage_lead": "log_wage_lead (lead)",
    "C(league)[T.Premier_League]": "Efekt stały: Premier League",
    "C(league)[T.La_Liga]": "Efekt stały: La Liga",
    "C(league)[T.Serie_A]": "Efekt stały: Serie A",
}


def ame(res) -> pd.DataFrame:
    """Przecietne efekty czastkowe: roznica dyskretna dla 0/1, pochodna dla ciaglych.

    Bledy standardowe dziedzicza klastrowanie po klubie z macierzy modelu.
    """
    frame = res.get_margeff(at="overall", method="dydx", dummy=True).summary_frame()
    frame.columns = AME_COLUMNS
    return frame


def discrete_effect(res, data: pd.DataFrame, variable: str) -> float:
    """Roznica dyskretna usredniona po probie dla zmiennej zero-jedynkowej.

    Dla zmiennej 0/1 poprawnym efektem jest Phi(.|1) - Phi(.|0) policzone dla
    kazdej obserwacji i usrednione, a nie pochodna w punkcie srednim.
    """
    d1 = data.copy()
    d1[variable] = 1.0
    d0 = data.copy()
    d0[variable] = 0.0
    return float((res.predict(d1) - res.predict(d0)).mean())


def mundlak_test(res, terms=("Y_0", "mean_log_wage")) -> tuple[float, float]:
    """Laczny test istotnosci warunku poczatkowego i srednich Mundlaka.

    H0: efekt klubu nieskorelowany z regresorami, czyli RE wystarcza.
    """
    w = res.wald_test([f"{t} = 0" for t in terms], scalar=True)
    return float(w.statistic), float(w.pvalue)


def link_test(res, est: pd.DataFrame) -> float:
    """Test Pregibona: kwadrat predykcji liniowej nie powinien byc istotny."""
    d = est.copy()
    d["_xb"] = res.fittedvalues
    d["_xb2"] = res.fittedvalues**2
    aux = smf.probit("Y ~ _xb + _xb2", data=d).fit(disp=False)
    return float(aux.pvalues["_xb2"])


def exogeneity_test(res, variable: str) -> float:
    """Wartosc p dla lead-a regresora - test scislej egzogenicznosci."""
    return float(res.pvalues[variable])


def league_difference_test(res_interaction) -> tuple[float, float, list[str]]:
    """Laczny test Walda rownosci `Y_lag` miedzy ligami."""
    terms = [n for n in res_interaction.params.index if "Y_lag:" in n]
    w = res_interaction.wald_test(terms, scalar=True)
    return float(w.statistic), float(w.pvalue), terms


def wage_decomposition(res) -> dict:
    """Rozbicie efektu plac na krotkookresowy i trwaly.

    `log_wage` jest czescia within, `mean_log_wage` czescia between, a ich
    suma efektem trwalego podniesienia funduszu plac.
    """
    frame = ame(res)
    within = float(frame.loc["log_wage", "AME"])
    between = float(frame.loc["mean_log_wage", "AME"])
    permanent = within + between
    return {
        "within": within,
        "between": between,
        "trwaly": permanent,
        # Przeliczenie na wzrost plac o 10 procent, w punktach procentowych.
        "trwaly_pp_na_10pct": permanent * np.log(1.10) * 100,
    }


def transition_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Empiryczna macierz przejsc stanu kwalifikacji, wiersze normalizowane."""
    d = panel.dropna(subset=["Y_lag"])
    return pd.crosstab(d["Y_lag"], d["Y"], normalize="index")
