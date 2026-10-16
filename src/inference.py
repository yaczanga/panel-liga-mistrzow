"""Efekty cząstkowe, testy diagnostyczne i dekompozycja Mundlaka.

W probicie współczynniki nie są prawdopodobieństwami, więc interpretowalną
wielkością jest przeciętny efekt cząstkowy (AME), nie `beta`.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.model import CLUSTER

# Nazwa kolumn zwracanych przez statsmodels w summary_frame efektów cząstkowych.
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
    """Przeciętne efekty cząstkowe: różnica dyskretna dla 0/1, pochodna dla ciągłych.

    Błędy standardowe dziedziczą klastrowanie po klubie z macierzy modelu.
    """
    frame = res.get_margeff(at="overall", method="dydx", dummy=True).summary_frame()
    frame.columns = AME_COLUMNS
    return frame


def discrete_effect(res, data: pd.DataFrame, variable: str) -> float:
    """Różnica dyskretna uśredniona po próbie dla zmiennej zero-jedynkowej.

    Dla zmiennej 0/1 poprawnym efektem jest Phi(.|1) - Phi(.|0) policzone dla
    każdej obserwacji i uśrednione, a nie pochodna w punkcie średnim.
    """
    d1 = data.copy()
    d1[variable] = 1.0
    d0 = data.copy()
    d0[variable] = 0.0
    return float((res.predict(d1) - res.predict(d0)).mean())


def mundlak_test(res, terms=("Y_0", "mean_log_wage")) -> tuple[float, float]:
    """Łączny test istotności warunku początkowego i średnich Mundlaka.

    H0: efekt klubu nieskorelowany z regresorami, czyli RE wystarcza.
    Odrzucenie oznacza, że CRE jest konieczne, a RE obciążony.
    """
    w = res.wald_test([f"{t} = 0" for t in terms], scalar=True)
    return float(w.statistic), float(w.pvalue)


def link_test(res, est: pd.DataFrame) -> float:
    """Test Pregibona: kwadrat predykcji liniowej nie powinien być istotny.

    Zwraca wartość p. Istotny kwadrat sygnalizuje złą postać funkcyjną.
    """
    d = est.copy()
    d["_xb"] = res.fittedvalues
    d["_xb2"] = res.fittedvalues**2
    aux = smf.probit("Y ~ _xb + _xb2", data=d).fit(disp=False)
    return float(aux.pvalues["_xb2"])


def exogeneity_test(res, variable: str) -> float:
    """Wartość p dla lead-a regresora - test ścisłej egzogeniczności.

    Istotny lead oznacza feedback: regresor reaguje na przeszłe wyniki, więc
    ścisła egzogeniczność nie zachodzi.
    """
    return float(res.pvalues[variable])


def league_difference_test(res_interaction) -> tuple[float, float, list[str]]:
    """Łączny test Walda równości `Y_lag` między ligami.

    Brak odrzucenia H0 uzasadnia estymację zunifikowaną na czterech ligach.
    """
    terms = [n for n in res_interaction.params.index if "Y_lag:" in n]
    w = res_interaction.wald_test(terms, scalar=True)
    return float(w.statistic), float(w.pvalue), terms


def wage_decomposition(res) -> dict:
    """Rozbicie efektu płac na krótkookresowy i trwały.

    `log_wage` to efekt within (zmiana funduszu płac w klubie),
    `mean_log_wage` to część between (trwały poziom zamożności klubu).
    Suma obu jest efektem trwałego podniesienia funduszu płac.
    """
    frame = ame(res)
    within = float(frame.loc["log_wage", "AME"])
    between = float(frame.loc["mean_log_wage", "AME"])
    permanent = within + between
    return {
        "within": within,
        "between": between,
        "trwaly": permanent,
        # Przeliczenie na wzrost płac o 10 procent, w punktach procentowych.
        "trwaly_pp_na_10pct": permanent * np.log(1.10) * 100,
    }


def transition_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Empiryczna macierz przejść stanu kwalifikacji, wiersze normalizowane."""
    d = panel.dropna(subset=["Y_lag"])
    return pd.crosstab(d["Y_lag"], d["Y"], normalize="index")
