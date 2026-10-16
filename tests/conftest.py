"""Generator syntetycznego panelu klub x sezon o kontrolowanych własnościach.

Proces generujący jest dynamicznym probitem, więc wiadomo, czego diagnostyka
ma szukać. Dwa pokrętła:

- `state_dependence` - siła zależności stanu, czyli wpływ `Y_lag` na `Y`.
- `club_sd` - rozrzut nieobserwowanego efektu klubu.

Testy oparte na tym panelu sprawdzają własności mechaniczne kodu, nie moc testów
statystycznych: czy dana hipoteza zostanie odrzucona na konkretnym losowaniu,
zależy od ziarna, a nie od poprawności implementacji.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

SEED = 20261002

N_CLUBS_PER_LEAGUE = 18
N_SEASONS = 11
LEAGUES = ["Premier_League", "Bundesliga", "La_Liga", "Serie_A"]

FIRST_SEASON = 2013

# Stała dobrana tak, żeby udział kwalifikacji był rzędu jednej piątej, jak
# w rzeczywistym panelu (cztery miejsca na osiemnaście klubów).
INTERCEPT = -1.6
TRUE_WAGE = 0.8

# Rozrzut funduszu płac: wewnątrz klubu między sezonami i między klubami.
WITHIN_WAGE_SD = 0.35
BETWEEN_WAGE_SD = 0.5


def make_panel(
    state_dependence: float = 1.0,
    club_sd: float = 0.8,
    seed: int = SEED,
) -> pd.DataFrame:
    """Syntetyczny panel z `Y` generowanym dynamicznym probitem."""
    rng = np.random.default_rng(seed)
    rows = []
    club_id = 0
    for league in LEAGUES:
        for _ in range(N_CLUBS_PER_LEAGUE):
            club_id += 1
            effect = rng.normal(0.0, club_sd) if club_sd > 0 else 0.0
            wage_level = 18.0 + rng.normal(0.0, BETWEEN_WAGE_SD)
            y_prev = int(rng.random() < 0.22)
            for s in range(N_SEASONS):
                # Rozrzut wewnątrzklubowy porównywalny z międzyklubowym; przy
                # bardzo małym `log_wage` i `mean_log_wage` byłyby współliniowe
                # i oszacowania obu stawałyby się nieidentyfikowalne.
                wage = wage_level + rng.normal(0.0, WITHIN_WAGE_SD)
                lin = (
                    INTERCEPT
                    + TRUE_WAGE * (wage - 18.0)
                    + state_dependence * y_prev
                    + effect
                )
                y = int(rng.random() < norm.cdf(lin))
                rows.append(
                    {
                        "club_id": club_id,
                        "league": league,
                        "competition_id": league[:3].upper(),
                        "season": FIRST_SEASON + s,
                        "Y": y,
                        "rank": 1 + int((1 - y) * 10),
                        "log_wage": wage,
                        "log_mv": wage + rng.normal(0.0, 0.2),
                        "net_spend_m": rng.normal(0.0, 20.0),
                    }
                )
                y_prev = y

    from src.pre_processing import add_cre_variables

    return add_cre_variables(pd.DataFrame(rows))


@pytest.fixture(scope="session")
def panel():
    """Panel z zależnością stanu i nieskorelowanym efektem klubu."""
    return make_panel()


@pytest.fixture(scope="session")
def est(panel):
    from src.pre_processing import to_estimation_frame

    return to_estimation_frame(panel)
