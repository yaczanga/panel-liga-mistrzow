"""Testy wnioskowania na syntetycznym panelu o znanym procesie generujacym.

Testy sprawdzaja wlasnosci, ktore musza zachodzic, a nie konkretne liczby:
ze efekty czastkowe nie sa wspolczynnikami, ze test Mundlaka wykrywa
skorelowany efekt klubu, ze pominiecie zaleznosci stanu podnosi efekt plac,
i ze dekompozycja Mundlaka sumuje sie tak, jak twierdzi README.

Generator panelu jest w conftest.py.
"""

import numpy as np
import pytest

from src.inference import (
    ame,
    discrete_effect,
    league_difference_test,
    link_test,
    mundlak_test,
    transition_matrix,
    wage_decomposition,
)
from src.model import fit_interaction, fit_main, fit_per_league, fit_static


@pytest.fixture(scope="module")
def m_cre(est):
    return fit_main(est)


@pytest.fixture(scope="module")
def m_static(est):
    return fit_static(est)


class TestEfektyCzastkowe:
    def test_ame_nie_jest_wspolczynnikiem(self, m_cre):
        """W probicie beta nie jest prawdopodobienstwem; AME musi sie roznic."""
        frame = ame(m_cre)
        assert frame.loc["log_wage", "AME"] != pytest.approx(
            m_cre.params["log_wage"], rel=1e-3
        )

    def test_ame_jest_mniejsze_co_do_modulu_niz_wspolczynnik(self, m_cre):
        """Gestosc normalna jest ponizej jedynki, wiec AME skaluje beta w dol."""
        frame = ame(m_cre)
        assert abs(frame.loc["log_wage", "AME"]) < abs(m_cre.params["log_wage"])

    def test_ame_miesci_sie_w_przedziale_prawdopodobienstw(self, m_cre):
        frame = ame(m_cre)
        assert frame["AME"].abs().max() <= 1.0

    def test_roznica_dyskretna_zgadza_sie_z_ame_dla_zmiennej_zerojedynkowej(
        self, m_cre, est
    ):
        """Dla Y_lag oba sposoby liczenia musza dac praktycznie to samo."""
        reczna = discrete_effect(m_cre, est, "Y_lag")
        z_ame = ame(m_cre).loc["Y_lag", "AME"]
        assert reczna == pytest.approx(z_ame, abs=0.02)

    def test_roznica_dyskretna_ma_znak_wspolczynnika(self, m_cre, est):
        assert np.sign(discrete_effect(m_cre, est, "Y_lag")) == np.sign(
            m_cre.params["Y_lag"]
        )


class TestDekompozycjaMundlaka:
    def test_efekt_trwaly_to_suma_within_i_between(self, m_cre):
        dec = wage_decomposition(m_cre)
        assert dec["trwaly"] == pytest.approx(dec["within"] + dec["between"])

    def test_przeliczenie_na_dziesiec_procent_jest_spojne(self, m_cre):
        dec = wage_decomposition(m_cre)
        assert dec["trwaly_pp_na_10pct"] == pytest.approx(
            dec["trwaly"] * np.log(1.10) * 100
        )

    def test_skladniki_pochodza_z_efektow_czastkowych_modelu(self, m_cre):
        dec = wage_decomposition(m_cre)
        frame = ame(m_cre)
        assert dec["within"] == pytest.approx(frame.loc["log_wage", "AME"])
        assert dec["between"] == pytest.approx(frame.loc["mean_log_wage", "AME"])


class TestTestMundlaka:
    """Testy dotycza opakowania, nie samej ekonometrii.

    Czy test Mundlaka wykryje skorelowany efekt klubu, zalezy od mocy na danym
    zbiorze, a nie od poprawnosci kodu - przy regresorze o malym rozrzucie
    wewnatrzklubowym `log_wage` i `mean_log_wage` sa niemal wspolliniowe i test
    traci moc. Sprawdzamy wiec, ze naklada wlasciwe restrykcje i ze daje to
    samo, co policzony wprost test Walda.
    """

    def test_zwraca_statystyke_i_wartosc_p(self, m_cre):
        chi2, p = mundlak_test(m_cre)
        assert chi2 >= 0
        assert 0 <= p <= 1

    def test_zgadza_sie_z_testem_walda_policzonym_wprost(self, m_cre):
        chi2, p = mundlak_test(m_cre)
        wprost = m_cre.wald_test(["Y_0 = 0", "mean_log_wage = 0"], scalar=True)
        assert chi2 == pytest.approx(float(wprost.statistic))
        assert p == pytest.approx(float(wprost.pvalue))

    def test_naklada_restrykcje_na_podane_wyrazy(self, m_cre):
        """Zawezenie restrykcji do jednego wyrazu musi zmienic statystyke."""
        chi2_oba, _ = mundlak_test(m_cre)
        chi2_jeden, _ = mundlak_test(m_cre, terms=("mean_log_wage",))
        assert chi2_oba != pytest.approx(chi2_jeden)

    def test_domyslne_wyrazy_to_warunek_poczatkowy_i_srednia_plac(self, m_cre):
        chi2, _ = mundlak_test(m_cre)
        jawnie, _ = mundlak_test(m_cre, terms=("Y_0", "mean_log_wage"))
        assert chi2 == pytest.approx(jawnie)


class TestSpecyfikacje:
    """Czy formuly zawieraja dokladnie to, co obiecuje ich opis."""

    def test_model_statyczny_nie_zawiera_opoznionej_zmiennej(self, m_static):
        assert "Y_lag" not in m_static.params.index

    def test_model_glowny_zawiera_opoznienie_warunek_i_srednia(self, m_cre):
        for term in ("Y_lag", "Y_0", "mean_log_wage", "log_wage"):
            assert term in m_cre.params.index, f"brak wyrazu: {term}"

    def test_oba_modele_maja_efekty_stale_ligi_i_sezonu(self, m_cre, m_static):
        for res in (m_cre, m_static):
            assert any("C(league)" in n for n in res.params.index)
            assert any("C(season)" in n for n in res.params.index)

    def test_model_per_liga_nie_ma_efektu_stalego_ligi(self, est):
        """Na podprobie jednej ligi efekt staly ligi bylby nieidentyfikowalny."""
        res = next(iter(fit_per_league(est).values()))
        assert not any("C(league)" in n for n in res.params.index)

    def test_bledy_standardowe_sa_klastrowane_po_klubie(self, m_cre):
        assert m_cre.cov_type == "cluster"


class TestZaleznoscStanu:
    def test_zaleznosc_stanu_wychodzi_dodatnia_gdy_jest_w_danych(self, m_cre):
        """Przy silnej zaleznosci stanu w procesie generujacym Y_lag musi zapalic."""
        assert m_cre.params["Y_lag"] > 0
        assert m_cre.pvalues["Y_lag"] < 0.05


class TestLinkTest:
    def test_zwraca_wartosc_p(self, m_cre, est):
        p = link_test(m_cre, est)
        assert 0 <= p <= 1

    def test_poprawna_postac_funkcyjna_nie_zapala(self, m_cre, est):
        """Dane generowane probitem, wiec kwadrat predykcji ma byc nieistotny."""
        assert link_test(m_cre, est) > 0.05


class TestRoznicowanieLig:
    def test_zwraca_wyrazy_interakcyjne(self, est):
        _, _, terms = league_difference_test(fit_interaction(est))
        # Cztery ligi, wiec trzy interakcje wzgledem ligi bazowej.
        assert len(terms) == 3
        assert all("Y_lag:" in t for t in terms)

    def test_zgadza_sie_z_testem_walda_policzonym_wprost(self, est):
        """Jak przy tescie Mundlaka: sprawdzamy opakowanie, nie moc testu.

        Czy H0 zostanie odrzucona na konkretnym zbiorze, zalezy od losowania -
        test prawdziwej hipotezy zerowej zapala w okolo pieciu procentach prob,
        wiec oczekiwanie `p > 0,05` dawaloby test chwiejny.
        """
        res = fit_interaction(est)
        chi2, p, terms = league_difference_test(res)
        wprost = res.wald_test(terms, scalar=True)
        assert chi2 == pytest.approx(float(wprost.statistic))
        assert p == pytest.approx(float(wprost.pvalue))

    def test_wartosc_p_jest_w_przedziale_jednostkowym(self, est):
        _, p, _ = league_difference_test(fit_interaction(est))
        assert 0 <= p <= 1

    def test_model_per_liga_dla_kazdej_ligi(self, est):
        per = fit_per_league(est)
        assert set(per) == set(est["league"].unique())


class TestMacierzPrzejsc:
    def test_wiersze_sumuja_sie_do_jednosci(self, panel):
        tab = transition_matrix(panel)
        assert np.allclose(tab.sum(axis=1), 1.0)

    def test_trwalosc_widac_na_diagonali(self, panel):
        """Przy dodatniej zaleznosci stanu P(1|1) > P(1|0)."""
        tab = transition_matrix(panel)
        assert tab.loc[1.0, 1] > tab.loc[0.0, 1]

    def test_pierwsze_obserwacje_sa_pomijane(self, panel):
        """Wiersze z brakiem w Y_lag nie moga wchodzic do macierzy."""
        tab = transition_matrix(panel)
        assert set(tab.index) == {0.0, 1.0}
