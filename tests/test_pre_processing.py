"""Testy gwarancji metodologicznych budowy panelu klub x sezon.

Kazdy test dziala na malej probie syntetycznej, bez danych zrodlowych, wiec
wykonuje sie w sekundy i dziala w CI, gdzie 65 MB plikow Transfermarkt nie
istnieje.

Sprawdzane sa wlasnosci, na ktorych stoja wnioski: wyznaczanie zmiennej celu
w obrebie ligi i sezonu, brak wycieku przyszlych wycen do wartosci kadry na
start sezonu, brak przecieku miedzy klubami w zmiennych dynamicznych oraz
poprawnosc warunku poczatkowego i srednich Mundlaka.
"""

import numpy as np
import pandas as pd
import pytest

from src.pre_processing import (
    QUAL_SPOTS,
    add_cre_variables,
    add_net_spend,
    add_promoted,
    add_wages,
    balanced_clubs,
    league_table,
    squad_value,
)


def make_club_games(rows):
    """Mecze z perspektywy klubu: (game_id, club_id, gole_wlasne, gole_przeciwnika)."""
    return pd.DataFrame(
        [
            {
                "game_id": g,
                "club_id": c,
                "own_goals": og,
                "opponent_goals": pg,
                "is_win": int(og > pg),
                "own_manager_name": f"T{c}",
            }
            for g, c, og, pg in rows
        ]
    )


def make_lg(rows):
    """Mecze ligowe: (game_id, sezon, competition_id)."""
    return pd.DataFrame(
        [{"game_id": g, "season": s, "competition_id": comp} for g, s, comp in rows]
    )


class TestZmiennaCeluWObrebieLigiISezonu:
    """`Y` musi byc wyznaczane w parze liga x sezon, nie w rankingu globalnym."""

    def test_kazda_liga_ma_wlasna_czworke(self):
        """Dwie ligi, w kazdej szesc klubow - razem osiem jedynek, nie cztery."""
        cg_rows, lg_rows = [], []
        gid = 0
        for comp, base in (("GB1", 100), ("L1", 200)):
            for i in range(6):
                # Klub o nizszym indeksie wygrywa wiecej meczow.
                for _ in range(6 - i):
                    cg_rows.append((gid, base + i, 3, 0))
                    lg_rows.append((gid, 2015, comp))
                    gid += 1
        agg, _ = league_table(make_club_games(cg_rows), make_lg(lg_rows))
        assert agg["Y"].sum() == 2 * QUAL_SPOTS
        for comp in ("GB1", "L1"):
            assert agg[agg["competition_id"] == comp]["Y"].sum() == QUAL_SPOTS

    def test_slabsza_liga_tez_ma_swoich_kwalifikantow(self):
        """Przy rankingu globalnym kluby slabszej ligi mialyby zawsze Y = 0."""
        cg_rows, lg_rows = [], []
        gid = 0
        # Liga GB1 zdobywa duzo punktow, L1 wyraznie mniej - ale kazdy klub
        # musi rozegrac co najmniej jeden mecz, inaczej zniknie z panelu.
        for comp, base, wins in (("GB1", 100, 30), ("L1", 200, 6)):
            for i in range(5):
                for _ in range(wins - i):
                    cg_rows.append((gid, base + i, 3, 0))
                    lg_rows.append((gid, 2015, comp))
                    gid += 1
        agg, _ = league_table(make_club_games(cg_rows), make_lg(lg_rows))
        slabsza = agg[agg["competition_id"] == "L1"]
        assert slabsza["Y"].sum() == QUAL_SPOTS

    def test_kazdy_sezon_ma_wlasna_czworke(self):
        cg_rows, lg_rows = [], []
        gid = 0
        for season in (2015, 2016):
            for i in range(6):
                for _ in range(6 - i):
                    cg_rows.append((gid, 100 + i, 3, 0))
                    lg_rows.append((gid, season, "GB1"))
                    gid += 1
        agg, _ = league_table(make_club_games(cg_rows), make_lg(lg_rows))
        for season in (2015, 2016):
            assert agg[agg["season"] == season]["Y"].sum() == QUAL_SPOTS

    def test_pozycja_rozstrzygana_bilansem_przy_rownych_punktach(self):
        """Dwa klub po jednym zwyciestwie; lepszy bilans daje wyzsze miejsce."""
        cg = make_club_games([(1, 100, 5, 0), (2, 200, 1, 0)])
        lg = make_lg([(1, 2015, "GB1"), (2, 2015, "GB1")])
        agg, _ = league_table(cg, lg)
        pos = agg.set_index("club_id")["rank"]
        assert pos[100] < pos[200]

    def test_punkty_liczone_trzy_za_zwyciestwo_jeden_za_remis(self):
        cg = make_club_games([(1, 100, 2, 0), (2, 100, 1, 1), (3, 100, 0, 3)])
        lg = make_lg([(g, 2015, "GB1") for g in (1, 2, 3)])
        agg, _ = league_table(cg, lg)
        assert agg.loc[0, "points"] == 4


class TestWartoscKadryBezWyciekuZPrzyszlosci:
    """Wycena na start sezonu nie moze zawierac wycen pozniejszych."""

    def test_wycena_pozniejsza_niz_pierwszy_lipca_jest_pomijana(self):
        val = pd.DataFrame(
            {
                "player_id": [1, 1],
                "current_club_id": [100, 100],
                "date": pd.to_datetime(["2015-06-01", "2015-12-01"]),
                "market_value_in_eur": [1_000_000.0, 50_000_000.0],
            }
        )
        out = squad_value(val, [2015]).set_index("club_id")
        assert out.loc[100, "squad_value"] == 1_000_000.0

    def test_brana_jest_ostatnia_wycena_przed_cutoffem(self):
        val = pd.DataFrame(
            {
                "player_id": [1, 1, 1],
                "current_club_id": [100, 100, 100],
                "date": pd.to_datetime(["2014-01-01", "2015-05-01", "2015-06-30"]),
                "market_value_in_eur": [1e6, 2e6, 3e6],
            }
        )
        out = squad_value(val, [2015]).set_index("club_id")
        assert out.loc[100, "squad_value"] == 3e6

    def test_wartosc_kadry_to_suma_po_zawodnikach(self):
        val = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "current_club_id": [100, 100, 200],
                "date": pd.to_datetime(["2015-06-01"] * 3),
                "market_value_in_eur": [1e6, 2e6, 5e6],
            }
        )
        out = squad_value(val, [2015]).set_index("club_id")
        assert out.loc[100, "squad_value"] == 3e6
        assert out.loc[200, "squad_value"] == 5e6

    def test_kolejne_sezony_maja_wlasne_cutoffy(self):
        val = pd.DataFrame(
            {
                "player_id": [1, 1],
                "current_club_id": [100, 100],
                "date": pd.to_datetime(["2015-06-01", "2016-06-01"]),
                "market_value_in_eur": [1e6, 4e6],
            }
        )
        out = squad_value(val, [2015, 2016]).set_index("season")
        assert out.loc[2015, "squad_value"] == 1e6
        assert out.loc[2016, "squad_value"] == 4e6


class TestZmienneDynamiczneBezPrzecieku:
    @staticmethod
    def base(rows):
        """Panel: (club_id, sezon, Y, log_wage)."""
        df = pd.DataFrame(
            [
                {"club_id": c, "season": s, "Y": y, "log_wage": w,
                 "log_mv": w, "net_spend_m": 0.0}
                for c, s, y, w in rows
            ]
        )
        return add_cre_variables(df)

    def test_pierwsza_obserwacja_klubu_ma_brak_w_opoznieniu(self):
        out = self.base([(1, 2015, 1, 18.0), (1, 2016, 0, 18.5)]).set_index("season")
        assert pd.isna(out.loc[2015, "Y_lag"])
        assert out.loc[2016, "Y_lag"] == 1

    def test_opoznienie_nie_przecieka_miedzy_klubami(self):
        out = self.base(
            [(1, 2015, 1, 18.0), (1, 2016, 1, 18.0), (2, 2015, 0, 17.0), (2, 2016, 0, 17.0)]
        ).set_index(["club_id", "season"])
        assert pd.isna(out.loc[(2, 2015), "Y_lag"])
        assert out.loc[(2, 2016), "Y_lag"] == 0

    def test_warunek_poczatkowy_to_stan_pierwszego_sezonu_klubu(self):
        out = self.base(
            [(1, 2015, 1, 18.0), (1, 2016, 0, 18.0), (2, 2016, 0, 17.0), (2, 2017, 1, 17.0)]
        ).set_index(["club_id", "season"])
        assert out.loc[(1, 2015), "Y_0"] == 1
        assert out.loc[(1, 2016), "Y_0"] == 1
        assert out.loc[(2, 2016), "Y_0"] == 0
        assert out.loc[(2, 2017), "Y_0"] == 0

    def test_warunek_poczatkowy_nie_zalezy_od_kolejnosci_wierszy(self):
        rosnaco = self.base([(1, 2015, 1, 18.0), (1, 2016, 0, 18.0)])
        malejaco = self.base([(1, 2016, 0, 18.0), (1, 2015, 1, 18.0)])
        assert list(rosnaco["Y_0"]) == list(malejaco["Y_0"]) == [1, 1]

    def test_srednie_mundlaka_liczone_w_obrebie_klubu(self):
        out = self.base(
            [(1, 2015, 0, 10.0), (1, 2016, 0, 20.0), (2, 2015, 0, 100.0), (2, 2016, 0, 200.0)]
        ).set_index("club_id")
        assert out.loc[1, "mean_log_wage"].tolist() == [15.0, 15.0]
        assert out.loc[2, "mean_log_wage"].tolist() == [150.0, 150.0]

    def test_lead_to_wartosc_z_nastepnego_sezonu_tego_samego_klubu(self):
        out = self.base(
            [(1, 2015, 0, 10.0), (1, 2016, 0, 20.0), (2, 2015, 0, 99.0)]
        ).set_index(["club_id", "season"])
        assert out.loc[(1, 2015), "log_wage_lead"] == 20.0
        # Ostatni sezon klubu nie ma nastepnika, a nie wartosc innego klubu.
        assert pd.isna(out.loc[(1, 2016), "log_wage_lead"])
        assert pd.isna(out.loc[(2, 2015), "log_wage_lead"])


class TestBeniaminek:
    @staticmethod
    def panel(rows):
        """(competition_id, club_id, sezon)."""
        return pd.DataFrame(
            [{"competition_id": comp, "club_id": c, "season": s} for comp, c, s in rows]
        )

    def test_pierwszy_sezon_panelu_ma_brak_a_nie_zero(self):
        """Cenzura lewostronna: nie wiemy, czy klub byl wczesniej w lidze."""
        out = add_promoted(self.panel([("GB1", 100, 2013)]))
        assert pd.isna(out.loc[0, "promoted"])

    def test_klub_nieobecny_rok_wczesniej_jest_beniaminkiem(self):
        out = add_promoted(self.panel([("GB1", 100, 2014)])).set_index("club_id")
        assert out.loc[100, "promoted"] == 1

    def test_klub_obecny_rok_wczesniej_nie_jest_beniaminkiem(self):
        out = add_promoted(
            self.panel([("GB1", 100, 2014), ("GB1", 100, 2015)])
        ).set_index("season")
        assert out.loc[2015, "promoted"] == 0

    def test_obecnosc_liczy_sie_w_obrebie_tej_samej_ligi(self):
        """Klub, ktory rok wczesniej gral w innej lidze, jest tu beniaminkiem."""
        out = add_promoted(
            self.panel([("L1", 100, 2014), ("GB1", 100, 2015)])
        ).set_index("competition_id")
        assert out.loc["GB1", "promoted"] == 1


class TestSaldoTransferowe:
    COLUMNS = ["to_club_id", "from_club_id", "transfer_date", "transfer_fee",
               "transfer_season"]

    @classmethod
    def transfers(cls, rows):
        """(to_club, from_club, data, kwota, transfer_season).

        Kolumny podane jawnie, zeby pusta ramka miala ten sam uklad co plik
        zrodlowy - inaczej test pustego przypadku sprawdzalby brak kolumn,
        a nie zachowanie funkcji.
        """
        frame = pd.DataFrame(
            [
                {"to_club_id": t, "from_club_id": f, "transfer_date": pd.Timestamp(d),
                 "transfer_fee": fee, "transfer_season": ts}
                for t, f, d, fee, ts in rows
            ],
            columns=cls.COLUMNS,
        )
        frame["transfer_date"] = pd.to_datetime(frame["transfer_date"])
        frame["transfer_season"] = frame["transfer_season"].astype("object")
        return frame

    def test_transfer_zimowy_nie_wchodzi_do_okienka_letniego(self):
        """Styczen nowego roku kalendarzowego: poza okienkiem i poza rokiem startu."""
        panel = pd.DataFrame({"club_id": [100], "season": [2015]})
        tr = self.transfers([(100, 200, "2016-01-15", 10e6, "15/16")])
        out = add_net_spend(panel, tr)
        assert out.loc[0, "net_spend_m"] == 0.0

    def test_sam_filtr_miesiaca_wyklucza_transfer_z_wlasciwego_roku(self):
        """Transfer ze stycznia roku startu sezonu odrzuca wylacznie filtr miesiaca.

        Rok zgadza sie ze `start_year`, wiec gdyby okienko nie bylo zawezone do
        czerwca-sierpnia, kwota weszlaby do salda. Bez tego testu rozszerzenie
        okienka na caly rok przechodzi niezauwazone.
        """
        panel = pd.DataFrame({"club_id": [100], "season": [2015]})
        tr = self.transfers([(100, 200, "2015-01-15", 10e6, "15/16")])
        out = add_net_spend(panel, tr)
        assert out.loc[0, "net_spend_m"] == 0.0

    def test_sam_filtr_roku_wyklucza_transfer_z_wlasciwego_miesiaca(self):
        """Lipiec, ale rok inny niz rok startu sezonu - odrzuca filtr roku."""
        panel = pd.DataFrame({"club_id": [100], "season": [2015]})
        tr = self.transfers([(100, 200, "2016-07-15", 10e6, "15/16")])
        out = add_net_spend(panel, tr)
        assert out.loc[0, "net_spend_m"] == 0.0

    def test_zakup_letni_daje_dodatnie_saldo(self):
        panel = pd.DataFrame({"club_id": [100], "season": [2015]})
        tr = self.transfers([(100, 200, "2015-07-15", 10e6, "15/16")])
        out = add_net_spend(panel, tr)
        assert out.loc[0, "net_spend_m"] == pytest.approx(10.0)

    def test_sprzedaz_daje_ujemne_saldo(self):
        panel = pd.DataFrame({"club_id": [200], "season": [2015]})
        tr = self.transfers([(100, 200, "2015-07-15", 10e6, "15/16")])
        out = add_net_spend(panel, tr)
        assert out.loc[0, "net_spend_m"] == pytest.approx(-10.0)

    def test_brak_transferow_to_zero_a_nie_brak(self):
        """Zero jest poprawne: klub bez transferow nie wydal nic."""
        panel = pd.DataFrame({"club_id": [100], "season": [2015]})
        out = add_net_spend(panel, self.transfers([]))
        assert out.loc[0, "net_spend_m"] == 0.0
        assert out["net_spend_m"].notna().all()


class TestMapowanieNazwKlubow:
    def test_niezmapowana_nazwa_jest_zgloszona_a_nie_przypisana(self):
        panel = pd.DataFrame({"club_id": [11], "season": [2015]})
        wages = pd.DataFrame(
            {
                "League": ["Premier_League", "Premier_League"],
                "Season": ["2015-2016", "2015-2016"],
                "Squad": ["Arsenal", "Klub Nieistniejacy"],
                "Annual Wages": [100_000_000, 1_000],
            }
        )
        out, unmatched = add_wages(panel, wages)
        assert list(unmatched["Squad"]) == ["Klub Nieistniejacy"]
        assert out.loc[0, "wage_bill"] == 100_000_000

    def test_sezon_czytany_z_pierwszych_czterech_znakow(self):
        panel = pd.DataFrame({"club_id": [11], "season": [2015]})
        wages = pd.DataFrame(
            {
                "League": ["Premier_League"], "Season": ["2015-2016"],
                "Squad": ["Arsenal"], "Annual Wages": [100_000_000],
            }
        )
        out, _ = add_wages(panel, wages)
        assert out.loc[0, "log_wage"] == pytest.approx(np.log(100_000_000))

    def test_ta_sama_nazwa_w_innej_lidze_nie_jest_mylona(self):
        """Mapowanie jest zagniezdzone w ligach, wiec kolizje nazw nie szkodza."""
        from src.data_load import WAGE2ID

        wspolne = set(WAGE2ID["Premier_League"]) & set(WAGE2ID["Serie_A"])
        assert not wspolne


class TestPodprobaZbalansowana:
    def test_wybiera_tylko_kluby_obecne_we_wszystkich_sezonach(self):
        panel = pd.DataFrame(
            {
                "club_id": [1, 1, 1, 2, 2, 3],
                "season": [2015, 2016, 2017, 2015, 2016, 2017],
            }
        )
        assert list(balanced_clubs(panel)) == [1]
