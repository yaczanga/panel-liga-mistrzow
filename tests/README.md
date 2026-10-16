# Testy

Testy sprawdzają **gwarancje metodologiczne**, bo na nich stoją wnioski projektu.
Nie weryfikują konkretnych oszacowań - to byłoby testowanie `statsmodels`.

Wszystkie działają na małej próbie syntetycznej, bez danych źródłowych, więc
wykonują się w sekundy i przechodzą w CI, gdzie 65 MB plików Transfermarkt nie
istnieje.

## Uruchomienie

```bash
pip install -r requirements-dev.txt
pytest
```

## Co jest sprawdzane

`test_pre_processing.py` - budowa panelu:

- **zmienna celu w obrębie ligi i sezonu** - przy rankingu globalnym kluby
  słabszej ligi nigdy nie trafiłyby do pierwszej czwórki;
- **brak wycieku przyszłych wycen** - wartość kadry na start sezonu bierze
  wyłącznie wyceny nie późniejsze niż 1 lipca, czyli przed rundą transferową,
  która jest już reakcją na poprzedni sezon;
- **brak przecieku między klubami** w `Y_lag`, `Y_0` i lead-ach;
- **warunek początkowy z pierwszego, nie ostatniego sezonu** klubu, niezależnie
  od kolejności wierszy w pliku;
- **średnie Mundlaka w obrębie klubu**, nie na całej ramce;
- **cenzura lewostronna beniaminka** - pierwszy sezon panelu dostaje brak, a nie
  zero, bo nie wiadomo, czy klub był wcześniej w lidze;
- **okienko transferowe** - oba filtry (miesiąc i rok startu sezonu) sprawdzane
  osobno, bo każdy z nich sam wyklucza inny zbiór transferów;
- **niezmapowane nazwy klubów są zgłaszane**, a nie cicho pomijane.

`test_inference.py` - wnioskowanie na panelu syntetycznym (generator w
`conftest.py`):

- efekty cząstkowe nie są współczynnikami i mieszczą się w przedziale
  prawdopodobieństw;
- różnica dyskretna dla zmiennej zero-jedynkowej zgadza się z AME;
- dekompozycja Mundlaka sumuje się tak, jak twierdzi README;
- opakowania testów Mundlaka i Walda dają to samo, co testy policzone wprost;
- specyfikacje zawierają dokładnie to, co obiecuje ich opis (model statyczny
  bez `Y_lag`, model per liga bez efektu stałego ligi, błędy klastrowane).

### Czego testy świadomie nie sprawdzają

Nie ma testów typu „test Mundlaka odrzuci H0" ani „test Walda jej nie odrzuci".
Czy hipoteza zostanie odrzucona na konkretnym losowaniu, zależy od mocy testu
i ziarna generatora, a nie od poprawności kodu - test prawdziwej hipotezy
zerowej zapala w około pięciu procentach prób. Takie oczekiwania dawałyby suitę
chwiejną, która psuje się przy zmianie ziarna i nie mówi nic o kodzie.
Sprawdzane jest więc, że opakowanie nakłada właściwe restrykcje i zwraca to, co
test policzony wprost.

## Weryfikacja mutacyjna

Test, który zawsze przechodzi, nie jest testem. Suita została sprawdzona przez
celowe uszkodzenie szesnastu gwarancji na kopii kodu - m.in. ranking globalny
zamiast w obrębie ligi, wycena kadry bez odcięcia na 1 lipca, opóźnienie i lead
bez podziału na kluby, warunek początkowy z ostatniego sezonu, średnie Mundlaka
liczone globalnie, beniaminek bez rozróżnienia ligi, okienko transferowe
rozszerzone na cały rok, błędy standardowe bez klastrowania, efekt trwały bez
części between, macierz przejść bez normalizacji.

Piętnaście uszkodzeń zapaliło właściwy test od razu. Szesnaste - rozszerzenie
okienka transferowego na cały rok - **przeszło niezauważone**, bo jedyny test
zimowego transferu wykluczał go drugim warunkiem (rokiem), nie miesiącem. Stąd
dwa dodatkowe testy izolujące każdy filtr osobno.

Weryfikację warto powtórzyć po każdej zmianie w `src/pre_processing.py` lub
`src/inference.py`.

## Kompletność zależności testowych

`requirements-dev.txt` jest węższy od `requirements.txt`, więc trzeba sprawdzać,
czy nie brakuje w nim czegoś, co testy mimo wszystko wołają:

```bash
python tests/sprawdz_zaleznosci.py
```

Skrypt blokuje import bibliotek, których `pip install -r requirements-dev.txt`
nie zainstaluje, i uruchamia testy.
