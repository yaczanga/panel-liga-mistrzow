# -*- coding: utf-8 -*-
"""Przebieg wszystkich specyfikacji na gotowym panelu; wynik do reports/model_results.txt.

Skrypt nie buduje panelu od danych surowych - czyta zapisany
`data/processed/panel_klub_sezon.csv`. To jest jego sens: powtorzenie modeli
bez pobierania 65 MB plikow Transfermarkt. Panel buduje
notebooks/02_panel_cre_probit.ipynb.

Uruchomienie z korzenia projektu:

    python scripts/run_models.py
"""

import io
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data_load import PROJECT_ROOT
from src.inference import (
    ame,
    discrete_effect,
    exogeneity_test,
    league_difference_test,
    link_test,
    mundlak_test,
    transition_matrix,
    wage_decomposition,
)
from src.model import (
    LEAGUE_ORDER,
    fit_extensions,
    fit_interaction,
    fit_main,
    fit_per_league,
    fit_static,
)
from src.pre_processing import (
    QUAL_SPOTS,
    balanced_clubs,
    describe_panel,
    load_panel,
    to_estimation_frame,
)

OUTPUT = PROJECT_ROOT / "reports" / "model_results.txt"


def main() -> None:
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 40)

    panel = load_panel()
    est = to_estimation_frame(panel)

    print(f"Panel: {len(panel)} obserwacji")
    print(f"Proba estymacyjna: {len(est)} obs., {est['club_id'].nunique()} klubow, 4 ligi")
    print(describe_panel(est).to_string())

    print("\n=== Empiryczna macierz przejsc stanu kwalifikacji ===")
    print(transition_matrix(panel).round(3).to_string())

    m_cre = fit_main(est)
    print("\n=== Model glowny: dynamiczny probit CRE ===")
    print(m_cre.summary())
    print(f"Pseudo R2 = {m_cre.prsquared:.3f}; LLR p-value = {m_cre.llr_pvalue:.2e}")

    m_static = fit_static(est)
    print("\n=== Punkt odniesienia: statyczny probit (bez Y_lag) ===")
    print(m_static.summary())
    print(
        f"log_wage: statyczny {m_static.params['log_wage']:+.3f} "
        f"(p={m_static.pvalues['log_wage']:.3f}) | "
        f"CRE {m_cre.params['log_wage']:+.3f} (p={m_cre.pvalues['log_wage']:.3f})"
    )

    print("\n=== Efekty czastkowe (AME) ===")
    sf = ame(m_cre)
    show = [n for n in ["Y_lag", "log_wage", "Y_0", "mean_log_wage"] if n in sf.index]
    print(sf.loc[show, ["AME", "se", "p"]].round(3).to_string())
    print(f"APE Y_lag (roznica dyskretna): {discrete_effect(m_cre, est, 'Y_lag'):+.3f}")

    dec = wage_decomposition(m_cre)
    print("\n=== Dekompozycja Mundlaka efektu plac ===")
    print(f"   within  (log_wage):      {dec['within']:+.3f}")
    print(f"   between (mean_log_wage): {dec['between']:+.3f}")
    print(f"   trwaly  (within+between):{dec['trwaly']:+.3f}"
          f" -> +10% plac = {dec['trwaly_pp_na_10pct']:+.1f} pp")

    chi2_m, p_m = mundlak_test(m_cre)
    print("\n=== Testy poprawnosci specyfikacji ===")
    print(f"   Laczny test Mundlaka: chi2={chi2_m:.2f}, df=2, p={p_m:.2e}")
    print(f"   Link test (Pregibon): p={link_test(m_cre, est):.3f}")

    ext = fit_extensions(est, panel, balanced_clubs(panel))
    print(f"   Lead log_wage(t+1):  p={exogeneity_test(ext['wage_lead'], 'log_wage_lead'):.3f}")
    print(f"   Lead net_spend(t+1): p={exogeneity_test(ext['net_spend_lead'], 'net_spend_lead'):.3f}")

    print("\n=== Rozszerzenia i odpornosc ===")
    for label, key in [
        ("+ zmiana trenera", "mgr"),
        ("+ saldo transferowe", "net_spend"),
        ("Wartosc rynkowa kadry", "market_value"),
        ("Podproba zbalansowana", "balanced"),
    ]:
        if key not in ext:
            continue
        res = ext[key]
        print(
            f"   {label:24s} N={int(res.nobs):4d} "
            f"Y_lag={res.params['Y_lag']:+.3f} (p={res.pvalues['Y_lag']:.3f})"
        )

    e_prom = est.dropna(subset=["promoted"])
    prom_top = int(((e_prom["promoted"] == 1) & (e_prom["Y"] == 1)).sum())
    n_prom = int((e_prom["promoted"] == 1).sum())
    print(f"   Beniaminkow w top-{QUAL_SPOTS}: {prom_top} z {n_prom}"
          + (" [SEPARACJA - wspolczynnika nie da sie oszacowac]" if prom_top == 0 else ""))

    print("\n=== Rozbicie per liga ===")
    per = fit_per_league(est)
    rows = []
    for lg in LEAGUE_ORDER:
        if lg not in per:
            continue
        m, sub = per[lg], est[est["league"] == lg]
        rows.append({
            "liga": lg, "obs": len(sub), "klubow": sub["club_id"].nunique(),
            "Y_lag": m.params["Y_lag"], "p(Y_lag)": m.pvalues["Y_lag"],
            "mean_log_wage": m.params.get("mean_log_wage"), "Y_0": m.params.get("Y_0"),
        })
    print(pd.DataFrame(rows).set_index("liga").round(3).to_string())

    chi2_w, p_w, terms = league_difference_test(fit_interaction(est))
    print(f"\n   Laczny test Walda (H0: brak roznic Y_lag miedzy ligami): "
          f"chi2={chi2_w:.2f}, df={len(terms)}, p={p_w:.3f}")

    print("\nDONE")


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    original = sys.stdout
    sys.stdout = io.open(OUTPUT, "w", encoding="utf-8")
    try:
        main()
    finally:
        sys.stdout.close()
        sys.stdout = original
    print("Zapisano:", OUTPUT)
