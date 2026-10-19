"""Wykresy i tabele publikacyjne. Funkcje bezstanowe, katalog docelowy globalny.

Barwy pochodza wylacznie z tokenow modulu `plot_style` i sa wybierane przez
role, jaka seria pelni w danych: cynober to przedmiot uwagi, blekit to
odniesienie. Kwalifikacja do Ligi Mistrzow jest przedmiotem badania, wiec
sezony z awansem nosza ACCENT na tle MUTED, a krzywa dla klubu bez awansu
w poprzednim sezonie jest odniesieniem. Macierz przejsc niesie wielkosc,
nie tozsamosc, wiec dostaje rampe sekwencyjna.
"""

import numpy as np
import pandas as pd

from src import plot_style as style
from src.data_load import PROJECT_ROOT
from src.inference import LABELS, ame
from src.model import LEAGUE_LABELS, LEAGUE_ORDER

FIGURES = PROJECT_ROOT / "reports" / "figures"
TABLES = PROJECT_ROOT / "reports" / "tables"

# Zakres siatki predykcji: kwantyle 5 i 95 procent, zeby nie ekstrapolowac
# poza obserwowane fundusze plac.
GRID_QUANTILES = (0.05, 0.95)
GRID_POINTS = 60

# Sezon i liga ustalone dla wykresu predykcji - efekty stale trzeba na czyms
# oprzec, a wybor nie zmienia ksztaltu zaleznosci, tylko poziom.
PRED_SEASON = 2019
PRED_LEAGUE = "Premier_League"


def setup_style() -> None:
    """Ustawia system wizualny projektu. Wywolac raz, przed pierwszym rysunkiem."""
    style.apply_style()


def _finish(fig, name: str | None, show: bool):
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if name:
        FIGURES.mkdir(parents=True, exist_ok=True)
        path = FIGURES / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        print("Zapisano wykres:", path.name)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_transition_matrix(tab: pd.DataFrame, name="macierz_przejsc", show=True):
    """Empiryczna macierz przejsc: jak trwala jest kwalifikacja do LM."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=style.SQUARE)
    im = ax.imshow(tab.values, cmap=style.SEQUENTIAL_CMAP, vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["poza top-4 (t)", "top-4 (t)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["poza top-4 (t–1)", "top-4 (t–1)"])
    ax.grid(visible=False)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            ax.text(
                j, i, f"{v:.0%}",
                ha="center", va="center", fontsize=13,
                color=style.BACKGROUND if v > 0.5 else style.TEXT,
            )
    bar = fig.colorbar(im, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=9, colors=style.TEXT_SECONDARY)
    style.decimal_comma(bar.ax, axis="y", places=1)
    return _finish(fig, name, show)


def plot_predicted_probability(res, est, name="prognoza_plac_stan", show=True):
    """Przewidywane P(awans) wzgledem funduszu plac, osobno dla Y_lag = 0 i 1.

    Odstep miedzy krzywymi to zaleznosc stanu: ile daje sam fakt awansu
    w poprzednim sezonie, przy tym samym poziomie plac.
    """
    import matplotlib.pyplot as plt

    lo, hi = GRID_QUANTILES
    grid = np.linspace(
        est["mean_log_wage"].quantile(lo),
        est["mean_log_wage"].quantile(hi),
        GRID_POINTS,
    )

    def predict(y_lag):
        frame = pd.DataFrame(
            {
                "Y_lag": y_lag,
                "log_wage": grid,
                "mean_log_wage": grid,
                "Y_0": 0,
                "season": PRED_SEASON,
                "league": PRED_LEAGUE,
            }
        )
        return res.predict(frame)

    fig, ax = plt.subplots(figsize=style.STANDARD)
    ax.plot(grid, predict(0), color=style.REFERENCE,
            label="Y_lag = 0 (brak awansu w t–1)")
    ax.plot(grid, predict(1), color=style.ACCENT,
            label="Y_lag = 1 (awans w t–1)")
    ax.set_xlabel("Średni (log) fundusz płac klubu — mean_log_wage")
    ax.set_ylabel("Przewidywane P(awans do Ligi Mistrzów)")
    style.decimal_comma(ax, axis="xy", places=1)
    ax.legend()
    return _finish(fig, name, show)


def plot_wage_vs_rank(panel: pd.DataFrame, name="place_a_pozycja", show=True):
    """Fundusz plac a pozycja koncowa, panel 2x2 po ligach.

    Os pozycji odwrocona raz dla calej siatki, bo panele dziela te os przez
    `sharey` i odwrocenie w petli zniosloby samo siebie. Sezony z awansem
    niosa akcent, pozostale sa masa bez tozsamosci.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 2, figsize=style.SQUARE, sharey=True)
    for ax, lg in zip(axes.flat, LEAGUE_ORDER):
        s = panel[panel["league"] == lg].dropna(subset=["log_wage", "rank"])
        colours = [style.ACCENT if v == 1 else style.MUTED for v in s["Y"]]
        ax.scatter(s["log_wage"], s["rank"], c=colours, s=26,
                   edgecolor=style.BACKGROUND, linewidth=0.3)
        ax.set_title(LEAGUE_LABELS[lg])
        ax.set_xlabel("log(fundusz płac)")
        ax.set_ylabel("Pozycja końcowa")
        # Pozycja w tabeli jest liczba calkowita, wiec os nie dzieli miejsc.
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        style.decimal_comma(ax, axis="x", places=1)

    axes.flat[0].invert_yaxis()
    # W legendzie stoi sam akcent: masa bez tozsamosci nie jest kategoria.
    fig.legend(
        handles=[Line2D([], [], marker="o", linestyle="none",
                        color=style.ACCENT,
                        label="sezon z awansem do Ligi Mistrzów")],
        loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02),
    )
    return _finish(fig, name, show)


# --- Tabele publikacyjne -------------------------------------------------

CSS = """<style>
table.sci { border-collapse: collapse; margin: 8px 0 4px 0; font-family:'Times New Roman',Times,serif; font-size:10pt; color:#000; }
table.sci th, table.sci td { padding:4px 8px; text-align:center; border:none; }
table.sci th { border-top:2px solid #000; border-bottom:1px solid #000; font-weight:bold; }
table.sci tr:last-child td { border-bottom:2px solid #000; }
table.sci td:first-child, table.sci th:first-child { text-align:left; }
.sci .sub td { padding-top:0; padding-bottom:8px; color:#333; }
.sci .div td { border-top:1px solid #000; font-weight:bold; text-align:left; padding-top:10px; padding-bottom:5px; }
.caption { font-family:'Times New Roman',Times,serif; font-weight:bold; font-size:11pt; margin-bottom:10px; }
.note { font-family:'Times New Roman',Times,serif; font-size:9pt; max-width:900px; margin:5px 0 20px 0; text-align:left; }
</style>"""

SOURCE_NOTE = "<em>Źródło:</em> Opracowanie własne."

# Nota powtarzana pod kazda tabela - jedno miejsce, jedna tresc.
CLUSTER_NOTE = (
    "W nawiasach błędy standardowe klastrowane na poziomie klubu. "
    "Istotność: *** p&lt;0,01, ** p&lt;0,05, * p&lt;0,1. Liga bazowa: Bundesliga."
)


def stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def fmt(x: float) -> str:
    """Polski separator dziesietny i minus typograficzny."""
    return f"{x:.3f}".replace("-", "−").replace(".", ",")


def save_table(html: str, fname: str, show: bool = True) -> str:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / fname
    path.write_text(
        "<!DOCTYPE html><html lang='pl'><head><meta charset='UTF-8'></head><body>"
        + html
        + "</body></html>",
        encoding="utf-8",
    )
    print("Zapisano tabelę:", path.name)
    if show:
        try:
            from IPython.display import HTML, display

            display(HTML(html))
        except ImportError:
            pass
    return html


def _cell(res, key: str) -> tuple[str, str]:
    """Para (wspolczynnik z gwiazdkami, blad standardowy w nawiasie)."""
    if key in res.params.index:
        return f"{fmt(res.params[key])}{stars(res.pvalues[key])}", f"({fmt(res.bse[key])})"
    return "—", ""


def coef_ame_table(title, res, keys, stats, note, label_map=None) -> str:
    """Tabela jednego modelu: wspolczynnik i efekt czastkowy obok siebie."""
    labels = label_map or LABELS
    frame = ame(res)
    body = ""
    for key in keys:
        main, sub = _cell(res, key)
        if key in frame.index:
            a, ase, ap = frame.loc[key, ["AME", "se", "p"]]
            ame_main, ame_sub = f"{fmt(a)}{stars(ap)}", f"({fmt(ase)})"
        else:
            ame_main, ame_sub = "—", ""
        body += (
            f"<tr class='main'><td><strong>{labels.get(key, key)}</strong></td>"
            f"<td>{main}</td><td>{ame_main}</td></tr>"
            f"<tr class='sub'><td></td><td>{sub}</td><td>{ame_sub}</td></tr>"
        )
    body += "<tr class='div'><td colspan='3'>Statystyki i diagnostyka modelu</td></tr>"
    for k, v in stats:
        body += f"<tr><td>{k}</td><td colspan='2'>{v}</td></tr>"
    return (
        CSS + f"<div class='caption'>{title}</div><table class='sci'>"
        "<tr><th>Zmienna</th><th>Współczynnik</th><th>Efekt cząstkowy (AME)</th></tr>"
        + body
        + "</table>"
        + f"<div class='note'>{SOURCE_NOTE}<br>{note}</div>"
    )


def multi_model_table(title, models, keys, stat_rows, note, label_map=None) -> str:
    """Tabela wielu modeli: kolumna na model, wiersz na zmienna.

    `models` to lista (etykieta, wynik). Wiersz pomijany, jesli zaden model
    nie zawiera danej zmiennej.
    """
    labels = label_map or LABELS
    names = [lab for lab, _ in models]
    ncol = len(names) + 1
    head = "<tr><th>Zmienna</th>" + "".join(f"<th>{n}</th>" for n in names) + "</tr>"
    body = ""
    for key in keys:
        if not any(key in res.params.index for _, res in models):
            continue
        mains = "".join(f"<td>{_cell(res, key)[0]}</td>" for _, res in models)
        subs = "".join(f"<td>{_cell(res, key)[1]}</td>" for _, res in models)
        body += f"<tr class='main'><td><strong>{labels.get(key, key)}</strong></td>{mains}</tr>"
        body += f"<tr class='sub'><td></td>{subs}</tr>"
    body += f"<tr class='div'><td colspan='{ncol}'>Statystyki i diagnostyka modelu</td></tr>"
    for lab, vals in stat_rows:
        body += f"<tr><td>{lab}</td>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>"
    return (
        CSS
        + f"<div class='caption'>{title}</div><table class='sci'>"
        + head
        + body
        + "</table>"
        + f"<div class='note'>{SOURCE_NOTE}<br>{note}</div>"
    )
