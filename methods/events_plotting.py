import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from methods.events import compute_normalized_event_curve
from typing import Sequence

# ============================================================
#  ВСПОМОГАТЕЛЬНАЯ ПРОЦЕДУРА: добавляет вторую легенду
#  с параметрами моделирования «κ,  r,  N»
# ============================================================
def _add_param_legend(ax, params: dict | None) -> None:
    if not params:
        return
    txt = []
    if "kappa"    in params: txt.append(rf"$\kappa={params['kappa']}$")
    if "r"        in params: txt.append(rf"$r={params['r']}$")
    if "N_events" in params: txt.append(rf"$N={params['N_events']}$")
    if txt:
        legend = ax.legend(handles=[Line2D([], [], color="none")],
                           labels=["   ".join(txt)],
                           loc="upper right", frameon=False,
                           handlelength=0, handletextpad=0,
                           prop={"size": 12})
        ax.add_artist(legend)

# --------------------------------------------------------------------------- #
#   События от времени                                                        #
# --------------------------------------------------------------------------- #    

def plot_events_and_pressure_vs_time(time, events_vs_time, pore_pressure_vs_time, time_unit=None, fname=None):
    """
    Визуализация микросейсмических событий и давления от времени с опцией агрегирования.

    Параметры:
    ----------
    time : np.ndarray (T,)
        Массив времени в секундах

    events_vs_time : np.ndarray (T,)
        Массив количества микросейсмических событий во времени

    pore_pressure_vs_time : np.ndarray (T,)
        Массив давления (МПа) во времени

    time_unit : str or None
        Единица агрегации времени: 'm', 'h', 'd'. Если None — без агрегирования.
    """
    unit_labels = {'s': 'Время, с', 'm': 'Время, мин', 'h': 'Время, ч', 'd': 'Время, дн'}
    time_label = unit_labels.get(time_unit, 'Время, с')

    total_events = np.sum(events_vs_time)
    time_ev, ev_curve = compute_normalized_event_curve(time, events_vs_time, time_unit)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    if time_unit is None:
        ax1.plot(time_ev, ev_curve, color='steelblue', label=f'События (всего: {total_events})', lw=2)
    else:
        ax1.bar(time_ev, ev_curve, width=(time_ev[1] - time_ev[0]), align='center',
                alpha=0.7, color='steelblue', label=f'События (всего: {total_events})')

    ax1.set_ylabel("Отн. число событий", fontsize=12)
    ax1.set_xlabel(time_label, fontsize=12)
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Вторая ось — давление
    scale = {'m': 60, 'h': 3600, 'd': 86400}.get(time_unit, 1)
    time_scaled = time / scale
    ax2 = ax1.twinx()
    ax2.plot(time_scaled, pore_pressure_vs_time, color='firebrick', linewidth=2,
             label="Поровое давление")
    ax2.set_ylabel("Давление, МПа", fontsize=12)
    ax2.tick_params(axis='y', labelcolor='firebrick')

    # Легенды
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10)

    ax1.grid(True, linestyle=':', linewidth=0.5)
    ax1.set_xlim(time_scaled.min(), time_scaled.max())
    plt.tight_layout()
    plt.title("Микросейсмические события и давление во времени", fontsize=13)
    
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# --------------------------------------------------------------------------- #
#   События от давления график со всемипирогами                               #
# --------------------------------------------------------------------------- #    

def plot_events_vs_pressure(pore_pressure_vs_time,
                            events_vs_time,
                            *,
                            bins_count: int | None = None,         
                            cpps: tuple[float, float, float] | None = None,
                            cpps_preds: float | Sequence[float] | None = None,
                            params: dict | None = None,
                            fname: str | None = None) -> None:
    """
    Гистограмма относительного числа событий как функции порового давления
    с отображением истинных и предсказанных критических давлений.

    Если ``bins_count`` не задан (``None``), столбцы строятся непосредственно
    по исходным точкам ``events_vs_time`` без биннинга.

    Parameters
    ----------
    pore_pressure_vs_time : array_like, shape (T,)
        Давление во времени (МПа).

    events_vs_time : array_like, shape (T,)
        Количество событий во времени.

    bins_count : int | None, default None
        Число бинов по оси давления.  Если ``None`` — столбцы строятся
        по каждой исходной точке.

    ... (остальные параметры неизменны)
    """
    # ------------------------------------------------------------------ data
    pressure = np.asarray(pore_pressure_vs_time, dtype=float)
    events   = np.asarray(events_vs_time,        dtype=float)

    total_events = events.sum()
    norm_events  = events / total_events if total_events > 0 else events

    # ------------------------------------------------------------------ prepare bars
    if bins_count is None:                                      # ► без биннинга
        x_vals   = pressure
        y_vals   = norm_events
        width    = - (pressure.max() - pressure.min()) / (pressure.shape[-1]-1)
        align    = "edge"

    else:                                                       # ► c биннингом
        bin_edges = np.linspace(pressure.min(), pressure.max(),
                            bins_count + 1, endpoint=True) + pressure.min()*1e-8 # чтобы первое значение попадало в нулевой бин немного сдвинем
        x_vals    = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        y_vals, _ = np.histogram(pressure, bins=bin_edges, weights=norm_events)
        width     = (pressure.max() - pressure.min()) / bins_count
        align     = "center"

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_vals, y_vals,
           width=width, align=align,
           alpha=0.7, color='steelblue',
           label=f'События (всего: {int(total_events):,d})'.replace(",", " "))

    p_min, p_max = 0.95 * pressure.min(), pressure.max()

    # --- истинные критические давления ------------------------------------
    if cpps is not None:
        cpp_31, cpp_21, cpp_32 = cpps
        true_lines = [
            (cpp_21, r'$P_{21}$ (истина)', 'green'),
            (cpp_32, r'$P_{32}$ (истина)', 'orange'),
            (cpp_31, r'$P_{31}$ (истина)', 'navy')
        ]
        for p, lab, col in true_lines:
            if p_min <= p <= p_max:
                ax.axvline(p, color=col, linestyle='-', lw=1.4, label=lab)

    # --- предсказанные давления -------------------------------------------
    if cpps_preds is not None:
        preds = np.atleast_1d(cpps_preds)
        label_used = False
        for ii, p in enumerate(preds):
            if p_min <= p <= p_max:
                lab = "модель" if not label_used else ""
                label_used = True
                ax.axvline(p, color='crimson', linestyle='--', lw=2*(1-0.2*ii), label=lab) # разная немного толщина у главного и

    # ------------------------------------------------------------------ axes
    ax.set_xlabel("Поровое давление, МПа", fontsize=12)
    ax.set_ylabel("Отн. число событий",    fontsize=12)
    ax.set_xlim(p_min, p_max)
    ax.grid(True, linestyle=':', linewidth=0.5)

    # ----------------------- первая легенда -------------------------------
    main_legend = ax.legend(fontsize=10, loc='upper left')
    ax.add_artist(main_legend)

    # ----------------------- вторая легенда – параметры --------------------
    _add_param_legend(ax, params)

    plt.tight_layout()

    # ---------------------------------------------------------------- save/show
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# ============================================================
#  1. Гистограмма предсказанных / истинных критических давлений
# ============================================================
def plot_peak_histograms(preds_aligned: np.ndarray,
                         true_vals: tuple[float, float],
                         *,
                         bins: int = 40,
                         params: dict | None = None,
                         fname: str | None = None) -> None:
    """
    Две гистограммы предсказанных pиков P21-/P32-и вертикальные линии истин.

    preds_aligned : (..., 2) – столбец 0 ⇒ P21 (model), столбец 1 ⇒ P32 (model)
    """
    P21_true, P32_true = map(float, true_vals)

    P21_pred = preds_aligned[:, 0][~np.isnan(preds_aligned[:, 0])]
    P32_pred = preds_aligned[:, 1][~np.isnan(preds_aligned[:, 1])]

    p_min = np.nanmin([P21_pred.min() if P21_pred.size else P21_true,
                       P32_pred.min() if P32_pred.size else P32_true,
                       P21_true, P32_true])
    p_max = np.nanmax([P21_pred.max() if P21_pred.size else P21_true,
                       P32_pred.max() if P32_pred.size else P32_true,
                       P21_true, P32_true])

    fig, ax = plt.subplots(figsize=(7, 4))

    if P21_pred.size:
        ax.hist(P21_pred, bins=bins, range=(p_min, p_max),
                color="green", alpha=0.4, label=r"$P_{21}$ (модель)")
    if P32_pred.size:
        ax.hist(P32_pred, bins=bins, range=(p_min, p_max),
                color="orange", alpha=0.4, label=r"$P_{32}$ (модель)")

    ax.axvline(P21_true, color="green",  lw=1.5, label=r"$P_{21}$ (истина)")
    ax.axvline(P32_true, color="orange", lw=1.5, label=r"$P_{32}$ (истина)")

    ax.set_xlabel("Поровое давление, МПа", fontsize=12)
    ax.set_ylabel("Частота", fontsize=12)
    ax.set_xlim(p_min, p_max)
    ax.grid(True, ls=":", lw=0.5)

    main_leg = ax.legend(fontsize=10, loc="upper left")
    ax.add_artist(main_leg)
    _add_param_legend(ax, params)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# ============================================================
#  2. Гистограмма коэффициента Лоде-Надаи r
# ============================================================
def plot_r_histograms(r_pred,
                      r_true: float,
                      *,
                      bins: int = 40,
                      params: dict | None = None,
                      fname: str | None = None) -> None:
    """Гистограмма предсказанных r и линия истинного r."""
    r_pred = np.asarray(r_pred, dtype=float).ravel()
    r_pred = r_pred[~np.isnan(r_pred)]
    if r_pred.size == 0:
        print("plot_r_histograms: нет валидных r_pred"); return

    fig, ax = plt.subplots(figsize=(6, 4))
    if r_pred.size >= 2:
        ax.hist(r_pred, bins=bins, range=(0, 1),
                color="slateblue", alpha=0.5, label="r (модель)")
    else:
        ax.axvline(r_pred[0], color="slateblue", ls="--", lw=2,
                   alpha=0.6, label="r (модель)")

    ax.axvline(r_true, color="slateblue", lw=1.5, label="r (истина)")

    ax.set_xlabel(r"Коэффициент Лоде-Надаи $r$", fontsize=12)
    ax.set_ylabel("Частота", fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(True, ls=":", lw=0.5)

    main_leg = ax.legend(fontsize=10, loc="upper left")
    ax.add_artist(main_leg)
    _add_param_legend(ax, params)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# ============================================================
#  3. Гистограмма максимального горизонтального напряжения σ₂
# ============================================================
def plot_sigma2_histograms(s2_pred,
                           s2_true: float,
                           *,
                           bins: int = 40,
                           params: dict | None = None,
                           fname: str | None = None) -> None:
    """Гистограмма предсказанных σ₂ и линия истинного σ₂."""
    s2_pred = np.asarray(s2_pred, dtype=float).ravel()
    s2_pred = s2_pred[~np.isnan(s2_pred)]
    if s2_pred.size == 0:
        print("plot_sigma2_histograms: нет предсказанных σ₂."); return

    p_min = min(s2_pred.min(), s2_true) * 0.95
    p_max = max(s2_pred.max(), s2_true) * 1.05

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s2_pred, bins=bins, range=(p_min, p_max),
            color="firebrick", alpha=0.5, label=r"$\sigma_2$ (модель)")
    ax.axvline(s2_true, color="firebrick", lw=1.5,
               label=r"$\sigma_2$ (истина)")

    ax.set_xlabel(r"$\sigma_2$, МПа", fontsize=12)
    ax.set_ylabel("Частота",      fontsize=12)
    ax.set_xlim(p_min, p_max)
    ax.grid(True, ls=":", lw=0.5)

    main_leg = ax.legend(fontsize=10, loc="upper left")
    ax.add_artist(main_leg)
    _add_param_legend(ax, params)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------- #
# Отрисовка сводных графиков с ошибками восстанволения                        #
# давлений, Лоде-Надаи и sH                                                   #
# --------------------------------------------------------------------------- #        

# --------------------------------------------------------------------------- #
# 1. Ошибки критических давлений P21 / P32                                    #
# --------------------------------------------------------------------------- #
def plot_cpp_err_single_kappa(df_kappa, *, fname=None):
    """
    df_kappa  ─ DataFrame с колонками
        r, N_events, cpp_21_err, cpp_32_err, kappa
    Для фиксированного κ рисуются:
        • сплошная линия  –  P21-ошибка
        • пунктир         –  P32-ошибка
    Цвет ⇒ различное r.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap("tab10")

    for i, (r_val, df_r) in enumerate(df_kappa.groupby("r")):
        df_r = df_r.sort_values("N_events")
        col  = cmap(i % 10)

        ax.plot(df_r["N_events"], df_r["cpp_21_err"],
                color=col, lw=2, marker="o",
                label=rf"$r={r_val}$ – $P_{{21}}$")
        ax.plot(df_r["N_events"], df_r["cpp_32_err"],
                color=col, lw=2, ls="--", marker="s",
                label=rf"$r={r_val}$ – $P_{{32}}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"$N_{\mathrm{events}}$", fontsize=12)
    ax.set_ylabel("Ошибка, МПа", fontsize=12)
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(fontsize=9, ncols=2)
    ax.set_title(rf"$\kappa = {df_kappa['kappa'].iloc[0]}$")

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------- #
# 2. Ошибка коэффициента Лоде–Надаи  r                                        #
# --------------------------------------------------------------------------- #
def plot_r_err_single_kappa(df_kappa, *, fname=None):
    """df_kappa: r, N_events, r_err, kappa"""
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap("tab10")

    for i, (r_val, df_r) in enumerate(df_kappa.groupby("r")):
        df_r = df_r.sort_values("N_events")
        ax.plot(df_r["N_events"], df_r["r_err"],
                color=cmap(i % 10), lw=2, marker="o",
                label=rf"$r={r_val}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"$N_{\mathrm{events}}$", fontsize=12)
    ax.set_ylabel("Ошибка  $r$", fontsize=12)
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(fontsize=9)
    ax.set_title(rf"$\kappa = {df_kappa['kappa'].iloc[0]}$")

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------- #
# 3. Ошибка максимального горизонтального напряжения  σ₂                      #
# --------------------------------------------------------------------------- #
def plot_sigma2_err_single_kappa(df_kappa, *, fname=None):
    """df_kappa: r, N_events, s2_pred_err, kappa"""
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap("tab10")

    for i, (r_val, df_r) in enumerate(df_kappa.groupby("r")):
        df_r = df_r.sort_values("N_events")
        ax.plot(df_r["N_events"], df_r["s2_pred_err"],
                color=cmap(i % 10), lw=2, marker="o",
                label=rf"$r={r_val}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"$N_{\mathrm{events}}$", fontsize=12)
    ax.set_ylabel(r"Ошибка  $\sigma_2$, МПа", fontsize=12)
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(fontsize=9)
    ax.set_title(rf"$\kappa = {df_kappa['kappa'].iloc[0]}$")

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()