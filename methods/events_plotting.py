import numpy as np
import matplotlib.pyplot as plt

def compute_normalized_event_curve(time, events_vs_time, time_unit=None):
    """
    Вычисляет нормированную кривую микросейсмических событий, возможно с агрегацией по времени.

    Параметры:
    ----------
    time : np.ndarray (T,)
        Массив времени в секундах

    events_vs_time : np.ndarray (T,)
        Массив количества микросейсмических событий во времени

    time_unit : str or None
        Единица агрегации времени: 'm', 'h', 'd' — минуты, часы или дни.
        Если None, возвращается нормированная кривая без агрегации.

    Возвращает:
    -----------
    time_out : np.ndarray
        Временные метки (в нужной размерности)

    event_curve : np.ndarray
        Нормированная кривая событий по бинам
    """
    unit_multipliers = {'m': 60, 'h': 3600, 'd': 86400}

    total_events = np.sum(events_vs_time)
    norm_events = events_vs_time / total_events if total_events > 0 else events_vs_time

    if time_unit is None:
        return time, norm_events

    if time_unit not in unit_multipliers:
        raise ValueError("time_unit должен быть одним из: 'm', 'h', 'd' или None")

    scale = unit_multipliers[time_unit]
    time_scaled = time / scale

    bin_edges = np.arange(np.floor(time_scaled.min()), np.ceil(time_scaled.max()) + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    binned_events, _ = np.histogram(time_scaled, bins=bin_edges, weights=norm_events)

    return bin_centers, binned_events


def plot_events_and_pressure(time, events_vs_time, pore_pressure_vs_time, time_unit=None):
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
    plt.show()
    
def plot_events_vs_pressure(pore_pressure_vs_time, events_vs_time):
    """
    Визуализация микросейсмических событий как функции от порового давления с разбиением по бинам.
    Предполагается, что давление уже передано в нужных единицах (обычно МПа), и ширина бина = 1.

    Параметры:
    ----------
    pore_pressure_vs_time : np.ndarray (T,)
        Поровое давление во времени (в нужных единицах, напр. МПа)

    events_vs_time : np.ndarray (T,)
        Количество микросейсмических событий во времени
    """
    pressure = pore_pressure_vs_time
    bin_width = 1.0

    total_events = np.sum(events_vs_time)
    norm_events = events_vs_time / total_events if total_events > 0 else events_vs_time

    # Биннинг с шагом 1
    bin_edges = np.arange(np.floor(pressure.min()), np.ceil(pressure.max()) + bin_width, bin_width)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned_events, _ = np.histogram(pressure, bins=bin_edges, weights=norm_events)

    # Визуализация
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bin_centers, binned_events, width=bin_width, align='center',
           alpha=0.7, color='steelblue', label=f'События (всего: {total_events})')

    ax.set_xlabel("Поровое давление", fontsize=12)
    ax.set_ylabel("Отн. число событий", fontsize=12)
    ax.set_title("Микросейсмические события в зависимости от давления", fontsize=13)
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(pressure.min(), pressure.max())
    plt.tight_layout()
    plt.show()