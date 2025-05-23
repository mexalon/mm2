import numpy as np
from tqdm.notebook import tqdm

from methods.colomb_things import check_coulomb_failure

def simulate_microseismicity_over_time(sigma_n, tau, mu, cohesion, pore_pressure_vs_time):

    """
    Расчёт количества микросейсмических событий во времени
    на основе изменения порового давления и критерия Кулона.

    Для бетча тензоров напряжения.
    """   
    T = pore_pressure_vs_time.shape[0] # сколько шагов по времени
    N = sigma_n.shape[0] # сколько тензоров в батче
    M = sigma_n.shape[1] # количество трещин

    # Храним массив нарушений критерия Кулона
    failures_vs_time = np.zeros((T, N, M), dtype=bool)

    print("Проверка критерия Кулона-Мора по времени...")
    for t in tqdm(range(T), desc="Временные шаги", leave=False):
        failures_vs_time[t] = check_coulomb_failure(sigma_n, tau, mu, cohesion, pore_pressure=pore_pressure_vs_time[t])

    # Исключаем трещины, которые уже были нестабильны в начальный момент
    ever_failed = failures_vs_time[0].copy()
    event_counts = np.zeros((N, T), dtype=int)

    for t in range(T):
        new_failures = np.logical_and(failures_vs_time[t], ~ever_failed)
        event_counts[:, t] = np.sum(new_failures, axis=1)
        ever_failed |= new_failures

    return event_counts


def compute_normalized_event_curve(time, events_vs_time, time_unit=None):
    """
    Векторизованная версия: вычисляет нормированную кривую событий
    для одного вектора или батча

    Параметры:
    ----------
    time : np.ndarray (T,)
        Массив времени в секундах

    events_vs_time : np.ndarray (T,) или (N, T)
        Число событий по времени

    time_unit : str or None
        'm','h','d' или None — агрегирование

    Возвращает:
    ---------
    time_out : np.ndarray (B,) или (T,)
    event_curve : np.ndarray (N, B) или (B,) или (T,) или (N, T)
    """
    # Приводим к батчу
    ev = np.atleast_2d(events_vs_time)
    if ev.shape[1] != time.shape[0]:
        ev = ev.T  # если передали как (T,N)

    # Нормировка
    totals = ev.sum(axis=1, keepdims=True)
    norm_ev = np.divide(ev, totals, where=(totals>0))

    # Без агрегации
    if time_unit is None:
        out = norm_ev if ev.shape[0]>1 else norm_ev[0]
        return time, out

    # Шкала
    multipliers = {'m':60,'h':3600,'d':86400}
    if time_unit not in multipliers:
        raise ValueError("time_unit должен быть 'm','h','d' или None")
    ts = time / multipliers[time_unit]

    # Бины
    bin_edges = np.arange(np.floor(ts.min()), np.ceil(ts.max())+1)
    B = len(bin_edges)-1

    # Индексы бинов: 0..B-1
    idx = np.clip(np.digitize(ts, bin_edges)-1, 0, B-1)

    # One-hot матрица (T, B)
    oh = np.eye(B)[idx]

    # Векторизация биннинга: (N,T) @ (T,B) -> (N,B)
    binned = norm_ev @ oh

    # Если одиночный вектор, разворачиваем
    if events_vs_time.ndim == 1:
        return 0.5*(bin_edges[:-1]+bin_edges[1:]), binned[0]
    return 0.5*(bin_edges[:-1]+bin_edges[1:]), binned



