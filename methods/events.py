import numpy as np
from tqdm.notebook import tqdm

from colomb_things import check_coulomb_failure

def simulate_microseismicity_over_time(sigma_n, tau, mu, cohesion, pore_pressure_in_place):

    """
    Расчёт количества микросейсмических событий во времени
    на основе изменения порового давления и критерия Кулона.

    Для бетча тензоров напряжения.
    """
    
    T = pore_pressure_in_place.shape[0] # сколько шагов по времени
    N = sigma_n.shape[0] # сколько тензоров в батче
    M = mu.shape[0] # количество трещин

    # Храним массив нарушений критерия Кулона
    failures_vs_time = np.zeros((T, N, M), dtype=bool)

    print("Проверка критерия Кулона-Мора по времени...")
    for t in tqdm(range(T), desc="Временные шаги", leave=False):
        failures_vs_time[t] = check_coulomb_failure(sigma_n, tau, mu, cohesion, pore_pressure=pore_pressure_in_place[t])

    # Исключаем трещины, которые уже были нестабильны в начальный момент
    ever_failed = failures_vs_time[0].copy()
    event_counts = np.zeros((N, T), dtype=int)

    for t in range(T):
        new_failures = np.logical_and(failures_vs_time[t], ~ever_failed)
        event_counts[:, t] = np.sum(new_failures, axis=1)
        ever_failed |= new_failures

    return event_counts

