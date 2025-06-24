import itertools
import os
import gc
import pickle
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

# наше добро
from methods.frac_classes import FisherFractureSeed
from methods.frac_plotting import plot_fracture_ensemble_on_sphere, plot_fracture_normals_and_planes, plot_fracture_density, plot_fracture_strike_rose, plot_mu_cohesion_histograms
from methods.stress_classes import FrictionalStressTensor
# from methods.stress_plotting import plot_stress_tensor_with_rotated_ensemble 
from methods.colomb_things import compute_normal_and_shear_ensemble, get_critical_strike_dip, critical_pore_pressure
from methods.events import simulate_microseismicity_over_time, compute_normalized_event_curve
from methods.events_plotting import plot_events_and_pressure_vs_time, plot_events_vs_pressure
from methods.colomb_things_plotting import plot_coulomb_diagram, plot_coulomb_diagram_density


def generate_combinations(params):
    ''' Генерирует все комбинации параметров в словаре парамс '''
    # Преобразуем значения словаря в списки, если они не являются списками
    params_lists = {k: v if isinstance(v, list) else [v] for k, v in params.items()}
    
    # Получаем все возможные комбинации параметров
    keys = params_lists.keys()
    values = params_lists.values()
    combinations = itertools.product(*values)
    
    # Создаем список словарей с комбинациями параметров
    result = [dict(zip(keys, combination)) for combination in combinations]
    
    return result


def save_simulation_results(save_path, params,
                            pore_press_list, events_vs_pore_press,
                            cpps, ps, strike, dip, cstrike, cdip,
                            sigma_n, tau, normals_all, strikes_all, seed):
    """
    Сохраняет результаты моделирования в папке, имя которой включает основные параметры.

    save_path: базовый путь для сохранения (str)
    params: словарь параметров моделирования
    остальные аргументы: полученные массивы и значения
    """
    # Формируем читаемое имя подпапки по параметрам
    parts = [f"{k}={params[k]}" for k in ['depth', 'mu', 'cohesion', 'kappa', 'r']]
    folder = os.path.join(save_path, "__".join(parts) + "__" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)

    # Сохраняем изображения
    plot_fracture_ensemble_on_sphere(normals_all, seed=seed,
                                     fname=os.path.join(folder, "fracture_ensemble_on_sphere.png"))
    plot_fracture_normals_and_planes(normals_all, seed=seed,
                                     fname=os.path.join(folder, "normals_planes.png"))
    plot_fracture_density(normals_all,
                          fname=os.path.join(folder, "fracture_density.png"))
    plot_fracture_strike_rose(strikes_all,
                              fname=os.path.join(folder, "fracture_strike_rose.png"))
    plot_coulomb_diagram(sigma_n, tau,
                         params['mu'], params['cohesion'],
                         pore_pressure=min(cpps[1], cpps[2]),
                         principal_stresses=ps,
                         fname=os.path.join(folder, "coulomb.png"))
    plot_coulomb_diagram_density(sigma_n, tau,
                                 params['mu'], params['cohesion'],
                                 pore_pressure=min(cpps[1], cpps[2]),
                                 principal_stresses=ps,
                                 fname=os.path.join(folder, "coulomb_density.png"))
    
    plot_events_vs_pressure(pore_press_list, events_vs_pore_press, cpps=cpps, bins_count=len(pore_press_list)-1, fname=os.path.join(folder, "events_vs_pressure.png"))

    # CSV с параметрами и ключевыми величинами
    df_params = pd.DataFrame([{**params,
                               'sigma1': ps[0], 'sigma2': ps[1], 'sigma3': ps[2],
                               'strike': strike, 'dip': dip,
                               'cstrike': cstrike, 'cdip': cdip,
                               'cpp_31': cpps[0], 'cpp_21': cpps[1], 'cpp_32': cpps[2]}])
    df_params.to_csv(os.path.join(folder, "params.csv"), index=False, sep=';', decimal=',')

    # CSV с событиями vs давлением
    pd.DataFrame({
        'pore_pressure': pore_press_list,
        'events': events_vs_pore_press
    }).to_csv(os.path.join(folder, "events_vs_pressure.csv"), index=False, sep=';', decimal=',')

    # CSV с sigma_n и tau
    pd.DataFrame({'sigma_n': sigma_n, 'tau': tau}).to_csv(os.path.join(folder, "stress_norm_tau_components.csv"), index=False, sep=';', decimal=',')

    plt.close('all')
    
    

def calculate_events_vs_pore_pressure(params, press_steps=100, save_path=None):
    ''' 
    Считает кривую событий от давления для одного набора параметров
    '''
    sv = params['ro_s'] * 9.81 * params['depth'] * 1e-6  # вертикальное напряжение (литостатика), МПа 
    ini_pore_press = params['ro_f'] * 9.81 * params['depth'] * 1e-6  # начальное поровое давление глубине, МПа

    # Шаг 1: Создаём объект материнского тензора напряжений, 
    # из условия фрикционного равновесия для данного порового давленияб
    # максимальная главная компонента - вертикальная, минмальная - по x
    tensor = FrictionalStressTensor(s1=sv, mode='zyx', ratio=params['r'], 
                                    pore_pressure=ini_pore_press, 
                                    mu=params['mu'], cohesion=params['cohesion'], 
                                    trend=0, plunge=0, rake=0)

    # Шаг 2: Готовим лист значений порового давления
    ps = tensor.principal_stresses # получившиеся главные компоненты
    s3 = np.min(ps) # минимальное напряжение
    cpp_31, cpp_21, cpp_32 = critical_pore_pressure(ps, params['mu'], params['cohesion']) 
    cpps = (cpp_31, cpp_21, cpp_32) # Критические давления касания кругов Мора
    
    pore_press_list = np.linspace(ini_pore_press, s3, int(press_steps)) # давление меняется от начального до занчения минимального напряжения

    # Шаг 3: Создаём объект материнской трещины и сэмплируем ансамбль терщин
    # находим углы наибольшего сдвига (обычный и сопряженной)
    ((strike, dip), (cstrike, cdip)) = get_critical_strike_dip(tensor, params['mu']) 

    # объектs материнской трещины (обычный и сопряженной)
    seed = FisherFractureSeed(strike, dip, mu=params['mu'], cohesion=params['cohesion'])
    cseed = FisherFractureSeed(cstrike, cdip, mu=params['mu'], cohesion=params['cohesion'])

    # ансамбли нормалей
    N_events=int(params['N_frac'])
    normals, strikes, _, _, _ = seed.generate_ensemble(N=N_events//2, kappa=params['kappa'])
    cnormals, cstrikes, _, _, _ = cseed.generate_ensemble(N=N_events//2, kappa=params['kappa'])
    normals_and_cnormals = np.vstack((normals, cnormals)) # стакаем вместе сопряженные наборы нормалей
    strikes_and_cstrikes = np.vstack((strikes, cstrikes))

    # Шаг 4: Расчёт нормальных и касательных напряжений
    sigma_n, tau = compute_normal_and_shear_ensemble(tensor.tensor, normals_and_cnormals)

    # Шаг 5: Расчёт кривой количества событий
    events_vs_pore_press = simulate_microseismicity_over_time(sigma_n, tau, params['mu'], params['cohesion'], pore_press_list)

    # сохранение данный если передан путь к папке, опционально
    if save_path:
        save_simulation_results(
            save_path, params,
            pore_press_list, events_vs_pore_press[0],
            cpps, ps,
            strike, dip, cstrike, cdip,
            sigma_n[0], tau[0],
            normals_and_cnormals, strikes_and_cstrikes, seed
        )
        
    return events_vs_pore_press[0], pore_press_list, cpps, ps


def resample_events(events: np.ndarray,
                    N: int,
                    *,
                    random_state: int | None = None) -> np.ndarray:
    """
    Из большого набора «гладких» кривых events (B×T) формирует
    новый набор, где в каждой кривой ровно N событий.

    Parameters
    ----------
    events : ndarray, shape (B, T)
        Наблюдённые счётчики событий (любые неотрицательные числа).
        B — число кривых, T — число бинов давления.

    N : int
        Сколько событий должно оказаться в каждой итоговой кривой.

    random_state : int | None, default None
        Seed генератора NumPy для воспроизводимости.

    Returns
    -------
    sampled : ndarray, shape (B, T)
        Каждая строка — выборка из многомерного биномиального
        распределения, сумма по строке ровно N.
    """
    rng = np.random.default_rng(random_state)

    # нормируем каждую кривую → вектор вероятностей длиной T
    probs = events / events.sum(axis=1, keepdims=True)

    # для каждой строки тянем выборку из многомерного биномиального
    sampled = np.vstack(
        [rng.multinomial(N, p) for p in probs]
    ).astype(int)

    return sampled


def detect_peaks_findpeaks(events: np.ndarray,
                           pressure: np.ndarray,
                           *,
                           n_peaks: int = 2,
                           win: int = 5,
                           poly: int = 2,
                           prominence: float = 1e-3,
                           width: float = 1):
    
    """
    Находит n_peaks локальных максимумов на кривых микросейсмических
    событий от давления с помощью сглаживания Savitzky–Golay и
    scipy.signal.find_peaks, причём первые нулевые бины поднимаются
    до первого ненулевого, чтобы избежать артефактов на границах.

    Parameters
    ----------
    events : array_like, shape (B, T) или (T,)
        Матрица (или вектор) счётчиков событий по бинам давления.
        Если передан одномерный массив, считается одна кривая.
    pressure : array_like, shape (T,) или (B, T)
        Значения центров бинов давления, соответствующие оси events.
        Если одномерный, применяется ко всем кривым.
    n_peaks : int, default=2
        Число пиков, которое нужно найти в каждой кривой.
    win : int, default=5
        Длина окна (должно быть нечётным) для сглаживания фильтром
        Савицкого–Голея.
    poly : int, default=2
        Степень полинома для фильтра Савицкого–Голея.
    prominence : float, default=1e-3
        Минимальная «заметность» пика (prominence) для scipy.find_peaks.
    width : float, default=1
        Минимальная ширина пика (в бинах) для scipy.find_peaks.

    Returns
    -------
    idx_out : ndarray of int, shape (B, n_peaks)
        Индексы найденных пиков в каждой строке events. Если пиков меньше
        n_peaks, лишние позиции заполняются -1.
    peak_pressures : ndarray of float, shape (B, n_peaks)
        Значения давлений в найденных пиках. Для отсутствующих пиков
        ставится NaN.

    Notes
    -----
    - Ведущие нулевые значения в каждой строке `events` поднимаются до
      первого ненулевого, чтобы при сглаживании не получалось
      искажений на границе.
    - Сглаживание выполняется единым вызовом Savitzky–Golay по оси
      давления с режимом `mode='nearest'`.
    - Поиск пиков делается классическим `scipy.signal.find_peaks`
      с порогами `prominence` и `width`.
    """
    
    events = np.atleast_2d(events)             # (B,T)
    B, T   = events.shape

    # 1) нормировка
    probs = events / events.sum(axis=1, keepdims=True)

    # 2) «поднимаем» ведущие нули
    #    для каждой строки находим первый ненулевой индекс
    first_nz = np.argmax(probs>0, axis=1)      # если всё нули – вернёт 0, но тогда ничего и не фильтруется
    #    создаём копию для фильтра
    probs_f = probs.copy()
    for b in range(B):
        k = first_nz[b]
        if k>0:
            probs_f[b, :k] = probs_f[b, k]

    # 3) сглаживаем всю строку сразу, mode='nearest' повторяет граничные значения
    smoothed = savgol_filter(
        probs_f,
        window_length=win,
        polyorder=poly,
        axis=1,
        mode='nearest'    
    )

    # 4) находим пики
    idx_out = np.full((B, n_peaks), -1, dtype=int)
    for b in range(B):
        peaks, props = find_peaks(
            smoothed[b],
            prominence=prominence,
            width=width
        )
        if peaks.size:
            order = np.argsort(props["prominences"])[::-1][:n_peaks]
            idx_out[b, :len(order)] = peaks[order]

    # 5) переводим индексы → давления
    #    учтём, что pressure может быть (T,) или (B,T)
    p = np.broadcast_to(pressure, (B, T)) if pressure.ndim==1 else pressure
    peak_pressures = np.full_like(idx_out, np.nan, dtype=float)
    for b in range(B):
        good = idx_out[b]>=0
        peak_pressures[b, good] = p[b, idx_out[b, good]]

    return idx_out, peak_pressures


def align_peaks(cpps_preds: np.ndarray,
                        true_vals: tuple[float, float]) -> np.ndarray:
    """
    Расставляет два предсказанных пика (или один, если второй не найден)
    в последовательности (P21, P32).

    Логика по строке:
      • оба пика NaN   → возвращаются NaN-ы;
      • один пик       → присваивается ближайшему из {P21, P32};
      • оба пика       → оцениваются 4 расстояния |predᵢ−Pⱼ|,
                        выбирается конфигурация с минимальным расстоянием;
                        при необходимости пара меняется местами.

    Parameters
    ----------
    cpps_preds : ndarray, shape (N, 2)
        Пики, найденные на каждой кривой;
        NaN там, где пик не детектирован.
    true_vals  : tuple(float, float)
        Истинные критические давления (P21, P32).

    Returns
    -------
    aligned : ndarray, shape (N, 2)
        [предсказание для P21, предсказание для P32]
        (NaN, если соответствующего пика нет).
    """
    preds      = np.asarray(cpps_preds, dtype=float)
    P21, P32   = map(float, true_vals)
    aligned    = np.full_like(preds, np.nan)

    valid_mask     = ~np.isnan(preds)
    valid_count    = valid_mask.sum(axis=1)

    # ------------------------------------------------------------------ 2 пика
    rows_two = valid_count == 2
    if np.any(rows_two):
        p2 = preds[rows_two]                        # (M,2)
        d  = np.abs(p2[:, :, None] - np.array([P21, P32]))  # (M,2,2)

        idx_flat  = np.argmin(d.reshape(-1, 4), axis=1)     # 0..3
        pred_idx  = idx_flat // 2                           # 0|1
        true_idx  = idx_flat  % 2                           # 0|1  (0→P21)

        need_flip = pred_idx != true_idx
        p2[need_flip] = p2[need_flip, ::-1]

        aligned[rows_two] = p2

    # ------------------------------------------------------------------ 1 пик
    rows_one = valid_count == 1
    if np.any(rows_one):
        p1_vals   = preds[rows_one][valid_mask[rows_one]]   # (K,)
        rows_one_idx = np.flatnonzero(rows_one)

        dist_to_21 = np.abs(p1_vals - P21)
        dist_to_32 = np.abs(p1_vals - P32)
        closer_to_21 = dist_to_21 <= dist_to_32

        aligned[rows_one_idx, 0] = np.where(closer_to_21, p1_vals, np.nan)
        aligned[rows_one_idx, 1] = np.where(~closer_to_21, p1_vals, np.nan)

    # ------------------------------------------------------------------ 0 пиков (остаются NaN)
    return aligned