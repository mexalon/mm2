import numpy as np

def compute_normal_and_shear_ensemble(tensors, normals):
    """
    Вычисляет нормальные и касательные напряжения на ансамбле трещин
    для каждого тензора напряжений.

    Параметры:
    ----------
    tensors : np.ndarray (N, 3, 3) или (3, 3)
        Один или несколько тензоров напряжений (сжимающие напряжения положительные)

    normals : np.ndarray (M, 3)
        Нормали к трещинам (единичные вектора, направлены вниз для горизонтальных трещин)

    Возвращает:
    -----------
    sigma_n : np.ndarray (N, M)
        Нормальные напряжения (сжимающие положительные)

    tau : np.ndarray (N, M)
        Касательные напряжения (модули)
    """
    # Приведение к батчу, если передан один тензор (3, 3)
    if tensors.ndim == 2 and tensors.shape == (3, 3):
        tensors = tensors[np.newaxis, :]  # (1, 3, 3)

    N = tensors.shape[0]
    M = normals.shape[0]

    # Проверка нормировки нормалей
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Расчёт тракции: t = σ · n для каждого тензора и нормали
    traction_vectors = np.einsum('nij,mj->nmi', tensors, normals)

    # Нормальное напряжение: sigma_n = t · n
    sigma_n = np.einsum('nmi,mi->nm', traction_vectors, normals)

    # Касательное напряжение: tau = || t - sigma_n * n ||
    shear_vectors = traction_vectors - sigma_n[..., np.newaxis] * normals[np.newaxis, :, :]
    tau = np.linalg.norm(shear_vectors, axis=2)

    return sigma_n, tau


def check_coulomb_failure(sigma_n, tau, mu, cohesion, pore_pressure=0.0):
    """
    Проверка критерия Кулона-Мора с учётом порового давления.

    Параметры:
    ----------
    sigma_n : np.ndarray (N, M)
        Нормальные напряжения

    tau : np.ndarray (N, M)
        Касательные напряжения

    mu : np.ndarray (M,) или float
        Коэффициент трения

    cohesion : np.ndarray (M,) или float
        Сцепление

    pore_pressure : float или np.ndarray (N, M)
        Поровое давление, вычитаемое из нормального напряжения

    Возвращает:
    -----------
    failures : np.ndarray (N, M)
        0 — стабильная трещина, 1 — нестабильная
    """
    sigma_eff = sigma_n - pore_pressure
    mu = np.atleast_1d(mu)[np.newaxis, :]
    cohesion = np.atleast_1d(cohesion)[np.newaxis, :]
    tau_crit = mu * sigma_eff + cohesion
    failures = (tau > tau_crit).astype(int)
    return failures


def critical_pore_pressure(principal_stresses, mu, cohesion):
    """
    Вычисляет критические поровые давления, при котором соответсвующие круги Мора касаются критерия Кулона,
    на основе объектов StressTensor и FractureSeed.

    Параметры:
    ----------
    tensor : StressTensor
        Объект, содержащий тензор главных напряжений

    seed : FractureSeed
        Объект, содержащий параметры трения и сцепления

    Возвращает:
    -----------
    p_crit : tuple (3,)
        Критические поровые давления для трех кругов
    """
    def get_p_crit(smin, smax):
        # Центр и радиус большого круга Мора
        C = (smax + smin)/2 
        R = (smax - smin)/2

        # Расстояние от центра круга до прямой Кулона при p = 0 из геометрии
        # d = (mu * C + cohesion) / (mu**2 + 1)**0.5 
        # d = R при касании, когда центр круга смещен на p_crit
        # R = (mu * (C - p_crit) + cohesion) / (mu**2 + 1)**0.5
        # откуда:

        # Критическое поровое давление
        p_crit = C - (R * (mu**2 + 1)**0.5 - cohesion) / mu
        return p_crit
    
    s1, s2, s3 = sorted(principal_stresses, reverse=True)

    return get_p_crit(s3, s1), get_p_crit(s2, s1), get_p_crit(s3, s2)


def normal_to_strike_dip(n):
    """
    Перевод нормали (геодезическая система: x-восток, y-север, z-вверх)
    в углы strike (азимут от севера по часовой) и dip (0…90°, вниз).
    """
    # Dip: 0° – горизонтальная плоскость (n_z = –1); 90° – вертикальная.
    dip = np.degrees(np.arccos(-n[2]))

    # Strike: направление линии пересечения плоскости с горизонтом
    # s = n × k  ,  k=(0,0,1)
    s_x = n[1]
    s_y = -n[0]
    if np.isclose(dip, 0.0):
        strike = 0.0          # горизонталь: strike не определён
    else:
        strike = (np.degrees(np.arctan2(s_x, s_y)) + 360) % 360
    return strike, dip


def get_critical_strike_dip(tensor, mu):
    """
    Возвращает два сопряжённых решения (strike, dip) для критической трещины
    
    """
    # индексы максимального и минимального главных напряжений
    ps = tensor.principal_stresses
    idx_max = int(np.argmax(ps))
    idx_min = int(np.argmin(ps))

    alpha = np.pi/4 + 1/2 * np.arctan(mu)          # угол между нормалью и σ₁

    normals_geo = []
    signs = (+1, -1)               # два сопряжённых решения
    for s in signs:
        n_pr = np.zeros(3)
        n_pr[idx_max] =  np.cos(alpha)
        n_pr[idx_min] =  s * np.sin(alpha)

        # перевод из главных в геодезические оси
        n_geo = tensor.rotation_matrix @ n_pr

        # нормаль должна смотреть ВНИЗ (n_z ≤ 0)
        if n_geo[2] > 0:
            n_geo = -n_geo

        n_geo /= np.linalg.norm(n_geo)
        normals_geo.append(n_geo)

    # переводим обе нормали в strike/dip
    strikes_dips = [normal_to_strike_dip(n) for n in normals_geo]

    return strikes_dips    # [(strike1, dip1), (strike2, dip2)]

