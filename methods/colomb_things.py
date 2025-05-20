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


def critical_pore_pressure(tensor, seed):
    """
    Вычисляет критическое поровое давление, при котором круг Мора касается критерия Кулона,
    на основе объектов StressTensor и FractureSeed.

    Параметры:
    ----------
    tensor : StressTensor
        Объект, содержащий тензор главных напряжений

    seed : FractureSeed
        Объект, содержащий параметры трения и сцепления

    Возвращает:
    -----------
    p_crit : float
        Критическое поровое давление
    """
    s1, s2, s3 = sorted(tensor.principal_stresses, reverse=True)
    mu = seed.mu
    cohesion = seed.cohesion

    # Центр и радиус большого круга Мора
    C = (s1 + s3)/2
    R = (s1 - s3)/2

    # Расстояние от центра круга до прямой Кулона при p = 0
    # d = abs(-mu * C - cohesion) / (mu**2 + 1)**0.5

    # Критическое расстояние до касания
    d_crit = R * (1 + mu**2)**0.5

    # Критическое поровое давление
    p_crit = C + (-1 * d_crit + cohesion) / mu

    return p_crit