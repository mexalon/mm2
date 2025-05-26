import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_coulomb_diagram_density(sigma_n, tau, mu=0.6, cohesion=0.0, 
                                 pore_pressure=0.0, principal_stresses=None):
    """
    Визуализация диаграммы Кулона-Мора с кругами Мора и контурной картой 
    относительной плотности нормальных и касательных напряжений,
    с обрезкой за пределами большого круга и внутри малых.

    Параметры:
    ----------
    sigma_n : np.ndarray (M,)
        Нормальные напряжения

    tau : np.ndarray (M,)
        Касательные напряжения

    mu : float
        Коэффициент трения

    cohesion : float
        Сцепление

    principal_stresses : array-like (3,), optional
        Главные напряжения

    pore_pressure : float
        Поровое давление, вычитаемое из σₙ
    """
    sigma_eff = sigma_n - pore_pressure

    # Рассчёт кругов Мора
    if principal_stresses is not None:
        s1, s2, s3 = np.sort(principal_stresses)[::-1]
        C_big = 0.5 * (s1 + s3) - pore_pressure
        R_big = 0.5 * (s1 - s3)
        C_small1 = 0.5 * (s1 + s2) - pore_pressure
        R_small1 = 0.5 * (s1 - s2)
        C_small2 = 0.5 * (s2 + s3) - pore_pressure
        R_small2 = 0.5 * (s2 - s3)
        sigma_min, sigma_max = 0, (s1 - pore_pressure) * 1.05
        tau_min, tau_max = 0, R_big * 1.05
    else:
        sigma_min, sigma_max = 0, np.max(sigma_eff) * 1.05
        tau_min, tau_max = 0, np.max(tau) * 1.05

    # Прямая Кулона
    sigma_line = np.linspace(sigma_min, sigma_max, 500)
    tau_failure = mu * sigma_line + cohesion

    # Гистограмма плотности
    H, xedges, yedges = np.histogram2d(
        sigma_eff, tau,
        bins=64,
        range=[[sigma_min, sigma_max], [tau_min, tau_max]]
    )
    H_rel = H / H.sum() if H.sum() > 0 else H

    X = 0.5 * (xedges[:-1] + xedges[1:])
    Y = 0.5 * (yedges[:-1] + yedges[1:])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cf = ax.contourf(X, Y, H_rel.T, levels=10, cmap='cubehelix_r', alpha=0.8)

    # Обрезка за пределами большого круга
    big_circle = Circle((C_big, 0), R_big, transform=ax.transData)
    cf.set_clip_path(big_circle)

    # Маскировка внутри малых кругов
    if principal_stresses is not None:
        for C, R in [(C_small1, R_small1), (C_small2, R_small2)]:
            ax.add_patch(Circle((C, 0), R, transform=ax.transData,
                                facecolor='white', edgecolor='none', zorder=10))

    # Колорбар
    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, shrink=0.5)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Отн. плотность', fontsize=12)

    # Круги Мора
    if principal_stresses is not None:
        for (C, R, color) in [(C_big, R_big, 'black'),
                               (C_small1, R_small1, 'gray'),
                               (C_small2, R_small2, 'lightgray')]:
            theta = np.linspace(0, np.pi, 300)
            ax.plot(C + R*np.cos(theta), R*np.sin(theta), '--',
                    color=color, linewidth=1)

    # Линия Кулона
    ax.plot(sigma_line, tau_failure, 'r--',
            label=fr'$\tau = {mu:.2f}\,\sigma_{{\mathrm{{eff}}}} + {cohesion:.2f}$',
            linewidth=1.2, zorder=15)  

    # Оформление шрифтов 12
    ax.set_xlim(sigma_min, sigma_max)
    ax.set_ylim(tau_min, tau_max)
    ax.set_xlabel(r'$\sigma_{\mathrm{eff}}$, МПа', fontsize=12)
    ax.set_ylabel(r'$\tau$, МПа', fontsize=12)
    ax.set_title('Диаграмма Кулона-Мора (плотность)', fontsize=12)
    leg = ax.legend(fontsize=12, loc='lower right', frameon=True)
    leg.set_zorder(20)
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()


def plot_coulomb_diagram(sigma_n, tau, mu=0.6, cohesion=0.0, pore_pressure=0.0, principal_stresses=None, failures=None, ):
    """
    Визуализация диаграммы Кулона-Мора с кругами Мора и учётом порового давления.

    Параметры:
    ----------
    sigma_n : np.ndarray (M,)
        Нормальные напряжения

    tau : np.ndarray (M,)
        Касательные напряжения

    mu : float
        Коэффициент трения

    cohesion : float
        Сцепление

    principal_stresses : array-like (3,), optional
        Главные напряжения

    failures : np.ndarray (M,), optional
        0 — стабильная, 1 — нестабильная

    pore_pressure : float
        Поровое давление, вычитаемое из σₙ
    """

    # Эффективное напряжение
    sigma_eff = sigma_n - pore_pressure

    # Если заданы главные напряжения — определим пределы из них
    if principal_stresses is not None:
        s1, s2, s3 = np.sort(principal_stresses)[::-1]
        max_sigma_eff = s1 - pore_pressure
        min_sigma_eff = s3 - pore_pressure
        sigma_n_max = max_sigma_eff * 1.05
        sigma_n_min = 0
        tau_max = 0.5 * (s1 - s3) * 1.05
    else:
        sigma_n_min = 0
        sigma_n_max = np.max(sigma_eff) * 1.05
        tau_max = np.max(tau) * 1.05

    # Линия разрушения
    sigma_line = np.linspace(sigma_n_min, sigma_n_max, 500)
    tau_failure = mu * sigma_line + cohesion

    # Фигура
    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()

    # Круги Мора
    if principal_stresses is not None:
        mora_pairs = [(s1, s2), (s2, s3), (s1, s3)]
        colors = ['gray', 'gray', 'lightgray']
        for (s_max, s_min), color in zip(mora_pairs, colors):
            center = 0.5 * (s_max + s_min - 2 * pore_pressure)
            radius = 0.5 * (s_max - s_min)
            theta = np.linspace(0, np.pi, 300)
            circle_sigma = center + radius * np.cos(theta)
            circle_tau = radius * np.sin(theta)
            ax.plot(circle_sigma, circle_tau, linestyle='--', color=color, linewidth=1)

    # Линия Кулона
    ax.plot(sigma_line, tau_failure, 'r--',
            label=fr'$\tau = {mu:.2f} \cdot \sigma_{{\mathrm{{eff}}}} + {cohesion:.2f}$',
            linewidth=1.2)

    # Точки
    if failures is not None:
        failures = np.asarray(failures).astype(bool)
        stable = ~failures
        ax.scatter(sigma_eff[stable], tau[stable], s=2, alpha=0.4, label='Стабильные', color='lightblue')
        ax.scatter(sigma_eff[failures], tau[failures], s=4, alpha=0.8, label='Нестабильные', color='steelblue')
    else:
        tau_crit = mu * sigma_eff + cohesion
        unstable = tau > tau_crit
        stable = ~unstable
        ax.scatter(sigma_eff[stable], tau[stable], s=2, alpha=0.4, label='Стабильные', color='lightblue')
        ax.scatter(sigma_eff[unstable], tau[unstable], s=4, alpha=0.8, label='Нестабильные', color='steelblue')

    # Оформление
    ax.set_xlim(0, sigma_n_max)
    ax.set_ylim(0, tau_max)
    ax.set_xlabel(r'$\sigma_{\mathrm{eff}}$, МПа', fontsize=12)
    ax.set_ylabel(r'$\tau$, МПа', fontsize=12)
    ax.set_title('Диаграмма Кулона-Мора', fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
