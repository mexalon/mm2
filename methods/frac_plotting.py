import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scipy.stats import gaussian_kde

"""  
Отрисовка всего связанного с трещинами 
"""


''' 
Красивая 3д сфера со свем добром
'''
def plot_fracture_ensemble_on_sphere(normals=None,
                                     seed=None,
                                     max_to_plot=40):
    """
    3D-визуализация ансамбля трещин:
      - Сфера (тектонная модель)
      - Дуги пересечения трещин с поверхностью сферы (ансамбль и материнская трещина)
      - Векторы нормалей (ансамбль и материнская)
      - Точки пересечения нормалей со сферой
      - Плоскость материнской трещины с сеткой
      - Обозначение сторон света (E, N, Up)
    """
    # Создаём фигуру и 3D‑ось
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    def plot_sphere(ax, radius=1.0, alpha=0.1, color='lightblue'):
        """
        Рисует полупрозрачную единичную сферу:
          - u, v — параметры для построения сетки
          - x, y, z — координаты точек поверхности
        """
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    def plot_circle_on_sphere(ax, normal, color='gray', linewidth=1.0, alpha=0.5, n_points=200):
        """
        Рисует дугу большого круга, определяемую нормалью:
          - normal — вектор нормали к плоскости разлома
          - генерируем два ортогональных вектора v1, v2 в этой плоскости
          - theta задаёт угол по дуге
          - circle — массив точек на окружности
        """
        # Нормализуем вектор нормали
        normal = normal / np.linalg.norm(normal)
        # Выбираем опорный вектор для построения базиса
        if np.abs(normal[2]) < 0.9:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [0, 1, 0])
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)

        # Построение дуги
        theta = np.linspace(0, 2 * np.pi, n_points)
        circle = np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2)
        ax.plot(circle[:, 0], circle[:, 1], circle[:, 2],
                color=color, linewidth=linewidth, alpha=alpha)

    def plot_plane_with_grid(ax, normal, size=1.0, resolution=20, color='cyan', alpha=0.2):
        """
        Рисует сетку в плоскости трещины:
          - строится равномерная сетка (uu, vv) в базисе плоскости
          - затем переводится в глобальные координаты
        """
        normal = normal / np.linalg.norm(normal)
        if np.abs(normal[2]) < 0.9:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [0, 1, 0])
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)

        # Параметрическая сетка по двум направлениям
        u = np.linspace(-size, size, resolution)
        v = np.linspace(-size, size, resolution)
        uu, vv = np.meshgrid(u, v)
        points = uu[..., np.newaxis] * v1 + vv[..., np.newaxis] * v2

        X, Y, Z = points[..., 0], points[..., 1], points[..., 2]
        ax.plot_wireframe(X, Y, Z,
                          color=color,
                          alpha=alpha,
                          linewidth=0.5,
                          rstride=2,
                          cstride=2)

    # Если нет заранее вычисленных нормалей, генерируем из strike/dip
    # if normals is None and strike_samples is not None and dip_samples is not None:
    #     normals = seed.build_normal_batch(strike_samples, dip_samples)

    # Нарисовать сферу
    plot_sphere(ax)

    # Построить ансамбль нормалей
    if normals is not None:
        k = min(len(normals), max_to_plot)
        for i in range(k):
            # 1) Дуга пересечения трещины с сферой
            plot_circle_on_sphere(ax,
                                  normals[i],
                                  color='gray',
                                  linewidth=1.0,
                                  alpha=0.3)
            # 2) Вектор нормали (стрелка)
            ax.quiver(0, 0, 0,
                      normals[i, 0],
                      normals[i, 1],
                      normals[i, 2],
                      length=1.0,
                      normalize=True,
                      arrow_length_ratio=0.15,
                      color='gray',
                      alpha=0.3,
                      linewidth=1.2)
            # 3) Конечная точка нормали
            ax.scatter(normals[i, 0],
                       normals[i, 1],
                       normals[i, 2],
                       color='black',
                       s=5,
                       alpha=0.4)

    # Построение материнской трещины
    if seed is not None:
        # — дуга большого круга
        plot_circle_on_sphere(ax,
                            seed.normal,
                            color='blue',
                            linewidth=2.0,
                            alpha=1.0)
        # — плоскость в сетке
        plot_plane_with_grid(ax,
                            seed.normal,
                            size=1.0,
                            resolution=20,
                            color='blue',
                            alpha=0.4)
        # — вектор нормали материнской трещины
        ax.quiver(0, 0, 0,
                seed.normal[0],
                seed.normal[1],
                seed.normal[2],
                length=1.0,
                normalize=True,
                arrow_length_ratio=0.15,
                color='blue',
                linewidth=2.5)
        # — конечная точка вектора
        ax.scatter(seed.normal[0],
                seed.normal[1],
                seed.normal[2],
                color='blue',
                s=40,
                alpha=1.0)

    # Настройка границ и пропорций axes
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])  # единая шкала во всех осях
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_zticks([-1, 1])
    ax.grid(False)

    # Ручное нанесение меток сторон света
    ax.text(0, -1.4, -1.4, r'$E$', fontsize=12)
    ax.text(1, 0,    -1.4, r'$N$', fontsize=12)
    ax.text(1, 1.4,  -0.2, r'$Up$', fontsize=12)

    # При необходимости можно раскомментировать легенду
    # custom_lines = [
    #     Line2D([0], [0], color='blue', lw=2, label='Материнская трещина'),
    #     Line2D([0], [0], color='gray', lw=1, alpha=0.4, label='Ансамбль трещин'),
    #     Line2D([0], [0], color='black', marker='o', lw=0, label='Нормали на сфере')
    # ]
    # ax.legend(handles=custom_lines, loc='upper left', fontsize=12)

    # plt.tight_layout()  # обычно не нужен в 3D
    plt.show()

"""
Стереогрммы без тензора напряжений
"""
def plot_fracture_normals_and_planes(normals, seed=None, max_to_plot=1000):
    """
    2D‐визуализация полюсов (нормалей) трещин и их контуров:
      - Точки полюсов на нижней полусфере
      - Дуги пересечения каждой плоскости трещины с единичной окружностью
      - Особая дуга для «материнской» трещины (seed)
      - Подписи сторон света

    Аргументы:
    ----------
    normals : np.ndarray (M, 3)
        Нормали к трещинам
    seed : FractureSeed, optional
        Объект материнской трещины для выделения
    max_to_plot : int
        Максимальное количество нормалей для визуализации
    """
    # Нормализация векторов и отбор только нижней полусферы (nz < 0)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    normals[normals[:, 2] > 0] = - normals[normals[:, 2] > 0] # переворачиваем нормали, которые смотрят вверх на противоположные, чотобы смотрели вниз
    
    # Ограничение количества нормалей для отрисовки
    if len(normals) > max_to_plot:
        idx = np.random.choice(len(normals), size=max_to_plot, replace=False)
        normals = normals[idx]

    x, y = normals[:, 0], normals[:, 1]

    # Создаём фигуру и квадратную ось
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_aspect('equal')

    # Рисуем точки‐полюсы
    ax.scatter(x, y, s=10, color='dodgerblue', alpha=0.6)
    
    # Дуги ансамбля
    for n in normals:
        plot_great_circle(ax, n,
                            color='gray',
                            alpha=0.2,
                            linewidth=0.6)
        
    # Если задан seed, рисуем его дугу поверх ансамбля
    if seed is not None:
        # Дуга материнской трещины
        plot_great_circle(ax, seed.normal,
                          color='blue',
                          linewidth=2.0,
                          alpha=0.9)

    # Граница единичной окружности
    ax.add_patch(Circle((0, 0), 1.0,
                        edgecolor='black',
                        facecolor='none',
                        lw=1))
    # Устанавливаем лимиты чуть за окружностью
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    # Убираем подписи осей
    ax.set_xticks([])
    ax.set_yticks([])
    # Заголовок с отступом
    ax.set_title("Нормали и плоскости",
                 fontsize=12,
                 pad=25)
    # Подписи N‐E‐S‐W
    label_cardinal_directions(ax, r=1.1)
    # Убираем рамку вокруг графика
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()

"""Плотность распределения полюсов - Kamb Contours in Standard Deviations"""
def plot_fracture_density(normals):
    """
    2D‐контур плотности полюсов на нижней полусфере:
      - Оценка плотности методом KDE
      - Запрет значений вне единичного круга
      - colorbar с подписью
      - Подписи сторон света
    """
    # Нормализация и фильтрация по z < 0
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    normals[normals[:, 2] > 0] = - normals[normals[:, 2] > 0] # переворачиваем нормали, которые смотрят вверх на противоположные, чотобы смотрели вниз
    x, y = normals[:, 0], normals[:, 1]

    # Фигура с двумя осями: основная и под colorbar
    fig = plt.figure(figsize=(5, 3))
    ax  = fig.add_axes([0.05, 0.05, 0.74, 0.90])  # левый большой прямоугольник
    cax = fig.add_axes([0.83, 0.15, 0.06, 0.70])  # узкая полоса справа
    ax.set_aspect('equal')

    # Сетка точек для KDE
    xi, yi = np.mgrid[-1:1:100j, -1:1:100j]
    # Оценка плотности
    dens = gaussian_kde(np.vstack([x, y]))(
        np.vstack([xi.ravel(), yi.ravel()])
    ).reshape(xi.shape)
    # Обнуляем значения вне круга
    dens[(xi**2 + yi**2) > 1] = np.nan

    # Заливка и контуры плотности
    cf = ax.contourf(xi, yi, dens,
                     levels=20,
                     cmap='GnBu')
    ax.contour(xi, yi, dens,
               levels=10,
               colors='k',
               linewidths=0.3,
               alpha=0.5)

    # Отрисовка colorbar и подпись
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label("Плотность (отн. ед.)", fontsize=12)

    # Граница единичного круга поверх плотности
    ax.add_patch(Circle((0, 0), 1,
                        ec='k',
                        fc='none',
                        lw=1,
                        zorder=10))
    # Настройка лимитов и отключение осей
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    # Заголовок с отступом
    ax.set_title("Контурная плотность нормалей",
                 pad=25,
                 fontsize=12)
    # Подписи сторон света
    label_cardinal_directions(ax, r=1.10)
    # Скрываем рамку
    for sp in ax.spines.values():
        sp.set_visible(False)

    plt.show()

"""Роза направлений страйк"""
def plot_fracture_strike_rose(strike_samples, n_bins=36):
    """
    Полярная диаграмма «роза направлений» strike:
      - Гистограмма направлений с равномерными бинами
      - 0° → North, по часовой
      - Подписи N, E, S, W
    """
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))

    # Считаем количество в каждом бине
    bin_edges = np.linspace(0, 360, n_bins + 1)
    counts, _ = np.histogram(strike_samples % 360, bins=bin_edges)
    theta = np.radians((bin_edges[:-1] + bin_edges[1:]) / 2)

    # Столбики розы
    ax.bar(theta,
           counts,
           width=np.radians(360 / n_bins),
           color='mediumseagreen',
           edgecolor='black',
           alpha=0.8)

    # Геологические настройки полярной оси
    ax.set_theta_zero_location("N")  # 0° наверху
    ax.set_theta_direction(-1)       # по часовой
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(['N', 'E', 'S', 'W'])
    ax.set_title("Роза направлений (strike)", fontsize=12)

    plt.show()


def plot_great_circle(ax, normal, n_points=200, **kwargs):
    """
    Рисует дуги большого круга для заданной нормали:
      - Разбиваем полную окружность на сегменты по условию z<0
      - Каждую смежную часть рисуем отдельно
    """
    normal = normal / np.linalg.norm(normal)
    # Базис в плоскости нормали
    if np.abs(normal[2]) < 0.9:
        v1 = np.cross(normal, [0, 0, 1])
    else:
        v1 = np.cross(normal, [0, 1, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    # Параметрическое описание круга
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2)

    # Оставляем только нижнюю полусферу и разрезаем на сегменты
    mask = circle[:, 2] < 0
    segments = np.split(circle, np.where(np.diff(mask.astype(int)) != 0)[0] + 1)
    # Рисуем каждую непрерывную дугу
    for seg in segments:
        if len(seg) > 1 and np.all(seg[:, 2] < 0):
            ax.plot(seg[:, 0], seg[:, 1], **kwargs)


def label_cardinal_directions(ax, r=1.08, fontsize=12):
    """
    Наносит подписи сторон света вокруг круга:
      - r задаёт расстояние от центра
      - выравнивание текста так, чтобы буквы уходили наружу
    """
    ax.text( 0,  r, 'N', ha='center', va='bottom', fontsize=fontsize)
    ax.text( 0, -r, 'S', ha='center', va='top',    fontsize=fontsize)
    ax.text( r,  0, 'E', ha='left',   va='center', fontsize=fontsize)
    ax.text(-r,  0, 'W', ha='right',  va='center', fontsize=fontsize)


def plot_mu_cohesion_histograms(mu, cohesion, bins=50):
    """
    Отображает маленькие гистограммы распределения коэффициента трения и сцепления.

    Параметры:
    ----------
    mu : np.ndarray
        Массив коэффициентов внутреннего трения

    cohesion : np.ndarray
        Массив сцеплений

    bins : int
        Количество бинов в гистограммах
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), constrained_layout=True)

    axes[0].hist(mu, bins=bins, color='skyblue', edgecolor='k', alpha=0.8)
    axes[0].set_title('Коэффициент трения μ', fontsize=10)
    axes[0].set_xlabel('μ', fontsize=10)
    axes[0].set_ylabel('Частота', fontsize=10)
    axes[0].tick_params(labelsize=9)

    axes[1].hist(cohesion, bins=bins, color='lightcoral', edgecolor='k', alpha=0.8)
    axes[1].set_title('Сцепление C', fontsize=10)
    axes[1].set_xlabel('C, МПа', fontsize=10)
    axes[1].set_ylabel('Частота', fontsize=10)
    axes[1].tick_params(labelsize=9)

    for ax in axes:
        ax.grid(True, linestyle=':', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.show()