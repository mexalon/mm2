import matplotlib.pyplot as plt

"""  
Отрисовка тензора напряжений
"""

def plot_stress_tensor_with_rotated_ensemble(tensor,
                                              ensemble_stresses=None,
                                              ensemble_trends=None,
                                              ensemble_plunges=None,
                                              ensemble_rakes=None,
                                              max_ensemble_to_plot=100):
    """
    Визуализирует исходный и повернутый тензор напряжений, а также ансамбль случайных реализаций
    на основе ориентации, заданной через углы Эйлера: trend, plunge, rake (по Zoback).

    Аргументы:
    - tensor: StressTensor — основной эталонный тензор
    - ensemble_stresses: (n, 3) — массив сгенерированных главных напряжений
    - ensemble_trends: (n,) — углы trend (градусы)
    - ensemble_plunges: (n,) — углы plunge (градусы)
    - ensemble_rakes: (n,) — углы rake (градусы)
    - max_ensemble_to_plot: int — количество случайных реализаций для визуализации
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection='3d')

    # Отключаем стандартную сетку осей
    ax.grid(False)

    colors = ['red', 'green', 'blue']
    marker_size = 60

    # Исходные главные оси
    original_vectors = np.diag(tensor.principal_stresses)
    rotated_vectors = tensor.rotation_matrix @ original_vectors

    # Рисуем исходные и повернутые оси
    for i in range(3):
        # Исходные
        ax.plot([0, original_vectors[0, i]],
                [0, original_vectors[1, i]],
                [0, original_vectors[2, i]],
                color=colors[i], alpha=0.3, linewidth=2)
        ax.scatter(*original_vectors[:, i],
                   color=colors[i], alpha=0.3, s=marker_size)
        # Повернутые
        ax.plot([0, rotated_vectors[0, i]],
                [0, rotated_vectors[1, i]],
                [0, rotated_vectors[2, i]],
                color=colors[i], alpha=1.0, linewidth=3)
        ax.scatter(*rotated_vectors[:, i],
                   color=colors[i], alpha=1.0, s=marker_size)

    # Ансамбль случайных тензоров
    if (ensemble_stresses is not None and
        ensemble_trends is not None and
        ensemble_plunges is not None and
        ensemble_rakes is not None):

        n = min(max_ensemble_to_plot, ensemble_stresses.shape[0])

        R = tensor.build_rotation_matrix(
            ensemble_trends[:n], ensemble_plunges[:n], ensemble_rakes[:n]
        )
        unit_axes = np.eye(3)[np.newaxis, :, :]
        rotated_axes = np.einsum('nij,njk->nik', R, unit_axes)
        scaled_axes = rotated_axes * ensemble_stresses[:n, np.newaxis, :]

        for axes in scaled_axes:
            for i in range(3):
                ax.plot([0, axes[0, i]],
                        [0, axes[1, i]],
                        [0, axes[2, i]],
                        color=colors[i], alpha=0.05, linewidth=1)

    # Настройка масштаба и тиков
    max_val = np.max(np.abs(tensor.principal_stresses))
    scale = max(10, np.ceil(max_val / 5) * 2.5)
    ticks = [-scale, scale]

    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    # Добавляем горизонтальную плоскость z=0 с сеткой
    grid_vals = np.linspace(-scale, scale, 11)
    xx, yy = np.meshgrid(grid_vals, grid_vals)
    zz = np.zeros_like(xx)
    ax.plot_wireframe(xx, yy, zz, color='gray', alpha=0.7, linewidth=0.5)

    # Подписи к осям
    ax.text(0, -1.4 * scale, -1.4 * scale, r'$X$', fontsize=12)
    ax.text(scale, 0, -1.4 * scale, r'$Y$', fontsize=12)
    ax.text(scale, 1.4 * scale, -0.2 * scale, r'$Z$', fontsize=12)

    ax.view_init(elev=15)
    plt.show()