import numpy as np

class StressTensor:
    def __init__(self,
                 principal_stresses: np.ndarray = np.array([1.0, 1.0, 1.0]),
                 trend: float = 0.0,
                 plunge: float = 0.0,
                 rake: float = 0.0):
        """
        Базовый класс для эталонного тензора напряжений.
        
        Создание тензора напряжений на основе главных компонент (σ₁, σ₂, σ₃) и их поворота в 
        пространстве на углы Эйлера: trend, plunge, rake.
        В начальной системе координат ось x направлена на восток, y - на север, z - вверх 

        Далее над исходным тензором совершается три поворота, для которых задаётся три угла вращения:
        1) trend - угол поворота вокруг оси z — задаёт направление проекции σ₁ на горизонтальную плоскость.
        Отсчитывается по часовой стрелке от севера (оси Y) в градусах. При повороте ось x переходит в ось x',
        y переходит в y'
        
        2) plunge - угол наклона σ₁ вниз от горизонтали (в градусах).
        Поворот происходит вокруг новой оси y' в положительном направлении.
        При этом ось z переходит в ось Z, x' переходит в x"

        3) rake - угол поворота в положительном направлении вокруг новой оси Z, 
        задающий ориентацию векторов в плоскости x"-y', которые переходят в оси X и Y соотвественно
        """
        self.principal_stresses = np.asarray(principal_stresses, dtype=float)
        self.trend = trend 
        self.plunge = plunge
        self.rake = rake

        # Ротация и сам тензор в глобальной системе
        R = StressTensor.build_rotation_matrix(
            np.array([self.trend]),
            np.array([self.plunge]),
            np.array([self.rake])
        )[0]
        self.rotation_matrix = R
        self.tensor = R @ np.diag(self.principal_stresses) @ R.T

    @staticmethod
    def build_rotation_matrix(trend_batch, plunge_batch, rake_batch):
        """
        Собирает массив (n,3,3) матриц поворота по углам Эйлера:
            R = Rz(trend) · Ry(plunge) · Rz(rake)
        """
        a = np.radians( - trend_batch) # минус, чтобы задавать вращение по часовой, как азимут, как в трещине, а сама матрица как в положительном повороте 
        b = np.radians(plunge_batch)
        c = np.radians(rake_batch)
        n = len(a)

        # Rz(trend)
        Rz1 = np.zeros((n,3,3))
        Rz1[:,0,0] = np.cos(a)
        Rz1[:,0,1] = -np.sin(a)
        Rz1[:,1,0] = np.sin(a)
        Rz1[:,1,1] = np.cos(a)
        Rz1[:,2,2] = 1.0

        # Ry(plunge)
        Ry = np.zeros((n,3,3))
        Ry[:,0,0] = np.cos(b)
        Ry[:,0,2] = np.sin(b)
        Ry[:,1,1] = 1.0
        Ry[:,2,0] = -np.sin(b)
        Ry[:,2,2] = np.cos(b)

        # Rz(rake)
        Rz2 = np.zeros((n,3,3))
        Rz2[:,0,0] = np.cos(c)
        Rz2[:,0,1] = -np.sin(c)
        Rz2[:,1,0] = np.sin(c)
        Rz2[:,1,1] = np.cos(c)
        Rz2[:,2,2] = 1.0

        # Композиция: сначала Rz1, затем Ry, затем Rz2
        R_temp = np.einsum('nij,njk->nik', Rz1, Ry)
        return np.einsum('nij,njk->nik', R_temp, Rz2)
    
    def __str__(self):
        return self.tensor.__str__()


class RandomStressTensor(StressTensor):
    """
    Генерация ансамбля случайных тензоров и параметров:
        - principal_stress_samples (N,3) по нормальному распределению вокруг self.principal_stresses
        - trend_samples, plunge_samples, rake_samples по нормальному отклонению вокруг исходных углов
        - tensors (N,3,3) поворачивается R(trend,plunge,rake)
    Стандартные отклонения значений принципиальных компонент тензора задаются в долях, углов эйлера - в градусах
    """
    @staticmethod
    def build_tensor_batch(ps_batch, trend_batch, plunge_batch, rake_batch):
        """
        Построение массива повернутых тензоров (n,3,3) для заданных
        паок главных напряжений и триплета углов.
        """
        R = StressTensor.build_rotation_matrix(trend_batch, plunge_batch, rake_batch)
        T = np.zeros((len(ps_batch), 3, 3))
        T[:,0,0] = ps_batch[:,0]
        T[:,1,1] = ps_batch[:,1]
        T[:,2,2] = ps_batch[:,2]
        # σ' = R · T · R^T
        return np.einsum('nij,njk,nlk->nil', R, T, R)

    def generate_ensemble(self,
                          N: int = 10,
                          stress_std_frac = 0.0, # в долях от среднего, может быть одним значением на всех или листом из трех
                          angle_std_deg = 0.0, # в градусах, может быть одним значением на всех или листом из трех
                          random_seed: int = None):
        '''
        Сама генерация ансамбля тензоров
        Результаты сохраняются в атрибуты:
        self.ensemble_tensors,
        self.ensemble_principal_stresses,
        self.ensemble_trends,
        self.ensemble_plunges,
        self.ensemble_rakes
        и возвращаются в том же порядке.
        '''

        if random_seed is not None:
            np.random.seed(random_seed)

        # Стресс-сэмплирование
        frac = np.array(stress_std_frac, ndmin=1)
        if frac.size == 1:
            frac = np.full(3, frac.item())
        ps_samples = np.random.normal(
            loc=self.principal_stresses,
            scale=self.principal_stresses * frac,
            size=(N,3)
        )

        # Углы-сэмплирование
        if isinstance(angle_std_deg, (int, float)):
            t_std = p_std = r_std = angle_std_deg
        else:
            t_std, p_std, r_std = angle_std_deg
        trend_samples  = np.random.normal(self.trend,  t_std, size=N)
        plunge_samples = np.random.normal(self.plunge, p_std, size=N)
        rake_samples   = np.random.normal(self.rake,   r_std, size=N)

        # Поворот и сбор тензоров
        tensors = RandomStressTensor.build_tensor_batch(
            ps_samples,
            trend_samples,
            plunge_samples,
            rake_samples
        )

        # Сохранение в атрибуты
        self.ensemble_principal_stresses = ps_samples
        self.ensemble_trends             = trend_samples
        self.ensemble_plunges            = plunge_samples
        self.ensemble_rakes              = rake_samples
        self.ensemble_tensors            = tensors

        return (tensors,
                ps_samples,
                trend_samples,
                plunge_samples,
                rake_samples)
