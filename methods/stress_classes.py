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
    

class FrictionalStressTensor(StressTensor):
    """
    Тензор в фрикционном равновесии (Mohr–Coulomb) с известным σ₁ и вычисленным σ₃.

    Инициализация:
      - s1: float
          Максимальное главное напряжение σ₁ (MPa).
      - mode: str, длина=3, буквы 'x','y','z'
          Коды осей, куда направлены σ₁, σ₂, σ₃ (от макс. к мин.):
          e.g. 'zxy' → σ₁→z, σ₂→x, σ₃→y.
      - ratio: float ∈ [0,1]
          Отношение (σ₂-σ₃)/(σ₁-σ₃)  0→σ₂=σ₃, 1→σ₂=σ₁
      - pore_pressure: float
          Поровое давление p (MPa).
      - mu: float
          Коэффициент трения μ.
      - C: float
          Сцепление C (MPa).
      - trend, plunge, rake: float
          Углы Эйлера для ориентации тензора, как в StressTensor.
    """
    def __init__(self,
                 s1: float,
                 mode: str,
                 ratio: float,
                 pore_pressure: float,
                 mu: float,
                 cohesion: float,
                 trend: float = 0.0,
                 plunge: float = 0.0,
                 rake: float = 0.0):
        
        # 0) Сохраняем входные параметры
        self.s1 = float(s1)
        self.mode = mode.lower()
        self.ratio = float(ratio)
        self.pore_pressure = float(pore_pressure)
        self.mu = float(mu)
        self.C = float(cohesion)
        self.trend = trend
        self.plunge = plunge
        self.rake = rake

        # 1) Эффективное σ₁
        s1_eff = self.s1 - self.pore_pressure

        # 2) По строгому условию касания окружности Мора:
        #    σ₁' = [σ₃'(√(1+μ²)+μ) + 2C] / (√(1+μ²) - μ)
        #    → решаем обратную задачу: находим σ₃' из известного σ₁'
        #    σ₃' = [ (√(1+μ²)-μ)·σ₁' - 2C ] / (√(1+μ²)+μ)
        coeff = np.sqrt(1 + self.mu**2)
        s3_eff = ( (coeff - self.mu) * s1_eff - 2*self.C ) / (coeff + self.mu)
        s3 = s3_eff + self.pore_pressure

        # 3) Теперь: σ₂ = σ₃ + ratio * (σ₁ - σ₃)
        s2 = s3 + self.ratio * (self.s1 - s3)

        # 4) Собираем главные в локальный вектор [σ₁,σ₂,σ₃] по mode
        ordered = [self.s1, s2, s3]
        local_ps = np.empty(3, dtype=float)
        for i, ax in enumerate(('x','y','z')):
            j = self.mode.index(ax)
            local_ps[i] = ordered[j]

        # 5) Дальнейшее — как в StressTensor
        super().__init__(principal_stresses=local_ps,
                         trend=trend, plunge=plunge, rake=rake)
        
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
                          N: int = 100,
                          stress_std_frac = 0.0,
                          angle_std_deg = 0.0,
                          random_s2: bool = False,
                          random_seed: int = None):
        """
        Генерация ансамбля N тензоров с варьируемым σ₁ (и, при random_s2, σ₂).

        Параметры:
        - N: int — число реализаций
        - s1_std_frac: float или (3,) — STD σ₁ (в долях от self.s1)
        - angle_std_deg: float или (3,) — STD для (trend, plunge, rake)
        - random_s2: bool — если True, σ₂~Uniform(σ₃,σ₁), иначе по fixed ratio
        - random_seed: int|None — seed для воспроизводимости

        Возвращает (как у RandomStressTensor):
        - tensors: (N,3,3)
        - principal_stresses: (N,3)  — локальные [σₓ,σᵧ,σ_z]
        - trends:   (N,)
        - plunges:  (N,)
        - rakes:    (N,)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # 1) Семплируем σ₁ вокруг self.s1
        s1_samples = np.random.normal(
            loc=self.s1,
            scale=self.s1 * np.atleast_1d(stress_std_frac).min(),
            size=N
        )
        s1_eff = s1_samples - self.pore_pressure

        # 2) Из σ₁_eff восстанавливаем σ₃_eff той же формулой
        coeff = np.sqrt(1 + self.mu**2)
        s3_eff_arr = ( (coeff - self.mu)*s1_eff - 2*self.C ) / (coeff + self.mu)
        s3_samples = s3_eff_arr + self.pore_pressure

        # 3) σ₂: либо random, либо по ratio
        if random_s2:
            s2_samples = np.random.uniform(low=s3_samples, high=s1_samples, size=N)
        else:
            # σ₂ = σ₃ + ratio * (σ₁ - σ₃)
            s2_samples = s3_samples + self.ratio * (s1_samples - s3_samples)

        # 4) Семплируем углы
        if isinstance(angle_std_deg, (int, float)):
            t_std = p_std = r_std = angle_std_deg
        else:
            t_std, p_std, r_std = angle_std_deg

        trends  = np.random.normal(self.trend,  scale=t_std, size=N)
        plunges = np.random.normal(self.plunge, scale=p_std, size=N)
        rakes   = np.random.normal(self.rake,   scale=r_std, size=N)

        # 5) Составляем principal_stresses_batch (N,3) по mode
        ordered_arr = np.stack([s1_samples, s2_samples, s3_samples], axis=1)
        ps_batch = np.empty_like(ordered_arr)
        for i, ax in enumerate(('x','y','z')):
            j = self.mode.index(ax)
            ps_batch[:, i] = ordered_arr[:, j]

        # 6) Поворачиваем в глобальную систему
        tensors = FrictionalStressTensor.build_tensor_batch(
            ps_batch, trends, plunges, rakes
        )

        # 7) Сохраняем и возвращаем
        self.ensemble_tensors            = tensors
        self.ensemble_principal_stresses = ps_batch
        self.ensemble_trends             = trends
        self.ensemble_plunges            = plunges
        self.ensemble_rakes              = rakes

        return tensors, ps_batch, trends, plunges, rakes
