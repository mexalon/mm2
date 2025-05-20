import numpy as np
from scipy.stats import weibull_min

class FractureSeed:
    def __init__(self, strike=0.0, dip=0.0, mu=0.6, cohesion=0.01):
        """
        Базовый класс: задаёт ориентацию «материнской» трещины.
        Хранит strike/dip, mu/cohesion, рассчитывает rotation_matrix и normal.
        """
        self.strike = strike
        self.dip = dip
        self.mu = mu
        self.cohesion = cohesion

        # Поворотная матрица (3×3) и нормаль в глобальной системе
        R = FractureSeed.build_rotation_matrix(
            np.array([self.strike]), np.array([self.dip])
        )[0]
        self.rotation_matrix = R
        self.normal = R @ np.array([0, 0, -1])

    @staticmethod
    def build_rotation_matrix(strike_batch, dip_batch):
        """
        Строит n матриц поворота (n,3,3) из массивов strike_batch и dip_batch [градусы]:
          1) dip вокруг локальной Y,
          2) strike вокруг глобальной Z (часовая стрелка → минус).
        """
        theta_v = np.radians(dip_batch)
        theta_h = np.radians(-strike_batch)
        n = len(strike_batch)

        # dip
        Rv = np.zeros((n,3,3))
        Rv[:,0,0] = np.cos(theta_v)
        Rv[:,0,2] = np.sin(theta_v)
        Rv[:,1,1] = 1.
        Rv[:,2,0] = -np.sin(theta_v)
        Rv[:,2,2] = np.cos(theta_v)

        # strike
        Rh = np.zeros((n,3,3))
        Rh[:,0,0] = np.cos(theta_h)
        Rh[:,0,1] = -np.sin(theta_h)
        Rh[:,1,0] = np.sin(theta_h)
        Rh[:,1,1] = np.cos(theta_h)
        Rh[:,2,2] = 1.

        # последовательное применение: сначала dip, затем strike
        return np.einsum('nij,njk->nik', Rh, Rv)




class RandomFractureSeed(FractureSeed):
    
    @staticmethod
    def build_normal_batch(strike_batch, dip_batch):
        """
        Генерирует массив нормалей (n,3) для плоскостей с указанными strike/dip.
        Локальный вектор нормали = [0,0,-1].
        """
        R = FractureSeed.build_rotation_matrix(strike_batch, dip_batch)
        n0 = np.array([0,0,-1])
        return np.einsum('nij,j->ni', R, n0)

    def generate_ensemble(self,
                          N: int,
                          strike_std: float = 0,
                          dip_std: float = 0,
                          mu_weibull_loc: float = None, # если не передано, то будет просто мю
                          mu_weibull_scale: float = 0, # если 0, будет постоянная величина
                          cohesion_weibull_loc: float = None, 
                          cohesion_weibull_scale: float = 0,
                          random_seed: int = None):
        """
        Ансамбль по нормальным отклонениям углов и Weibull-параметрам:
         - strike, dip ~ N(self.strike, strike_std), N(self.dip, dip_std)
         - mu, cohesion ~ Weibull
         - normals из strike/dip
        Результаты сохраняются в:
          self.ensemble_normals, self.ensemble_strikes, self.ensemble_dips,
          self.ensemble_mu, self.ensemble_cohesion
        и возвращаются в том же порядке.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # 1) Weibull-сэмплирование
        mu_loc = mu_weibull_loc or self.mu
        ensemble_mu = weibull_min.rvs(loc=mu_loc, scale=mu_weibull_scale, c=1.8, size=N)

        coh_loc = cohesion_weibull_loc or self.cohesion
        ensemble_cohesion = weibull_min.rvs(loc=coh_loc, scale=cohesion_weibull_scale, c=1.8, size=N)

        # 2) Углы из нормального распределения
        ensemble_strikes = np.random.normal(self.strike, strike_std, size=N)
        ensemble_dips    = np.random.normal(self.dip,    dip_std,    size=N)

        # 3) Нормали по strike/dip
        ensemble_normals = RandomFractureSeed.build_normal_batch(ensemble_strikes, ensemble_dips)

        # Сохраняем в атрибуты
        self.ensemble_normals  = ensemble_normals
        self.ensemble_strikes  = ensemble_strikes
        self.ensemble_dips     = ensemble_dips
        self.ensemble_mu       = ensemble_mu
        self.ensemble_cohesion = ensemble_cohesion

        return (ensemble_normals,
                ensemble_strikes,
                ensemble_dips,
                ensemble_mu,
                ensemble_cohesion)


class FisherFractureSeed(FractureSeed):
    def generate_ensemble(self,
                          N: int,
                          kappa: float = 5.0,
                          mu_weibull_loc: float = None, # если не передано, то будет просто мю
                          mu_weibull_scale: float = 0,  # если 0, будет постоянная величина
                          cohesion_weibull_loc: float = None,
                          cohesion_weibull_scale: float = 0, 
                          random_seed: int = None): # type: ignore
        """
        Ансамбль по распределению Фишера + Weibull:
         1) mu, cohesion ~ Weibull или просто постоянные
         2) θ по обратному CDF Фишера (kappa), φ ~ U[0,2π)
         3) локальные normals вокруг [0,0,-1]
         4) поворот self.rotation_matrix → normals_global
         5) strike из линии пересечения с горизонтом s=n×e_z
         6) dip из nz
        Результаты сохраняются в:
          self.ensemble_normals, self.ensemble_strikes, self.ensemble_dips,
          self.ensemble_mu, self.ensemble_cohesion
        и возвращаются в том же порядке.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # 1) Weibull-сэмплирование к-та трения и когезии. Если не переданы параметры scale то берутся значения для материнской трещины      
        mu_loc = mu_weibull_loc or self.mu
        ensemble_mu = weibull_min.rvs(loc=mu_loc, scale=mu_weibull_scale, c=1.8, size=N)

        coh_loc = cohesion_weibull_loc or self.cohesion
        ensemble_cohesion = weibull_min.rvs(loc=coh_loc, scale=cohesion_weibull_scale, c=1.8, size=N)

        # 2) Fisher θ и φ
        u = np.random.rand(N)
        cos_t = (1/kappa)*np.log(np.exp(kappa) - u*(np.exp(kappa)-np.exp(-kappa)))
        theta = np.arccos(cos_t)
        phi   = np.random.uniform(0,2*np.pi, size=N)

        # 3) Локальные нормали (θ=0 → [0,0,-1])
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = -np.cos(theta)
        normals_local = np.vstack([x,y,z]).T

        # 4) Поворот в глобальную систему
        R = self.rotation_matrix
        ensemble_normals = normals_local @ R.T

        # 5) strike: направление линии пересечения
        nx, ny, nz = ensemble_normals.T
        s_x, s_y  = ny, -nx
        ensemble_strikes = (np.degrees(np.arctan2(s_x, s_y)) + 360) % 360

        # 6) dip: угол падения плоскости
        ensemble_dips = np.degrees(np.arccos(-nz))

        # Сохраняем в атрибуты
        self.ensemble_normals  = ensemble_normals
        self.ensemble_strikes  = ensemble_strikes
        self.ensemble_dips     = ensemble_dips
        self.ensemble_mu       = ensemble_mu
        self.ensemble_cohesion = ensemble_cohesion

        return (ensemble_normals,
                ensemble_strikes,
                ensemble_dips,
                ensemble_mu,
                ensemble_cohesion)

