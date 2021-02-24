"""経験分布関数の作成とそこからのサンプル
"""

import numpy as np
from kernel_density_estimation import KDE


class EmpiricalDist:
    def __init__(self, minmax=(0., 1.), dx=0.01, kde_width=0.05):
        """EmpiricalDist

        Args:
            minmax (tuple, optional): 密度関数の定義域. Defaults to (0., 1.).
            dx (float, optional): 密度関数の量子化幅. Defaults to 0.01.
            kde_width (float, optional): KDEパラメータ(ガウス分布の標準偏差). Defaults to 0.05.
        """
        self.minmax = minmax
        self.xs = np.arange(minmax[0], minmax[1]+dx, dx)
        self.dxs = self.xs[1:] - self.xs[:-1]
        self.kde = KDE(kernel='gauss')
        self.kde_width = kde_width
        self.rnd_table = None
        self._X = None

    def _density_estimation(self):
        if self._X is None:
            raise ValueError('sample data does not set yet')
        px = self.kde.kde(self._X, self.xs, self.kde_width)
        self.px = px
        return px

    def _set_empirical_dist(self):
        if self._X is None:
            raise ValueError('sample data does not set yet')
        px_dx = self.px[1:] * self.dxs
        norm_const = px_dx.sum()
        empirical_dist = np.cumsum(px_dx) / norm_const
        self.empirical_dist = empirical_dist
        rnd_table = np.array(
            [self.xs[:-1] + self.dxs / 2., empirical_dist]).T
        self.rnd_table = rnd_table
        return empirical_dist

    def set_sample_data(self, X):
        """経験分布関数の基になるサンプルデータをセット

        Args:
            X ([type]): [description]
        """
        self._X = X
        _ = self._density_estimation()
        _ = self._set_empirical_dist()

    def sample(self, size=1):
        """経験分布関数からサンプルを取得
        """
        us = np.random.rand(size)

        def f_sample(u, rt):
            return rt[rt[:, 1] > u, 0].min()
        samples = np.array([f_sample(u, self.rnd_table) for u in us])
        return samples

    def get_px(self):
        """推定確率密度の取得

        Returns:
            px: 確率密度(量子化)
            xs: 確率密度の量子化座標
        """
        return self.px, self.xs
