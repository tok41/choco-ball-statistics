"""Kernel密度推定
- Parzen窓 : kde_parzen(X, tgt_point, h)
- Gaussian KDE : kde_gauss(X, tgt_point, h)
"""

import numpy as np
from scipy import stats


class KDE:
    def __init__(self, kernel='gauss'):
        if kernel in ['gauss']:
            self.f = self._kde_gauss
        else:
            self.f = self._kde_parzen

    def _kde_parzen(self, X, tgt_point, h):
        def parzen_window(x, xn, h):
            d = x - xn
            if np.abs(d) < h / 2.:
                ret = 1
            else:
                ret = 0
            return ret
        n = len(X)
        return sum([parzen_window(tgt_point, xn, h) / h for xn in X]) / n
    
    def _kde_gauss(self, X, tgt_point, h):
        def gauss_kernel(x, xn, h):
            ret = stats.norm(loc=x, scale=h).pdf(xn)
            return ret
        # n = len(X)
        # return sum([gauss_kernel(tgt_point, xn, h) for xn in X])/n
        return (stats.norm(loc=tgt_point, scale=h).pdf(X)).mean()

    def kde(self, X, xs, h):
        """Kernel Density Estimation

        Args:
            X (array-like): データ点群
            xs (array-like): 密度推定する位置
            h (float): 窓幅

        Returns:
            [numpy.array]: 推定確率密度
        """
        px = np.array([self.f(X, x, h) for x in xs])
        self.px = px
        self.xs = xs
        return px
    
    def mode(self):
        idx = np.argmax(self.px)+1
        x_mode = self.xs[idx]
        return x_mode
