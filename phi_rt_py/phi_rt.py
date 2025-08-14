import numpy as np
from .rolling import RollingCov
from .gaussian_mib import heuristic_mib
from .var1 import VAR1Online

class PhiRT:
    def __init__(self, window=4096, mode='gaussian', brute_maxN=12, interval=512):
        assert mode in ('gaussian', 'var1'), "mode must be 'gaussian' or 'var1'"
        self.mode = mode
        self.window = window
        self.brute_maxN = brute_maxN
        self.interval = interval
        self._N = None
        self._t = 0
        self._rcov = None
        self._var1 = None

    def update(self, x):
        x = np.asarray(x, float)
        if self._N is None:
            self._N = x.shape[0]
            self._rcov = RollingCov(self._N, window=self.window)
            self._var1 = VAR1Online(self._N, window=self.window)
        self._rcov.update(x)
        self._var1.update(x)
        self._t += 1
        if self._t % self.interval != 0:
            return None
        return self.current()

    def current(self, shuffle_control=False):
        if self.mode == 'gaussian':
            C = self._rcov.covariance()
            if C is None:
                return None
            if shuffle_control:
                X = self._rcov.sample_matrix().copy()
                T, N = X.shape
                rng = np.random.default_rng()
                for j in range(N):
                    rng.shuffle(X[:, j])
                Xc = X - X.mean(axis=0, keepdims=True)
                C = (Xc.T @ Xc) / (T - 1)
            return heuristic_mib(C, brute_maxN=self.brute_maxN)

        else:  # VAR(1)
            stats = self._var1.stats()
            if stats is None:
                return None
            Sigma = stats['Sigma_dyn']
            if shuffle_control:
                # Time-scramble: destroy Y_{t-1} -> Y_t mapping by permuting time independently per channel
                # Recompute VAR stats from a scrambled copy of the window
                # (cheap proxy: treat Sigma as mostly noise when order is broken)
                # We approximate control by using only residual covariance:
                Sigma = stats['Sigma_eps']
            return heuristic_mib(Sigma, brute_maxN=self.brute_maxN)
