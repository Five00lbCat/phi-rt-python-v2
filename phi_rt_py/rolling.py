
import numpy as np
from collections import deque

class RollingCov:
    def __init__(self, N, window=4096):
        self.N = N
        self.window = window
        self.buf = deque(maxlen=window)
        self._sum = np.zeros(N, dtype=float)
        self._sum2 = np.zeros((N, N), dtype=float)

    def update(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.N:
            raise ValueError(f"Expected vector of size {self.N}, got {x.shape}")
        if len(self.buf) == self.window:
            old = self.buf[0]
            self._sum -= old
            self._sum2 -= np.outer(old, old)
        self.buf.append(x)
        self._sum += x
        self._sum2 += np.outer(x, x)

    def covariance(self):
        T = len(self.buf)
        if T < 2:
            return None
        mean = self._sum / T
        C = (self._sum2 - T * np.outer(mean, mean)) / (T - 1)
        return C

    def sample_matrix(self):
        if not self.buf:
            return None
        X = np.stack(self.buf, axis=0)
        return X
