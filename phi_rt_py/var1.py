import numpy as np

class VAR1Online:
    """Rolling-window VAR(1) using covariance and lagged covariance.

    Tracks:
      C00 = cov(Y_t) over window
      C10 = cov(Y_t, Y_{t-1}) over window

    VAR coeff: A = C10 @ C00^{-1}
    Residual covariance: Sigma_eps = C00 - A @ C00 @ A.T   (Yule–Walker)
    We return a 'dynamic' covariance to measure integration on:
      Sigma_dyn = A @ C00 @ A.T + Sigma_eps  (one-step predictive)
    """
    def __init__(self, N, window=4096):
        self.N = N
        self.window = window
        self.buf = []
        self._sum0 = np.zeros(N)
        self._sum1 = np.zeros(N)
        self._sum00 = np.zeros((N, N))
        self._sum10 = np.zeros((N, N))

    def update(self, x):
        x = np.asarray(x, float)
        if len(self.buf) == 0:
            self.buf.append(x)
            return
        # new pair (x_prev, x_curr)
        xp = self.buf[-1]
        self.buf.append(x)

        if len(self.buf) > self.window:
            # remove (old_prev, old_curr) pair contributions
            old_curr = self.buf[-self.window-1]
            old_prev = self.buf[-self.window-2] if len(self.buf) > self.window+1 else None
            # remove only when we have a valid previous for the old_curr
            if old_prev is not None:
                self._sum0 -= old_curr
                self._sum1 -= old_prev
                self._sum00 -= np.outer(old_curr, old_curr)
                self._sum10 -= np.outer(old_curr, old_prev)

        # add contributions for (xp -> x)
        self._sum0 += x
        self._sum1 += xp
        self._sum00 += np.outer(x, x)
        self._sum10 += np.outer(x, xp)

    def stats(self):
        # number of valid pairs is len(buf)-1, but minus the ones dropped by window
        T = min(len(self.buf)-1, self.window-1) if len(self.buf) > 1 else 0
        if T < 2:
            return None
        mean0 = self._sum0 / T
        mean1 = self._sum1 / T
        C00 = (self._sum00 - T * np.outer(mean0, mean0)) / (T - 1)
        C10 = (self._sum10 - T * np.outer(mean0, mean1)) / (T - 1)
        # Regularize for stability
        eps = 1e-6
        C00r = C00 + eps * np.eye(self.N)
        A = C10 @ np.linalg.pinv(C00r)
        Sigma_eps = C00 - A @ C00 @ A.T
        # ensure SPD-ish
        Sigma_eps = 0.5 * (Sigma_eps + Sigma_eps.T)
        # dynamic covariance to evaluate MI on:
        Sigma_dyn = A @ C00 @ A.T + Sigma_eps
        Sigma_dyn = 0.5 * (Sigma_dyn + Sigma_dyn.T)
        return {"A": A, "C00": C00, "Sigma_dyn": Sigma_dyn, "Sigma_eps": Sigma_eps}
