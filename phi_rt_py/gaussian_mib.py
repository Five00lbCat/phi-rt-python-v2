
import numpy as np

def _logdet_psd(M):
    w = np.linalg.eigvalsh(M)
    w = np.clip(w, 1e-12, None)
    return np.sum(np.log(w))

def mutual_info_gaussian(C, A_idx, B_idx):
    A = C[np.ix_(A_idx, A_idx)]
    B = C[np.ix_(B_idx, B_idx)]
    logdetA = _logdet_psd(A)
    logdetB = _logdet_psd(B)
    logdetC = _logdet_psd(C)
    return 0.5 * (logdetA + logdetB - logdetC)

def _spectral_split(C):
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    R = C / (d[:, None] * d[None, :])
    D = np.diag(R.sum(axis=1))
    L = D - R
    w, V = np.linalg.eigh(L)
    if len(w) < 2:
        idx = np.arange(C.shape[0])
        mid = len(idx)//2
        return idx[:mid], idx[mid:]
    f = V[:, 1]
    A = np.where(f >= 0)[0]
    B = np.where(f < 0)[0]
    if len(A) == 0 or len(B) == 0:
        idx = np.arange(C.shape[0])
        mid = len(idx)//2
        A, B = idx[:mid], idx[mid:]
    return A, B

def _kl_refine(C, A, B, max_passes=5):
    A = list(A); B = list(B)
    improved = True
    passes = 0
    best_val = mutual_info_gaussian(C, A, B)
    while improved and passes < max_passes:
        improved = False
        gain_best = 0.0; move = None
        for i in list(A):
            if len(A) <= 1: continue
            A2 = [x for x in A if x != i]; B2 = B + [i]
            val = mutual_info_gaussian(C, A2, B2)
            gain = best_val - val
            if gain > gain_best:
                gain_best = gain; move = ("A2B", i, A2, B2, val)
        for j in list(B):
            if len(B) <= 1: continue
            A2 = A + [j]; B2 = [x for x in B if x != j]
            val = mutual_info_gaussian(C, A2, B2)
            gain = best_val - val
            if gain > gain_best:
                gain_best = gain; move = ("B2A", j, A2, B2, val)
        if move and gain_best > 1e-9:
            _, _, A, B, best_val = move
            improved = True
        passes += 1
    return np.array(A, dtype=int), np.array(B, dtype=int), float(best_val)

def _all_bipartitions(N):
    import itertools
    nodes = list(range(N))
    for r in range(1, N//2 + 1):
        for A in itertools.combinations(nodes[1:], r-1):
            A = (0,) + A
            B = tuple(sorted(set(nodes) - set(A)))
            yield tuple(sorted(A)), B

def mib_bruteforce(C):
    N = C.shape[0]
    best_v = float("inf"); best = None
    for A, B in _all_bipartitions(N):
        v = mutual_info_gaussian(C, list(A), list(B))
        if v < best_v:
            best_v, best = v, (A, B)
    return np.array(best[0], dtype=int), np.array(best[1], dtype=int), float(best_v)

def heuristic_mib(C, brute_maxN=12):
    N = C.shape[0]
    if N <= brute_maxN:
        A, B, v = mib_bruteforce(C)
        return {"phi": v, "A": A.tolist(), "B": B.tolist(), "method": "brute"}
    A, B = _spectral_split(C)
    A, B, v = _kl_refine(C, A, B, max_passes=8)
    return {"phi": v, "A": A.tolist(), "B": B.tolist(), "method": "kl"}
