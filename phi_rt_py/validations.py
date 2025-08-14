
import argparse, numpy as np, json
from .gaussian_mib import heuristic_mib

def synth_ablation(T=8000, N=12, cross=0.2, steps=5, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((T, N//2))
    B = rng.standard_normal((T, N - N//2))
    k = min(A.shape[1], B.shape[1])
    Xs = []
    for s in range(steps+1):
        frac = cross * (1 - s/steps)
        Bm = B.copy()
        Bm[:, :k] = Bm[:, :k] + frac * A[:, :k]
        Xs.append(np.concatenate([A, Bm], axis=1))
    return Xs

def demo_ablation():
    Xs = synth_ablation()
    vals = []
    for X in Xs:
        Xc = X - X.mean(axis=0, keepdims=True)
        C = (Xc.T @ Xc) / (X.shape[0]-1)
        out = heuristic_mib(C)
        vals.append(out['phi'])
    print(json.dumps({"ablation_phi": vals}))

def demo_shuffle():
    rng = np.random.default_rng(0)
    T, N = 10000, 16
    X = rng.standard_normal((T, N))
    k = N//2
    X[:, k:] += 0.2 * X[:, :k]
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / (T-1)
    base = heuristic_mib(C)
    idx = np.arange(T); rng.shuffle(idx)
    Xs = X[idx]
    Xcs = Xs - Xs.mean(axis=0, keepdims=True)
    Cs = (Xcs.T @ Xcs) / (T-1)
    ctrl = heuristic_mib(Cs)
    print(json.dumps({"baseline": base, "shuffle_control": ctrl}))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', choices=['ablation','shuffle'], required=True)
    args = ap.parse_args()
    if args.demo == 'ablation':
        demo_ablation()
    else:
        demo_shuffle()

if __name__ == '__main__':
    main()
