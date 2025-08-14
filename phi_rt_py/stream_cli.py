
import sys, json, argparse, numpy as np
from .phi_rt import PhiRT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=4096)
    ap.add_argument('--interval', type=int, default=512)
    ap.add_argument('--brute-maxN', type=int, default=12)
    ap.add_argument('--shuffle-control', action='store_true')
    args = ap.parse_args()

    rt = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = [float(p) for p in line.split(',')]
        x = np.array(parts, dtype=float)
        if rt is None:
            rt = PhiRT(window=args.window, brute_maxN=args.brute_maxN, interval=args.interval)
        out = rt.update(x)
        if out is not None:
            if args.shuffle_control:
                ctrl = rt.current(shuffle_control=True)
                out = {"live": out, "shuffle_control": ctrl}
            print(json.dumps(out))
            sys.stdout.flush()

if __name__ == '__main__':
    main()
