
# phi-rt-python-v2

A practical, **real-time** Φ-like integration estimator for multivariate time series (Gaussian mode),
with a **rolling window**, **MIB search (brute-force for small N; spectral + KL swaps for large N)**,
**negative controls**, and a **streaming CLI**.

> ⚠️ This is an *engineering approximation*, not an exact IIT-3.0 calculator.
> It estimates *integration across a bipartition* using Gaussian mutual information.

## Install (local)
```bash
pip install -e .
```

## Quick start (Python)
```python
import numpy as np
from phi_rt_py import PhiRT

rt = PhiRT(window=4096, mode="gaussian", brute_maxN=12)
for t in range(10000):
    x = np.random.randn(16)  # replace with your sample vector
    out = rt.update(x)       # returns None except every 'interval' steps
    if out is not None:
        print(out)           # {'phi': 1.23, 'A': [...], 'B': [...], 'method': 'kl'}
```

## Streaming CLI
Read CSV rows (floats per line) from stdin and print JSON updates:

```bash
python -m phi_rt_py.stream_cli --interval 1000 --window 4096 < data.csv
# or live:
some_generator | python -m phi_rt_py.stream_cli --interval 512 --window 4096
```

**Negative control** (time shuffle on the window):
```bash
python -m phi_rt_py.stream_cli --interval 1000 --window 4096 --shuffle-control
```

## Validation demos
```bash
python -m phi_rt_py.validations --demo ablation
python -m phi_rt_py.validations --demo shuffle
```

## What it computes
For a covariance `C` over the current window, it finds a bipartition A|B and computes:
`I(A;B) = 0.5 * log(det(C_A) det(C_B) / det(C))`

The **MIB** (minimum-information bipartition) is the partition minimizing `I(A;B)`.
Small-N uses brute-force; otherwise we warm-start with a **spectral split** and refine via **Kernighan–Lin (KL) swaps**.

## Roadmap
- Discrete/TPM mode with count-based TPM and D_KL divergence.
- CUDA kernels for batched logdets/MI eval.
- PCI-style perturbation complexity proxy.
