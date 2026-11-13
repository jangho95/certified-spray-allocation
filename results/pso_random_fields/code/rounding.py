from __future__ import annotations
import numpy as np

def _prepare(x_relax, budget, cmax):
    xr = np.clip(np.asarray(x_relax, dtype=float), 1.0, float(cmax))
    x = np.clip(np.floor(xr + 1e-10).astype(int), 1, cmax)
    return (xr, x, int(budget - int(x.sum())), xr - np.floor(xr))

def round_indexed(x_relax, budget: int, cmax: int) -> np.ndarray:
    xr, x, diff, frac = _prepare(x_relax, budget, cmax)
    idx = np.arange(len(x))
    if diff > 0:
        for i in np.lexsort((idx, -frac)):
            if diff == 0:
                break
            if x[i] < cmax:
                x[i] += 1
                diff -= 1
    elif diff < 0:
        for i in np.lexsort((idx, frac)):
            if diff == 0:
                break
            if x[i] > 1:
                x[i] -= 1
                diff += 1
    if diff != 0:
        raise RuntimeError(f'Could not round to budget; remaining diff={diff}')
    return x

def boundary_ties(x_relax, budget: int, cmax: int, tol: float=1e-09) -> dict:
    xr, x, diff, frac = _prepare(x_relax, budget, cmax)
    if diff == 0:
        return dict(k=0, cut=None, exact_at_cut=0, near_cut=0)
    elig = frac[x < cmax] if diff > 0 else frac[x > 1]
    order = np.sort(elig)[::-1] if diff > 0 else np.sort(elig)
    k = abs(diff)
    cut = float(order[k - 1])
    return dict(k=k, cut=cut, exact_at_cut=int(np.sum(elig == cut)), near_cut=int(np.sum(np.abs(elig - cut) <= tol)))
