from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
import osqp
import scipy.sparse as sp

@dataclass
class QPResult:
    primal_value: float
    dual_lower: float
    status: str
    elapsed_s: float
    x: np.ndarray
    dual_y: np.ndarray
    duality_gap: float

def dense_A(stamps, m: int, n: int) -> np.ndarray:
    A = np.zeros((m, n), dtype=float)
    for j, s in enumerate(stamps):
        for c, w in zip(s.cells, s.weights):
            A[c, j] += w
    return A

def solve_box_qp_relaxation(A: np.ndarray, initial: np.ndarray, budget: int, mass: float, cmax: int, *, eps: float=1e-08) -> QPResult:
    m, n = A.shape
    mean = (float(initial.sum()) + budget * mass) / m
    r0 = initial - mean
    gram = A.T @ A
    b = A.T @ r0
    const = float(r0 @ r0) / m
    H = 2.0 / m * gram
    q = 2.0 / m * b
    G = sp.vstack([sp.csc_matrix(np.ones((1, n))), sp.identity(n)]).tocsc()
    lo = np.concatenate([[float(budget)], np.full(n, 1.0)])
    hi = np.concatenate([[float(budget)], np.full(n, float(cmax))])
    prob = osqp.OSQP()
    prob.setup(sp.csc_matrix(H), q, G, lo, hi, eps_abs=eps, eps_rel=eps, max_iter=400000, polish=True, verbose=False)
    t0 = time.perf_counter()
    res = prob.solve()
    elapsed = time.perf_counter() - t0
    primal = float(res.info.obj_val + const)
    y = np.asarray(res.y, dtype=float)
    G_dense = G.toarray()
    v = q + G_dense.T @ y
    try:
        z = np.linalg.solve(H, v)
        resid = float(np.linalg.norm(H @ z - v))
    except np.linalg.LinAlgError:
        z, resid = (None, float('inf'))
    rel_resid = resid / (1.0 + float(np.linalg.norm(v)))
    if z is not None and rel_resid <= 1e-10:
        dual = const - 0.5 * float(v @ z)
        dual -= float(hi @ np.maximum(y, 0.0))
        dual += float(lo @ np.maximum(-y, 0.0))
    else:
        dual = 0.0
    dual_lower = max(0.0, min(float(dual), primal))
    return QPResult(primal_value=primal, dual_lower=dual_lower, status=res.info.status, elapsed_s=elapsed, x=np.asarray(res.x, dtype=float), dual_y=y, duality_gap=primal - dual_lower)

def largest_remainder_round(x_relax: np.ndarray, budget: int, cmax: int) -> np.ndarray:
    xr = np.clip(np.asarray(x_relax, dtype=float), 1.0, float(cmax))
    x = np.floor(xr + 1e-10).astype(int)
    x = np.clip(x, 1, cmax)
    diff = int(budget - int(x.sum()))
    frac = xr - np.floor(xr)
    if diff > 0:
        order = np.argsort(-frac)
        for idx in order:
            if diff == 0:
                break
            if x[idx] < cmax:
                x[idx] += 1
                diff -= 1
    elif diff < 0:
        order = np.argsort(frac)
        for idx in order:
            if diff == 0:
                break
            if x[idx] > 1:
                x[idx] -= 1
                diff += 1
    if diff != 0:
        raise RuntimeError(f'Could not round to budget; remaining diff={diff}')
    return x
