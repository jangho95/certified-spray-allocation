from __future__ import annotations
from package_paths import load_json
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
import osqp
import scipy.sparse as sp
THREAD_ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
DEFAULT_RANGE_TOL = 1e-10

def solve_qp_audit(A, initial, budget, mass, cmax, *, eps_abs=1e-08, eps_rel=1e-08, polish=True, max_iter=400000, range_tol=DEFAULT_RANGE_TOL):
    m, n = A.shape
    t0 = time.perf_counter()
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
    t1 = time.perf_counter()
    prob = osqp.OSQP()
    prob.setup(sp.csc_matrix(H), q, G, lo, hi, eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter, polish=polish, verbose=False)
    t2 = time.perf_counter()
    res = prob.solve()
    t3 = time.perf_counter()
    primal = float(res.info.obj_val + const)
    x = np.asarray(res.x, dtype=float)
    y = np.asarray(res.y, dtype=float)
    G_dense = G.toarray()
    v = q + G_dense.T @ y
    try:
        z = np.linalg.solve(H, v)
        resid = float(np.linalg.norm(H @ z - v))
    except np.linalg.LinAlgError:
        z, resid = (None, float('inf'))
    rel_resid = resid / (1.0 + float(np.linalg.norm(v)))
    if z is not None:
        raw = const - 0.5 * float(v @ z)
        raw -= float(hi @ np.maximum(y, 0.0))
        raw += float(lo @ np.maximum(-y, 0.0))
    else:
        raw = float('nan')
    t4 = time.perf_counter()
    out = dict(status=str(res.info.status), status_polish=int(res.info.status_polish), iter=int(res.info.iter), primal=primal, dual_raw=raw, range_rel_resid=rel_resid, osqp_prim_res=float(res.info.prim_res), osqp_dual_res=float(res.info.dual_res), osqp_primal_dual_gap=float(res.info.duality_gap), budget_viol=abs(float(x.sum()) - budget), box_viol=float(max(0.0, np.max(1.0 - x), np.max(x - cmax))), kkt_stationarity_inf=float(np.max(np.abs(H @ x + q + G.T @ y))), x_min=float(x.min()), x_max=float(x.max()), n_lower_active=int(np.sum(np.abs(x - 1.0) < 1e-05)), n_upper_active=int(np.sum(np.abs(x - cmax) < 1e-05)), t_gram=t1 - t0, t_setup=t2 - t1, t_solve=t3 - t2, t_dual=t4 - t3, settings=dict(eps_abs=eps_abs, eps_rel=eps_rel, polish=polish, max_iter=max_iter, range_tol=range_tol), x=x, y=y)
    out.update(lower_bound_at(out, range_tol))
    return out

def lower_bound_at(audit: dict, range_tol: float) -> dict:
    raw, primal = (audit['dual_raw'], audit['primal'])
    fallback = not (np.isfinite(raw) and audit['range_rel_resid'] <= range_tol)
    dual = 0.0 if fallback else raw
    L = max(0.0, min(float(dual), primal))
    return dict(dual_lower=L, fallback=fallback, clip_low=not fallback and raw < 0.0, clip_high=not fallback and raw > primal)

def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
RECORD_KEYS = ['stage', 'experiment', 'case', 'input', 'input_sha256', 'M', 'N', 'B', 'Cmax', 'kernel', 'sigma', 'spacing', 'boundary', 'seed', 'solver', 'settings', 'status', 'U', 'L', 'abs_gap', 'rel_gap', 'residuals', 'times', 'peak_rss_mb']

def make_record(**kw) -> dict:
    rec = {k: kw.pop(k, None) for k in RECORD_KEYS}
    if rec['U'] is not None and rec['L'] is not None:
        rec['abs_gap'] = rec['U'] - rec['L']
        rec['rel_gap'] = rec['abs_gap'] / rec['U'] if rec['U'] > 0 else None
    rec['peak_rss_mb'] = peak_rss_mb()
    rec['extra'] = kw
    return rec

def append_records(path, records) -> None:
    with open(path, 'a') as fh:
        for r in records:
            fh.write(json.dumps(r, default=float) + '\n')
