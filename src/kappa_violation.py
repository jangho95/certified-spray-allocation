from __future__ import annotations
import argparse
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from audit_equivalence import cpp_common, cpp_eval, cpp_initial_field, solver_args
from convex_qp_tools import dense_A
from scip_compare import build_stamps, make_waypoints
HERE = Path(__file__).resolve().parent
DEFAULT_SOLVER = HERE / 'build' / 'pufoam_solver'

def operator(ns: argparse.Namespace, boundary: str):
    sa = solver_args(ns)
    sa.boundary = boundary
    pts = make_waypoints(sa)
    stamps = build_stamps(sa, pts)
    m = ns.field_size ** 2
    A = dense_A(stamps, m, len(stamps))
    return (A, pts)

def extreme_mass(kappa: np.ndarray, budget: int, cmax: int) -> tuple[float, float]:
    n = kappa.size
    surplus = budget - n
    base = float(kappa.sum())

    def pour(order: np.ndarray) -> float:
        left, extra = (surplus, 0.0)
        for j in order:
            if left <= 0:
                break
            take = min(cmax - 1, left)
            extra += take * kappa[j]
            left -= take
        return base + extra
    return (pour(np.argsort(kappa)), pour(np.argsort(-kappa)))

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--field', choices=['zero', 'random', 'center'], default='zero')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--random-low', type=float, default=1.0)
    p.add_argument('--random-high', type=float, default=30.0)
    p.add_argument('--center-height', type=float, default=100.0)
    p.add_argument('--center-size', type=int, default=16)
    p.add_argument('--target', type=float, default=200.0)
    p.add_argument('--field-size', type=int, default=50)
    p.add_argument('--waypoint-rows', type=int, default=25)
    p.add_argument('--waypoint-cols', type=int, default=25)
    p.add_argument('--spray-interval', type=int, default=2)
    p.add_argument('--kernel-size', type=int, default=7)
    p.add_argument('--sigma-x', type=float, default=1.75)
    p.add_argument('--sigma-y', type=float, default=1.75)
    p.add_argument('--cmax', type=int, default=200)
    p.add_argument('--budget', type=int, default=None)
    p.add_argument('--shots', type=int, default=100, help='size k of the interior<->edge move')
    p.add_argument('--solver', type=Path, default=DEFAULT_SOLVER)
    ns = p.parse_args()
    ns.boundary = 'reflect'
    M = ns.field_size ** 2
    results = {}
    for boundary in ('reflect', 'truncate'):
        A, pts = operator(ns, boundary)
        results[boundary] = (A, pts, A.sum(axis=0))
    print('=' * 78)
    print('Column masses kappa = A^T 1')
    print('=' * 78)
    for boundary in ('reflect', 'truncate'):
        k = results[boundary][2]
        rel = (k.max() - k.min()) / k.mean()
        print(f'  {boundary:<9s} min={k.min():.6f} max={k.max():.6f} spread={k.max() - k.min():.3e} relative={rel:.3e} distinct={len(np.unique(np.round(k, 12)))}')
    print()
    A_t, pts, kappa_t = results['truncate']
    A_r, _, kappa_r = results['reflect']
    print('=' * 78)
    print('Layer 1 (operator): does truncation break convexity?')
    print('=' * 78)
    L = np.eye(M) - np.ones((M, M)) / M
    for boundary, A in (('reflect', A_r), ('truncate', A_t)):
        H = 2.0 * (A.T @ L @ A) / M
        H = 0.5 * (H + H.T)
        lo = float(np.linalg.eigvalsh(H).min())
        scale = float(np.abs(H).max())
        print(f'  {boundary:<9s} lambda_min(2 A^T L A / M) = {lo:+.3e} (|H|_max = {scale:.3e}, relative {lo / scale:+.3e})')
    print('  -> convexity does not depend on common column mass')
    print()
    n = len(pts)
    j_int = int(np.argmax(kappa_t))
    j_edge = int(np.argmin(kappa_t))
    k = ns.shots
    print('=' * 78)
    print('Layer 2 (index): one-unit interior <-> edge move at constant budget')
    print('=' * 78)
    print(f'  interior waypoint j={j_int} at {pts[j_int]}  kappa={kappa_t[j_int]:.6f}')
    print(f'  edge     waypoint j={j_edge} at {pts[j_edge]}  kappa={kappa_t[j_edge]:.6f}')
    print(f'  move size k={k} shots, budget held at B={n + k}')
    print()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ns.boundary = 'truncate'
        s0 = cpp_initial_field(ns, tmp)
        x_int = np.ones(n, dtype=np.int64)
        x_int[j_int] += k
        x_edge = np.ones(n, dtype=np.int64)
        x_edge[j_edge] += k
        _, mean_int, _ = cpp_eval(ns, tmp, x_int, 'pershot', 'kint')
        _, mean_edge, _ = cpp_eval(ns, tmp, x_edge, 'pershot', 'kedge')
        d_measured = mean_edge - mean_int
        d_predicted = k * (kappa_t[j_edge] - kappa_t[j_int]) / M
        print(f'  simulator mean, interior-loaded : {mean_int:.12f}')
        print(f'  simulator mean, edge-loaded     : {mean_edge:.12f}')
        print(f'  measured  delta-mu              : {d_measured:+.12e}')
        print(f'  predicted k(kappa_e-kappa_i)/M  : {d_predicted:+.12e}')
        print(f'  |measured - predicted|          : {abs(d_measured - d_predicted):.3e}')
        print()
        ns.boundary = 'reflect'
        _, r_int, _ = cpp_eval(ns, tmp, x_int, 'pershot', 'rint')
        _, r_edge, _ = cpp_eval(ns, tmp, x_edge, 'pershot', 'redge')
        print(f'  control (reflect) delta-mu      : {r_edge - r_int:+.3e}   <- budget still fixes the mean')
        print()
        ns.boundary = 'truncate'
        budget = ns.budget
        if budget is None:
            init_mean = float(s0.sum()) / M
            budget = int(round((ns.target - init_mean) * M / kappa_r[0]))
            budget = max(n, min(n * ns.cmax, budget))
        print('=' * 78)
        print(f'Spread of achievable means over one budget slice (B={budget})')
        print('=' * 78)
        for boundary, kap in (('reflect', kappa_r), ('truncate', kappa_t)):
            m_lo, m_hi = extreme_mass(kap, budget, ns.cmax)
            mu_lo = float(s0.sum()) / M + m_lo / M
            mu_hi = float(s0.sum()) / M + m_hi / M
            print(f'  {boundary:<9s} mean in [{mu_lo:.6f}, {mu_hi:.6f}]  width={mu_hi - mu_lo:.6e}')
        print('  -> under truncation a single budget spans a range of means,')
        print('     so the bi-objective problem no longer decomposes by budget')
        print()
        print('=' * 78)
        print('Recovery: does kappa^T x fix the mean under truncation?')
        print('=' * 78)
        mu0 = float(s0.sum()) / M
        for tag, x, mean_meas in (('interior-loaded', x_int, mean_int), ('edge-loaded', x_edge, mean_edge)):
            pred = mu0 + float(kappa_t @ x) / M
            print(f'  {tag:<16s} mu0 + kappa^T x / M = {pred:.12f}  simulator = {mean_meas:.12f}  diff={abs(pred - mean_meas):.3e}')
        print('  -> the mass level, not the shot budget, is the correct index')
        print()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
