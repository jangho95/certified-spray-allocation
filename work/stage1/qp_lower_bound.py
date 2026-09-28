from __future__ import annotations
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from convex_qp_tools import dense_A, largest_remainder_round, solve_box_qp_relaxation
from scip_compare import build_stamps, choose_budget, evaluate, load_warm_start, make_initial, make_waypoints
DEFAULTS = dict(field='zero', seed=1, random_low=1.0, random_high=30.0, center_height=100.0, center_size=16, target=200.0, field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, cmax=200, budget=None, initial_field=None)

def args(**kw) -> SimpleNamespace:
    d = dict(DEFAULTS)
    d.update(kw)
    return SimpleNamespace(**d)

def box_qp_lower_bound(stamps, initial, budget, m, cmax):
    mass = stamps[0].mass
    n = len(stamps)
    A = dense_A(stamps, m, n)
    return solve_box_qp_relaxation(A, initial, budget, mass, cmax)
INSTANCES = [('zero', 'zero_allocation.csv', args(field='zero')), ('random_1_30_seed7', 'random_1_30_seed7_sync_allocation.csv', args(field='random', seed=7, random_low=1.0, random_high=30.0, initial_field=Path('random_1_30_seed7_initial.csv'))), ('center30', 'center30_allocation.csv', args(field='center', center_height=30.0)), ('center100', 'center100_allocation.csv', args(field='center', center_height=100.0))]

def main():
    hdr = f"{'instance':20s}{'repair UB':>12s}{'round UB':>12s}{'V_UB=min':>12s}{'QP primal':>12s}{'dual LB':>12s}{'gap<=%':>9s}{'dual gap':>10s}{'QP(s)':>7s}"
    print(hdr)
    for name, warm, a in INSTANCES:
        pts = make_waypoints(a)
        stamps = build_stamps(a, pts)
        initial = make_initial(a)
        m = a.field_size ** 2
        n = len(stamps)
        mass = stamps[0].mass
        budget = choose_budget(a, initial, mass, n)
        wx = load_warm_start(Path(warm), n)
        _, wvar, _, _ = evaluate(wx, initial, stamps, a.field_size)
        t0 = time.time()
        res = box_qp_lower_bound(stamps, initial, budget, m, a.cmax)
        rx = largest_remainder_round(res.x, budget, a.cmax)
        _, rvar, _, _ = evaluate(rx, initial, stamps, a.field_size)
        dt = time.time() - t0
        v_ub = min(wvar, rvar)
        gap = (v_ub - res.dual_lower) / v_ub * 100.0
        print(f'{name:20s}{wvar:12.6f}{rvar:12.6f}{v_ub:12.6f}{res.primal_value:12.6f}{res.dual_lower:12.6f}{gap:9.4f}{res.duality_gap:10.2e}{dt:7.2f}')
if __name__ == '__main__':
    main()
