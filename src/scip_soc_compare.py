from __future__ import annotations
import argparse
import time
from types import SimpleNamespace
import numpy as np
from pyscipopt import Model, quicksum
from scip_compare import build_stamps, choose_budget, evaluate, load_warm_start, make_initial, make_waypoints
DEFAULTS = dict(field='zero', seed=1, random_low=1.0, random_high=30.0, center_height=100.0, center_size=16, target=200.0, field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, cmax=200, budget=None, initial_field=None)

def make_args(**kw) -> SimpleNamespace:
    d = dict(DEFAULTS)
    d.update(kw)
    return SimpleNamespace(**d)

def build_dense_A(stamps, m, n) -> np.ndarray:
    A = np.zeros((m, n), dtype=float)
    for j, st in enumerate(stamps):
        for cell, w in zip(st.cells, st.weights):
            A[cell, j] += w
    return A

def numpy_equality_lb(A, initial, budget, mass, m) -> float:
    mean = (float(initial.sum()) + budget * mass) / m
    r0 = initial - mean
    n = A.shape[1]
    P = A.T @ A
    b = A.T @ r0
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = 2.0 * P
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    rhs = np.concatenate([-2.0 * b, [float(budget)]])
    sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    x = sol[:n]
    val = (float(x @ (P @ x)) + 2.0 * float(b @ x) + float(r0 @ r0)) / m
    return val

def build_soc_model(stamps, initial, budget, m, cmax, integer: bool):
    n = len(stamps)
    mass = stamps[0].mass
    mean = (float(initial.sum()) + budget * mass) / m
    by_cell: list[list[tuple[int, float]]] = [[] for _ in range(m)]
    for j, st in enumerate(stamps):
        for cell, w in zip(st.cells, st.weights):
            by_cell[cell].append((j, w))
    model = Model('pufoam_soc')
    model.hideOutput(True)
    vt = 'I' if integer else 'C'
    xv = [model.addVar(vtype=vt, lb=1, ub=cmax, name=f'x_{j}') for j in range(n)]
    model.addCons(quicksum(xv) == budget, name='budget')
    const_var = 0.0
    tparts = []
    for c in range(m):
        entries = by_cell[c]
        base = float(initial[c]) - mean
        if not entries:
            const_var += base * base
            continue
        g = model.addVar(vtype='C', lb=-1e+20, ub=1e+20, name=f'g_{c}')
        model.addCons(g == base + quicksum((w * xv[j] for j, w in entries)))
        t = model.addVar(vtype='C', lb=0.0, name=f't_{c}')
        model.addCons(t >= g * g)
        tparts.append(t)
    model.setObjective((quicksum(tparts) + const_var) / m, 'minimize')
    return (model, xv)

def complete_warm_start(model, xv, warm_x, stamps, initial, budget, m):
    mass = stamps[0].mass
    mean = (float(initial.sum()) + budget * mass) / m
    dev = np.asarray(initial, dtype=float) - mean
    for j, st in enumerate(stamps):
        for cell, w in zip(st.cells, st.weights):
            dev[cell] += w * warm_x[j]
    sol = model.createSol()
    for v, val in zip(xv, warm_x):
        model.setSolVal(sol, v, val)
    for var in model.getVars():
        if var.name.startswith('g_'):
            model.setSolVal(sol, var, float(dev[int(var.name[2:])]))
        elif var.name.startswith('t_'):
            model.setSolVal(sol, var, float(dev[int(var.name[2:])]) ** 2)
    return sol

def run_instance(name, warm, args):
    pts = make_waypoints(args)
    stamps = build_stamps(args, pts)
    initial = make_initial(args)
    m = args.field_size * args.field_size
    n = len(stamps)
    mass = stamps[0].mass
    budget = choose_budget(args, initial, mass, n)
    warm_x = load_warm_start(warm, n)
    wmean, wvar, _, _ = evaluate(warm_x, initial, stamps, args.field_size)
    A = build_dense_A(stamps, m, n)
    lb_eq = numpy_equality_lb(A, initial, budget, mass, m)
    mC, _ = build_soc_model(stamps, initial, budget, m, args.cmax, integer=False)
    mC.setParam('limits/time', 120.0)
    t0 = time.time()
    mC.optimize()
    status_soc = mC.getStatus()
    lb_soc = mC.getDualbound()
    ub_soc = mC.getPrimalbound() if mC.getNSols() > 0 else float('nan')
    tC = time.time() - t0
    gap_eq = (wvar - lb_eq) / wvar * 100.0
    gap_soc = (wvar - lb_soc) / wvar * 100.0
    print(f'\n=== {name} ===')
    print(f'budget={budget}  warm_sum={sum(warm_x)}  mean={wmean:.6f}')
    print(f'C++ incumbent variance      = {wvar:.6f}')
    print(f'(A) numpy equality LB       = {lb_eq:.6f}   -> gap <= {gap_eq:6.3f}%')
    print(f'(B) SCIP cont. SOC dual bd  = {lb_soc:.6f}   -> gap <= {gap_soc:6.3f}%  ({tC:.1f}s, status={status_soc}, primal={ub_soc:.6f})')
    return (name, wvar, lb_eq, lb_soc, gap_eq, gap_soc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--integer-demo', action='store_true', help='also run the integer SOC model on the zero instance')
    a = ap.parse_args()
    instances = [('zero', 'zero_allocation.csv', make_args(field='zero')), ('random_1_30_seed7', 'random_1_30_seed7_sync_allocation.csv', make_args(field='random', seed=7, random_low=1.0, random_high=30.0, initial_field=__import__('pathlib').Path('random_1_30_seed7_initial.csv'))), ('center30', 'center30_allocation.csv', make_args(field='center', center_height=30.0)), ('center100', 'center100_allocation.csv', make_args(field='center', center_height=100.0))]
    from pathlib import Path
    rows = []
    for name, warm, args in instances:
        rows.append(run_instance(name, Path(warm), args))
    print('\n================ SUMMARY (certified upper gap) ================')
    print(f"{'instance':20s} {'incumbent':>12s} {'eqLB':>12s} {'socLB':>12s} {'gap%(soc)':>10s}")
    for name, wvar, lb_eq, lb_soc, g_eq, g_soc in rows:
        print(f'{name:20s} {wvar:12.6f} {lb_eq:12.6f} {lb_soc:12.6f} {g_soc:10.3f}')
    if a.integer_demo:
        print('\n--- integer SOC dual-bound demo (zero) ---')
        args = make_args(field='zero')
        pts = make_waypoints(args)
        stamps = build_stamps(args, pts)
        initial = make_initial(args)
        m = args.field_size ** 2
        n = len(stamps)
        budget = choose_budget(args, initial, stamps[0].mass, n)
        warm_x = load_warm_start(Path('zero_allocation.csv'), n)
        _, wvar, _, _ = evaluate(warm_x, initial, stamps, args.field_size)
        mI, xv = build_soc_model(stamps, initial, budget, m, args.cmax, integer=True)
        sol = complete_warm_start(mI, xv, warm_x, stamps, initial, budget, m)
        feasible = mI.checkSol(sol, printreason=False)
        accepted = mI.addSol(sol, free=True)
        print(f'warm start: checkSol={feasible} accepted={accepted}')
        mI.setParam('limits/time', 60.0)
        mI.optimize()
        print(f'status={mI.getStatus()} primal={mI.getPrimalbound():.6f} dual={mI.getDualbound():.6f} gap={mI.getGap() * 100:.3f}% nodes={mI.getNNodes()}')
if __name__ == '__main__':
    main()
