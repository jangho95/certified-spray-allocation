from __future__ import annotations
import csv
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pyscipopt import Model, quicksum
from convex_qp_tools import dense_A, largest_remainder_round, solve_box_qp_relaxation
from scip_compare import build_gram_and_linear, build_stamps, evaluate, make_initial, make_waypoints

def args(**kw) -> SimpleNamespace:
    base = dict(field='zero', seed=7, random_low=1.0, random_high=30.0, center_height=30.0, center_size=4, target=200.0, field_size=12, waypoint_rows=4, waypoint_cols=4, spray_interval=3, kernel_size=7, sigma_x=1.75, sigma_y=1.75, cmax=8, budget=64, initial_field=None)
    base.update(kw)
    return SimpleNamespace(**base)
INSTANCES = [('zero', args(field='zero')), ('random', args(field='random', seed=7, initial_field=Path('inputs/small_random_pcg64_seed7_F12.csv'))), ('center', args(field='center', center_height=30.0, center_size=4))]

def exact_miqp(a: SimpleNamespace, stamps, initial, budget: int):
    n = len(stamps)
    constant, linear, quad, _ = build_gram_and_linear(initial, stamps, budget, a.field_size)
    model = Model('convex_miqp')
    model.hideOutput()
    xs = [model.addVar(vtype='I', lb=1, ub=a.cmax, name=f'x{j}') for j in range(n)]
    model.addCons(quicksum(xs) == budget)
    t = model.addVar(vtype='C', lb=0.0, ub=None, name='t')
    obj = constant
    obj += quicksum((float(linear[j]) * xs[j] for j in range(n)))
    obj += quicksum((float(coef) * xs[i] * xs[j] for (i, j), coef in quad.items()))
    model.addCons(obj <= t)
    model.setObjective(t, 'minimize')
    model.setParam('limits/gap', 0.0)
    model.setParam('limits/time', 300.0)
    t0 = time.time()
    model.optimize()
    dt = time.time() - t0
    xsol = [int(round(model.getVal(v))) for v in xs]
    return (model.getStatus(), model.getObjVal(), model.getGap(), xsol, dt)

def main() -> None:
    rows = []
    hdr = f"{'instance':10s}{'N':>4s}{'B':>5s}{'dual_LB':>11s}{'V_QP*':>11s}{'V_Z*(SCIP)':>12s}{'V_UB':>11s}{'status':>10s}{'gap':>8s}{'relax':>9s}{'inc.sub':>9s}{'cert':>9s}"
    print(hdr)
    for name, a in INSTANCES:
        pts = make_waypoints(a)
        stamps = build_stamps(a, pts)
        initial = make_initial(a)
        m = a.field_size ** 2
        n = len(stamps)
        budget = a.budget
        mass = stamps[0].mass
        res = solve_box_qp_relaxation(dense_A(stamps, m, n), initial, budget, mass, a.cmax)
        x_round = largest_remainder_round(res.x, budget, a.cmax)
        _, v_round, _, _ = evaluate(list(x_round), initial, stamps, a.field_size)
        v_qp = res.primal_value
        dual = res.dual_lower
        v_ub = v_round
        status, v_z, scip_gap, xsol, dt = exact_miqp(a, stamps, initial, budget)
        _, v_z_eval, _, _ = evaluate(xsol, initial, stamps, a.field_size)
        rows.append(dict(instance=name, N=n, B=budget, dual_lower=dual, v_qp=v_qp, v_z=v_z, v_z_eval=v_z_eval, v_ub=v_ub, status=status, scip_gap=scip_gap, relax_gap=v_z - v_qp, incumbent_subopt=v_ub - v_z, reported_cert=v_ub - dual, scip_time_s=dt))
        print(f'{name:10s}{n:4d}{budget:5d}{dual:11.5f}{v_qp:11.5f}{v_z:12.5f}{v_ub:11.5f}{status:>10s}{scip_gap:8.1e}{v_z - v_qp:9.5f}{v_ub - v_z:9.5f}{v_ub - dual:9.5f}')
    print('\nchecks:')
    for r in rows:
        chain = r['dual_lower'] <= r['v_qp'] + 1e-06 <= r['v_z'] + 1e-06 and r['v_z'] <= r['v_ub'] + 1e-06
        match = abs(r['v_z'] - r['v_z_eval']) < 0.0001
        print(f"  {r['instance']:10s} chain_ok={chain} scip==evaluate={match} (V_Z*={r['v_z']:.5f}, evaluate={r['v_z_eval']:.5f}, status={r['status']})")
    with open('small_exact_miqp.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('\nwrote small_exact_miqp.csv')
if __name__ == '__main__':
    main()
