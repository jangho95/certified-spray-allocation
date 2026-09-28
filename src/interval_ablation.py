from __future__ import annotations
import csv
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import osqp
import scipy.sparse as sp
from convex_qp_tools import dense_A, largest_remainder_round, solve_box_qp_relaxation
from scip_compare import build_stamps, choose_budget, evaluate, make_initial, make_waypoints
CPP = './build/pufoam_solver'
FINAL = re.compile('mean=([\\d.eE+-]+) variance=([\\d.eE+-]+).*?total_shots=(\\d+)')
TIME = re.compile('total_ms=(\\d+)')
DEFAULTS = dict(field='zero', seed=1, random_low=1.0, random_high=30.0, center_height=100.0, center_size=16, target=200.0, field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, cmax=200, budget=None, initial_field=None)
INSTANCES = [('zero', ['--field', 'zero'], dict(field='zero')), ('random', ['--initial-field', 'inputs/random_u1_30_seed7_F50.csv'], dict(field='random', seed=7, random_low=1.0, random_high=30.0)), ('center30', ['--field', 'center', '--center-height', '30'], dict(field='center', center_height=30.0)), ('center100', ['--field', 'center', '--center-height', '100'], dict(field='center', center_height=100.0))]

def waypoint_count(field_size: int, interval: int) -> int:
    return (field_size - 1) // interval + 1

def run_cpp(cli: list[str], interval: int, out_prefix: str, initial_path: Path) -> dict[str, float]:
    rows = waypoint_count(50, interval)
    cmd = [CPP, *cli, '--spray-interval', str(interval), '--waypoint-rows', str(rows), '--waypoint-cols', str(rows), '--quiet', '--out-prefix', out_prefix, '--write-initial', str(initial_path), '--full-precision']
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    mf = FINAL.search(out)
    mt = TIME.search(out)
    if mf is None or mt is None:
        raise RuntimeError(out)
    return {'mean': float(mf.group(1)), 'var_upper': float(mf.group(2)), 'budget': int(mf.group(3)), 'time_int_s': int(mt.group(1)) / 1000.0, 'rows': rows, 'n': rows * rows}

def sparse_A(stamps, m: int, n: int) -> sp.csc_matrix:
    rr, cc, vv = ([], [], [])
    for j, st in enumerate(stamps):
        for cell, w in zip(st.cells, st.weights):
            rr.append(cell)
            cc.append(j)
            vv.append(w)
    return sp.coo_matrix((vv, (rr, cc)), shape=(m, n)).tocsc()

def qp_lb(args: SimpleNamespace, budget: int):
    pts = make_waypoints(args)
    stamps = build_stamps(args, pts)
    initial = make_initial(args)
    m = args.field_size ** 2
    n = len(stamps)
    mass = stamps[0].mass
    t0 = time.time()
    res = solve_box_qp_relaxation(dense_A(stamps, m, n), initial, budget, mass, args.cmax)
    qptime = time.time() - t0
    x_round = largest_remainder_round(res.x, budget, args.cmax)
    _, v_round, _, _ = evaluate(list(x_round), initial, stamps, args.field_size)
    return (res.dual_lower, v_round, qptime, res.status)

def main() -> None:
    rows = []
    for name, cli, ov in INSTANCES:
        for interval in (1, 2, 3):
            out_prefix = f'ablation_{name}_s{interval}'
            initial_path = Path(f'ablation_{name}_s{interval}_initial.csv')
            cpp = run_cpp(cli, interval, out_prefix, initial_path)
            args = SimpleNamespace(**{**DEFAULTS, **ov})
            args.spray_interval = interval
            args.waypoint_rows = args.waypoint_cols = int(cpp['rows'])
            args.initial_field = initial_path
            lb, v_round, qptime, status = qp_lb(args, int(cpp['budget']))
            v_ub = min(cpp['var_upper'], v_round)
            gap = (v_ub - lb) / v_ub * 100.0 if v_ub > 1e-12 else float('nan')
            row = {'instance': name, 'interval': interval, 'N': int(cpp['n']), 'budget': int(cpp['budget']), 'mean': cpp['mean'], 'var_repair': cpp['var_upper'], 'var_round': v_round, 'var_upper': v_ub, 'var_lower': lb, 'abs_gap': v_ub - lb, 'gap_pct': gap, 'int_s': cpp['time_int_s'], 'qp_s': qptime, 'status': status}
            rows.append(row)
            print(row)
    with open('interval_ablation.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for inst in [r['instance'] for r in rows if r['interval'] == 1]:
        sub = [r for r in rows if r['instance'] == inst]
        x = [r['interval'] for r in sub]
        axes[0].plot(x, [r['var_upper'] for r in sub], '-o', label=inst)
        axes[1].plot(x, [r['abs_gap'] for r in sub], '-o', label=inst)
        axes[2].plot(x, [r['int_s'] + r['qp_s'] for r in sub], '-o', label=inst)
    axes[0].set_ylabel('integer feasible variance')
    axes[1].set_ylabel('absolute gap (mm$^2$)')
    axes[2].set_ylabel('certification time (s)')
    for ax in axes:
        ax.set_xlabel('spray interval (cells)')
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.savefig('interval_ablation.png', dpi=300)
    print('wrote interval_ablation.csv and interval_ablation.png')
if __name__ == '__main__':
    main()
