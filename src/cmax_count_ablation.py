from __future__ import annotations
import csv
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from convex_qp_tools import dense_A, largest_remainder_round, solve_box_qp_relaxation
from scip_compare import build_stamps, evaluate, make_initial, make_waypoints
CPP = './build/pufoam_solver'
FINAL = re.compile('mean=([\\d.eE+-]+) variance=([\\d.eE+-]+).*?total_shots=(\\d+)')
TIME = re.compile('total_ms=(\\d+)')
DEFAULTS = dict(field='zero', seed=1, random_low=1.0, random_high=30.0, center_height=100.0, center_size=16, target=200.0, field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, cmax=200, budget=None, initial_field=None)
SCENARIOS = [(3, 938), (5, 1250), (8, 1875), (20, 5000), (50, 12500), (200, 28348)]

def run_cpp(cmax: int, budget: int) -> dict[str, float]:
    out_prefix = f'cmax_zero_c{cmax}_b{budget}'
    cmd = [CPP, '--field', 'zero', '--cmax', str(cmax), '--budget', str(budget), '--quiet', '--out-prefix', out_prefix]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    mf = FINAL.search(out)
    mt = TIME.search(out)
    if mf is None or mt is None:
        raise RuntimeError(out)
    return {'mean': float(mf.group(1)), 'var_upper': float(mf.group(2)), 'budget': int(mf.group(3)), 'time_s': int(mt.group(1)) / 1000.0}

def main() -> None:
    args = SimpleNamespace(**DEFAULTS)
    pts = make_waypoints(args)
    stamps = build_stamps(args, pts)
    initial = make_initial(args)
    m = args.field_size ** 2
    n = len(stamps)
    mass = stamps[0].mass
    A = dense_A(stamps, m, n)
    rows = []
    for cmax, budget in SCENARIOS:
        cpp = run_cpp(cmax, budget)
        qpres = solve_box_qp_relaxation(A, initial, budget, mass, cmax)
        rx = largest_remainder_round(qpres.x, budget, cmax)
        _, rvar, _, _ = evaluate(rx, initial, stamps, args.field_size)
        var_repair = cpp['var_upper']
        var_ub = min(var_repair, rvar)
        row = {'cmax': cmax, 'budget': budget, 'avg_count': budget / n, 'mean': cpp['mean'], 'var_repair': var_repair, 'var_round': rvar, 'var_upper': var_ub, 'var_lower': qpres.dual_lower, 'abs_gap': var_ub - qpres.dual_lower, 'gap_pct': (var_ub - qpres.dual_lower) / var_ub * 100.0, 'round_gap_pct': (rvar - qpres.dual_lower) / rvar * 100.0 if rvar > 0 else 0.0, 'dual_gap': qpres.duality_gap, 'time_s': cpp['time_s'] + qpres.elapsed_s}
        rows.append(row)
        print(row)
    with open('cmax_count_ablation.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    x = np.array([r['avg_count'] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axes[0].plot(x, [r['var_upper'] for r in rows], '-o', label='V_UB=min(round,repair)')
    axes[0].plot(x, [r['var_round'] for r in rows], '--s', label='QP rounding')
    axes[0].plot(x, [r['var_lower'] for r in rows], ':^', label='QP lower')
    axes[0].set_ylabel('variance')
    axes[0].legend(fontsize=8)
    axes[1].plot(x, [r['gap_pct'] for r in rows], '-o', label='V_UB gap')
    axes[1].plot(x, [r['round_gap_pct'] for r in rows], '--s', label='round gap')
    axes[1].set_ylabel('upper-bound relative gap (%)')
    axes[1].legend(fontsize=8)
    axes[2].plot(x, [r['time_s'] for r in rows], '-o')
    axes[2].set_ylabel('time (s)')
    for ax in axes:
        ax.set_xlabel('average shot count B/N')
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{round(v, 1):g}' for v in x])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.savefig('cmax_count_ablation.png', dpi=300)
    print('wrote cmax_count_ablation.csv and cmax_count_ablation.png')
if __name__ == '__main__':
    main()
