from __future__ import annotations
import argparse
import csv
import re
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from convex_qp_tools import dense_A, largest_remainder_round, solve_box_qp_relaxation
from scip_compare import build_stamps, make_initial, make_waypoints
from types import SimpleNamespace
CPP = './build/pufoam_solver'
TARGET = 200.0
CMAX = 200
FINAL = re.compile('mean=([\\d.eE+-]+) variance=([\\d.eE+-]+).*?total_shots=(\\d+)')
DEFAULTS = dict(field='zero', seed=1, random_low=1.0, random_high=30.0, center_height=100.0, center_size=16, target=TARGET, field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, cmax=CMAX, budget=None, initial_field=None)
INSTANCES = [('zero', ['--field', 'zero'], dict(field='zero')), ('random_1_30_seed7', ['--initial-field', 'inputs/random_u1_30_seed7_F50.csv'], dict(field='random', seed=7, random_low=1.0, random_high=30.0)), ('center30', ['--field', 'center', '--center-height', '30'], dict(field='center', center_height=30.0)), ('center100', ['--field', 'center', '--center-height', '100'], dict(field='center', center_height=100.0))]

def run_cpp(cli, budget=None, write_initial=None):
    cmd = [CPP, *cli, '--quiet', '--out-prefix', 'tmp_sw']
    if budget is not None:
        cmd += ['--budget', str(int(budget))]
    if write_initial is not None:
        cmd += ['--write-initial', str(write_initial), '--full-precision']
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = FINAL.search(out)
    return dict(mean=float(m.group(1)), var=float(m.group(2)), shots=int(m.group(3)))

def sweep_instance(name, cli, ov, npts):
    args = SimpleNamespace(**{**DEFAULTS, **ov})
    init_path = Path(f'tmp_sw_{name}_init.csv')
    base = run_cpp(cli, write_initial=init_path)
    bstar = base['shots']
    args.initial_field = init_path
    pts = make_waypoints(args)
    stamps = build_stamps(args, pts)
    initial = make_initial(args)
    m = args.field_size ** 2
    n = len(stamps)
    mass = stamps[0].mass
    A = dense_A(stamps, m, n)
    budgets = sorted(set((int(round(b)) for b in np.linspace(0.85 * bstar, 1.05 * bstar, npts))))
    rows = []
    for B in budgets:
        feas = run_cpp(cli, budget=B)
        qpres = solve_box_qp_relaxation(A, initial, B, mass, args.cmax)
        mean = (float(initial.sum()) + B * mass) / m
        rx = largest_remainder_round(qpres.x, B, args.cmax)
        from scip_compare import evaluate
        _, round_var, _, _ = evaluate(rx, initial, stamps, args.field_size)
        var_ub = min(feas['var'], round_var)
        rows.append(dict(budget=B, mean=mean, dev=abs(mean - TARGET), var_repair=feas['var'], var_round=round_var, var_upper=var_ub, var_lower=qpres.dual_lower, relax_primal=qpres.primal_value, duality_gap=qpres.duality_gap, gap=(var_ub - qpres.dual_lower) / var_ub * 100.0))
    return (name, bstar, rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npts', type=int, default=21)
    ap.add_argument('--pso', type=Path, default=None)
    a = ap.parse_args()
    pso = {}
    if a.pso and a.pso.exists():
        with a.pso.open() as fh:
            for r in csv.DictReader(fh):
                pso.setdefault(r['instance'], []).append((float(r['mean']), float(r['variance'])))
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.13, top=0.9, wspace=0.22, hspace=0.4)
    legend_handles = {}
    allrows = []
    for idx, (ax, (name, cli, ov)) in enumerate(zip(axes.ravel(), INSTANCES)):
        nm, bstar, rows = sweep_instance(name, cli, ov, a.npts)
        with open(f'pareto_{nm}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        mean = np.array([r['mean'] for r in rows])
        up = np.array([r['var_upper'] for r in rows])
        rnd = np.array([r['var_round'] for r in rows])
        lo = np.array([r['var_lower'] for r in rows])
        gap_pct = np.array([r['gap'] for r in rows])
        order = np.argsort(mean)
        mean = mean[order]
        up = up[order]
        rnd = rnd[order]
        lo = lo[order]
        gap_pct = gap_pct[order]
        ax.fill_between(mean, 0, gap_pct, color='tab:orange', alpha=0.28, label='certified relative gap')
        ax.plot(mean, gap_pct, '-o', ms=4.0, lw=2.0, color='tab:red', label='relative certificate gap')
        ax.axvline(TARGET, color='gray', ls='--', lw=1, label='target T')
        atgt = min(rows, key=lambda r: r['dev'])
        target_idx = int(np.argmin(np.abs(mean - TARGET)))
        ax.scatter([mean[target_idx]], [gap_pct[target_idx]], s=46, color='black', zorder=5, label='target-budget sample')
        ax.annotate(f'{gap_pct[target_idx]:.2f}%', xy=(mean[target_idx], gap_pct[target_idx]), xytext=(6, 8), textcoords='offset points', fontsize=8)
        ax.set_title(f"{name}\nB*={bstar}, target={atgt['gap']:.2f}%, max={max(gap_pct):.2f}%", fontsize=10)
        ax.grid(alpha=0.28)
        ax.tick_params(labelsize=9)
        ax.set_ylim(0, 0.62)
        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend_handles.setdefault(label, handle)
        if idx // 2 == 1:
            ax.set_xlabel('mean thickness (mm)')
        if idx % 2 == 0:
            ax.set_ylabel('certificate gap (%)')
        allrows.append((nm, rows))
        maxgap = max((r['gap'] for r in rows))
        print(f"{nm:20s} B*={bstar:6d}  gap@target={atgt['gap']:.3f}%  max_gap={maxgap:.3f}%")
    fig.legend(legend_handles.values(), legend_handles.keys(), loc='lower center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.savefig('pareto_fronts.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print('wrote pareto_fronts.png and pareto_<instance>.csv')
if __name__ == '__main__':
    main()
