from __future__ import annotations
import csv
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from convex_qp_tools import dense_A, largest_remainder_round, solve_box_qp_relaxation
from scip_compare import build_stamps, choose_budget, evaluate, make_initial, make_waypoints
ROOT = Path(__file__).resolve().parent
CPP = ROOT / 'build' / 'pufoam_solver'
FINAL = re.compile('mean=([\\d.eE+-]+) variance=([\\d.eE+-]+).*?total_shots=(\\d+)')
TIME = re.compile('total_ms=(\\d+)')
DEFAULTS = dict(field='random', seed=1, random_low=1.0, random_high=30.0, center_height=100.0, center_size=16, target=200.0, field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, cmax=200, budget=None, initial_field=None)

def args(**kw):
    data = dict(DEFAULTS)
    data.update(kw)
    return SimpleNamespace(**data)

def box_qp_lb(stamps, initial, budget, m, cmax):
    mass = stamps[0].mass
    n = len(stamps)
    return solve_box_qp_relaxation(dense_A(stamps, m, n), initial, budget, mass, cmax)

def run_cpp(seed: int):
    prefix = ROOT / f'robust_random_seed{seed}'
    init = ROOT / f'robust_random_seed{seed}_initial.csv'
    cmd = [str(CPP), '--initial-field', str(ROOT / 'inputs' / f'random_u1_30_seed{seed}_F50.csv'), '--quiet', '--out-prefix', str(prefix), '--write-initial', str(init), '--full-precision']
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    metric = FINAL.search(out)
    timing = TIME.search(out)
    if metric is None:
        raise RuntimeError(out)
    return {'seed': seed, 'mean': float(metric.group(1)), 'variance_ub': float(metric.group(2)), 'budget': int(metric.group(3)), 'int_time_s': int(timing.group(1)) / 1000.0 if timing else float('nan'), 'initial': init}

def main():
    rows = []
    for seed in range(1, 11):
        row = run_cpp(seed)
        a = args(seed=seed, initial_field=row['initial'])
        pts = make_waypoints(a)
        stamps = build_stamps(a, pts)
        initial = make_initial(a)
        m = a.field_size ** 2
        n = len(stamps)
        budget = choose_budget(a, initial, stamps[0].mass, n)
        if budget != row['budget']:
            raise RuntimeError(f"budget mismatch for seed {seed}: {budget} vs {row['budget']}")
        t0 = time.time()
        qpres = box_qp_lb(stamps, initial, budget, m, a.cmax)
        row['qp_time_s'] = time.time() - t0
        x_round = largest_remainder_round(qpres.x, budget, a.cmax)
        _, v_round, _, _ = evaluate(list(x_round), initial, stamps, a.field_size)
        row['variance_repair'] = row['variance_ub']
        row['variance_round'] = v_round
        row['variance_ub'] = min(row['variance_ub'], v_round)
        row['variance_lb'] = qpres.dual_lower
        row['duality_gap'] = qpres.duality_gap
        row['gap_pct'] = (row['variance_ub'] - qpres.dual_lower) / row['variance_ub'] * 100.0
        row['status'] = qpres.status
        rows.append(row)
    out = ROOT / 'random_seed_robustness.csv'
    fields = ['seed', 'budget', 'mean', 'variance_repair', 'variance_round', 'variance_ub', 'variance_lb', 'duality_gap', 'gap_pct', 'qp_time_s', 'int_time_s', 'status']
    with out.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    gaps = np.array([r['gap_pct'] for r in rows])
    qpt = np.array([r['qp_time_s'] for r in rows])
    it = np.array([r['int_time_s'] for r in rows])
    print(f'wrote {out}')
    print(f'gap mean={gaps.mean():.4f}% std={gaps.std(ddof=1):.4f}% min={gaps.min():.4f}% max={gaps.max():.4f}%')
    print(f'QP time mean={qpt.mean():.4f}s max={qpt.max():.4f}s')
    print(f'int time mean={it.mean():.4f}s max={it.max():.4f}s')
if __name__ == '__main__':
    main()
