from __future__ import annotations
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from scip_compare import build_stamps, evaluate, load_warm_start, make_initial, make_waypoints
DEFAULTS = dict(field='zero', seed=1, random_low=1.0, random_high=30.0, center_height=100.0, center_size=16, target=200.0, field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, cmax=200, budget=None, initial_field=None)
FIELD_CFG = {'zero': dict(field='zero'), 'random_1_30_seed7': dict(field='random', seed=7, random_low=1.0, random_high=30.0, initial_field=Path('random_1_30_seed7_initial.csv')), 'center30': dict(field='center', center_height=30.0), 'center100': dict(field='center', center_height=100.0)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', required=True, choices=list(FIELD_CFG))
    ap.add_argument('--alloc', required=True, type=Path, help="PSO allocation CSV with columns 'waypoint','shot_count'")
    ap.add_argument('--out', type=Path, default=Path('pso_points.csv'))
    a = ap.parse_args()
    args = SimpleNamespace(**{**DEFAULTS, **FIELD_CFG[a.instance]})
    pts = make_waypoints(args)
    stamps = build_stamps(args, pts)
    initial = make_initial(args)
    n = len(stamps)
    x = load_warm_start(a.alloc, n)
    mean, var, mn, mx = evaluate(x, initial, stamps, args.field_size)
    print(f'{a.instance}: mean={mean:.6f} variance={var:.6f} min={mn:.4f} max={mx:.4f} budget={sum(x)}')
    new = not a.out.exists()
    with a.out.open('a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['instance', 'mean', 'variance'])
        w.writerow([a.instance, f'{mean:.6f}', f'{var:.6f}'])
    print(f'appended to {a.out}')
if __name__ == '__main__':
    main()
