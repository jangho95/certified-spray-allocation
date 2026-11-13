from __future__ import annotations
import csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from paper_pso import run_pso
ROOT = Path(__file__).resolve().parent
DEFAULTS = dict(field='zero', seed=1, pso_seed=1, random_low=1.0, random_high=30.0, center_height=100.0, center_size=16, target=200.0, field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, cmax=200, budget=None, initial_field=None, pop_size=100, iters=500, w=0.7, c1=1.5, c2=1.5, strategy='proposed', gamma=None, alpha=0.5, beta=10.0, gradient_scale=10.0, adjust_iters=20, atf_min=0.05, atf_max=0.25, atf_initial=0.15, atf_tighten=0.005, atf_relax=0.01, relax_patience=3, diversify_patience=15, diversify_ratio=0.15, k_aw=5.0, w_min=0.1, w_max=0.9, archive_capacity=200, log_every=10 ** 9, quiet=True, out_prefix=Path('/tmp/pso'))
INSTANCES = [('zero', dict(field='zero')), ('random_1_30_seed7', dict(field='random', initial_field=ROOT / 'random_1_30_seed7_initial.csv')), ('center30', dict(field='center', center_height=30.0)), ('center100', dict(field='center', center_height=100.0))]
JOCS_VAR = {'zero': 166.1, 'random_1_30_seed7': 200.2, 'center30': 170.44, 'center100': 254.87}
IQP_FEAS_VAR = {'zero': 54.077695, 'random_1_30_seed7': 102.93372, 'center30': 55.86483, 'center100': 93.780756}

def make_args(instance_overrides: dict, run_seed: int) -> SimpleNamespace:
    data = dict(DEFAULTS)
    data.update(instance_overrides)
    data['pso_seed'] = run_seed
    return SimpleNamespace(**data)

def fmt_mean_std(values: np.ndarray, digits: int=3) -> str:
    return f'{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}'

def main() -> None:
    rows: list[dict[str, float | int | str]] = []
    summary: list[dict[str, float | str]] = []
    for instance, overrides in INSTANCES:
        print(f'=== {instance} ===', flush=True)
        for run in range(1, 26):
            args = make_args(overrides, run)
            _, _, metrics, _, _ = run_pso(args)
            row = {'instance': instance, 'run': run, 'mean': metrics['mean'], 'variance': metrics['variance'], 'runtime_s': metrics['elapsed_s'], 'min': metrics['min'], 'max': metrics['max'], 'total_shots': metrics['total_shots'], 'archive_size': metrics['archive_size']}
            rows.append(row)
            print(f"run={run:02d} mean={metrics['mean']:.6f} var={metrics['variance']:.6f} time={metrics['elapsed_s']:.3f}s", flush=True)
        sub = [r for r in rows if r['instance'] == instance]
        mean = np.array([float(r['mean']) for r in sub])
        var = np.array([float(r['variance']) for r in sub])
        runtime = np.array([float(r['runtime_s']) for r in sub])
        shots = np.array([float(r['total_shots']) for r in sub])
        summary.append({'instance': instance, 'runs': 25, 'mean_mean': mean.mean(), 'mean_std': mean.std(ddof=1), 'variance_mean': var.mean(), 'variance_std': var.std(ddof=1), 'runtime_mean': runtime.mean(), 'runtime_std': runtime.std(ddof=1), 'shots_mean': shots.mean(), 'shots_std': shots.std(ddof=1), 'jocs_variance': JOCS_VAR[instance], 'iqp_feasible_variance': IQP_FEAS_VAR[instance], 'variance_over_jocs': var.mean() / JOCS_VAR[instance], 'variance_over_iqp': var.mean() / IQP_FEAS_VAR[instance]})
    rows_path = ROOT / 'pso_repeat_runs.csv'
    with rows_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = ROOT / 'pso_repeat_summary.csv'
    with summary_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    points_path = ROOT / 'pso_points.csv'
    with points_path.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['instance', 'mean', 'variance'])
        for row in summary:
            writer.writerow([row['instance'], f"{row['mean_mean']:.6f}", f"{row['variance_mean']:.6f}"])
    print(f'wrote {rows_path}')
    print(f'wrote {summary_path}')
    print(f'updated {points_path} with mean PSO points')
    print('\nSummary:')
    for row in summary:
        print(f"{row['instance']:20s} mu={row['mean_mean']:.3f}±{row['mean_std']:.3f} V={row['variance_mean']:.3f}±{row['variance_std']:.3f} t={row['runtime_mean']:.3f}±{row['runtime_std']:.3f}s V/JoCS={row['variance_over_jocs']:.3f} V/IQP={row['variance_over_iqp']:.3f}")
if __name__ == '__main__':
    main()
