from __future__ import annotations
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from convex_qp_tools import dense_A
from scip_compare import build_stamps, make_waypoints
EPS = float(np.finfo(float).eps)
HERE = Path(__file__).resolve().parent
DEFAULT_SOLVER = HERE / 'build' / 'pufoam_solver'

def solver_args(ns: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(field=ns.field, seed=ns.seed, random_low=ns.random_low, random_high=ns.random_high, center_height=ns.center_height, center_size=ns.center_size, target=ns.target, field_size=ns.field_size, waypoint_rows=ns.waypoint_rows, waypoint_cols=ns.waypoint_cols, spray_interval=ns.spray_interval, kernel_size=ns.kernel_size, sigma_x=ns.sigma_x, sigma_y=ns.sigma_y, cmax=ns.cmax, budget=ns.budget, boundary=ns.boundary, initial_field=None)

def cpp_common(ns: argparse.Namespace) -> list[str]:
    return [str(ns.solver), '--field', ns.field, '--seed', str(ns.seed), '--random-low', str(ns.random_low), '--random-high', str(ns.random_high), '--center-height', str(ns.center_height), '--center-size', str(ns.center_size), '--target', str(ns.target), '--field-size', str(ns.field_size), '--waypoint-rows', str(ns.waypoint_rows), '--waypoint-cols', str(ns.waypoint_cols), '--spray-interval', str(ns.spray_interval), '--kernel-size', str(ns.kernel_size), '--sigma-x', str(ns.sigma_x), '--sigma-y', str(ns.sigma_y), '--cmax', str(ns.cmax), '--boundary', ns.boundary]

def read_field_csv(path: Path, m: int) -> np.ndarray:
    values: list[float] = []
    with path.open(newline='') as fh:
        for row in csv.reader(fh):
            values.extend((float(v) for v in row))
    if len(values) != m:
        raise ValueError(f'{path} has {len(values)} values, expected {m}')
    return np.array(values, dtype=float)

def write_allocation_csv(path: Path, x: np.ndarray) -> None:
    with path.open('w', newline='') as fh:
        fh.write('waypoint,shot_count\n')
        for j, v in enumerate(x):
            fh.write(f'{j},{int(v)}\n')

def cpp_initial_field(ns: argparse.Namespace, tmp: Path) -> np.ndarray:
    out = tmp / 'initial.csv'
    cmd = cpp_common(ns) + ['--write-initial', str(out), '--full-precision', '--budget', str(ns.field_size ** 2), '--out-prefix', str(tmp / 'throwaway'), '--quiet']
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return read_field_csv(out, ns.field_size ** 2)

def cpp_eval(ns: argparse.Namespace, tmp: Path, x: np.ndarray, mode: str, tag: str) -> tuple[np.ndarray, float, float]:
    alloc = tmp / f'alloc_{tag}_{mode}.csv'
    field_out = tmp / f'field_{tag}_{mode}.csv'
    write_allocation_csv(alloc, x)
    cmd = cpp_common(ns) + ['--eval-allocation', str(alloc), '--eval-mode', mode, '--eval-field-out', str(field_out)]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    mean = variance = float('nan')
    for tok in res.stdout.split():
        if tok.startswith('mean='):
            mean = float(tok.split('=', 1)[1])
        elif tok.startswith('variance='):
            variance = float(tok.split('=', 1)[1])
    return (read_field_csv(field_out, ns.field_size ** 2), mean, variance)

def sequential_tolerance(A: np.ndarray, supp: np.ndarray, s0: np.ndarray, x: np.ndarray) -> np.ndarray:
    n_terms = supp @ x + 1.0
    scale = A @ x + np.abs(s0)
    gamma = n_terms * EPS / np.maximum(1.0 - n_terms * EPS, 1e-12)
    return 2.0 * gamma * scale

def field_stats(sim: np.ndarray, model: np.ndarray) -> dict:
    d = sim - model
    denom = np.linalg.norm(model)
    return {'linf': float(np.max(np.abs(d))), 'rel_l2': float(np.linalg.norm(d) / denom) if denom > 0 else 0.0, 'mean_diff': float(sim.mean() - model.mean()), 'var_diff': float(sim.var() - model.var())}

def fill_by_priority(n: int, budget: int, cmax: int, order: np.ndarray) -> np.ndarray:
    x = np.ones(n, dtype=np.int64)
    remaining = budget - n
    if remaining < 0:
        raise ValueError('budget below the box lower bound')
    for j in order:
        if remaining <= 0:
            break
        take = min(cmax - 1, remaining)
        x[j] += take
        remaining -= take
    if remaining > 0:
        raise ValueError('budget exceeds box capacity')
    return x

def make_allocations(n: int, budget: int, cmax: int, pts, field_size: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_rand = np.ones(n, dtype=np.int64)
    remaining = budget - n
    while remaining > 0:
        room = (cmax - x_rand).astype(float)
        p = room / room.sum()
        draw = rng.multinomial(min(remaining, int(remaining)), p)
        add = np.minimum(draw, cmax - x_rand)
        if add.sum() == 0:
            free = np.flatnonzero(x_rand < cmax)
            add = np.zeros(n, dtype=np.int64)
            add[rng.choice(free)] = 1
        x_rand += add.astype(np.int64)
        remaining = budget - int(x_rand.sum())
    dist = np.array([min(r, c, field_size - 1 - r, field_size - 1 - c) for r, c in pts], dtype=float)
    x_boundary = fill_by_priority(n, budget, cmax, np.argsort(dist, kind='stable'))
    x_dense = fill_by_priority(n, budget, cmax, np.arange(n))
    return {'random': x_rand, 'boundary': x_boundary, 'dense': x_dense}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='audit equivalence', formatter_class=argparse.RawDescriptionHelpFormatter)
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
    p.add_argument('--boundary', choices=['reflect', 'truncate'], default='reflect')
    p.add_argument('--alloc-seed', type=int, default=7)
    p.add_argument('--solver', type=Path, default=DEFAULT_SOLVER)
    p.add_argument('--out', type=Path, default=None)
    return p.parse_args()

def main() -> int:
    ns = parse_args()
    if not ns.solver.exists():
        print(f'error: solver not found at {ns.solver}', file=sys.stderr)
        return 1
    sa = solver_args(ns)
    pts = make_waypoints(sa)
    stamps = build_stamps(sa, pts)
    n = len(stamps)
    m = ns.field_size ** 2
    A = dense_A(stamps, m, n)
    supp = (A > 0.0).astype(float)
    kappa = A.sum(axis=0)
    rows: list[dict] = []
    worst_ratio = 0.0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        s0 = cpp_initial_field(ns, tmp)
        budget = ns.budget
        if budget is None:
            mass = float(kappa[0])
            init_mean = float(s0.sum()) / m
            budget = int(round((ns.target - init_mean) * m / mass))
            budget = max(n, min(n * ns.cmax, budget))
        print(f'instance={ns.field}  M={m}  N={n}  budget={budget}  kappa[0]={kappa[0]:.6f}  kappa spread={kappa.max() - kappa.min():.3e}')
        print()

        def compare(tag: str, x: np.ndarray, mode: str='pershot') -> dict:
            nonlocal worst_ratio
            sim, c_mean, c_var = cpp_eval(ns, tmp, x, mode, tag)
            model = A @ x.astype(float) + s0
            tol = sequential_tolerance(A, supp, s0, x.astype(float))
            st = field_stats(sim, model)
            tol_inf = float(tol.max())
            ratio = st['linf'] / tol_inf if tol_inf > 0 else float('inf')
            worst_ratio = max(worst_ratio, ratio)
            row = {'check': tag, 'mode': mode, 'budget': int(x.sum()), 'linf': st['linf'], 'tol_linf': tol_inf, 'ratio': ratio, 'rel_l2': st['rel_l2'], 'mean_diff': st['mean_diff'], 'var_diff': st['var_diff'], 'cpp_mean': c_mean, 'cpp_var': c_var, 'model_mean': float(model.mean()), 'model_var': float(model.var())}
            rows.append(row)
            print(f"  {tag:<22s} mode={mode:<9s} linf={st['linf']:.3e} tol={tol_inf:.3e} ratio={ratio:.3f} rel_l2={st['rel_l2']:.3e} dmean={st['mean_diff']:+.3e} dvar={st['var_diff']:+.3e}")
            return row
        print('Step A: baseline allocation x0 = 1')
        x0 = np.ones(n, dtype=np.int64)
        for mode in ('pershot', 'scaled', 'shuffled'):
            compare('baseline_ones', x0, mode)
        print()
        print(f'Step B: operator columns, sim(x0+e_j) - sim(x0) for all {n} waypoints')
        col_path = tmp / 'operator.csv'
        cmd = cpp_common(ns) + ['--dump-operator', str(col_path)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        emp = np.zeros((m, n), dtype=float)
        with col_path.open(newline='') as fh:
            reader = csv.reader(fh)
            next(reader)
            for j_s, i_s, d_s in reader:
                emp[int(i_s), int(j_s)] = float(d_s)
        col_err = np.abs(emp - A)
        col_scale = A @ np.ones(n) + np.abs(s0)
        col_terms = supp @ np.ones(n) + 1.0
        col_tol = float((2.0 * col_terms * EPS * col_scale).max())
        col_linf = float(col_err.max())
        supp_mismatch = int(np.sum((emp > 0) != (A > 0)))
        worst_ratio = max(worst_ratio, col_linf / col_tol if col_tol > 0 else 0.0)
        per_col = col_err.max(axis=0)
        rows.append({'check': 'operator_columns', 'mode': 'pershot', 'budget': n, 'linf': col_linf, 'tol_linf': col_tol, 'ratio': col_linf / col_tol if col_tol > 0 else 0.0, 'rel_l2': float(np.linalg.norm(emp - A) / np.linalg.norm(A)), 'mean_diff': 0.0, 'var_diff': 0.0, 'cpp_mean': float('nan'), 'cpp_var': float('nan'), 'model_mean': float('nan'), 'model_var': float('nan')})
        print(f'  columns compared      : {n}')
        print(f'  support mismatches    : {supp_mismatch}')
        print(f'  max |emp - A|         : {col_linf:.3e}  (tol {col_tol:.3e})')
        print(f'  worst column          : j={int(np.argmax(per_col))} err={per_col.max():.3e}')
        print(f'  relative l2 over A    : {np.linalg.norm(emp - A) / np.linalg.norm(A):.3e}')
        print()
        print(f'Step C: budget-preserving allocations at B={budget}')
        allocs = make_allocations(n, budget, ns.cmax, pts, ns.field_size, ns.alloc_seed)
        for tag, x in allocs.items():
            compare(f'alloc_{tag}', x)
        print()
        print('Step D: accumulation-order spread on the random allocation')
        order_rows = [compare('order_random', allocs['random'], mode) for mode in ('pershot', 'scaled', 'shuffled')]
        means = [r['cpp_mean'] for r in order_rows]
        vars_ = [r['cpp_var'] for r in order_rows]
        print(f'  mean spread across orders     : {max(means) - min(means):.3e}')
        print(f'  variance spread across orders : {max(vars_) - min(vars_):.3e}')
        print()
    print('=' * 78)
    verdict = 'PASS' if worst_ratio <= 1.0 else 'REVIEW'
    print(f'worst measured/tolerated ratio: {worst_ratio:.4f}   ->   {verdict}')
    print('=' * 78)
    if ns.out is not None:
        with ns.out.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {ns.out}')
    return 0 if worst_ratio <= 1.0 else 2
if __name__ == '__main__':
    raise SystemExit(main())
