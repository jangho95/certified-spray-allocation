from __future__ import annotations
import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from pyscipopt import Model, quicksum

@dataclass
class Stamp:
    cells: list[int]
    weights: list[float]
    mass: float

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
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
    p.add_argument('--warm-start', type=Path, default=None)
    p.add_argument('--initial-field', type=Path, default=None)
    p.add_argument('--time-limit', type=float, default=30.0)
    p.add_argument('--gap-limit', type=float, default=0.01)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--write-solution', type=Path, default=None)
    return p.parse_args()

def reflect_index(p: int, n: int) -> int:
    while p < 0 or p >= n:
        if p < 0:
            p = -p - 1
        if p >= n:
            p = 2 * n - p - 1
    return p

def make_waypoints(args: argparse.Namespace) -> list[tuple[int, int]]:
    return [(r * args.spray_interval, c * args.spray_interval) for r in range(args.waypoint_rows) for c in range(args.waypoint_cols)]

def build_stamps(args: argparse.Namespace, pts: list[tuple[int, int]]) -> list[Stamp]:
    f = args.field_size
    radius = args.kernel_size // 2
    stamps: list[Stamp] = []
    boundary = getattr(args, 'boundary', 'reflect')
    for wr, wc in pts:
        dense = {}
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                z = dr * dr / (2.0 * args.sigma_y * args.sigma_y) + dc * dc / (2.0 * args.sigma_x * args.sigma_x)
                w = math.exp(-z)
                rr = wr + dr
                cc = wc + dc
                if boundary == 'truncate':
                    if rr < 0 or rr >= f or cc < 0 or (cc >= f):
                        continue
                else:
                    rr = reflect_index(rr, f)
                    cc = reflect_index(cc, f)
                idx = rr * f + cc
                dense[idx] = dense.get(idx, 0.0) + w
        cells = sorted(dense)
        weights = [dense[i] for i in cells]
        stamps.append(Stamp(cells=cells, weights=weights, mass=sum(weights)))
    return stamps

def make_initial(args: argparse.Namespace) -> np.ndarray:
    f = args.field_size
    if args.initial_field is not None:
        values: list[float] = []
        with args.initial_field.open(newline='') as fh:
            reader = csv.reader(fh)
            for row in reader:
                values.extend((float(v) for v in row))
        if len(values) != f * f:
            raise ValueError(f'{args.initial_field} has {len(values)} values, expected {f * f}')
        return np.array(values, dtype=float)
    field = np.zeros(f * f, dtype=float)
    if args.field == 'random':
        rng = np.random.default_rng(args.seed)
        field[:] = rng.uniform(args.random_low, args.random_high, size=f * f)
    elif args.field == 'center':
        side = min(args.center_size, f)
        r0 = (f - side) // 2
        c0 = (f - side) // 2
        for r in range(r0, r0 + side):
            for c in range(c0, c0 + side):
                field[r * f + c] = args.center_height
    return field

def choose_budget(args: argparse.Namespace, initial: np.ndarray, mass: float, n: int) -> int:
    if args.budget is not None:
        return args.budget
    m = args.field_size * args.field_size
    init_mean = float(initial.sum()) / m
    raw = (args.target - init_mean) * m / mass
    budget = int(round(raw))
    return max(n, min(n * args.cmax, budget))

def load_warm_start(path: Path, n: int) -> list[int]:
    x = [0] * n
    with path.open(newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            i = int(row['waypoint'])
            x[i] = int(row['shot_count'])
    if any((v == 0 for v in x)):
        raise ValueError(f'Warm start {path} did not fill all {n} waypoints')
    return x

def evaluate(x: list[int], initial: np.ndarray, stamps: list[Stamp], field_size: int) -> tuple[float, float, float, float]:
    field = initial.copy()
    for count, stamp in zip(x, stamps):
        if count == 0:
            continue
        for cell, weight in zip(stamp.cells, stamp.weights):
            field[cell] += count * weight
    mean = float(field.mean())
    var = float(((field - mean) ** 2).mean())
    return (mean, var, float(field.min()), float(field.max()))

def build_gram_and_linear(initial: np.ndarray, stamps: list[Stamp], budget: int, field_size: int) -> tuple[float, np.ndarray, dict[tuple[int, int], float], float]:
    m = field_size * field_size
    mass = stamps[0].mass
    mean = (float(initial.sum()) + budget * mass) / m
    residual0 = initial - mean
    constant = float(np.dot(residual0, residual0)) / m
    linear = np.zeros(len(stamps), dtype=float)
    by_cell: list[list[tuple[int, float]]] = [[] for _ in range(m)]
    for j, stamp in enumerate(stamps):
        dot = 0.0
        for cell, weight in zip(stamp.cells, stamp.weights):
            dot += residual0[cell] * weight
            by_cell[cell].append((j, weight))
        linear[j] = 2.0 * dot / m
    quad: dict[tuple[int, int], float] = {}
    for entries in by_cell:
        if not entries:
            continue
        for a, (i, wi) in enumerate(entries):
            for j, wj in entries[a:]:
                if i <= j:
                    key = (i, j)
                    coef = wi * wj / m
                else:
                    key = (j, i)
                    coef = wi * wj / m
                quad[key] = quad.get(key, 0.0) + coef
    for (i, j), val in list(quad.items()):
        if i != j:
            quad[i, j] = 2.0 * val
    return (constant, linear, quad, mean)

def write_solution(path: Path, pts: list[tuple[int, int]], x: list[int]) -> None:
    with path.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['waypoint', 'row', 'col', 'shot_count'])
        for i, ((r, c), val) in enumerate(zip(pts, x)):
            writer.writerow([i, r, c, val])

def main() -> None:
    args = parse_args()
    pts = make_waypoints(args)
    stamps = build_stamps(args, pts)
    initial = make_initial(args)
    n = len(stamps)
    budget = choose_budget(args, initial, stamps[0].mass, n)
    constant, linear, quad, fixed_mean = build_gram_and_linear(initial, stamps, budget, args.field_size)
    warm_x = None
    if args.warm_start is not None:
        warm_x = load_warm_start(args.warm_start, n)
        warm_mean, warm_var, warm_min, warm_max = evaluate(warm_x, initial, stamps, args.field_size)
        print(f'warm_start mean={warm_mean:.9f} variance={warm_var:.9f} min={warm_min:.9f} max={warm_max:.9f} budget={sum(warm_x)}')
    model = Model('pufoam_budget_iqp')
    model.hideOutput(not args.verbose)
    model.setParam('limits/time', args.time_limit)
    model.setParam('limits/gap', args.gap_limit)
    xvars = [model.addVar(vtype='I', lb=1, ub=args.cmax, name=f'x_{j}') for j in range(n)]
    z = model.addVar(vtype='C', lb=0.0, name='variance_epigraph')
    model.addCons(quicksum(xvars) == budget, name='shot_budget')
    expr = constant
    expr += quicksum((float(linear[j]) * xvars[j] for j in range(n) if linear[j] != 0.0))
    expr += quicksum((float(v) * xvars[i] * xvars[j] for (i, j), v in quad.items()))
    model.addCons(z >= expr, name='variance_epigraph_constraint')
    model.setObjective(z, 'minimize')
    if warm_x is not None:
        sol = model.createSol()
        for var, val in zip(xvars, warm_x):
            model.setSolVal(sol, var, val)
        model.setSolVal(sol, z, warm_var)
        accepted = model.addSol(sol, free=True)
        print(f'warm_start_accepted={accepted}')
    t0 = time.time()
    model.optimize()
    elapsed = time.time() - t0
    status = model.getStatus()
    has_sol = model.getNSols() > 0
    print(f'status={status} elapsed={elapsed:.3f}s nodes={model.getNNodes()}')
    print(f'budget={budget} fixed_mean={fixed_mean:.9f} quad_terms={len(quad)}')
    if has_sol:
        sol_x = [int(round(model.getVal(v))) for v in xvars]
        mean, var, mn, mx = evaluate(sol_x, initial, stamps, args.field_size)
        print(f'scip_solution mean={mean:.9f} variance={var:.9f} min={mn:.9f} max={mx:.9f} budget={sum(sol_x)}')
        try:
            print(f'primal={model.getPrimalbound():.9f} dual={model.getDualbound():.9f} gap={model.getGap():.9f}')
        except Exception as exc:
            print(f'bound_info_unavailable={exc}')
        if args.write_solution is not None:
            write_solution(args.write_solution, pts, sol_x)
            print(f'wrote {args.write_solution}')
    else:
        try:
            print(f'dual={model.getDualbound():.9f}')
        except Exception:
            pass
if __name__ == '__main__':
    main()
