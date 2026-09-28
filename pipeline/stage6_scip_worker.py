from package_paths import load_json
import determinism
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP / 'work' / 'stage1')]
from convex_qp_tools import dense_A
from pyscipopt import Model, quicksum
from scip_compare import build_gram_and_linear, build_stamps, make_waypoints
from scip_soc_compare import build_soc_model, complete_warm_start

def load_field(path):
    return np.array([float(t) for t in Path(path).read_text().replace('\n', ',').split(',') if t.strip()])

def read_alloc(path, n):
    x = np.zeros(n, dtype=int)
    for line in Path(path).read_text().splitlines()[1:]:
        f = line.split(',')
        x[int(f[0])] = int(f[-1])
    return x

def main(spec_path, form, out_path):
    spec = load_json(Path(spec_path).read_text())
    g = SimpleNamespace(field_size=spec['F'], waypoint_rows=spec['rows'], waypoint_cols=spec['rows'], spray_interval=spec['s'], kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
    st = build_stamps(g, make_waypoints(g))
    n, m = (len(st), spec['F'] ** 2)
    A = dense_A(st, m, n)
    s0 = load_field(spec['input_path'])
    B, cmax = (spec['B'], spec['cmax'])
    x0 = read_alloc(spec['warm_alloc'], n)
    res = dict(instance=spec['id'], form=form, time_limit=spec['time_limit'], mem_limit=spec['mem_limit'])
    t0 = time.perf_counter()
    if form == 'expanded':
        const, lin, quad, _ = build_gram_and_linear(s0, st, B, spec['F'])
        model = Model('expanded')
        xs = [model.addVar(vtype='I', lb=1, ub=cmax, name=f'x{j}') for j in range(n)]
        model.addCons(quicksum(xs) == B)
        t = model.addVar(vtype='C', lb=0.0, name='t')
        model.addCons(const + quicksum((float(lin[j]) * xs[j] for j in range(n))) + quicksum((float(c) * xs[i] * xs[j] for (i, j), c in quad.items())) <= t)
        model.setObjective(t, 'minimize')
        tval = const + float(lin @ x0) + sum((c * x0[i] * x0[j] for (i, j), c in quad.items()))
        sol = model.createSol()
        for v, val in zip(xs, x0):
            model.setSolVal(sol, v, float(val))
        model.setSolVal(sol, t, tval)
    elif form in ('soc', 'soc_cont'):
        model, xs = build_soc_model(st, s0, B, m, cmax, integer=form == 'soc')
        sol = complete_warm_start(model, xs, list(x0), st, s0, B, m) if form == 'soc' else None
    else:
        raise SystemExit(f'unknown form {form}')
    res['build_s'] = time.perf_counter() - t0
    model.hideOutput(True)
    model.setParam('limits/time', float(spec['time_limit']))
    model.setParam('limits/memory', float(spec['mem_limit']))
    if form != 'soc_cont':
        model.setParam('limits/gap', 0.0)
        res['warm_checksol'] = bool(model.checkSol(sol, printreason=False))
        res['warm_accepted'] = bool(model.addSol(sol, free=True))
    t1 = time.perf_counter()
    model.optimize()
    res.update(status=model.getStatus(), solve_wall_s=time.perf_counter() - t1, scip_time_s=model.getSolvingTime(), primal_bound=model.getPrimalbound() if model.getNSols() > 0 else None, dual_bound=model.getDualbound(), gap=model.getGap() if model.getNSols() > 0 else None, nodes=model.getNNodes(), nsols=model.getNSols())
    if form != 'soc_cont' and model.getNSols() > 0:
        best = model.getBestSol()
        x = np.array([int(round(model.getSolVal(best, v))) for v in xs])
        f = A @ x + s0
        res.update(V_eval=float(np.mean((f - f.mean()) ** 2)), feasible=bool(x.sum() == B and x.min() >= 1 and (x.max() <= cmax)), same_as_warm=bool(np.array_equal(x, x0)))
        ap = Path(out_path).with_suffix('.alloc.csv')
        ap.write_text('waypoint,shot_count\n' + ''.join((f'{j},{int(v)}\n' for j, v in enumerate(x))))
        res['alloc'] = str(ap)
    Path(out_path).write_text(json.dumps(res, indent=1, default=float))
if __name__ == '__main__':
    main(*sys.argv[1:4])
