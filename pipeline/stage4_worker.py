import time
T_START = time.perf_counter()
import determinism
import argparse
import json
import os
import re
import resource
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import osqp
import scipy.linalg as sla
import scipy.sparse as sp
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP / 'work' / 'stage1')]
from convex_qp_tools import dense_A, largest_remainder_round
from qp_audit import lower_bound_at, solve_qp_audit
from rounding import round_indexed
import progress
from scip_compare import build_stamps, make_waypoints
T_IMPORT = time.perf_counter() - T_START
BIN = EXP / 'build' / 'pufoam_solver'
INPUTS = EXP / 'inputs'
RAW = EXP / 'results' / 'stage4' / 'raw'
WORK = EXP / 'work' / 'stage4'
CAP = 200

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

class Clock:

    def __init__(self):
        self.t, self.rss = ({}, {})

    def __call__(self, name, fn, *a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0
        self.rss[name] = rss_mb()
        return out

def child(cmd, log):
    t0 = time.perf_counter()
    with open(log, 'w') as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        _, status, ru = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    wall = time.perf_counter() - t0
    out = Path(log).read_text()
    if p.returncode != 0:
        raise RuntimeError(f'{cmd[0]} failed: {out[-300:]}')
    return (out, wall, ru.ru_maxrss / 1024.0)

def cpp_parse(out):
    g = lambda k: int(re.search(f'{k}=(\\d+)', out).group(1))
    return dict(moves=g('exchange_moves'), gram_ms=g('gram_ms'), repair_ms=g('repair_ms'), total_ms=g('total_ms'), rss_mb=g('peak_rss_kb') / 1024.0)

def geometry(F):
    cnt = (F - 1) // 2 + 1
    return (SimpleNamespace(field_size=F, waypoint_rows=cnt, waypoint_cols=cnt, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect'), cnt)

def load_field(name):
    return np.array([float(t) for t in (INPUTS / name).read_text().replace('\n', ',').split(',') if t.strip()])

def target_budget(s0, M, mass, N):
    return int(max(N, min(N * CAP, round((200.0 - float(s0.mean())) * M / mass))))

def cpp_cmd(inp, F, cnt, B, prefix, init=None):
    cmd = [str(BIN), '--initial-field', str(INPUTS / inp), '--field-size', str(F), '--waypoint-rows', str(cnt), '--waypoint-cols', str(cnt), '--budget', str(B), '--cmax', str(CAP), '--quiet', '--full-precision', '--out-prefix', str(prefix)]
    return cmd + (['--init-allocation', str(init)] if init else [])

def read_alloc(path, n):
    x = np.zeros(n, dtype=int)
    for line in Path(path).read_text().splitlines()[1:]:
        f = line.split(',')
        x[int(f[0])] = int(f[-1])
    return x

def write_alloc(path, x):
    Path(path).write_text('waypoint,shot_count\n' + ''.join((f'{j},{int(v)}\n' for j, v in enumerate(x))))

def variance(A, x, s0):
    f = A @ np.asarray(x, float) + s0
    return float(np.mean((f - f.mean()) ** 2))

def one_budget(clk, A, s0, B, mass, inp, F, cnt, tag, n):
    au = clk('qp_total', solve_qp_audit, A, s0, B, mass, CAP)
    for k in ('t_gram', 't_setup', 't_solve', 't_dual'):
        clk.t[k[2:]] = clk.t.get(k[2:], 0.0) + au[k]
    xr = clk('round', lambda: (largest_remainder_round(au['x'], B, CAP), round_indexed(au['x'], B, CAP)))[1]
    rpath = WORK / f'{tag}_round.csv'
    write_alloc(rpath, xr)
    out_g, w_g, rss_g = child(cpp_cmd(inp, F, cnt, B, WORK / f'{tag}_greedy'), WORK / f'{tag}_greedy.log')
    out_r, w_r, rss_r = child(cpp_cmd(inp, F, cnt, B, WORK / f'{tag}_rr', rpath), WORK / f'{tag}_rr.log')
    clk.t['greedy_repair_cpp'] = clk.t.get('greedy_repair_cpp', 0.0) + w_g
    clk.t['round_repair_cpp'] = clk.t.get('round_repair_cpp', 0.0) + w_r
    xg = read_alloc(WORK / f'{tag}_greedy_allocation.csv', n)
    xrr = read_alloc(WORK / f'{tag}_rr_allocation.csv', n)
    vals = clk('eval', lambda: dict(V_greedy=variance(A, xg, s0), V_round=variance(A, xr, s0), V_round_repair=variance(A, xrr, s0)))
    return dict(B=B, L=au['dual_lower'], primal=au['primal'], iters=au['iter'], status=au['status'], cpp_greedy=cpp_parse(out_g) | dict(wall=w_g, wait4_maxrss_mb=rss_g), cpp_round_repair=cpp_parse(out_r) | dict(wall=w_r, wait4_maxrss_mb=rss_r), **vals)

def run(args):
    WORK.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    threads = determinism.assert_single_thread()
    clk = Clock()
    clk.rss['import'] = rss_mb()
    g, cnt = geometry(args.F)
    inp = f'{args.field}_F{args.F}.csv'
    tag = f'{args.mode}_F{args.F}_{args.field}_r{args.rep}'
    res = dict(mode=args.mode, F=args.F, field=args.field, rep=args.rep, input=inp, t_import=T_IMPORT, threads=threads, pid=os.getpid())
    if args.mode == 'pso':
        sys.path.insert(0, str(EXP / 'work' / 'stage1'))
        import pso_repeated_runs as P
        from paper_pso import run_pso
        a = P.make_args(dict(field='zero' if args.field == 'zero' else 'center', field_size=args.F, waypoint_rows=cnt, waypoint_cols=cnt, initial_field=INPUTS / inp), args.rep)
        t0 = time.perf_counter()
        sim, gbest, metrics, _, _ = run_pso(a)
        total = time.perf_counter() - t0
        res.update(N=sim.n, M=sim.m, B=int(gbest.sum()), t_total=total, t_opt=metrics['elapsed_s'], t_model=total - metrics['elapsed_s'], variance=metrics['variance'], mean=metrics['mean'], rss_python_peak=rss_mb(), P_bytes=sim.P.nbytes, A_bytes=sim.A.nbytes)
    else:
        pts = clk('waypoints', make_waypoints, g)
        stamps = clk('stamps', build_stamps, g, pts)
        n, m = (len(stamps), args.F ** 2)
        A = clk('dense_A', dense_A, stamps, m, n)
        s0 = clk('load_input', load_field, inp)
        mass = stamps[0].mass
        B0 = target_budget(s0, m, mass, n)
        res.update(N=n, M=m, B_target=B0, A_bytes=A.nbytes, gram_bytes=8 * n * n)
        if args.mode == 'single':
            res['budget'] = one_budget(clk, A, s0, B0, mass, inp, args.F, cnt, tag, n)
        elif args.mode == 'sweep':
            budgets = sorted({int(round(b)) for b in np.linspace(0.85 * B0, 1.05 * B0, 21)})
            res['budgets'] = []
            for i, B in enumerate(budgets, 1):
                progress.detail('4.2', f'{args.job} sweep · F={args.F} · rep {args.rep} · 예산 {i}/{len(budgets)} (B={B})')
                res['budgets'].append(one_budget(clk, A, s0, B, mass, inp, args.F, cnt, f'{tag}_b{B}', n))
        elif args.mode == 'sweep_reuse':
            budgets = sorted({int(round(b)) for b in np.linspace(0.85 * B0, 1.05 * B0, 21)})
            gram = clk('gram', lambda: A.T @ A)
            H = 2.0 / m * gram
            As0 = clk('gram', lambda: A.T @ s0)
            kappa = clk('gram', lambda: A.T @ np.ones(m))
            G = sp.vstack([sp.csc_matrix(np.ones((1, n))), sp.identity(n)]).tocsc()
            Gd = G.toarray()
            prob = osqp.OSQP()

            def vecs(B):
                mean = (float(s0.sum()) + B * mass) / m
                r0 = s0 - mean
                q = 2.0 / m * (As0 - mean * kappa)
                return (q, float(r0 @ r0) / m, np.concatenate([[float(B)], np.full(n, 1.0)]), np.concatenate([[float(B)], np.full(n, float(CAP))]))
            q0, c0, lo0, hi0 = vecs(budgets[0])
            clk('setup', prob.setup, sp.csc_matrix(H), q0, G, lo0, hi0, eps_abs=1e-08, eps_rel=1e-08, max_iter=400000, polishing=True, warm_starting=False, verbose=False)
            chol = clk('dual', sla.cho_factor, H)
            out = []
            for B in budgets:
                q, const, lo, hi = vecs(B)
                clk('update', prob.update, q=q, l=lo, u=hi)
                r = clk('solve', prob.solve)
                y = np.asarray(r.y)

                def dual():
                    v = q + Gd.T @ y
                    z = sla.cho_solve(chol, v)
                    rel = float(np.linalg.norm(H @ z - v)) / (1.0 + float(np.linalg.norm(v)))
                    raw = const - 0.5 * float(v @ z) - float(hi @ np.maximum(y, 0)) + float(lo @ np.maximum(-y, 0))
                    return lower_bound_at(dict(dual_raw=raw, primal=float(r.info.obj_val + const), range_rel_resid=rel), 1e-10)
                lb = clk('dual', dual)
                out.append(dict(B=B, L=lb['dual_lower'], primal=float(r.info.obj_val + const), iters=int(r.info.iter), status=str(r.info.status)))
            res['budgets'] = out
    res['times'] = clk.t
    res['rss_python_after'] = clk.rss
    res['rss_python_peak'] = rss_mb()
    res['t_wall_main'] = time.perf_counter() - T_START
    (RAW / f'{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ('mode', 'F', 'field', 'rep', 't_wall_main', 'rss_python_peak')}))
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['single', 'sweep', 'sweep_reuse', 'pso'])
    ap.add_argument('--F', type=int, required=True)
    ap.add_argument('--field', default='zero')
    ap.add_argument('--rep', type=int, default=1)
    ap.add_argument('--job', default='', help='runner position, e.g. [55/57], shown in live progress')
    run(ap.parse_args())
