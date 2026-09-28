import determinism
import concurrent.futures as cf
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP / 'work' / 'stage1')]
import incumbents
import progress
from convex_qp_tools import dense_A, largest_remainder_round
from qp_audit import append_records, make_record, sha256_file, solve_qp_audit
from rounding import round_indexed
from scip_compare import build_stamps, make_waypoints
BIN = EXP / 'build' / 'pufoam_solver'
INPUTS = EXP / 'inputs'
WORK = EXP / 'work' / 'stage7'
OUT = EXP / 'results' / 'stage7'
RUN_ID = incumbents.new_run_id(__file__)
F, S, CMAX = (50, 2, 200)
FIELDS = [('zero', 'zero_F50.csv'), ('random s7', 'random_u1_30_seed7_F50.csv'), ('center-30', 'center30_F50.csv'), ('center-100', 'center100_F50.csv')]
RHO_A = (1.5, 2, 3, 8, 20, 45)
SIGMA_B = (1.0, 1.25, 1.75, 2.5, 3.5)
RHO_B = (1.5, 3, 8, 45)

def support(sigma):
    return 2 * math.ceil(3 * sigma) + 1

def jobs():
    out = []
    for field, inp in FIELDS:
        for rho in RHO_A:
            out.append(dict(series='A', field=field, input=inp, kernel=7, sigma=1.75, rho=rho))
        for sg in SIGMA_B:
            for rho in RHO_B:
                out.append(dict(series='B', field=field, input=inp, kernel=support(sg), sigma=sg, rho=rho))
    return out

def load_field(name):
    return np.array([float(t) for t in (INPUTS / name).read_text().replace('\n', ',').split(',') if t.strip()])

def write_alloc(path, x):
    Path(path).write_text('waypoint,shot_count\n' + ''.join((f'{j},{int(v)}\n' for j, v in enumerate(x))))
    return str(path)

def read_alloc(path, n):
    x = np.zeros(n, dtype=int)
    for line in Path(path).read_text().splitlines()[1:]:
        f = line.split(',')
        x[int(f[0])] = int(f[-1])
    return x

def run(job):
    g = SimpleNamespace(field_size=F, waypoint_rows=25, waypoint_cols=25, spray_interval=S, kernel_size=job['kernel'], sigma_x=job['sigma'], sigma_y=job['sigma'], boundary='reflect')
    st = build_stamps(g, make_waypoints(g))
    n, m = (len(st), F * F)
    A = dense_A(st, m, n)
    s0 = load_field(job['input'])
    mass = st[0].mass
    B = int(round(job['rho'] * n))
    au = solve_qp_audit(A, s0, B, mass, CMAX)
    tag = WORK / f"{job['series']}_{job['input'][:-4]}_k{job['kernel']}_s{job['sigma']:g}_B{B}"
    cand = {'반올림(원고 규칙)': write_alloc(f'{tag}_round_paper.csv', largest_remainder_round(au['x'], B, CMAX))}
    ri = write_alloc(f'{tag}_round_idx.csv', round_indexed(au['x'], B, CMAX))
    base = [str(BIN), '--initial-field', str(INPUTS / job['input']), '--kernel-size', str(job['kernel']), '--sigma', f"{job['sigma']}", '--budget', str(B), '--cmax', str(CMAX), '--quiet']
    subprocess.run(base + ['--out-prefix', f'{tag}_greedy'], check=True, capture_output=True)
    subprocess.run(base + ['--out-prefix', f'{tag}_rr', '--init-allocation', ri], check=True, capture_output=True)
    cand['greedy+repair'] = f'{tag}_greedy_allocation.csv'
    cand['반올림→repair'] = f'{tag}_rr_allocation.csv'
    vals = {}
    for name, path in cand.items():
        x = read_alloc(path, n)
        f = A @ x + s0
        vals[name] = (float(np.mean((f - f.mean()) ** 2)), bool(x.sum() == B and x.min() >= 1 and (x.max() <= CMAX)), path)
    return dict(job, N=n, B=B, BN=B / n, mass=mass, L=au['dual_lower'], V_QP=au['primal'], status=au['status'], x_R=au['x'].tolist(), n_lower_active=au['n_lower_active'], n_upper_active=au['n_upper_active'], x_R_max=au['x_max'], residuals=dict(prim=au['osqp_prim_res'], dual=au['osqp_dual_res'], Hz_v=au['range_rel_resid']), settings=au['settings'], cands=vals)

def decompose(r):
    g = SimpleNamespace(field_size=F, waypoint_rows=25, waypoint_cols=25, spray_interval=S, kernel_size=r['kernel'], sigma_x=r['sigma'], sigma_y=r['sigma'], boundary='reflect')
    st = build_stamps(g, make_waypoints(g))
    n, m = (len(st), F * F)
    A = dense_A(st, m, n)
    s0 = load_field(r['input'])
    xR = np.array(r['x_R'])
    xU = read_alloc(EXP / r['U_alloc'], n)
    rB = s0 - (s0.sum() + r['B'] * r['mass']) / m
    grad = 2.0 / m * (A.T @ (A @ xR + rB))
    d = xU - xR
    first, curv = (float(grad @ d), float(A @ d @ (A @ d) / m))
    return dict(first=first, curv=curv, numeric=r['V_QP'] - r['L'], split_check=r['U'] - r['V_QP'] - (first + curv), U_at_lower=int(np.sum(xU == 1)), U_at_upper=int(np.sum(xU == CMAX)), U_max=int(xU.max()))

def main():
    threads = determinism.assert_single_thread()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    js = jobs()
    progress.task('7.1', 'running', '계열 A: F=50, s=2, 7×7, σ=1.75, C_max=200, 초기장 4개')
    progress.task('7.4', 'running', '계열 B(별도): 원시 Gaussian, 지지 ±⌈3σ⌉, σ 5개')
    res = []
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        for k, r in enumerate(ex.map(run, js), 1):
            res.append(r)
            if k % 8 == 0 or k == len(js):
                progress.detail('7.1', f'계열 A·B 인스턴스 {k}/{len(js)} 계산(8개 병렬)')
    for r in res:
        for name, (v, ok, path) in r['cands'].items():
            incumbents.register(inp=r['input'], B=r['B'], cap=CMAX, method=name, V=v, alloc_path=path, run_id=RUN_ID, feasible=ok, F=F, kernel=r['kernel'], sigma=(r['sigma'], r['sigma']), spacing=S, stage=7, experiment=f"range_{r['series']}")
    best = incumbents.rebuild_best()
    rows, records = ([], [])
    for r in res:
        k = incumbents.key(r['input'], r['B'], CMAX, F=F, kernel=r['kernel'], sigma=(r['sigma'], r['sigma']), spacing=S)
        r.update(U=best[k]['V'], U_method=best[k]['method'], U_alloc=best[k]['alloc_file'])
        dec = decompose(r)
        W = r['U'] - r['L']
        row = dict(series=r['series'], field=r['field'], kernel=f"{r['kernel']}×{r['kernel']}", sigma=r['sigma'], sigma_over_s=r['sigma'] / S, rho=r['rho'], B=r['B'], BN=r['BN'], mass=r['mass'], L=r['L'], V_QP=r['V_QP'], U=r['U'], U_method=r['U_method'], abs_width=W, rel_width_pct=100 * W / r['U'], first=dec['first'], curv=dec['curv'], numeric=dec['numeric'], curv_share=dec['curv'] / W if W > 0 else None, split_check=dec['split_check'], relax_lower_active=r['n_lower_active'], relax_upper_active=r['n_upper_active'], relax_x_max=r['x_R_max'], U_at_lower=dec['U_at_lower'], U_at_upper=dec['U_at_upper'], U_max=dec['U_max'], status=r['status'])
        row['flag'] = 'ok' if row['rel_width_pct'] <= 1 else 'warn' if row['rel_width_pct'] <= 5 else 'fail'
        rows.append(row)
        records.append(make_record(stage=7, experiment=f"range_{r['series']}", case=f"{r['field']} k{r['kernel']} sigma{r['sigma']:g} rho{r['rho']:g}", input=r['input'], input_sha256=sha256_file(INPUTS / r['input']), M=F * F, N=r['N'], B=r['B'], Cmax=CMAX, kernel=r['kernel'], sigma=r['sigma'], spacing=S, boundary='reflect', solver='OSQP 1.1.3 + C++ exchange repair', settings=r['settings'], status=r['status'], U=r['U'], L=r['L'], residuals=r['residuals'], U_method=r['U_method'], U_alloc_sha256=best[k]['alloc_sha256'], run_id=RUN_ID, threads=threads, decomposition=dec, stamp_mass=r['mass']))
    append_records(OUT / 'records.jsonl', records)
    (OUT / 'range_rows.json').write_text(json.dumps(dict(run_id=RUN_ID, threads=threads, rows=rows), indent=1, default=float, ensure_ascii=False))
    return rows
if __name__ == '__main__':
    main()
