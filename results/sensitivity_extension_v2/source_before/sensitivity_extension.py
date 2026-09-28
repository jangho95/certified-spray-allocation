import determinism
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erf, ndtr, ndtri
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / 'work' / 'stage1'))
import progress
import rigor_pilot as rp
from convex_qp_tools import dense_A
from qp_audit import sha256_file, solve_qp_audit
from rounding import round_indexed
from scip_compare import build_stamps, make_waypoints
OUT = EXP / 'results' / 'sensitivity_extension'
INPUT = EXP / 'inputs' / 'sensitivity_extension'
WORK = EXP / 'work' / 'sensitivity_extension'
FIELDS = dict(zero='zero_F50.csv', random='random_u1_30_seed7_F50.csv', center30='center30_F50.csv', center100='center100_F50.csv')
TARGETS = [50, 100, 150, 200, 300, 400]
FAMILIES = ['uniform', 'beta_half', 'trunc_normal', 'correlated_uniform']
CMAX = 200

def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(progress.strict_json(data), indent=1, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)

def register_progress():

    def fn(state):
        if not any((s['id'] == '10' for s in state['stages'])):
            state['stages'].append(dict(id='10', title='추가 민감도 · 목표 두께·분포·격자', tasks=[dict(id=f'10.{i}', title=t, status='pending') for i, t in enumerate(['구현·모델 검산', '목표 두께 24개', '초기장 분포 120개', '고정 영역의 격자 12개', '저장 자료 재검산·요약'], 1)]))
    progress._update(fn)

def load_input(path):
    return np.loadtxt(path, delimiter=',').ravel()

def one_d(F):
    h = 50 / F
    edges = np.linspace(0, 50, F + 1)
    pts = 0.5 + 2 * np.arange(25)
    b = np.zeros((F, 25))
    sig, rad = (1.75, 3.5)
    for j, p in enumerate(pts):
        for u in [p, -p, 100 - p]:
            lo = np.maximum(edges[:-1], u - rad)
            hi = np.minimum(edges[1:], u + rad)
            valid = hi > lo
            b[valid, j] += sig * np.sqrt(np.pi / 2) * (erf((hi[valid] - u) / (sig * np.sqrt(2))) - erf((lo[valid] - u) / (sig * np.sqrt(2)))) / h
    return b

def continuous_gram():
    pts, sig, rad = (0.5 + 2 * np.arange(25), 1.75, 3.5)
    g = np.zeros((25, 25))
    for i, p in enumerate(pts):
        for j, q in enumerate(pts):
            for u in [p, -p, 100 - p]:
                for v in [q, -q, 100 - q]:
                    lo, hi = (max(0, u - rad, v - rad), min(50, u + rad, v + rad))
                    if hi > lo:
                        c = (u + v) / 2
                        g[i, j] += math.exp(-(u - v) ** 2 / (4 * sig ** 2)) * sig * np.sqrt(np.pi) / 2 * (erf((hi - c) / sig) - erf((lo - c) / sig))
    return g

@lru_cache(maxsize=2)
def operator(kind, F=50):
    if kind == 'benchmark':
        arg = SimpleNamespace(field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
        stamps = build_stamps(arg, make_waypoints(arg))
        A = dense_A(stamps, 2500, 625)
    else:
        b = one_d(F)
        A = np.einsum('ri,cj->rcij', b, b).reshape(F * F, 625)
        stamps = []
        for j in range(625):
            idx = np.flatnonzero(A[:, j])
            stamps.append(SimpleNamespace(cells=idx.tolist(), weights=A[idx, j].tolist()))
    return (A, stamps, float(np.mean(A.sum(axis=0))))

def save_operator(kind, F):
    A, st, mass = operator(kind, F)
    path = OUT / f'operator_{kind}_{F}.npz'
    np.savez_compressed(path, col_ptr=np.cumsum([0] + [len(t.cells) for t in st]), cells=np.concatenate([t.cells for t in st]), weights=np.concatenate([t.weights for t in st]), M=A.shape[0], N=A.shape[1])
    return path

def matrix_run(A, s0, B, x, tag, mode):
    n, m = (A.shape[1], A.shape[0])
    H = 2 * (A.T @ A) / m
    q = 2 * (A.T @ (s0 - (s0.sum() + B * np.mean(A.sum(axis=0))) / m)) / m
    infile, outfile = (Path(str(tag) + '.bin'), Path(str(tag) + '_out.bin'))
    with infile.open('wb') as f:
        np.array([n, B, CMAX], dtype='<i8').tofile(f)
        for z in (H, q, np.asarray(x, float)):
            np.asarray(z, dtype='<f8').tofile(f)
    r = subprocess.run([str(EXP / 'build' / 'matrix_exchange'), str(infile), str(outfile), mode], check=True, capture_output=True, text=True, timeout=300)
    xx = np.fromfile(outfile, dtype='<f8')
    infile.unlink()
    outfile.unlink()
    return (xx, json.loads(r.stdout))

def write_alloc(path, x):
    path.write_text('waypoint,shot_count\n' + ''.join((f'{i},{int(v)}\n' for i, v in enumerate(x))))

def read_alloc(path):
    return np.loadtxt(path, delimiter=',', skiprows=1)[:, -1]

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    INPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        return
    register_progress()
    progress.task('10.1', 'running', '독립 입력과 계산 조건 준비')
    jobs = []
    for field, filename in FIELDS.items():
        for T in TARGETS:
            jobs.append(dict(id=f'target_{field}_T{T}', group='target', field=field, T=T, F=50, input=str((EXP / 'inputs' / filename).relative_to(EXP)), kind='benchmark'))
    children = np.random.SeedSequence(2026092901).spawn(len(FAMILIES) * 30)
    impulse = np.zeros((50, 50))
    impulse[0, 0] = 1
    norm = float(np.linalg.norm(gaussian_filter(impulse, 2, mode='wrap', truncate=4)))
    for fi, family in enumerate(FAMILIES):
        for i in range(30):
            rng = np.random.default_rng(children[30 * fi + i])
            if family == 'uniform':
                values = rng.uniform(1, 30, (50, 50))
            elif family == 'beta_half':
                values = 1 + 29 * rng.beta(0.5, 0.5, (50, 50))
            elif family == 'trunc_normal':
                sd = 29 / np.sqrt(12)
                lo, hi = (ndtr((1 - 15.5) / sd), ndtr((30 - 15.5) / sd))
                values = 15.5 + sd * ndtri(rng.uniform(lo, hi, (50, 50)))
            else:
                z = gaussian_filter(rng.standard_normal((50, 50)), 2, mode='wrap', truncate=4) / norm
                values = 1 + 29 * ndtr(z)
            cid = f'dist_{family}_{i + 1:02d}'
            path = INPUT / f'{cid}.csv'
            np.savetxt(path, values, delimiter=',', fmt='%.17g')
            jobs.append(dict(id=cid, group='distribution', field=family, rep=i + 1, T=200, F=50, input=str(path.relative_to(EXP)), kind='benchmark'))
    for field, filename in FIELDS.items():
        for F in [50, 100, 200]:
            jobs.append(dict(id=f'mesh_{field}_F{F}', group='mesh', field=field, T=200, F=F, input=str((EXP / 'inputs' / filename).relative_to(EXP)), kind='cell_average'))
    for job in jobs:
        job['input_sha256'] = sha256_file(EXP / job['input'])
    codepaths = [Path(__file__), EXP / 'src' / 'matrix_exchange.cpp', EXP / 'pipeline' / 'rigor_pilot.py', EXP / 'pipeline' / 'qp_audit.py', EXP / 'pipeline' / 'rounding.py', EXP / 'build' / 'pufoam_solver']
    dump(OUT / 'protocol.json', dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), jobs=jobs, code_sha256={str(p.relative_to(EXP)): sha256_file(p) for p in codepaths}, Cmax=CMAX, candidate_rule='indexed largest remainder, greedy + exchange, indexed round + exchange', guarantee='exact rational evaluation of stored float64 A and initial field; no ideal erf/physical guarantee', random=dict(master_entropy=2026092901, n_per_family=30, range=[1, 30], mean_population=15.5, variance_matched=False, correlation='Gaussian copula; Gaussian filter sigma=2, wrap, truncate=4, exact filter L2 normalisation', statistics='descriptive; no significance or Wilks claim'), mesh=dict(domain=[0, 50, 0, 50], waypoints='.5+2i for i=0..24, both axes', sigma=1.75, half_support=3.5, initial='same piecewise constant 50x50 field at every mesh', note='separate continuous extension; h=1 differs from the point-sampled benchmark'), failure_policy='retain failures in the case list; do not exclude from denominators', environment=rp.env_versions()))

def solve_job(job):
    t0 = time.perf_counter()
    try:
        assert sha256_file(EXP / job['input']) == job['input_sha256']
        F = job['F']
        rp.M = F * F
        A, st, mass = operator(job['kind'], F)
        n, m = (A.shape[1], A.shape[0])
        sbase = load_input(EXP / job['input']).reshape(50, 50)
        s0 = np.repeat(np.repeat(sbase, F // 50, axis=0), F // 50, axis=1).ravel()
        raw = (job['T'] - float(s0.mean())) * m / mass
        B = max(n, min(n * CMAX, round(raw)))
        au = solve_qp_audit(A, s0, B, mass, CMAX)
        if au['status'] not in ('solved', 'solved inaccurate'):
            raise RuntimeError('OSQP: ' + au['status'])
        tag = WORK / job['id']
        ri = round_indexed(au['x'], B, CMAX).astype(float)
        write_alloc(Path(str(tag) + '_round.csv'), ri)
        cands = {'round': ri}
        solver_details = {}
        if job['kind'] == 'benchmark':
            base = [str(EXP / 'build' / 'pufoam_solver'), '--initial-field', str(EXP / job['input']), '--budget', str(B), '--target', str(job['T']), '--cmax', str(CMAX), '--quiet', '--full-precision']
            for method, extra in [('greedy_exchange', []), ('round_exchange', ['--init-allocation', str(tag) + '_round.csv'])]:
                prefix = str(tag) + '_' + method
                r = subprocess.run(base + extra + ['--out-prefix', prefix], check=True, capture_output=True, text=True, timeout=300)
                (WORK / (job['id'] + '_' + method + '.log')).write_text(r.stdout + r.stderr)
                cands[method] = read_alloc(prefix + '_allocation.csv')
        else:
            for method, mode in [('greedy_exchange', 'greedy'), ('round_exchange', 'exchange')]:
                cands[method], solver_details[method] = matrix_run(A, s0, B, ri, Path(str(tag) + '_' + method), mode)
        ex = rp.Exact(st, s0, B, CMAX)
        for x in cands.values():
            assert all(rp.check_int(x, B, CMAX).values())
        vals = {k: ex.value(x) for k, x in cands.items()}
        best = min(vals, key=vals.get)
        xu = cands[best]
        U = vals[best]
        (L, xb), ref_log = rp.refine(A, s0, B, CMAX, ex, au['x'])
        extra = False
        if float((ex.value(xb) - L) / max(U, Fraction(1))) > 1e-09:
            extra = True
            xc, cstatus, _ = rp.clarabel_x(A, s0, B, CMAX)
            solver_details['clarabel'] = cstatus
            (Lc, xbc), logc = rp.refine(A, s0, B, CMAX, ex, xc)
            if Lc > L:
                L, xb = (Lc, xbc)
        L = max(Fraction(0), L)
        xf = rp.feasible_point(xb, B, CMAX)
        if xf is None:
            raise RuntimeError('no exact feasible relaxation point')
        Vf = ex.value(xf)
        assert L <= Vf and L <= U
        f = A @ xu + s0
        wd = OUT / 'witness'
        wd.mkdir(exist_ok=True)
        witness = wd / (job['id'] + '.npz')
        np.savez_compressed(witness, s0=s0, xbar=xb, allocation=xu.astype(np.int64), B=B, Cmax=CMAX, M=m, candidate_names=np.array(list(cands)), candidates=np.array(list(cands.values()), dtype=np.int64), feasible_relaxation=np.array([rp.frs(v) for v in xf]))
        row = dict(job, B=B, BN=B / n, N=n, M=m, mass=mass, physical_mass=mass * (50 / F) ** 2, budget_clipped=not n <= round(raw) <= n * CMAX, mean_initial=float(s0.mean()), sd_initial=float(s0.std()), initial_neighbor_corr=float(np.corrcoef(sbase[:, :-1].ravel(), sbase[:, 1:].ravel())[0, 1]) if sbase.std() > 0 else None, mean_final=float(f.mean()), target_error=float(f.mean() - job['T']), L=rp.down(L), U=rp.up(U), width_pct=rp.up(100 * (U - L) / U) if U else 0, abs_width=rp.up(U - L), L_exact=rp.frs(L), U_exact=rp.frs(U), Vfeas_exact=rp.frs(Vf), continuous_interval_width=rp.up(Vf - L), bound_refine_relative_gap=float((Vf - L) / max(U, Fraction(1))), L_float=au['dual_lower'], osqp_status=au['status'], osqp_primal_residual=au['osqp_prim_res'], L_float_above_feasible_exact=bool(Fraction(au['dual_lower']) > Vf), extra_reference=extra, selected=best, candidate_U={k: rp.up(v) for k, v in vals.items()}, lower_active=int(np.sum(np.abs(au['x'] - 1) < 1e-05)), upper_active=int(np.sum(np.abs(au['x'] - CMAX) < 1e-05)), allocation_cap_active=int(np.sum(xu == CMAX)), allocation_min=int(xu.min()), allocation_max=int(xu.max()), column_mass_spread=float(np.ptp(A.sum(axis=0))), solver_details=solver_details, witness=str(witness.relative_to(OUT)), elapsed_s=time.perf_counter() - t0, failed=False, threads=determinism.assert_single_thread())
        dump(OUT / 'cases' / (job['id'] + '.json'), row)
        return row
    except Exception:
        row = dict(job, failed=True, error=traceback.format_exc(), elapsed_s=time.perf_counter() - t0)
        dump(OUT / 'cases' / (job['id'] + '.json'), row)
        return row

def verify_row(row):
    if row['failed']:
        return False
    op = np.load(OUT / f"operator_{row['kind']}_{row['F']}.npz")
    ptr = op['col_ptr']
    st = [SimpleNamespace(cells=op['cells'][ptr[j]:ptr[j + 1]], weights=op['weights'][ptr[j]:ptr[j + 1]]) for j in range(int(op['N']))]
    w = np.load(OUT / row['witness'])
    rp.M = int(w['M'])
    B = int(w['B'])
    C = int(w['Cmax'])
    assert rp.M == int(op['M'])
    ex = rp.Exact(st, w['s0'], B, C)
    assert all(rp.check_int(w['allocation'].astype(float), B, C).values())
    L = max(Fraction(0), ex.bound(w['xbar'])[0])
    U = ex.value(w['allocation'].astype(float))
    assert rp.frs(L) == row['L_exact'] and rp.frs(U) == row['U_exact']
    xf = [Fraction(str(v)) for v in w['feasible_relaxation']]
    assert sum(xf) == B and all((1 <= v <= C for v in xf))
    assert rp.frs(ex.value(xf)) == row['Vfeas_exact']
    for k, x in zip(w['candidate_names'], w['candidates']):
        assert all(rp.check_int(x.astype(float), B, C).values())
        assert rp.up(ex.value(x.astype(float))) == row['candidate_U'][str(k)]
    return True

def checks():
    register_progress()
    progress.task('10.1', 'running', '질량·격자 집계·연속 적분·exchange 검산')
    out = {}
    b50, b100, b200 = (one_d(50), one_d(100), one_d(200))
    exactmass = (1.75 * np.sqrt(2 * np.pi) * erf(3.5 / (1.75 * np.sqrt(2)))) ** 2
    for F, b in [(50, b50), (100, b100), (200, b200)]:
        out[f'mass_error_{F}'] = float(np.max(np.abs((b.sum(axis=0) * (50 / F)) ** 2 - exactmass)))
    out['nested_average_error'] = float(max(np.max(np.abs(b100.reshape(50, 2, 25).mean(axis=1) - b50)), np.max(np.abs(b200.reshape(100, 2, 25).mean(axis=1) - b100))))
    assert max(out.values()) < 1e-12
    G = continuous_gram()
    b800 = one_d(800)
    out['gram_800_relative_error'] = float(np.linalg.norm(50 / 800 * (b800.T @ b800) - G) / np.linalg.norm(G))
    assert out['gram_800_relative_error'] < 0.001
    rng = np.random.default_rng(1309)
    A = rng.uniform(0, 1, (9, 5))
    s0 = rng.uniform(0, 1, 9)
    B = 13
    x = np.array([3, 3, 3, 2, 2], dtype=float)
    y, info = matrix_run(A, s0, B, x, WORK / 'check', 'exchange')
    V = lambda z: float(np.mean((A @ z + s0 - (s0.sum() + B * np.mean(A.sum(axis=0))) / 9) ** 2))
    assert V(y) <= V(x) + 1e-10
    changes = []
    for i in range(5):
        for j in range(5):
            if i != j and y[i] > 1 and (y[j] < CMAX):
                z = y.copy()
                z[i] -= 1
                z[j] += 1
                changes.append(V(z) - V(y))
    out['exchange_min_direct_change'] = min(changes)
    assert min(changes) >= -1e-09
    A, st, mass = operator('benchmark')
    rp.M = 2500
    s0 = load_input(EXP / 'inputs' / 'zero_F50.csv')
    B = 28348
    au = solve_qp_audit(A, s0, B, mass, CMAX)
    ri = round_indexed(au['x'], B, CMAX)
    xm, info = matrix_run(A, s0, B, ri, WORK / 'benchmark_check', 'exchange')
    write_alloc(WORK / 'benchmark_check_round.csv', ri)
    subprocess.run([str(EXP / 'build' / 'pufoam_solver'), '--initial-field', str(EXP / 'inputs' / 'zero_F50.csv'), '--budget', str(B), '--init-allocation', str(WORK / 'benchmark_check_round.csv'), '--out-prefix', str(WORK / 'benchmark_cpp'), '--quiet'], check=True, capture_output=True)
    xc = read_alloc(WORK / 'benchmark_cpp_allocation.csv')
    ex = rp.Exact(st, s0, B, CMAX)
    out['generic_vs_original_cpp_U_diff'] = float(ex.value(xm) - ex.value(xc))
    out['generic_vs_original_alloc_equal'] = bool(np.array_equal(xm, xc))
    write_alloc(WORK / 'benchmark_matrix_allocation.csv', xm)
    ev = subprocess.run([str(EXP / 'build' / 'pufoam_solver'), '--initial-field', str(EXP / 'inputs' / 'zero_F50.csv'), '--eval-allocation', str(WORK / 'benchmark_matrix_allocation.csv'), '--quiet', '--full-precision'], check=True, capture_output=True, text=True)
    vcpp = float(re.search('variance=([0-9.eE+-]+)', ev.stdout).group(1))
    out['same_allocation_cpp_evaluation_error'] = float(vcpp - float(ex.value(xm)))
    assert abs(out['same_allocation_cpp_evaluation_error']) < 1e-09
    H = 2 * (A.T @ A) / 2500
    q = 2 * (A.T @ (s0 - (s0.sum() + B * mass) / 2500)) / 2500
    for label, x in [('generic', xm), ('original_cpp', xc)]:
        g = H @ x + q
        changes = g[None, :] - g[:, None] + 0.5 * (np.diag(H)[:, None] + np.diag(H)[None, :] - 2 * H)
        changes[x <= 1, :] = np.inf
        changes[:, x >= CMAX] = np.inf
        np.fill_diagonal(changes, np.inf)
        out[label + '_min_exchange_change'] = float(changes.min())
        assert changes.min() >= -1e-09
    dump(OUT / 'implementation_checks.json', out)
    progress.task('10.1', 'done', '질량 보존·해상도 집계·exchange 독립 평가 및 기존 C++ 비교 통과')
    return out

def mesh_compare(rows):
    rows = [r for r in rows if r['group'] == 'mesh' and (not r['failed'])]
    G = continuous_gram()
    A50, _, _ = operator('cell_average', 50)
    results = []
    for field in FIELDS:
        rr = sorted([r for r in rows if r['field'] == field], key=lambda r: r['F'])
        if len(rr) != 3:
            continue
        s0 = load_input(EXP / rr[0]['input'])
        xf = np.load(OUT / rr[-1]['witness'])['allocation'].astype(float)

        def cont(x):
            X = x.reshape(25, 25)
            sq = float(np.sum(X * (G @ X @ G))) / 2500
            cross = 2 * float(A50.T @ s0 @ x) / 2500
            mean = float(s0.mean() + x.sum() * rr[0]['physical_mass'] / 2500)
            return sq + cross + float(np.mean(s0 * s0)) - mean * mean
        vf = cont(xf)
        for r in rr:
            x = np.load(OUT / r['witness'])['allocation'].astype(float)
            b = one_d(r['F'])
            k = r['F'] // 50
            si = np.repeat(np.repeat(s0.reshape(50, 50), k, axis=0), k, axis=1)
            vfix = float(np.var(b @ xf.reshape(25, 25) @ b.T + si))
            vref = cont(x)
            results.append(dict(field=field, F=r['F'], h=50 / r['F'], B=r['B'], L=r['L'], U=r['U'], width_pct=r['width_pct'], continuous_evaluation=vref, continuous_relative_to_finest_pct=100 * (vref - vf) / vf, own_allocation_quadrature_error_pct=100 * (vref - r['U']) / vref, fixed_finest_grid_V=vfix, fixed_finest_continuous_V=vf, fixed_allocation_quadrature_error_pct=100 * (vf - vfix) / vf, allocation_L1_from_finest=int(np.abs(x - xf).sum())))
        assert len({r['B'] for r in rr}) == 1
    dump(OUT / 'mesh_comparison.json', results)
    return results

def publish(rows, complete=False):
    columns = [('field', '초기장/분포'), ('T', '목표 두께'), ('F', '격자 F'), ('B', '예산'), ('rep', '표본'), ('mean_final', '최종 평균'), ('U', '정수해 분산'), ('L', '검증 하한'), ('width_pct', '검증 폭 %'), ('sd_initial', '초기 표준편차'), ('initial_neighbor_corr', '인접 셀 상관'), ('upper_active', '연속해 상한 활성'), ('failed', '실패')]
    for group, title in [('target', '목표 두께'), ('distribution', '초기장 분포'), ('mesh', '고정 영역 격자')]:
        rs = [r for r in rows if r['group'] == group]
        progress.result('s10_' + group, 'S10 · ' + title, dict(columns=[dict(key=k, label=v) for k, v in columns], rows=rs, note='저장 float64 모델에 대한 정확 산술 인증. 격자 계열은 셀 평균 Gaussian 확장 모델. ' + ('완료 결과' if complete else '실행 중: 완료한 사례를 순차 표시')))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'check', 'pilot', 'run', 'verify'])
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    prepare()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    if args.command == 'prepare':
        return
    for rel, h in protocol['code_sha256'].items():
        assert sha256_file(EXP / rel) == h, f'changed code: {rel}; preserve protocol and record changes explicitly'
    if args.command == 'check':
        print(json.dumps(checks()))
        return
    for kind, F in [('benchmark', 50), ('cell_average', 50), ('cell_average', 100), ('cell_average', 200)]:
        if not (OUT / f'operator_{kind}_{F}.npz').exists():
            save_operator(kind, F)
    operator.cache_clear()
    jobs = protocol['jobs']
    if args.command == 'pilot':
        names = ['target_zero_T50', 'target_center100_T400', 'dist_correlated_uniform_01', 'mesh_center30_F50']
        jobs = [j for j in jobs if j['id'] in names]
    rows = []
    if args.command != 'verify':
        for group, tid in [('target', '10.2'), ('distribution', '10.3'), ('mesh', '10.4')]:
            gj = [j for j in jobs if j['group'] == group]
            if not gj:
                continue
            progress.task(tid, 'running', f'{len(gj)}개 사례 계산')
            pending = []
            group_rows = []
            for j in gj:
                path = OUT / 'cases' / (j['id'] + '.json')
                if path.exists():
                    group_rows.append(json.loads(path.read_text()))
                else:
                    pending.append(j)
            with cf.ProcessPoolExecutor(max_workers=min(args.workers, 2 if group == 'mesh' else args.workers)) as executor:
                for r in executor.map(solve_job, pending):
                    group_rows.append(r)
                    progress.detail(tid, f"{len(group_rows)}/{len(gj)} 완료, 실패 {sum((x['failed'] for x in group_rows))}")
                    publish(rows + group_rows)
                    print(r['id'], 'FAILED' if r['failed'] else f"width={r['width_pct']:.6g}%", flush=True)
            rows += group_rows
            failed = sum((r['failed'] for r in group_rows))
            progress.task(tid, 'warn' if args.command == 'pilot' or failed else 'done', f'{len(group_rows)}/{len(gj)} 계산, 실패 {failed}' + (' (예비 실행)' if args.command == 'pilot' else ''))
        dump(OUT / ('pilot_rows.json' if args.command == 'pilot' else 'rows.json'), rows)
        if args.command == 'pilot':
            return
    else:
        rows = json.loads((OUT / 'rows.json').read_text())
    progress.task('10.5', 'running', '저장한 연산자·배분·기준점으로 정확 산술 재검산')
    with cf.ProcessPoolExecutor(max_workers=args.workers) as executor:
        verified = list(executor.map(verify_row, rows))
    dump(OUT / 'verification.json', dict(n=len(rows), passed=sum(verified), failed=len(rows) - sum(verified), per_case={r['id']: v for r, v in zip(rows, verified)}))
    mesh_compare(rows)
    publish(rows, complete=True)
    progress.task('10.5', 'done' if all(verified) else 'warn', f'검산 통과 {sum(verified)}/{len(rows)}; 요약은 별도 보고')
if __name__ == '__main__':
    main()
