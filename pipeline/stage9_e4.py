from package_paths import load_json, data_path
import determinism
import argparse
import concurrent.futures as cf
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
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
from stage2_e1 import clarabel_solve
BIN = EXP / 'build' / 'pufoam_solver'
INPUTS = EXP / 'inputs'
NEW = INPUTS / 'stage9'
WORK = EXP / 'work' / 'stage9'
OUT = EXP / 'results' / 'stage9'
PREREG = OUT / 'preregistration.json'
F, S, K, SIGMA, CMAX, TARGET, M = (50, 2, 7, 1.75, 200, 200.0, 2500)
N_SAMPLES, NPTS = (59, 21)
ENTROPY = 16607498802698493863187215274244358790
DEV = [f'random_u1_30_seed{k}_F50.csv' for k in range(1, 11)]
CODE = ['pipeline/stage9_e4.py', 'pipeline/qp_audit.py', 'pipeline/rounding.py', 'pipeline/stage2_e1.py', 'pipeline/determinism.py', 'work/stage1/convex_qp_tools.py', 'work/stage1/scip_compare.py', 'build/pufoam_solver']
VALID_TOL = 1e-09
FAILURE_RULES = ['OSQP status other than solved, or the range test / fallback of qp_audit: L_osqp = 0 for that instance (width as computed, 100% when L = 0); the instance is kept.', 'Clarabel status other than Solved or its range test failing: L_c = 0 and no validity check for that instance.', 'C++ non-zero exit or missing allocation: rerun once; if it fails again the candidate is missing and U uses the remaining feasible candidates (paper rounding always exists).', 'Candidate with sum != B or outside [1, C_max]: discarded and recorded.', 'U <= 0: relative width undefined, recorded as +inf (worst) in order statistics; absolute width reported.', 'Validity flag: L_osqp or L_c > P_c + 1e-9 max(1, |P_c|) (P_c = Clarabel relaxation objective): counted and reported; L is not replaced.', 'Worker exception for a field: the whole field is rerun once; if it fails again T1 = T2 = +inf for it. The reported sample size is always 59.', 'No field, budget or instance is excluded; changes after registration are listed as deviations in the report.']
FINAL = re.compile('total_shots=(\\d+)')

def entropy():
    return ENTROPY

def operator():
    g = SimpleNamespace(field_size=F, waypoint_rows=25, waypoint_cols=25, spray_interval=S, kernel_size=K, sigma_x=SIGMA, sigma_y=SIGMA, boundary='reflect')
    st = build_stamps(g, make_waypoints(g))
    return (dense_A(st, M, len(st)), st[0].mass, len(st))

def load_field(inp):
    return np.array([float(t) for t in (INPUTS / inp).read_text().replace('\n', ',').split(',') if t.strip()])

def target_budget(s0, mass, n):
    raw = (TARGET - float(s0.sum()) / M) * M / mass
    return max(n, min(n * CMAX, int(round(raw))))

def grid(bstar):
    return sorted(set((int(round(b)) for b in np.linspace(0.85 * bstar, 1.05 * bstar, NPTS))))

def write_alloc(path, x):
    Path(path).write_text('waypoint,shot_count\n' + ''.join((f'{j},{int(v)}\n' for j, v in enumerate(x))))
    return str(path)

def read_alloc(path, n):
    x = np.zeros(n, dtype=int)
    for line in data_path(path).read_text().splitlines()[1:]:
        f = line.split(',')
        x[int(f[0])] = int(f[-1])
    return x

def cpp(cmd, prefix):
    for attempt in (1, 2):
        r = subprocess.run(cmd + ['--out-prefix', prefix], capture_output=True, text=True)
        path = Path(f'{prefix}_allocation.csv')
        if r.returncode == 0 and path.exists():
            return (str(path), r.stdout, attempt)
    return (None, r.stdout + r.stderr, 2)

def rel(U, L):
    return 100.0 * (U - L) / U if U > 0 else math.inf

def run_field(job):
    t0 = time.perf_counter()
    name, inp = (job['name'], job['input'])
    A, mass, n = operator()
    s0 = load_field(inp)
    bstar = target_budget(s0, mass, n)
    wd = WORK / job['set'] / name
    wd.mkdir(parents=True, exist_ok=True)
    base = [str(BIN), '--initial-field', str(INPUTS / inp), '--kernel-size', str(K), '--sigma', f'{SIGMA}', '--cmax', str(CMAX), '--quiet']
    rows, cpp_bstar = ([], None)
    assert bstar in grid(bstar)
    for B in grid(bstar):
        au = solve_qp_audit(A, s0, B, mass, CMAX)
        L = au['dual_lower'] if au['status'] == 'solved' else 0.0
        cl = clarabel_solve(dict(A=A, s0=s0, B=B, mass=mass, cmax=CMAX, n=n))
        cl_ok = cl['status'] == 'Solved' and (not cl['fallback'])
        Lc = cl['L_lagr'] if cl_ok else 0.0
        tol = VALID_TOL * max(1.0, abs(cl['primal']))
        tag = wd / f'B{B}'
        cand = {'반올림(원고 규칙)': write_alloc(f'{tag}_round_paper.csv', largest_remainder_round(au['x'], B, CMAX))}
        ri = write_alloc(f'{tag}_round_idx.csv', round_indexed(au['x'], B, CMAX))
        gcmd = base + ([] if B == bstar else ['--budget', str(B)])
        cand['greedy+repair'], gout, ga = cpp(gcmd, f'{tag}_greedy')
        cand['반올림→repair'], _, ra = cpp(base + ['--budget', str(B), '--init-allocation', ri], f'{tag}_rr')
        if B == bstar:
            m = FINAL.search(gout or '')
            cpp_bstar = int(m.group(1)) if m else None
        vals = {}
        for cname, path in cand.items():
            if path is None:
                vals[cname] = dict(V=None, feasible=False, path=None)
                continue
            x = read_alloc(path, n)
            f = A @ x + s0
            vals[cname] = dict(V=float(np.mean((f - f.mean()) ** 2)), feasible=bool(x.sum() == B and x.min() >= 1 and (x.max() <= CMAX)), path=path)
        feas = {k: v['V'] for k, v in vals.items() if v['feasible']}
        U_rev = min(feas.values()) if feas else math.inf
        U_paper = min((v for k, v in feas.items() if k != '반올림→repair')) if any((k != '반올림→repair' for k in feas)) else math.inf
        U_method = min(feas, key=feas.get) if feas else None
        Lb = max(L, Lc)
        rows.append(dict(B=B, BN=B / n, is_target=B == bstar, L_osqp=L, L_c=Lc, L_best=Lb, V_QP=au['primal'], P_c=cl['primal'], osqp_status=au['status'], fallback=au['fallback'], clip_low=au['clip_low'], clip_high=au['clip_high'], range_rel=au['range_rel_resid'], prim_res=au['osqp_prim_res'], dual_res=au['osqp_dual_res'], clar_status=cl['status'], clar_fallback=cl['fallback'], valid_flag_osqp=bool(cl_ok and L > cl['primal'] + tol), valid_flag_clar=bool(cl_ok and Lc > cl['primal'] + tol), cands=vals, cpp_attempts=dict(greedy=ga, rr=ra), U_rev=U_rev, U_paper=U_paper, U_method=U_method, W_rev=rel(U_rev, L), W_paper=rel(U_paper, L), W_best=rel(U_rev, Lb), abs_rev=U_rev - L, settings=au['settings']))
    return dict(name=name, input=inp, input_sha256=sha256_file(INPUTS / inp), bstar=bstar, cpp_bstar=cpp_bstar, mean_s0=float(s0.mean()), rows=rows, elapsed_s=time.perf_counter() - t0, attempts=1)

def per_field(res):
    rs = res['rows']
    tgt = next((r for r in rs if r['is_target']))
    return dict(name=res['name'], input=res['input'], bstar=res['bstar'], cpp_bstar=res['cpp_bstar'], mean_s0=res['mean_s0'], n_budgets=len(rs), T1=tgt['W_rev'], T2=max((r['W_rev'] for r in rs)), T1_paper=tgt['W_paper'], T2_paper=max((r['W_paper'] for r in rs)), T1_best=tgt['W_best'], T2_best=max((r['W_best'] for r in rs)), T2_B=max(rs, key=lambda r: r['W_rev'])['B'], abs_T1=tgt['abs_rev'], U_target=tgt['U_rev'], L_target=tgt['L_osqp'], U_method_target=tgt['U_method'], n_fallback=sum((r['fallback'] or r['osqp_status'] != 'solved' for r in rs)), n_clar_fail=sum((r['clar_status'] != 'Solved' or r['clar_fallback'] for r in rs)), n_valid_flag=sum((r['valid_flag_osqp'] or r['valid_flag_clar'] for r in rs)), n_cpp_retry=sum((v > 1 for r in rs for v in r['cpp_attempts'].values())), n_infeasible=sum((not c['feasible'] for r in rs for c in r['cands'].values())), attempts=res['attempts'], elapsed_s=res['elapsed_s'])

def quantile(v, p):
    v = sorted(v)
    h = (len(v) - 1) * p
    lo, fr = (int(math.floor(h)), h - math.floor(h))
    if fr == 0 or v[lo] == v[min(lo + 1, len(v) - 1)]:
        return v[lo]
    return v[lo] + fr * (v[lo + 1] - v[lo])

def summarize(fields, keys=('T1', 'T2', 'T1_paper', 'T2_paper', 'T1_best', 'T2_best', 'abs_T1')):
    out = {}
    for k in keys:
        v = [f[k] for f in fields]
        q1, q3 = (quantile(v, 0.25), quantile(v, 0.75))
        out[k] = dict(n=len(v), mean=float(np.mean(v)), median=quantile(v, 0.5), q1=q1, q3=q3, iqr=q3 - q1, min=min(v), max=max(v), argmax=fields[int(np.argmax(v))]['name'])
    return out

def run_set(jobs, label):
    res, failed = ({}, [])
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_field, j): j for j in jobs}
        for k, fu in enumerate(cf.as_completed(futs), 1):
            j = futs[fu]
            try:
                res[j['name']] = fu.result()
            except Exception as e:
                failed.append((j, repr(e)))
            progress.detail('9.2', f'{label}: 초기장 {k}/{len(jobs)} 완료 (예산 {NPTS}개씩, 8개 병렬)')
    for j, err in failed:
        try:
            r = run_field(j)
            r['attempts'] = 2
            res[j['name']] = r
        except Exception as e:
            res[j['name']] = dict(name=j['name'], input=j['input'], failed=True, errors=[err, repr(e)])
    return [res[j['name']] for j in jobs]

def register_candidates(results, experiment, run_id):
    for res in results:
        if res.get('failed'):
            continue
        for r in res['rows']:
            for cname, c in r['cands'].items():
                if c['path'] is not None:
                    incumbents.register(inp=res['input'], B=r['B'], cap=CMAX, method=cname, V=c['V'], alloc_path=c['path'], run_id=run_id, feasible=c['feasible'], F=F, kernel=K, sigma=(SIGMA, SIGMA), spacing=S, stage=9, experiment=experiment)
    incumbents.rebuild_best()

def records(results, experiment, run_id, threads):
    recs = []
    for res in results:
        if res.get('failed'):
            continue
        for r in res['rows']:
            best = r['cands'].get(r['U_method']) if r['U_method'] else None
            recs.append(make_record(stage=9, experiment=experiment, case=f"{res['name']} B{r['B']}", input=res['input'], input_sha256=res['input_sha256'], M=M, N=625, B=r['B'], Cmax=CMAX, kernel=K, sigma=SIGMA, spacing=S, boundary='reflect', solver='OSQP 1.1.3 (qp_audit) + Clarabel check + C++ exchange repair', settings=r['settings'], status=r['osqp_status'], U=r['U_rev'], L=r['L_osqp'], residuals=dict(prim=r['prim_res'], dual=r['dual_res'], Hz_v=r['range_rel']), U_method=r['U_method'], U_alloc_sha256=sha256_file(best['path']) if best else None, run_id=run_id, threads=threads, L_c=r['L_c'], P_c=r['P_c'], is_target=r['is_target']))
    return recs

def finish(results, set_label, experiment, prefix, threads):
    run_id = incumbents.new_run_id(__file__)
    register_candidates(results, experiment, run_id)
    fields = []
    for res in results:
        if res.get('failed'):
            fields.append(dict(name=res['name'], input=res['input'], **{k: math.inf for k in ('T1', 'T2', 'T1_paper', 'T2_paper', 'T1_best', 'T2_best', 'abs_T1')}, failed=True))
        else:
            fields.append(per_field(res))
    summ = summarize(fields)
    OUT.mkdir(parents=True, exist_ok=True)
    append_records(OUT / 'records.jsonl', records(results, experiment, run_id, threads))
    (OUT / f'{prefix}_runs.json').write_text(json.dumps(dict(run_id=run_id, set=set_label, threads=threads, results=results), indent=1, default=float, ensure_ascii=False))
    (OUT / f'{prefix}_fields.json').write_text(json.dumps(fields, indent=1, default=float, ensure_ascii=False))
    (OUT / f'{prefix}_summary.json').write_text(json.dumps(dict(run_id=run_id, set=set_label, summary=summ), indent=1, default=float, ensure_ascii=False))
    return (fields, summ)

def code_hashes():
    return {p: sha256_file(EXP / p) for p in CODE}

def do_dev():
    threads = determinism.assert_single_thread()
    progress.task('9.2', 'running', '파일럿: 개발용 초기장 10개(seed 1–10, C++ mt19937)로 같은 절차 실행 — 본 표본 아님')
    jobs = [dict(set='dev', name=f'dev{k:02d}', input=inp) for k, inp in enumerate(DEV, 1)]
    t0 = time.perf_counter()
    results = run_set(jobs, '개발용')
    fields, summ = finish(results, 'development seeds 1-10 (C++ std::mt19937), not the E4 sample', 'e4_dev', 'dev', threads)
    progress.log(f"Stage 9 파일럿(개발용 10개) {time.perf_counter() - t0:.0f}s: T1 최대 {summ['T1']['max']:.4f}%, T2 최대 {summ['T2']['max']:.4f}%, B* 불일치 {sum((f['bstar'] != f['cpp_bstar'] for f in fields))}")
    return (fields, summ)

def do_register():
    if PREREG.exists():
        sys.exit(f'{PREREG} exists; the protocol is fixed')
    NEW.mkdir(parents=True, exist_ok=True)
    ent = entropy()
    manifest = {}
    for k, ss in enumerate(np.random.SeedSequence(ent).spawn(N_SAMPLES), 1):
        v = np.random.Generator(np.random.PCG64(ss)).uniform(1.0, 30.0, size=M).reshape(F, F)
        p = NEW / f'e4_{k:02d}.csv'
        p.write_text('\n'.join((','.join((repr(float(t)) for t in r)) for r in v)) + '\n')
        manifest[f'stage9/{p.name}'] = dict(spawn_key=list(ss.spawn_key), sha256=sha256_file(p), mean=float(v.mean()))
    (NEW / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    prereg = dict(stage=9, experiment='E4', created=datetime.now(timezone.utc).isoformat(timespec='seconds'), population='i.i.d. Uniform[1, 30] per cell on the F=50 grid (M=2500), row-major', generator='numpy.random.Generator(PCG64(child)).uniform(1.0, 30.0, 2500) for child in SeedSequence(ENTROPY).spawn(59)', entropy=dict(value=str(ent)), n_samples=N_SAMPLES, development_inputs='random_u1_30_seed1..10_F50 (C++ std::mt19937, used in Stages 1-7) are excluded from the sample; run with the same pipeline beforehand as a pilot and reported separately', model=dict(F=F, spacing=S, waypoints='25x25', N=625, kernel=K, sigma=SIGMA, boundary='reflect', stamp_mass='raw', Cmax=CMAX, target_mean=TARGET), budgets=dict(target='B* = clamp(round((200 - mean(s0)) M / kappa), N, N Cmax); the C++ choice at B* is recorded and compared', grid='sorted(set(int(round(b)) for b in linspace(0.85 B*, 1.05 B*, 21))) (manuscript sweep)'), lower_bound=dict(primary='qp_audit: OSQP eps_abs = eps_rel = 1e-8 (manuscript setting), Lagrangian dual bound, range test 1e-10, fallback L = 0', check='Clarabel tol 1e-10 on every instance: relaxation objective P_c and Lagrangian bound L_c from its multipliers', sensitivity='L_best = max(L_osqp, L_c)', refinement='none per sample'), upper_bound=dict(candidates=['paper largest-remainder rounding of the OSQP solution', 'C++ greedy+repair', 'C++ round->repair from the index-tie rounding'], primary='U_rev = min over the feasible candidates of this run', paper='U_paper = min(greedy+repair, paper rounding) (manuscript Table S5 rule)', evaluator='V(x) = mean((A x + s0 - mean)^2) in Python for every candidate'), statistics=dict(T1='100 (U_rev - L_osqp) / U_rev at B* (primary)', T2='max over the 21-budget grid of 100 (U_rev - L_osqp) / U_rev (primary for budget-set claims)', secondary=['T1_paper, T2_paper with U_paper', 'T1_best, T2_best with L_best', 'abs_T1 = U_rev - L_osqp at B*'], summary='n, mean, median, Q1, Q3 (linear interpolation of order statistics), IQR, min, max'), claim='For this population, model, budget grid and procedure, the sample maximum of T over 59 independent fields is a one-sided upper confidence bound at level 1 - 0.95^59 = 0.9515 for the population 95th percentile of T. It is not a worst-case guarantee over all fields, budgets or kernels and does not replace the numerical validity of each L.', reference_check='The manuscript abstract reports sampled sweep widths below 0.6% (4 fields). max T2_paper and max T2 are compared with 0.6% as an observation only; the procedure does not depend on it.', failure_rules=FAILURE_RULES, threads='single-thread BLAS forced by determinism.py; 8 worker processes, one field each', code_sha256=code_hashes(), inputs_manifest_sha256=sha256_file(NEW / 'manifest.json'), outputs=['results/stage9/e4_runs.json', 'e4_fields.json', 'e4_summary.json', 'records.jsonl'])
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(json.dumps(prereg, indent=1, ensure_ascii=False))
    h = sha256_file(PREREG)
    progress.task('9.1', 'done', f"사전 고정 {prereg['created']}: 모집단 Uniform[1,30] i.i.d.(F=50), PCG64 SeedSequence spawn(59), entropy = {ENTROPY}, 개발용 seed 1–10 제외(별도 보고), 예산 B*와 원고 격자 21점, L = OSQP 1e-8(원고 설정)+Clarabel 검사, U = 후보 3종 최소, 통계량 T1(B*)·T2(격자 최대), 실패 규칙 {len(FAILURE_RULES)}개. preregistration.json sha256 {h[:16]}…")
    progress.log(f'Stage 9 사전 고정: results/stage9/preregistration.json sha256 {h}')
    print(h)

def do_run():
    threads = determinism.assert_single_thread()
    pre = load_json(PREREG.read_text())
    changed = [p for p, h in pre['code_sha256'].items() if sha256_file(EXP / p) != h]
    if changed:
        sys.exit(f'code changed since registration: {changed}')
    if sha256_file(NEW / 'manifest.json') != pre['inputs_manifest_sha256']:
        sys.exit('input manifest changed since registration')
    man = load_json((NEW / 'manifest.json').read_text())
    bad = [k for k, v in man.items() if sha256_file(INPUTS / k) != v['sha256']]
    if bad:
        sys.exit(f'inputs changed since registration: {bad}')
    if (OUT / 'e4_runs.json').exists():
        sys.exit('results/stage9/e4_runs.json exists; not rerunning')
    progress.task('9.2', 'running', f'본 표본 {N_SAMPLES}개 × 예산 {NPTS}개 실행')
    jobs = [dict(set='e4', name=Path(k).stem, input=k) for k in sorted(man)]
    t0 = time.perf_counter()
    results = run_set(jobs, '본 표본')
    fields, summ = finish(results, 'E4 sample (59 PCG64 fields)', 'e4', 'e4', threads)
    progress.log(f"Stage 9 본 표본 {len(fields)}개 완료 {time.perf_counter() - t0:.0f}s: T1 최대 {summ['T1']['max']:.4f}%, T2 최대 {summ['T2']['max']:.4f}%")
    return (fields, summ)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['dev', 'register', 'run'])
    a = ap.parse_args()
    {'dev': do_dev, 'register': do_register, 'run': do_run}[a.mode]()
