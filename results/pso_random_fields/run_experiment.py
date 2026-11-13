from common import *
import argparse
import concurrent.futures as cf
import subprocess
import numpy as np
from paper_pso import run_pso
import pso_repeated_runs as repeated

def pso_job(job):
    os.chdir(ROOT)
    field, seed, attempt, gate = (job['field'], job['seed'], job['attempt'], job.get('gate', False))
    started = now()
    t0 = time.perf_counter()
    pool = threads()
    args = repeated.make_args(dict(field='random', initial_field=Path(job['input']), iters=5000, log_every=10, out_prefix=Path('unused')), seed)
    sim, x, met, hist, _ = run_pso(args)
    folder = Path('reproduction') if gate else Path('fields') / field / 'pso'
    folder.mkdir(parents=True, exist_ok=True)
    ap = folder / f'seed_{seed}_allocation.csv'
    hp = folder / f'seed_{seed}_history.csv'
    write_alloc(ap, x)
    with hp.open('w', newline='') as f:
        keys = list(hist[0]) + ['gbest_budget']
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for h in hist:
            w.writerow(dict(h, gbest_budget=int(round((h['gbest_mean'] * sim.m - sim.initial_sum) / sim.mass))))
    fv = sim.A @ x + sim.initial
    P = float(np.mean((fv - fv.mean()) ** 2))
    B = int(x.sum())
    feasible = bool(np.isfinite(x).all() and np.all(x == np.rint(x)) and np.all((x >= 1) & (x <= 200)))
    h4500 = next((h for h in hist if h['iter'] == 4500))
    h5000 = next((h for h in hist if h['iter'] == 5000))
    row = dict(field=field, seed=seed, attempt=attempt, iterations=5000, B=B, P_float=P, P_reported=met['variance'], feasible=feasible, allocation=str(ap), history=str(hp), allocation_sha256=digest(ap), history_sha256=digest(hp), variance_4500=h4500['gbest_variance'], variance_5000=h5000['gbest_variance'], variance_lower_at_5000=bool(h5000['gbest_variance'] < h4500['gbest_variance']), started_utc=started, finished_utc=now(), elapsed_s=time.perf_counter() - t0, optimizer_s=met['elapsed_s'], threadpools=pool)
    assert abs(P - met['variance']) <= 1e-08 * max(1, abs(P))
    atomic(folder / f'seed_{seed}.json', row)
    return row

def reproduction():
    protocol = check_protocol()
    ref = json.loads((ROOT / 'reproduction/reference.json').read_text())
    r = pso_job(dict(field='benchmark_random', seed=2, attempt=1, gate=True, input='inputs/benchmark_random.csv'))
    allocation_identical = np.array_equal(read_alloc(ROOT / r['allocation']), read_alloc(ROOT / 'reproduction/reference_allocation.csv'))
    rel = abs(r['P_float'] - ref['P']) / max(1, abs(ref['P']))
    result = dict(checked_utc=now(), protocol_sha256=digest(ROOT / 'protocol.json'), expected_B=ref['B'], reproduced_B=r['B'], expected_P=ref['P'], reproduced_P=r['P_float'], relative_P_difference=rel, P_tolerance=1e-12, allocation_identical=bool(allocation_identical), passed=bool(r['feasible'] and r['B'] == ref['B'] and (rel <= 1e-12) and allocation_identical))
    atomic(ROOT / 'reproduction/check.json', result)
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise RuntimeError('Reproduction gate failed; main experiment prohibited')

def pso_all():
    p = check_protocol()
    gate = json.loads((ROOT / 'reproduction/check.json').read_text())
    assert gate['passed'] and gate['protocol_sha256'] == digest(ROOT / 'protocol.json')
    assert not (ROOT / 'pso_started.json').exists(), 'Use existing records; never silently restart a sample'
    t0 = time.perf_counter()
    atomic(ROOT / 'pso_started.json', dict(started_utc=now(), planned=1475, protocol_sha256=digest(ROOT / 'protocol.json')))
    jobs = [dict(field=f['field'], input=f['input'], seed=100000 + 100 * f['index'] + r, attempt=1) for r in range(1, 26) for f in p['fields']]
    rows = []
    errors = []
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        pending = {ex.submit(pso_job, j): j for j in jobs}
        while pending:
            done, _ = cf.wait(pending, timeout=15, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                j = pending.pop(fut)
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    failure = dict(field=j['field'], seed=j['seed'], attempt=j['attempt'], time_utc=now(), error=error_text(exc))
                    errors.append(failure)
                    atomic(ROOT / 'pso_failures.json', errors)
                    if j['attempt'] == 1:
                        retry = dict(j, attempt=2)
                        pending[ex.submit(pso_job, retry)] = retry
            atomic(ROOT / 'progress.json', dict(phase='PSO', completed=len(rows), planned=1475, failed_attempts=len(errors), retry_attempts=sum((e['attempt'] == 1 for e in errors)), pending=len(pending), wall_s=time.perf_counter() - t0, updated_utc=now()))
            print(f'PSO completed={len(rows)}/1475 failed_attempts={len(errors)} wall_s={time.perf_counter() - t0:.1f}', flush=True)
    atomic(ROOT / 'runs.json', dict(rows=sorted(rows, key=lambda r: (r['field'], r['seed'])), failures=errors, wall_s=time.perf_counter() - t0, finished_utc=now()))
    if len(rows) != 1475 or any((not r['feasible'] for r in rows)):
        raise RuntimeError('Incomplete or infeasible planned sample; aggregate reporting prohibited')

def operator():
    from scip_compare import build_stamps, make_waypoints
    from convex_qp_tools import dense_A
    stamps = build_stamps(geometry(), make_waypoints(geometry()))
    return (stamps, dense_A(stamps, 2500, 625))

def rigorous_lower(A, st, s0, B, U, rp):
    from qp_audit import solve_qp_audit
    ex = rp.Exact(st, s0, B, 200)
    au = solve_qp_audit(A, s0, B, st[0].mass, 200)
    points = {'OSQP_1e-8': au['x']}
    bounds = {'OSQP_1e-8': ex.bound(au['x'])[0]}
    (Lr, xr), _ = rp.refine(A, s0, B, 200, ex, au['x'])
    points['refined_1e-8'] = xr
    bounds['refined_1e-8'] = Lr
    best = max(bounds, key=bounds.get)
    Vf, xf, fsrc = rp.feasible_upper(ex, points, B, 200)
    extra = bool((Vf - bounds[best]) / U > Fraction(1, 10 ** 9))
    if extra:
        a10 = solve_qp_audit(A, s0, B, st[0].mass, 200, eps_abs=1e-10, eps_rel=1e-10)
        xc, status, _ = rp.clarabel_x(A, s0, B, 200)
        for name, x in [('OSQP_1e-10', a10['x']), ('Clarabel_1e-10', xc)]:
            points[name] = x
            bounds[name] = ex.bound(x)[0]
        start = max(['OSQP_1e-10', 'Clarabel_1e-10'], key=bounds.get)
        (Lr, xr), _ = rp.refine(A, s0, B, 200, ex, points[start])
        points['refined_extra'] = xr
        bounds['refined_extra'] = Lr
        best = max(bounds, key=bounds.get)
    Vf, xf, fsrc = rp.feasible_upper(ex, points, B, 200)
    return (max(Fraction(0), bounds[best]), points[best], Vf, xf, best, extra, au)

def cpp_candidate(input_path, B, prefix, init=None):
    binary = ROOT.parents[1] / 'build' / 'pufoam_solver'
    if not binary.is_file():
        raise FileNotFoundError('Build the C++ solver from the repository root: python tools/build.py')
    cmd = [str(binary), '--initial-field', input_path, '--field-size', '50', '--waypoint-rows', '25', '--waypoint-cols', '25', '--spray-interval', '2', '--kernel-size', '7', '--sigma', '1.75', '--boundary', 'reflect', '--budget', str(B), '--cmax', '200', '--max-exchanges', '200000', '--quiet', '--full-precision', '--out-prefix', prefix]
    if init:
        cmd += ['--init-allocation', init]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    Path(prefix + '.log').write_text(cp.stdout + cp.stderr)
    if cp.returncode:
        raise RuntimeError(f'Candidate executable failed: {cp.returncode}')
    return prefix + '_allocation.csv'

def verify_job(job):
    os.chdir(ROOT)
    pool = threads()
    rp = load_math()
    t0 = time.perf_counter()
    from convex_qp_tools import largest_remainder_round
    from rounding import round_indexed
    from qp_audit import solve_qp_audit
    st, A = operator()
    s0 = field_values(job['input'])
    B = job['B']
    field = job['field']
    folder = Path('fields') / field / 'budgets' / f'B{B}'
    folder.mkdir(parents=True, exist_ok=True)
    ex = rp.Exact(st, s0, B, 200)
    au = solve_qp_audit(A, s0, B, st[0].mass, 200)
    pr = folder / 'rounded.csv'
    ri = folder / 'rounded_index.csv'
    write_alloc(pr, largest_remainder_round(au['x'], B, 200))
    write_alloc(ri, round_indexed(au['x'], B, 200))
    candidates = [('rounding', str(pr)), ('greedy_exchange', cpp_candidate(job['input'], B, str(folder / 'greedy'))), ('rounded_exchange', cpp_candidate(job['input'], B, str(folder / 'rounded_exchange'), str(ri)))]
    checked = []
    for name, path in candidates:
        x = read_alloc(path)
        assert all(rp.check_int(x, B, 200).values())
        checked.append((name, path, x, ex.value(x)))
    Uthree = min((v for _, _, _, v in checked))
    psos = []
    for r in job['runs']:
        x = read_alloc(r['allocation'])
        assert digest(r['allocation']) == r['allocation_sha256']
        assert all(rp.check_int(x, B, 200).values())
        P = ex.value(x)
        psos.append((r, x, P))
        checked.append((f"PSO_seed_{r['seed']}", r['allocation'], x, P))
    name, path, xu, U = min(checked, key=lambda item: item[3])
    L, xb, Vf, xf, lsource, extra, audit = rigorous_lower(A, st, s0, B, U, rp)
    assert 0 <= L <= U and L <= Vf
    witness = dict(field=field, B=B, Cmax=200, input=job['input'], input_sha256=digest(job['input']), xbar_hex=[float(v).hex() for v in xb], continuous_feasible=[rp.frs(v) for v in xf], continuous_value_exact=rp.frs(Vf), L_exact=rp.frs(L), U_exact=rp.frs(U), U_three_exact=rp.frs(Uthree), U_source=name, U_allocation=path, candidates=[dict(method=n, allocation=p, sha256=digest(p), V_exact=rp.frs(v)) for n, p, x, v in checked], lower_source=lsource, extra_reference_points=extra)
    atomic(folder / 'witness.json', witness)
    rows = []
    for r, x, P in psos:
        lo = P / U - 1 if U > 0 else None
        hi = P / L - 1 if L > 0 else math.inf
        rows.append(dict(r, P_exact=rp.frs(P), L_exact=rp.frs(L), U_exact=rp.frs(U), excess_low_exact=rp.frs(lo) if lo is not None else None, excess_high_exact=rp.frs(hi) if hi != math.inf else '+Infinity', excess_low_pct=pct(lo), excess_high_pct=pct(hi, True), P_better_than_three=bool(P < Uthree)))
    width = (U - L) / U if U > 0 else None
    result = dict(field=field, B=B, runs=rows, n_runs=len(rows), witness=str(folder / 'witness.json'), L_exact=rp.frs(L), U_exact=rp.frs(U), U_three_exact=rp.frs(Uthree), relative_width_pct_low=pct(width), relative_width_pct_up=pct(width, True), pso_better_than_three=[r['seed'] for r in rows if r['P_better_than_three']], elapsed_s=time.perf_counter() - t0, threadpools=pool)
    atomic(folder / 'result.json', result)
    return result

def verify_all():
    p = check_protocol()
    data = json.loads((ROOT / 'runs.json').read_text())
    runs = data['rows']
    assert len(runs) == 1475 and all((r['feasible'] for r in runs))
    st, A = operator()
    atomic(ROOT / 'operator.json', dict(M=2500, N=625, columns=[[[int(i), float(v).hex()] for i, v in zip(s.cells, s.weights)] for s in st]))
    jobs = []
    for f in p['fields']:
        rs = [r for r in runs if r['field'] == f['field']]
        assert len(rs) == 25
        for B in sorted({r['B'] for r in rs}):
            jobs.append(dict(field=f['field'], input=f['input'], B=B, runs=[r for r in rs if r['B'] == B]))
    atomic(ROOT / 'verification_started.json', dict(started_utc=now(), planned_pairs=len(jobs)))
    rows = []
    errors = []
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        pending = {ex.submit(verify_job, j): j for j in jobs}
        while pending:
            done, _ = cf.wait(pending, timeout=15, return_when=cf.FIRST_COMPLETED)
            for fu in done:
                j = pending.pop(fu)
                try:
                    rows.append(fu.result())
                except Exception as exc:
                    errors.append(dict(field=j['field'], B=j['B'], error=error_text(exc)))
                    atomic(ROOT / 'verification_failures.json', errors)
            atomic(ROOT / 'progress.json', dict(phase='exact_verification', completed=len(rows), planned=len(jobs), failures=len(errors), wall_s=time.perf_counter() - t0, updated_utc=now()))
            print(f'Verified pairs={len(rows)}/{len(jobs)} failures={len(errors)} wall_s={time.perf_counter() - t0:.1f}', flush=True)
    atomic(ROOT / 'verified_instances.json', dict(instances=sorted(rows, key=lambda r: (r['field'], r['B'])), failures=errors, planned_pairs=len(jobs), wall_s=time.perf_counter() - t0, finished_utc=now()))
    assert not errors and len(rows) == len(jobs), 'Incomplete verification; no aggregate report'
if __name__ == '__main__':
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['reproduce', 'pso', 'verify'])
    phase = parser.parse_args().phase
    {'reproduce': reproduction, 'pso': pso_all, 'verify': verify_all}[phase]()
