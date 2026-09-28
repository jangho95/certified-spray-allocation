import determinism
import argparse
import concurrent.futures as cf
import json
import shutil
import time
import traceback
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import progress
import rigor_pilot as rp
import sensitivity_extension as se
from qp_audit import sha256_file, solve_qp_audit
EXP = se.EXP
V1 = EXP / 'results' / 'sensitivity_extension'
OUT = EXP / 'results' / 'sensitivity_extension_v2'
TOL = 1e-09

def protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'protocol.json'
    if p.exists():
        d = json.loads(p.read_text())
        for rel, h in d['code_sha256'].items():
            assert sha256_file(EXP / rel) == h, f'Code changed after v2 protocol: {rel}'
        return d
    files = [Path(__file__), Path(rp.__file__), Path(se.__file__), EXP / 'pipeline' / 'qp_audit.py']
    d = dict(version=2, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_protocol_sha256=sha256_file(V1 / 'protocol.json'), source_rows_sha256=sha256_file(V1 / 'rows.json'), jobs=json.loads((V1 / 'protocol.json').read_text())['jobs'], code_sha256={str(p.relative_to(EXP)): sha256_file(p) for p in files}, scope='New continuous solves and exact certification of the same 156 inputs and preserved integer candidates.', lower='maximum of v1 exact lower bound, current OSQP reference and refinements, optional additional references', upper='minimum exact value among preserved feasible point, integer allocation, budget-box projections and refined feasible points', projection='bisection for clip(x+t,1,C), followed by exact dyadic residual correction', fallback=f'if (best feasible continuous upper - lower)/max(integer upper,1) > {TOL:g}, add OSQP 1e-10 and Clarabel references and refine each with up to 20 active-set rounds', invariant='L_v2 >= L_v1, integer U and candidates unchanged, feasible continuous upper_v2 <= upper_v1', guarantee='Exact bounds for stored float64 operators and initial fields. Analytic mesh evaluation remains numerical.', historical_results='v1 and prior Stages remain preserved; source_before contains pre-change common routines.', environment=rp.env_versions())
    se.dump(p, d)
    for path in V1.glob('operator_*.npz'):
        shutil.copy2(path, OUT / path.name)
    return d

def tests():
    rng = np.random.default_rng(9321)
    ntests = 0
    max_shift_error = 0.0
    for n in [1, 2, 3, 7, 625]:
        for C in [1, 3, 200]:
            for k in range(12):
                B = [n, n * C, int(rng.integers(n, n * C + 1))][k % 3]
                x = rng.normal(0, 1000, n)
                xf = rp.feasible_point(x, B, C)
                assert xf is not None and sum(xf) == B and all((1 <= v <= C for v in xf))
                assert rp.feasible_point(xf, B, C) == xf
                z = np.array([float(v) for v in xf])
                free = (z > 1 + 1e-09) & (z < C - 1e-09)
                if free.any():
                    t = float(np.median((z - x)[free]))
                    err = float(np.max(np.abs(z - np.clip(x + t, 1, C))))
                    max_shift_error = max(max_shift_error, err)
                    assert err < 1e-09
                ntests += 1
    assert rp.feasible_point([0.0, 1.0], 1, 3) is None
    assert rp.feasible_point([float('nan'), 1.0], 3, 3) is None
    saved_M = rp.M
    rp.M = 2
    st = [SimpleNamespace(cells=[0], weights=[1.0]), SimpleNamespace(cells=[1], weights=[2.0])]
    ex = rp.Exact(st, np.array([10.0, 0.0]), 6, 5)
    for _ in range(40):
        xf = rp.feasible_point(rng.normal(0, 50, 2), 6, 5)
        assert ex.value(xf) >= Fraction(1, 4)
    assert ex.value(rp.feasible_point([-10.0, 10.0], 6, 5)) == Fraction(1, 4)
    xb = np.array([2 / 3, 16 / 3])
    L = ex.bound(xb)[0]
    Vf, _, _ = rp.feasible_upper(ex, {'outside': xb}, 6, 5)
    assert float(ex.value(xb) - L) < 1e-09 and Vf - L > Fraction(1, 5)
    rp.M = saved_M
    legacy = {}
    for name in ['rigor_pilot', 'stage5_rigor', 'stage9_rigor', 'bench_sweep']:
        count, bad, maxviol = (0, 0, 0.0)
        for p in (EXP / 'results' / name).rglob('*.npz'):
            with np.load(p) as w:
                if 'xbar' not in w:
                    continue
                xs = np.atleast_2d(w['xbar'])
                C = float(w['Cmax']) if 'Cmax' in w else 200.0
                v = np.maximum(np.maximum(1 - xs.min(axis=1), xs.max(axis=1) - C), 0)
                count += len(xs)
                bad += int(np.sum(v > 0))
                maxviol = max(maxviol, float(v.max()))
        legacy[name] = dict(n=count, strict_box_violations=bad, max_box_violation=maxviol)
    result = dict(projection_cases=ntests, projection_kkt_max_error=max_shift_error, known_qp_upper_cases=41, infeasible_reference_trigger_regression=True, legacy=legacy)
    se.dump(OUT / 'tests.json', result)
    return result

def run_case(old):
    t0 = time.perf_counter()
    cid = old['id']
    try:
        assert sha256_file(EXP / old['input']) == old['input_sha256']
        A, st, mass = se.operator(old['kind'], old['F'])
        w = np.load(V1 / old['witness'])
        s0 = w['s0']
        B = int(w['B'])
        C = int(w['Cmax'])
        rp.M = int(w['M'])
        ex = rp.Exact(st, s0, B, C)
        xu = w['allocation'].astype(float)
        assert all(rp.check_int(xu, B, C).values())
        U = ex.value(xu)
        assert U == Fraction(old['U_exact'])
        for name, x in zip(w['candidate_names'], w['candidates']):
            assert all(rp.check_int(x.astype(float), B, C).values())
            assert rp.up(ex.value(x.astype(float))) == old['candidate_U'][str(name)]
        lower_points = {'v1': w['xbar'].copy()}
        lower_values = {'v1': max(Fraction(0), ex.bound(lower_points['v1'])[0])}
        assert lower_values['v1'] == Fraction(old['L_exact'])
        feas_points = {'v1': [Fraction(str(v)) for v in w['feasible_relaxation']], 'integer': xu}
        assert ex.value(feas_points['v1']) == Fraction(old['Vfeas_exact'])
        au = solve_qp_audit(A, s0, B, mass, C)
        logs = {}

        def add_reference(name, x, rounds=6):
            lower_points[name] = np.array(x, float)
            lower_values[name] = ex.bound(x)[0]
            feas_points[name] = x
            keep = {}
            (Lr, xr), log = rp.refine(A, s0, B, C, ex, np.array(x, float), rounds=rounds, feasible_out=keep)
            lower_points[name + '_refined'] = xr
            lower_values[name + '_refined'] = Lr
            feas_points[name + '_lower_projection'] = xr
            if keep:
                feas_points[name + '_refined_feasible'] = keep['point']
            logs[name] = log
        add_reference('OSQP_1e8', au['x'])

        def select():
            src = max(lower_values, key=lower_values.get)
            L = max(Fraction(0), lower_values[src])
            Vf, xf, fsrc = rp.feasible_upper(ex, feas_points, B, C)
            return (L, lower_points[src], src, Vf, xf, fsrc)
        L, xb, lsrc, Vf, xf, fsrc = select()
        gap_initial = float((Vf - L) / max(U, Fraction(1)))
        extra = gap_initial > TOL
        extra_status = {}
        if extra:
            a10 = solve_qp_audit(A, s0, B, mass, C, eps_abs=1e-10, eps_rel=1e-10)
            extra_status['OSQP_1e10'] = a10['status']
            add_reference('OSQP_1e10', a10['x'], rounds=20)
            xc, status, _ = rp.clarabel_x(A, s0, B, C)
            extra_status['Clarabel'] = status
            add_reference('Clarabel', xc, rounds=20)
            L, xb, lsrc, Vf, xf, fsrc = select()
        assert L >= Fraction(old['L_exact']) and L <= Vf <= Fraction(old['Vfeas_exact'])
        assert Vf <= U and all((1 <= v <= C for v in xf)) and (sum(xf) == B)
        width = rp.up(100 * (U - L) / U) if U else 0.0
        assert width <= old['width_pct']
        dest = OUT / old['witness']
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, s0=s0, xbar=xb, lower_reference=xb, allocation=xu.astype(np.int64), B=B, Cmax=C, M=rp.M, candidate_names=w['candidate_names'], candidates=w['candidates'], feasible_relaxation=np.array([rp.frs(v) for v in xf]), raw_osqp_reference=au['x'])
        row = dict(old, version=2, L=rp.down(L), L_exact=rp.frs(L), U=rp.up(U), U_exact=rp.frs(U), width_pct=width, abs_width=rp.up(U - L), Vfeas_exact=rp.frs(Vf), continuous_interval_width=rp.up(Vf - L), bound_refine_relative_gap=float((Vf - L) / max(U, Fraction(1))), lower_reference_source=lsrc, feasible_source=fsrc, extra_reference=extra, feasible_gap_before_extra=gap_initial, extra_solver_status=extra_status, lower_reference_box_violation=float(max(0, 1 - xb.min(), xb.max() - C)), v1_width_pct=old['width_pct'], v1_relative_interval=old['bound_refine_relative_gap'], v1_L_exact=old['L_exact'], v1_Vfeas_exact=old['Vfeas_exact'], L_float_above_feasible_exact=bool(Fraction(old['L_float']) > Vf), float_lower_reproduced=bool(old['L_float'] == au['dual_lower']), projection_checks=dict(budget=True, box=True), elapsed_v2_s=time.perf_counter() - t0, failed=False)
        se.dump(OUT / 'cases' / (cid + '.json'), row)
        se.dump(OUT / 'refinement_logs' / (cid + '.json'), logs)
        return row
    except Exception:
        row = dict(old, failed=True, error=traceback.format_exc())
        se.dump(OUT / 'cases' / (cid + '.json'), row)
        return row

def summary(rows):
    ok = [r for r in rows if not r['failed']]
    d = dict(n=len(rows), failed=len(rows) - len(ok), extra_reference_count=sum((r['extra_reference'] for r in ok)), max_continuous_interval_width=max((r['continuous_interval_width'] for r in ok)), max_relative_interval=max((r['bound_refine_relative_gap'] for r in ok)), n_above_relative_tolerance=sum((r['bound_refine_relative_gap'] > TOL for r in ok)), lower_reference_outside_box=sum((r['lower_reference_box_violation'] > 0 for r in ok)), n_float_exceedance=sum((r['L_float_above_feasible_exact'] for r in ok)), all_float_lower_reproduced=all((r['float_lower_reproduced'] for r in ok)), max_width_change_pp=max((r['v1_width_pct'] - r['width_pct'] for r in ok)), n_width_changed=sum((r['L_exact'] != r['v1_L_exact'] for r in ok)), groups={g: dict(n=len(rs), width_min=min((r['width_pct'] for r in rs)), width_max=max((r['width_pct'] for r in rs))) for g in ['target', 'distribution', 'mesh'] if (rs := [r for r in ok if r['group'] == g])})
    se.dump(OUT / 'summary.json', d)
    return d

def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['test', 'pilot', 'run', 'verify'])
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    protocol()
    if args.command == 'test':
        print(json.dumps(tests(), indent=1))
        return
    old = json.loads((V1 / 'rows.json').read_text())
    if args.command == 'pilot':
        names = ['dist_correlated_uniform_25', 'dist_correlated_uniform_03', 'target_center30_T400', 'mesh_zero_F200']
        old = [r for r in old if r['id'] in names]
    progress.task('10.5', 'running', f'v2: 실행가능한 연속해와 하한을 분리하여 {len(old)}개 재검증')
    rows = []
    pending = []
    if args.command != 'verify':
        for r in old:
            path = OUT / 'cases' / (r['id'] + '.json')
            if path.exists():
                rows.append(json.loads(path.read_text()))
            else:
                pending.append(r)
        with cf.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for r in executor.map(run_case, pending):
                rows.append(r)
                print(r['id'], 'FAILED' if r['failed'] else f"relative interval={r['bound_refine_relative_gap']:.3g}", flush=True)
                progress.detail('10.5', f"v2 인증 {len(rows)}/{len(old)}, 실패 {sum((r['failed'] for r in rows))}")
        se.dump(OUT / ('pilot_rows.json' if args.command == 'pilot' else 'rows.json'), rows)
        if args.command == 'pilot':
            print(json.dumps(summary(rows), indent=1))
            return
    else:
        rows = json.loads((OUT / 'rows.json').read_text())
    se.OUT = OUT
    progress.detail('10.5', 'v2 저장 자료 재검산')
    with cf.ProcessPoolExecutor(max_workers=args.workers) as executor:
        verified = list(executor.map(se.verify_row, rows))
    se.dump(OUT / 'verification.json', dict(n=len(rows), passed=sum(verified), failed=len(rows) - sum(verified), per_case={r['id']: v for r, v in zip(rows, verified)}))
    se.mesh_compare(rows)
    se.publish(rows, complete=True)
    result = summary(rows)
    progress.task('10.5', 'done' if all(verified) and (not result['n_above_relative_tolerance']) else 'warn', f"v2 검산 {sum(verified)}/{len(rows)}, 연속 구간 상대 폭 최대 {result['max_relative_interval']:.3g}")
    print(json.dumps(result, indent=1))
if __name__ == '__main__':
    main()
