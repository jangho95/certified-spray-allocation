import determinism
import concurrent.futures as cf
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP / 'work' / 'stage1')]
import progress
import rigor_pilot as rp
import stage9_e4 as s9
from qp_audit import sha256_file, solve_qp_audit
OUT = EXP / 'results' / 'stage5_rigor'
S5 = EXP / 'results' / 'stage5'
WORK5 = EXP / 'work' / 'stage5'
ALLOC = EXP / 'results' / 'incumbents' / 'alloc'
GAP_TOL = 1e-09
FIELD_INPUT = {'zero': 'zero_F50.csv', 'random_1_30_seed7': 'random_u1_30_seed7_F50.csv', 'center30': 'center30_F50.csv', 'center100': 'center100_F50.csv'}

def pct_out(v, direction, d=1):
    x = Fraction(v) * 100 * 10 ** d
    n = math.floor(x) if direction == 'down' else math.ceil(x)
    return f'{n // 10 ** d}.{n % 10 ** d:0{d}d}'

def interval_str(lo, hi):
    return f"[{pct_out(lo, 'down')}, {pct_out(hi, 'up')}]%" if lo is not None and hi is not None else None

def median_out(v, direction):
    v = sorted(v)
    n = len(v)
    if n % 2:
        return v[n // 2]
    m = (Fraction(v[n // 2 - 1]) + Fraction(v[n // 2])) / 2
    return rp.down(m) if direction == 'down' else rp.up(m)

def read_alloc(path, n):
    x = np.zeros(n)
    for line in Path(path).read_text().splitlines()[1:]:
        f_ = line.split(',')
        x[int(f_[0])] = int(f_[-1])
    return x

def rigorous_L(A, st, s0, B, C, mass, U):
    ex = rp.Exact(st, s0, B, C)
    au = solve_qp_audit(A, s0, B, mass, C)
    pts = {'OSQP 1e-8': au['x']}
    Ls = {'OSQP 1e-8': ex.bound(au['x'])[0]}
    (Lr, xr), _ = rp.refine(A, s0, B, C, ex, au['x'])
    pts['정제(1e-8 출발)'], Ls['정제(1e-8 출발)'] = (xr, Lr)
    best = max(Ls, key=Ls.get)
    extra = bool((ex.value(pts[best]) - Ls[best]) / U > GAP_TOL)
    if extra:
        a10 = solve_qp_audit(A, s0, B, mass, C, eps_abs=1e-10, eps_rel=1e-10)
        xc, _, _ = rp.clarabel_x(A, s0, B, C)
        for k, x in (('OSQP 1e-10', a10['x']), ('Clarabel 1e-10', xc)):
            pts[k], Ls[k] = (x, ex.bound(x)[0])
        st_k = max(('OSQP 1e-10', 'Clarabel 1e-10'), key=Ls.get)
        (L2, x2), _ = rp.refine(A, s0, B, C, ex, pts[st_k])
        pts['정제(추가 기준점 출발)'], Ls['정제(추가 기준점 출발)'] = (x2, L2)
        best = max(Ls, key=Ls.get)
    return (max(Ls[best], Fraction(0)), best, extra, pts[best], au, ex)

def instance_job(job):
    t0 = time.perf_counter()
    inp, B, U_sha, U_float, L_float, runs = (job['input'], job['B'], job['U_sha'], job['U'], job['L'], job['runs'])
    A, mass, n = s9.operator()
    g = SimpleNamespace(field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
    st = rp.build_stamps(g, rp.make_waypoints(g))
    s0 = s9.load_field(inp)
    ex = rp.Exact(st, s0, B, 200)
    xu = read_alloc(ALLOC / f'{U_sha}.csv', n)
    chk = rp.check_int(xu, B, 200)
    assert all(chk.values()), (inp, B, chk)
    U = ex.value(xu)
    L, src, extra, xb, au, _ = rigorous_L(A, st, s0, B, 200, mass, U)
    L_osqp = au['dual_lower']
    xf = rp.feasible_point(xb, B, 200)
    Vf = ex.value(xf) if xf is not None else None
    prow = []
    for r in runs:
        xp = read_alloc(WORK5 / f"pso_{r['field']}_s{r['seed']}_it{r['iters']}_allocation.csv", n)
        c = rp.check_int(xp, B, 200)
        assert all(c.values()), (r, c)
        P = ex.value(xp)
        prow.append(dict(field=r['field'], seed=r['seed'], iters=r['iters'], P_exact=rp.frs(P), P_up=rp.up(P), P_float=r['P'], P_float_minus_exact=float(Fraction(r['P']) - P), excess_low_down=rp.down(P / U - 1), excess_high_up=rp.up(P / L - 1) if L > 0 else None, excess_low_float=r['rel_low'], excess_high_float=r['rel_high'], alloc_checks=c))
    return dict(input=inp, B=B, L_rig=rp.down(L), L_exact=rp.frs(L), L_source=src, extra_points=extra, L_float=L_float, L_osqp_reproduced=bool(L_osqp == L_float), L_float_minus_rig=float(Fraction(L_float) - L), float_clip_high=bool(au['clip_high']), float_fallback=bool(au['fallback']), float_dual_raw=au['dual_raw'], float_primal=au['primal'], U_rig=rp.up(U), U_exact=rp.frs(U), U_float=U_float, U_float_minus_exact=float(Fraction(U_float) - U), U_checks=chk, W_rig_pct_up=rp.up(100 * (U - L) / U), W_float_pct=100 * (U_float - L_float) / U_float, V_feas_up=rp.up(Vf) if Vf is not None else None, QP_enclosure_width_up=rp.up(Vf - L) if Vf is not None else None, L_float_above_Vfeas=bool(Vf is not None and Fraction(L_float) > Vf), L_float_minus_Vfeas=float(Fraction(L_float) - Vf) if Vf is not None else None, pso=prow, xbar=np.asarray(xb, dtype=np.float64).tolist(), t_s=time.perf_counter() - t0)

def main():
    threads = determinism.assert_single_thread()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'protocol.json').write_text(json.dumps(dict(status='post hoc additional analysis of the Stage 5 same-budget PSO evaluation; Stage 5 results unchanged', procedure='as stage9_rigor.py: OSQP 1e-8 point + active-set refinement; extra points if (V(xb) - L)/U > 1e-9; U = Stage 5 reported allocation (records.jsonl sha256) after integrality/budget/box checks; P = every PSO allocation of the budget, checked and evaluated exactly; intervals rounded outward', pso='same-simulator PSO-style reimplementation from the published algorithm description (work/stage1/paper_pso.py), not the original code of the prior paper', code_sha256=dict(stage5_rigor=sha256_file(Path(__file__)), rigor_pilot=sha256_file(EXP / 'pipeline' / 'rigor_pilot.py'), stage9_rigor=sha256_file(EXP / 'pipeline' / 'stage9_rigor.py')), environment=rp.env_versions()), indent=1, ensure_ascii=False))
    budgets = json.loads((S5 / 'pso_budgets.json').read_text())
    recs = [json.loads(l) for l in (S5 / 'records.jsonl').read_text().splitlines()]
    sha = {(r['input'], r['B']): r['extra']['U_alloc_sha256'] for r in recs}
    runs = json.loads((S5 / 'pso_runs.json').read_text())['rows']
    jobs = []
    for b in budgets:
        rs = [r for r in runs if FIELD_INPUT[r['field']] == b['input'] and r['B'] == b['B']]
        jobs.append(dict(input=b['input'], B=b['B'], U_sha=sha[b['input'], b['B']], U=b['U'], L=b['L'], runs=rs))
    assert sum((len(j['runs']) for j in jobs)) == len(runs) == 104
    progress.task('F.2', 'running', f'Stage 5 추가 분석: (초기장, 예산) {len(jobs)}개의 엄밀 L·U와 PSO 재구현 실행 {len(runs)}개의 정확 P')
    t0 = time.perf_counter()
    out = []
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        for k, r in enumerate(ex.map(instance_job, jobs), 1):
            out.append(r)
            progress.detail('F.2', f'Stage 5 추가 분석: 인스턴스 {k}/{len(jobs)} 완료(8개 병렬)')
    wall = time.perf_counter() - t0
    (OUT / 'witness').mkdir(exist_ok=True)
    for r in out:
        np.savez(OUT / 'witness' / f"{Path(r['input']).stem}_B{r['B']}.npz", xbar=np.array(r.pop('xbar')), B=np.array(r['B']), s0=np.asarray(s9.load_field(r['input']), dtype=np.float64), Cmax=np.array(200))
    summ = []
    for field, inp in FIELD_INPUT.items():
        ps = [p for r in out for p in r['pso'] if p['field'] == field and p['iters'] == 500]
        lo = [p['excess_low_down'] for p in ps]
        hi = [p['excess_high_up'] for p in ps]
        s5 = next((x for x in json.loads((S5 / 'pso_summary.json').read_text()) if x['field'] == field))
        longr = [p for r in out for p in r['pso'] if p['field'] == field and p['iters'] != 500]
        summ.append(dict(field=field, runs=len(ps), excess_low_med_rig=median_out(lo, 'down'), excess_high_med_rig=median_out(hi, 'up'), excess_low_min_rig=min(lo), excess_high_max_rig=max(hi), excess_low_med_float=s5['rel_low_med'], excess_high_med_float=s5['rel_high_med'], long_low_rig=longr[0]['excess_low_down'] if longr else None, long_high_rig=longr[0]['excess_high_up'] if longr else None, interval_500=interval_str(median_out(lo, 'down'), median_out(hi, 'up')), interval_5000=interval_str(longr[0]['excess_low_down'], longr[0]['excess_high_up']) if longr else None, n_budgets=sum((1 for r in out if r['input'] == inp)), n_extra=sum((r['extra_points'] for r in out if r['input'] == inp)), n_float_above_rig=sum((r['L_float_minus_rig'] > 0 for r in out if r['input'] == inp)), max_W_diff_pp=max((r['W_rig_pct_up'] - r['W_float_pct'] for r in out if r['input'] == inp)), P_float_minus_exact_max=max((abs(p['P_float_minus_exact']) for r in out for p in r['pso'] if p['field'] == field)), flag='ok'))
    meta = dict(n_instances=len(out), n_runs=len(runs), wall_s=wall, workers=8, threads=threads, n_extra=sum((r['extra_points'] for r in out)), n_not_reproduced=sum((not r['L_osqp_reproduced'] for r in out)), n_float_above_rig=sum((r['L_float_minus_rig'] > 0 for r in out)), n_float_above_Vfeas=sum((r['L_float_above_Vfeas'] for r in out)), n_clip_high=sum((r['float_clip_high'] for r in out)), n_above_Vfeas_clipped=sum((r['L_float_above_Vfeas'] and r['float_clip_high'] for r in out)), max_float_excess=max((r['L_float_minus_Vfeas'] for r in out if r['L_float_above_Vfeas']), default=0.0), max_W_underestimate_pp=max((r['W_rig_pct_up'] - r['W_float_pct'] for r in out), default=0.0))
    (OUT / 'rows.json').write_text(json.dumps(dict(meta=meta, instances=out), indent=1, default=float, ensure_ascii=False))
    (OUT / 'summary.json').write_text(json.dumps(dict(meta=meta, summary=summ), indent=1, default=float, ensure_ascii=False))
    publish(out, summ, meta)

def publish(out, summ, meta):
    cols = [('field', '초기장'), ('runs', '500회 실행'), ('excess_low_med_float', 'P/U−1 중앙값 (Stage 5 float)'), ('excess_low_med_rig', 'P/U−1 중앙값 (엄밀, 내림)'), ('excess_high_med_float', 'P/L−1 중앙값 (Stage 5 float)'), ('excess_high_med_rig', 'P/L−1 중앙값 (엄밀, 올림)'), ('excess_low_min_rig', 'P/U−1 최소(엄밀)'), ('excess_high_max_rig', 'P/L−1 최대(엄밀)'), ('interval_500', '500회 중앙값 구간(표시용, 바깥쪽 반올림)'), ('long_low_rig', '5000회 seed1 P/U−1'), ('long_high_rig', '5000회 seed1 P/L−1'), ('interval_5000', '5000회 구간(표시용)'), ('n_budgets', '예산 수'), ('n_extra', '추가 기준점'), ('n_float_above_rig', 'float L > 엄밀 L'), ('max_W_diff_pp', '인증폭 차이 최대 (%p)'), ('P_float_minus_exact_max', '|P float − 정확| 최대'), ('flag', '판정')]
    progress.result('s5_rigor', 'S5 · PSO 재구현 평가의 엄밀 검증(추가 분석)', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=summ, note=f"Stage 5 이후의 추가 분석(Stage 5 결과는 그대로). PSO는 선행 논문의 공개 설명에 따른 재구현(paper_pso.py)이며 원 코드가 아님. 고정 float64 입력에 대해 Stage 9와 같은 규칙으로 엄밀 L(볼록성 1차 하한, 정확 정수 산술)을 구하고, Stage 5가 보고한 U 배분과 PSO 배분 104개를 정수성·예산·box 검사 후 정확히 평가. 구간 P/U−1 ≤ P/V_Z*−1 ≤ P/L−1은 끝점을 바깥쪽으로 반올림한 뒤 중앙값(중앙값 보간도 바깥쪽); 표시용 구간 문자열은 소수 1자리로 다시 바깥쪽 반올림. 인스턴스 {meta['n_instances']}개, 8개 병렬 {meta['wall_s']:.0f}s(인증 생성 구간). Stage 5 float 하한 재현 불일치 {meta['n_not_reproduced']}건, float L > 엄밀 L {meta['n_float_above_rig']}건(그중 V_QP* 상한 초과 {meta['n_float_above_Vfeas']}건, 최대 {meta['max_float_excess']:.2g}; 그중 clipping 사례 {meta['n_above_Vfeas_clipped']}건, 나머지는 선형계로 평가한 float dual 값 자체가 V_QP*를 넘은 경우 — 잔차 검사 통과가 보수성을 보장하지 않는다는 실례), 추가 기준점 {meta['n_extra']}건. 초과량은 인증폭에 최대 {meta['max_W_underestimate_pp']:.2g}%p의 과소 추정을 만들었고 결론은 바뀌지 않음."))
    ic = [('input', '초기장'), ('B', 'B'), ('L_float', 'L float (Stage 5)'), ('L_rig', 'L 엄밀(내림)'), ('L_float_minus_rig', 'float − 엄밀'), ('float_clip_high', 'float L = OSQP primal로 clipping'), ('V_feas_up', 'V_QP* 상한'), ('QP_enclosure_width_up', 'V_QP* 포함구간 폭'), ('L_float_above_Vfeas', 'float L > V_QP* 상한'), ('L_float_minus_Vfeas', 'float L − V_QP* 상한'), ('U_rig', 'U 정확(올림)'), ('U_float_minus_exact', 'U float − 정확'), ('W_float_pct', '폭 float (%)'), ('W_rig_pct_up', '폭 엄밀(올림, %)'), ('L_source', 'L 기준점'), ('extra_points', '추가 기준점'), ('t_s', '시간 (s)')]
    progress.result('s5_rigor_inst', 'S5 · 엄밀 검증 인스턴스별', dict(columns=[dict(key=k, label=l) for k, l in ic], rows=[dict(r, pso=None, flag='warn' if r['L_float_above_Vfeas'] or not r['L_osqp_reproduced'] else 'ok') for r in out], note="40개 (초기장, B). V_QP* 상한 = 기준점 근처의 이진 유리수 실행가능점에서의 정확한 V. 판정 warn = Stage 5의 float 하한이 V_QP* 상한을 넘음(그 float 하한은 유효한 하한이 아니었음; 초과량은 'float L − V_QP* 상한'). 시간은 병렬 실행 중 인스턴스 처리시간."))
    lo = {s['field']: s for s in summ}
    progress.task('F.2', 'partial', f"Stage 5 추가 분석 완료: 40개 (초기장, B) 중 Stage 5 float 하한이 엄밀 하한을 넘은 사례 {meta['n_float_above_rig']}건, 그중 V_QP* 상한까지 넘은(유효하지 않았던) 사례 {meta['n_float_above_Vfeas']}건(최대 초과 {meta['max_float_excess']:.2g}, 인증폭 과소 최대 {meta['max_W_underestimate_pp']:.2g}%p). float 하한 재현 불일치 {meta['n_not_reproduced']}. PSO 재구현 104개 실행의 P를 정확히 평가. 500회 실행의 최적값 대비 초과분 중앙값(엄밀 구간, 표시는 바깥쪽 반올림): " + ', '.join((f"{f} {s['interval_500']}" for f, s in lo.items())) + '; 5000회 seed 1: ' + ', '.join((f"{f} {s['interval_5000']}" for f, s in lo.items())) + f". 인증 생성 {meta['wall_s']:.0f}s(8개 병렬). 엄밀 보장 범위: Stage 9·파일럿 6개·Stage 5 40개 예산")
    progress.log('Stage 5 엄밀 검증 추가 분석 게시: s5_rigor, s5_rigor_inst')
if __name__ == '__main__':
    main()
