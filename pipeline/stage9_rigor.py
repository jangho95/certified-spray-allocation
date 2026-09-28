from package_paths import load_json
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
OUT = EXP / 'results' / 'stage9_rigor'
GAP_TOL = 1e-09
PROTOCOL = dict(status='post hoc additional analysis of the preregistered E4 sample; the preregistered results are unchanged', guarantee='fixed float64 data: A from build_stamps (F=50, s=2, 7x7, sigma 1.75, reflect), s0 as stored; ideal Gaussian (same support and folding) through eps', lower_bound='max over points xb of V(xb) + min_X grad V(xb)^T (x - xb), exact integer arithmetic, rounded down', points=f'OSQP 1e-8 point of qp_audit (recomputed) and the active-set refinement started from it; if the best (V_feas - L)/U > {GAP_TOL:g}, add OSQP 1e-10 and Clarabel points and refine from the best of them', upper_bound='stored U_rev allocation of e4_runs.json after integrality/budget/box checks, exact V, rounded up', widths='100 (U - L)/U from exact rationals, rounded up; T1_rig at B*, T2_rig = max over the 21-budget grid', summary='same statistics as the preregistered report; sample maxima rounded up to 4 decimals', reproducibility='L_osqp recomputed must equal the stored value bit for bit (recorded otherwise)')

def field_job(res):
    t_field = time.perf_counter()
    A, mass, n = s9.operator()
    g = SimpleNamespace(field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
    st = rp.build_stamps(g, rp.make_waypoints(g))
    s0 = s9.load_field(res['input'])
    rows, xbars, allocs, feasible_points = ([], [], [], [])
    for r in res['rows']:
        t0 = time.perf_counter()
        B = r['B']
        ex = rp.Exact(st, s0, B, s9.CMAX)
        au = solve_qp_audit(A, s0, B, mass, s9.CMAX)
        same = au['dual_lower'] == r['L_osqp']
        pts = {'OSQP 1e-8': au['x']}
        Ls = {'OSQP 1e-8': ex.bound(au['x'])[0]}
        (Lr, xr), _ = rp.refine(A, s0, B, s9.CMAX, ex, au['x'])
        pts['정제(1e-8 출발)'], Ls['정제(1e-8 출발)'] = (xr, Lr)
        xu = np.zeros(n)
        for line in Path(r['cands'][r['U_method']]['path']).read_text().splitlines()[1:]:
            f_ = line.split(',')
            xu[int(f_[0])] = int(f_[-1])
        chk = rp.check_int(xu, B, s9.CMAX)
        assert all(chk.values()), (res['name'], B, chk)
        U = ex.value(xu)
        best = max(Ls, key=Ls.get)
        V_best, _, _ = rp.feasible_upper(ex, pts, B, s9.CMAX)
        extra = bool((V_best - Ls[best]) / U > GAP_TOL)
        if extra:
            a10 = solve_qp_audit(A, s0, B, mass, s9.CMAX, eps_abs=1e-10, eps_rel=1e-10)
            xc, _, _ = rp.clarabel_x(A, s0, B, s9.CMAX)
            for k, x in (('OSQP 1e-10', a10['x']), ('Clarabel 1e-10', xc)):
                pts[k], Ls[k] = (x, ex.bound(x)[0])
            st_k = max(('OSQP 1e-10', 'Clarabel 1e-10'), key=Ls.get)
            (L2, x2), _ = rp.refine(A, s0, B, s9.CMAX, ex, pts[st_k])
            pts['정제(추가 기준점 출발)'], Ls['정제(추가 기준점 출발)'] = (x2, L2)
            best = max(Ls, key=Ls.get)
        L = max(Ls[best], Fraction(0))
        V_feas, xf, fsrc = rp.feasible_upper(ex, pts, B, s9.CMAX)
        W = rp.up(100 * (U - L) / U)
        rows.append(dict(B=B, is_target=r['is_target'], L_rig=rp.down(L), U_rig=rp.up(U), W_rig_pct_up=W, W_prereg_pct=r['W_rev'], dW_pp=W - r['W_rev'], L_osqp=r['L_osqp'], L_osqp_minus_rig=float(Fraction(r['L_osqp']) - L), L_source=best, extra_points=extra, rel_gap_after=float((V_feas - L) / U), L_osqp_reproduced=same, U_float_minus_exact=float(Fraction(r['U_rev']) - U), U_checks=chk, L_exact=rp.frs(L), U_exact=rp.frs(U), V_feas_exact=rp.frs(V_feas), feasible_source=fsrc, t_s=time.perf_counter() - t0))
        xbars.append(np.asarray(pts[best], dtype=np.float64))
        allocs.append(xu.astype(np.int64))
        feasible_points.append([rp.frs(v) for v in xf])
    wd = OUT / 'witness'
    wd.mkdir(parents=True, exist_ok=True)
    np.savez(wd / f"{res['name']}.npz", B=np.array([r['B'] for r in rows]), xbar=np.array(xbars), alloc=np.array(allocs), s0=np.asarray(s0, dtype=np.float64), Cmax=np.array(s9.CMAX), M=np.array(s9.M), feasible_relaxation=np.array(feasible_points))
    return dict(name=res['name'], input=res['input'], bstar=res['bstar'], rows=rows, t_field=time.perf_counter() - t_field)

def verify(fr):
    op = np.load(OUT / 'operator.npz')
    ptr = op['col_ptr']
    st = [SimpleNamespace(cells=op['cells'][ptr[j]:ptr[j + 1]].tolist(), weights=op['weights'][ptr[j]:ptr[j + 1]].tolist()) for j in range(len(ptr) - 1)]
    w = np.load(OUT / 'witness' / f"{fr['name']}.npz")
    s0, C = (w['s0'], int(w['Cmax']))
    assert int(w['M']) == rp.M == len(s0)
    ok = []
    for k, r in enumerate(fr['rows']):
        ex = rp.Exact(st, s0, int(w['B'][k]), C)
        L = max(ex.bound(w['xbar'][k])[0], Fraction(0))
        U = ex.value(w['alloc'][k].astype(float))
        if 'feasible_relaxation' in w:
            xf = [Fraction(str(v)) for v in w['feasible_relaxation'][k]]
            assert sum(xf) == int(w['B'][k]) and all((1 <= v <= C for v in xf))
            assert rp.frs(ex.value(xf)) == r['V_feas_exact']
        ok.append(rp.frs(L) == r['L_exact'] and rp.frs(U) == r['U_exact'] and all(rp.check_int(w['alloc'][k].astype(float), int(w['B'][k]), C).values()))
    return (fr['name'], ok)

def main():
    threads = determinism.assert_single_thread()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'protocol.json').write_text(json.dumps(dict(PROTOCOL, code_sha256=dict(stage9_rigor=sha256_file(Path(__file__)), rigor_pilot=sha256_file(EXP / 'pipeline' / 'rigor_pilot.py')), environment=rp.env_versions()), indent=1, ensure_ascii=False))
    runs = load_json((EXP / 'results' / 'stage9' / 'e4_runs.json').read_text())
    todo = [r for r in runs['results'] if not r.get('failed')]
    g = SimpleNamespace(field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
    pts = rp.make_waypoints(g)
    st = rp.build_stamps(g, pts)
    np.savez(OUT / 'operator.npz', col_ptr=np.cumsum([0] + [len(t.cells) for t in st]), cells=np.concatenate([np.asarray(t.cells, dtype=np.int64) for t in st]), weights=np.concatenate([np.asarray(t.weights, dtype=np.float64) for t in st]))
    eps, dmax = rp.ideal_eps(st, pts, 2, s9.CMAX)
    progress.task('F.2', 'running', f'Stage 9 추가 분석: 본 표본 {len(todo)}개 × 21 예산에 엄밀 사후 검증(고정 절차) 적용')
    t0 = time.perf_counter()
    out = []
    with cf.ProcessPoolExecutor(max_workers=8) as exr:
        for k, fr in enumerate(exr.map(field_job, todo), 1):
            out.append(fr)
            progress.detail('F.2', f'Stage 9 추가 분석: 초기장 {k}/{len(todo)} 완료(8개 병렬)')
    wall = time.perf_counter() - t0
    progress.detail('F.2', 'Stage 9 추가 분석: 검증 자료만으로 전 사례 재계산 중')
    with cf.ProcessPoolExecutor(max_workers=8) as exr:
        ver = dict(exr.map(verify, out))
    for fr in out:
        for r in fr['rows']:
            L, U = (Fraction(r['L_exact']), Fraction(r['U_exact']))
            dL = max(Fraction(0), Fraction(rp.sqrt_down(L)) - Fraction(eps))
            L_id, U_id = (rp.down(dL * dL), rp.up((Fraction(rp.sqrt_up(U)) + Fraction(eps)) ** 2))
            r['W_ideal_pct_up'] = rp.up(100 * (Fraction(U_id) - Fraction(L_id)) / Fraction(U_id))
    fields = []
    for fr in out:
        rs = fr['rows']
        tgt = next((r for r in rs if r['is_target']))
        fields.append(dict(name=fr['name'], bstar=fr['bstar'], T1_rig=tgt['W_rig_pct_up'], T2_rig=max((r['W_rig_pct_up'] for r in rs)), T1_prereg=tgt['W_prereg_pct'], T2_prereg=max((r['W_prereg_pct'] for r in rs)), max_dW_pp=max((r['dW_pp'] for r in rs)), min_dW_pp=min((r['dW_pp'] for r in rs)), n_extra=sum((r['extra_points'] for r in rs)), n_not_reproduced=sum((not r['L_osqp_reproduced'] for r in rs)), n_osqp_above_rig=sum((r['L_osqp_minus_rig'] > 0 for r in rs)), verified=all(ver[fr['name']]), T1_ideal=tgt['W_ideal_pct_up'], T2_ideal=max((r['W_ideal_pct_up'] for r in rs)), t_field=fr['t_field']))
    summ = s9.summarize([dict(name=f['name'], T1=f['T1_rig'], T2=f['T2_rig'], T1_ideal=f['T1_ideal'], T2_ideal=f['T2_ideal'], T1_prereg=f['T1_prereg'], T2_prereg=f['T2_prereg']) for f in fields], keys=('T1', 'T2', 'T1_ideal', 'T2_ideal', 'T1_prereg', 'T2_prereg'))
    meta = dict(n_fields=len(fields), n_instances=sum((len(o['rows']) for o in out)), wall_s=wall, workers=8, instance_time_s=dict(median=float(np.median([r['t_s'] for o in out for r in o['rows']])), max=max((r['t_s'] for o in out for r in o['rows']))), eps_ideal=eps, delta_A_max=dmax, threads=threads)
    (OUT / 'rows.json').write_text(json.dumps(dict(meta=meta, fields=out), indent=1, default=float, ensure_ascii=False))
    (OUT / 'fields.json').write_text(json.dumps(fields, indent=1, default=float, ensure_ascii=False))
    (OUT / 'summary.json').write_text(json.dumps(dict(meta=meta, summary=summ), indent=1, default=float, ensure_ascii=False))
    publish(fields, summ, meta, out)
    return (fields, summ, meta)

def publish(fields, summ, meta, out, post=None):
    from stage9_report import LEVEL, ceil_pct
    lab = {'T1': 'T1 엄밀(B*)', 'T2': 'T2 엄밀(격자 최대)', 'T1_ideal': 'T1 이상 Gaussian', 'T2_ideal': 'T2 이상 Gaussian', 'T1_prereg': 'T1 사전 고정(float L)', 'T2_prereg': 'T2 사전 고정(float L)'}
    target = {'T1': 'G1 = B*에서 사전 고정 배분의 실제 최적성 갭', 'T2': 'G2 = 격자 최대 실제 최적성 갭', 'T1_ideal': '이상 모델의 G1', 'T2_ideal': '이상 모델의 G2', 'T1_prereg': 'T1(사전 고정 float 폭)', 'T2_prereg': 'T2(사전 고정 float 폭)'}
    rows = [dict(statistic=lab[k], n=v['n'], mean=v['mean'], median=v['median'], q1=v['q1'], q3=v['q3'], iqr=v['iqr'], min=v['min'], max=v['max'], max_up=ceil_pct(v['max']), ucl_target=target[k], argmax=v['argmax']) for k, v in summ.items()]
    allr = [r for o in out for r in o['rows']]
    cols = [('statistic', '통계량 (%)'), ('n', 'n'), ('mean', '평균'), ('median', '중앙값'), ('q1', 'Q1'), ('q3', 'Q3'), ('iqr', 'IQR (Q3−Q1)'), ('min', '최소'), ('max', '표본 최대'), ('max_up', '검증된 표본 최대(올림)'), ('ucl_target', '올림값이 95번째 백분위 상측 신뢰한계가 되는 대상'), ('argmax', '최대 초기장')]
    pt = f" 사후 단계(별도 측정): 결과 폴더만으로 전 사례 재검산 {post['verify_wall_s']:.0f}s(8개 병렬), 공통 연산자 ε 계산 {post['eps_s']:.2g}s." if post else ''
    progress.result('s9_rigor', 'S9 · 엄밀 사후 검증(추가 분석)', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=rows, note=f"사전 고정 이후의 추가 분석이며 사전 고정 결과가 주 결과. 추가 분석 실행 전에 절차와 코드 해시를 protocol.json에 기록(재실행 시 덮어쓰므로 원래 Stage 9 등록과 같은 변경 방지 절차는 아님). 원고 설정으로 계산한 Stage 9의 float 하한(제출 이후 표본)을 고정 float64 입력에 대한 엄밀 하한(볼록성 1차 하한, 정확 정수 산술; OSQP 1e-8 해와 그 정제, gap이 크면 추가 기준점)으로 검증하고, 정수해 검사 후 정확한 U로 상대폭을 유리수에서 올림. 이상 Gaussian(같은 지지·경계 접기): max(0, √L − ε)² 하한과 (√U + ε)² 상한, ε = {meta['eps_ideal']:.2g}. 통계 해석: 사후에 정한 엄밀 폭 자체의 백분위에는 사전 고정의 Wilks 해석을 붙이지 않고, 사전 고정 배분 절차의 실제 최적성 갭 G(s) = max_B 100(U − V_Z*)/U(T1은 B*에서)를 대상으로 함 — 검증된 하한 ≤ V_Z*이므로 G(s_i) ≤ T_rig(s_i), 따라서 Pr[q_0.95(G) ≤ max_i T_rig(s_i)] ≥ 1 − 0.95^59 = {LEVEL:.4f}(표본 수 59, T1·T2 동시 보장 아님). 사전 고정 행은 원래 T1·T2에 대한 해석. 인증 생성 {meta['wall_s']:.0f}s(8개 프로세스; 재검산·공통 ε·최종 보고 제외), 사례당 중앙값 {meta['instance_time_s']['median']:.2f}s는 병렬 실행 중 처리시간이며 단일 프로세스 벤치마크가 아님.{pt} 추가 기준점 규칙 발동 {sum((r['extra_points'] for r in allr))}건, float 하한 재현 불일치 {sum((not r['L_osqp_reproduced'] for r in allr))}건, float 하한 > 엄밀 하한 {sum((r['L_osqp_minus_rig'] > 0 for r in allr))}건, 재검산 실패 {sum((not f['verified'] for f in fields))}개 초기장. 엄밀 보장 범위는 Stage 9 표본과 F.2 파일럿 6개에 한정(Stage 5·7 등 다른 결과는 미적용). 재검산 파일 목록: results/stage9_rigor/VERIFY_FILES.json."))
    fc = [('name', '초기장'), ('bstar', 'B*'), ('T1_rig', 'T1 엄밀'), ('T1_prereg', 'T1 사전 고정'), ('T2_rig', 'T2 엄밀'), ('T2_prereg', 'T2 사전 고정'), ('max_dW_pp', '폭 차이 최대 (%p)'), ('min_dW_pp', '폭 차이 최소 (%p)'), ('n_extra', '추가 기준점'), ('n_osqp_above_rig', 'float > 엄밀'), ('verified', '재계산 일치'), ('flag', '판정')]
    progress.result('s9_rigor_fields', 'S9 · 엄밀 사후 검증 초기장별', dict(columns=[dict(key=k, label=l) for k, l in fc], rows=[dict(f, flag='ok' if f['verified'] and (not f['n_not_reproduced']) else 'warn') for f in fields], note='폭 차이 = 엄밀(올림) − 사전 고정 float 폭, 21개 예산의 최대·최소. float > 엄밀 = 원고 설정 float 하한이 엄밀 하한보다 큰 예산 수.'))
    s = summ
    progress.task('F.2', 'partial', f"Stage 9 추가 분석 완료: 원고 설정으로 계산한 Stage 9 float 하한 {meta['n_instances']:,}개 모두 엄밀 하한 이하, 재검산 {sum((f['verified'] for f in fields))}/{len(fields)} 초기장 일치(결과 폴더만 사용). T2 엄밀 표본 최대 {s['T2']['max']:.7f}% → 올림 {ceil_pct(s['T2']['max']):.4f}%, 이는 사전 고정 배분 절차의 실제 최적성 갭 G2의 95번째 백분위 상측 신뢰한계(≥ {LEVEL:.4f}). 인증 생성 {meta['wall_s']:.0f}s(8개 프로세스, 재검산·ε·보고 제외)" + (f", 재검산 {post['verify_wall_s']:.0f}s" if post else '') + '. 엄밀 보장은 Stage 9·파일럿 6개에 한정')
    progress.log('Stage 9 엄밀 사후 검증 보고 보완 게시: s9_rigor, s9_rigor_fields')
if __name__ == '__main__':
    main()
