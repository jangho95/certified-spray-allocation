from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
import determinism
import clarabel
import numpy as np
import scipy.sparse as sp
EXP = Path(__file__).resolve().parents[1]
WORK1 = EXP / 'work' / 'stage1'
sys.path[:0] = [str(WORK1), str(EXP / 'pipeline')]
import progress
from convex_qp_tools import dense_A, largest_remainder_round
from qp_audit import append_records, lower_bound_at, make_record, sha256_file, solve_qp_audit
from scip_compare import build_stamps, evaluate, load_warm_start, make_waypoints
INP = EXP / 'inputs'
OUT = EXP / 'results' / 'stage2'
OUT.mkdir(parents=True, exist_ok=True)
CASES = [('T-zero', '목표 · zero', '목표 예산', 2, 7, 'zero_F50.csv', 200, 'zero_allocation.csv'), ('T-random', '목표 · random s7', '목표 예산', 2, 7, 'random_u1_30_seed7_F50.csv', 200, 'random_1_30_seed7_sync_allocation.csv'), ('T-c30', '목표 · center-30', '목표 예산', 2, 7, 'center30_F50.csv', 200, 'center30_allocation.csv'), ('T-c100', '목표 · center-100', '목표 예산', 2, 7, 'center100_F50.csv', 200, 'center100_allocation.csv'), ('L-938', '저샷 · B=938 (Cmax 3)', '저샷', 2, 7, 'zero_F50.csv', 3, 'cmax_zero_c3_b938_allocation.csv'), ('L-1250', '저샷 · B=1250 (Cmax 5)', '저샷', 2, 7, 'zero_F50.csv', 5, 'cmax_zero_c5_b1250_allocation.csv'), ('L-1875', '저샷 · B=1875 (Cmax 8)', '저샷', 2, 7, 'zero_F50.csv', 8, 'cmax_zero_c8_b1875_allocation.csv'), ('V-k11zero', '작은 분산 · 11×11 zero', '작은 분산', 2, 11, 'zero_F50.csv', 200, 'kernel_zero_isotropic-11_allocation.csv'), ('V-s1c30', '작은 분산 · s=1 center-30', '작은 분산', 1, 7, 'center30_F50.csv', 200, 'ablation_center30_s1_allocation.csv'), ('V-s1zero', '작은 분산 · s=1 zero', '작은 분산', 1, 7, 'zero_F50.csv', 200, 'ablation_zero_s1_allocation.csv')]
SETTINGS = [(eps, pol) for eps in (1e-06, 1e-08, 1e-10) for pol in (True, False)]
REF = (1e-08, True)
THRESH = (1e-08, 1e-10, 1e-12)

def load_field(name):
    return np.array([float(t) for t in (INP / name).read_text().replace('\n', ',').split(',') if t.strip()])

def prepare(case):
    cid, label, group, s, k, inp, cmax, alloc = case
    cnt = (50 - 1) // s + 1
    g = SimpleNamespace(field_size=50, waypoint_rows=cnt, waypoint_cols=cnt, spray_interval=s, kernel_size=k, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
    st = build_stamps(g, make_waypoints(g))
    n = len(st)
    A = dense_A(st, 2500, n)
    s0 = load_field(inp)
    x_rep = np.array(load_warm_start(WORK1 / alloc, n))
    B = int(x_rep.sum())
    H = 2.0 / 2500 * (A.T @ A)
    ev = np.linalg.eigvalsh(H)
    return dict(cid=cid, label=label, group=group, s=s, k=k, inp=inp, cmax=cmax, alloc=alloc, stamps=st, A=A, s0=s0, n=n, B=B, mass=st[0].mass, V_repair=float(evaluate(list(x_rep), s0, st, 50)[1]), lam_min=float(ev[0]), lam_max=float(ev[-1]), cond_H=float(ev[-1] / ev[0]) if ev[0] > 0 else float('inf'))

def lagrangian_bound(A, s0, B, mass, cmax, y, primal, range_tol):
    m, n = A.shape
    mean = (float(s0.sum()) + B * mass) / m
    r0 = s0 - mean
    H = 2.0 / m * (A.T @ A)
    q = 2.0 / m * (A.T @ r0)
    const = float(r0 @ r0) / m
    lo = np.concatenate([[float(B)], np.full(n, 1.0)])
    hi = np.concatenate([[float(B)], np.full(n, float(cmax))])
    v = q + y[0] + y[1:]
    z = np.linalg.solve(H, v)
    rel = float(np.linalg.norm(H @ z - v)) / (1.0 + float(np.linalg.norm(v)))
    raw = const - 0.5 * float(v @ z) - float(hi @ np.maximum(y, 0.0)) + float(lo @ np.maximum(-y, 0.0))
    return lower_bound_at(dict(dual_raw=raw, primal=primal, range_rel_resid=rel), range_tol) | dict(raw=raw, rel=rel)

def clarabel_solve(P):
    A, s0, B, mass, cmax, n = (P['A'], P['s0'], P['B'], P['mass'], P['cmax'], P['n'])
    m = A.shape[0]
    mean = (float(s0.sum()) + B * mass) / m
    r0 = s0 - mean
    H = 2.0 / m * (A.T @ A)
    q = 2.0 / m * (A.T @ r0)
    const = float(r0 @ r0) / m
    Pm = sp.triu(sp.csc_matrix(H)).tocsc()
    Ac = sp.vstack([sp.csc_matrix(np.ones((1, n))), sp.identity(n), -sp.identity(n)]).tocsc()
    b = np.concatenate([[float(B)], np.full(n, float(cmax)), -np.ones(n)])
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.tol_gap_abs = st.tol_gap_rel = 1e-10
    st.tol_feas = 1e-10
    st.max_iter = 400
    t0 = time.perf_counter()
    sol = clarabel.DefaultSolver(Pm, q, Ac, b, [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(2 * n)], st).solve()
    t = time.perf_counter() - t0
    x = np.asarray(sol.x)
    z = np.asarray(sol.z)
    y = np.concatenate([[z[0]], z[1:n + 1] - z[n + 1:]])
    primal = float(0.5 * x @ H @ x + q @ x + const)
    lb = lagrangian_bound(A, s0, B, mass, cmax, y, primal, 1e-10)
    return dict(status=str(sol.status), iters=int(sol.iterations), time_s=t, primal=primal, dual_obj=float(sol.obj_val_dual) + const, L_lagr=lb['dual_lower'], L_lagr_raw=lb['raw'], range_rel=lb['rel'], fallback=lb['fallback'], budget_viol=abs(float(x.sum()) - B), box_viol=float(max(0, np.max(1 - x), np.max(x - cmax))))

def main():
    t_all = time.time()
    progress.task('2.1', 'running', '사례 10개 준비: 격자·커널·정본 입력·Stage 1 incumbent 불러오기')
    prepared = []
    for c in CASES:
        P = prepare(c)
        prepared.append(P)
        progress.detail('2.1', f"{len(prepared)}/10 {P['label']}: N={P['n']}, B={P['B']}, cond(H)={P['cond_H']:.2e}")
    prep_rows = [dict(case=P['label'], group=P['group'], N=P['n'], B=P['B'], Cmax=P['cmax'], BN=P['B'] / P['n'], kernel=f"{P['k']}×{P['k']}", spacing=P['s'], V_repair=P['V_repair'], lam_min_H=P['lam_min'], cond_H=P['cond_H'], input=P['inp'], flag='ok') for P in prepared]
    progress.result('s2_cases', 'S2 · 사례', dict(columns=[dict(key='case', label='사례'), dict(key='group', label='군'), dict(key='N', label='N'), dict(key='B', label='B'), dict(key='Cmax', label='Cmax'), dict(key='BN', label='B/N'), dict(key='kernel', label='kernel'), dict(key='spacing', label='s'), dict(key='V_repair', label='V_repair (Stage 1)'), dict(key='lam_min_H', label='λmin(H)'), dict(key='cond_H', label='cond(H)'), dict(key='input', label='입력')], rows=prep_rows, note='incumbent는 Stage 1 재현 결과(work/stage1), 입력은 inputs/ 정본 CSV.'))
    progress.task('2.1', 'done', f"사례 10개 준비 완료 (목표 4 · 저샷 3 · 작은 분산 3). cond(H) {min((p['cond_H'] for p in prepared)):.1e}–{max((p['cond_H'] for p in prepared)):.1e}")
    progress.task('2.2', 'running', 'OSQP 60회 풀이 시작')
    runs, records = ([], [])
    total = len(prepared) * len(SETTINGS)
    done = 0
    for P in prepared:
        for eps, pol in SETTINGS:
            done += 1
            progress.detail('2.2', f"{done}/{total} · {P['label']} · eps={eps:.0e}, polish={('on' if pol else 'off')}")
            au = solve_qp_audit(P['A'], P['s0'], P['B'], P['mass'], P['cmax'], eps_abs=eps, eps_rel=eps, polish=pol)
            xr = largest_remainder_round(au['x'], P['B'], P['cmax'])
            vr = float(evaluate(list(xr), P['s0'], P['stamps'], 50)[1])
            runs.append(dict(cid=P['cid'], case=P['label'], group=P['group'], eps=eps, polish=pol, status=au['status'], status_polish=au['status_polish'], iters=au['iter'], primal=au['primal'], dual_raw=au['dual_raw'], L=au['dual_lower'], range_rel=au['range_rel_resid'], fallback=au['fallback'], clip=au['clip_low'] or au['clip_high'], prim_res=au['osqp_prim_res'], dual_res=au['osqp_dual_res'], kkt_inf=au['kkt_stationarity_inf'], budget_viol=au['budget_viol'], box_viol=au['box_viol'], V_round=vr, t_solve=au['t_solve'], t_total=au['t_gram'] + au['t_setup'] + au['t_solve'] + au['t_dual']))
            records.append(make_record(stage=2, experiment='E1', case=P['label'], input=P['inp'], input_sha256=sha256_file(INP / P['inp']), M=2500, N=P['n'], B=P['B'], Cmax=P['cmax'], kernel=P['k'], sigma=1.75, spacing=P['s'], boundary='reflect', solver='OSQP 1.1.3', settings=au['settings'], status=au['status'], L=au['dual_lower'], residuals=dict(prim=au['osqp_prim_res'], dual=au['osqp_dual_res'], Hz_v=au['range_rel_resid'], kkt_inf=au['kkt_stationarity_inf']), times=dict(gram=au['t_gram'], setup=au['t_setup'], solve=au['t_solve'], dual=au['t_dual']), primal=au['primal'], dual_raw=au['dual_raw'], iters=au['iter']))
    U = {}
    for P in prepared:
        ref = next((r for r in runs if r['cid'] == P['cid'] and (r['eps'], r['polish']) == REF))
        U[P['cid']] = min(P['V_repair'], ref['V_round'])
    for r in runs:
        r['U'] = U[r['cid']]
        r['abs_gap'] = r['U'] - r['L']
        r['gap_pct'] = 100 * r['abs_gap'] / r['U']
    for rec, r in zip(records, runs):
        rec['U'] = r['U']
        rec['abs_gap'] = r['abs_gap']
        rec['rel_gap'] = r['abs_gap'] / r['U']
    append_records(OUT / 'records.jsonl', records)
    n_iter_cap = sum((1 for r in runs if 'maximum' in r['status'].lower()))
    progress.task('2.2', 'done', f'60회 완료. 상태: ' + ', '.join(sorted({r['status'] for r in runs})) + f'. 반복 상한 도달 {n_iter_cap}회', {'solves': 60})
    progress.task('2.3', 'running', '임계값 1e-8 / 1e-10 / 1e-12 재적용')
    thr_rows = []
    for r in runs:
        for thr in THRESH:
            lb = lower_bound_at(dict(dual_raw=r['dual_raw'], primal=r['primal'], range_rel_resid=r['range_rel']), thr)
            thr_rows.append(dict(case=r['case'], cid=r['cid'], eps=r['eps'], polish=r['polish'], threshold=thr, range_rel=r['range_rel'], L=lb['dual_lower'], fallback=lb['fallback'], gap_pct=100 * (r['U'] - lb['dual_lower']) / r['U']))
    changed = sum((1 for t in thr_rows if t['fallback'] != next((x for x in thr_rows if x['cid'] == t['cid'] and x['eps'] == t['eps'] and (x['polish'] == t['polish']) and (x['threshold'] == 1e-10)))['fallback']))
    progress.task('2.3', 'done', f"{len(thr_rows)}개 조합. 기준(1e-10) 대비 fallback 판정이 바뀐 조합 {changed}개; range 잔차 범위 {min((r['range_rel'] for r in runs)):.1e}–{max((r['range_rel'] for r in runs)):.1e}")
    progress.task('2.4', 'running', 'Clarabel 0.11.1 교차검증')
    clar = {}
    for i, P in enumerate(prepared, 1):
        progress.detail('2.4', f"{i}/10 {P['label']}")
        clar[P['cid']] = clarabel_solve(P)
    progress.task('2.4', 'done', '상태: ' + ', '.join(sorted({c['status'] for c in clar.values()})))
    progress.task('2.5', 'running', '민감도 요약 작성')
    summary = []
    for P in prepared:
        rs = [r for r in runs if r['cid'] == P['cid']]
        ref = next((r for r in rs if (r['eps'], r['polish']) == REF))
        C = clar[P['cid']]
        Ls = [r['L'] for r in rs]
        vtol = 1e-08 * max(1.0, abs(C['primal']))
        over = [r for r in rs if r['L'] > C['primal'] + vtol]
        thr_ref = [t for t in thr_rows if t['cid'] == P['cid'] and t['eps'] == REF[0] and (t['polish'] == REF[1])]
        row = dict(case=P['label'], group=P['group'], N=P['n'], B=P['B'], cond_H=P['cond_H'], U=U[P['cid']], L_ref=ref['L'], gap_ref_pct=ref['gap_pct'], L_min=min(Ls), L_max=max(Ls), L_spread_rel_U=(max(Ls) - min(Ls)) / U[P['cid']], gap_range_pp=max((r['gap_pct'] for r in rs)) - min((r['gap_pct'] for r in rs)), fallback_any=any((r['fallback'] for r in rs)), clip_any=any((r['clip'] for r in rs)), fallback_settings=sum((r['fallback'] for r in rs)), iters_range=f"{min((r['iters'] for r in rs))}–{max((r['iters'] for r in rs))}", V_clarabel=C['primal'], clarabel_status=C['status'], Lref_minus_Vclar=ref['L'] - C['primal'], L_clarabel=C['L_lagr'], L_clar_minus_Lref=C['L_lagr'] - ref['L'], valid_all=len(over) == 0, thr_L=' / '.join((f"{t['L']:.6g}" for t in sorted(thr_ref, key=lambda t: -t['threshold']))), t_solve_ref=ref['t_solve'], polish_success=f"{sum((r['status_polish'] == 1 for r in rs if r['polish']))}/3", clip_high_any=any((r['clip'] and r['dual_raw'] > r['primal'] for r in rs)))
        row['flag'] = 'fail' if not row['valid_all'] else 'warn' if row['fallback_any'] or row['gap_range_pp'] > 0.01 else 'ok'
        summary.append(row)
    (OUT / 'e1_runs.json').write_text(json.dumps(runs, indent=1, default=float, ensure_ascii=False))
    (OUT / 'e1_thresholds.json').write_text(json.dumps(thr_rows, indent=1, default=float, ensure_ascii=False))
    (OUT / 'e1_clarabel.json').write_text(json.dumps(clar, indent=1, default=float, ensure_ascii=False))
    (OUT / 'e1_summary.json').write_text(json.dumps(summary, indent=1, default=float, ensure_ascii=False))
    run_cols = [('case', '사례'), ('eps', 'eps'), ('polish', 'polish'), ('status', '상태'), ('iters', '반복'), ('primal', 'V_QP (OSQP)'), ('dual_raw', 'raw dual'), ('L', 'L'), ('range_rel', 'Hz=v 잔차'), ('fallback', 'fallback'), ('clip', 'clip'), ('status_polish', 'polish 상태'), ('prim_res', 'prim res'), ('dual_res', 'dual res'), ('U', 'U'), ('gap_pct', 'gap (%)'), ('V_round', 'V_round'), ('t_solve', 'solve (s)')]
    progress.result('s2_runs', 'S2 · tolerance×polish 60회', dict(columns=[dict(key=k, label=l) for k, l in run_cols], rows=[{**r, 'polish': 'on' if r['polish'] else 'off', 'fallback': 'yes' if r['fallback'] else 'no', 'clip': 'yes' if r['clip'] else 'no', 'flag': 'warn' if r['fallback'] else 'ok'} for r in runs], note='U는 사례마다 고정: min(Stage 1 repair, 기준 설정 1e-8·polish on의 반올림). L은 range 임계값 1e-10 기준.'))
    progress.result('s2_thresholds', 'S2 · range 임계값', dict(columns=[dict(key=k, label=l) for k, l in [('case', '사례'), ('eps', 'eps'), ('polish', 'polish'), ('threshold', '임계값'), ('range_rel', 'Hz=v 잔차'), ('fallback', 'fallback'), ('L', 'L'), ('gap_pct', 'gap (%)')]], rows=[{**t, 'polish': 'on' if t['polish'] else 'off', 'fallback': 'yes' if t['fallback'] else 'no', 'flag': 'warn' if t['fallback'] else 'ok'} for t in thr_rows], note='같은 풀이 결과에 range-test 임계값만 바꿔 다시 적용. fallback이면 L=0(자명한 하한).'))
    sum_cols = [('case', '사례'), ('N', 'N'), ('B', 'B'), ('cond_H', 'cond(H)'), ('U', 'U'), ('L_ref', 'L (기준)'), ('gap_ref_pct', 'gap 기준 (%)'), ('L_min', 'L 최소'), ('L_max', 'L 최대'), ('L_spread_rel_U', 'L 범위/U'), ('gap_range_pp', 'gap 범위 (%p)'), ('fallback_settings', 'fallback 설정 수'), ('polish_success', 'polish 성공'), ('clip_high_any', 'primal로 잘림'), ('iters_range', 'OSQP 반복'), ('V_clarabel', 'V_QP* (Clarabel)'), ('Lref_minus_Vclar', 'L기준 − V_Clarabel'), ('L_clarabel', 'L (Clarabel 승수)'), ('thr_L', 'L @ 1e-8/1e-10/1e-12'), ('valid_all', '하한 타당(6설정)'), ('flag', '판정')]
    progress.result('s2_summary', 'S2 · 민감도 요약', dict(columns=[dict(key=k, label=l) for k, l in sum_cols], rows=[{**r, 'valid_all': '예' if r['valid_all'] else '아니오'} for r in summary], note='하한 타당: 6개 설정의 L이 모두 Clarabel primal(독립 내점법, 허용오차 1e-10) + 1e-8·max(1,V) 이하. 설정 간 값이 비슷하다는 것은 구간 연산 증명이 아니다. gap 범위는 6개 설정에서 gap(%)의 최대−최소. polish 성공은 OSQP status_polish=1인 설정 수(polish on 3개 중). primal로 잘림: raw dual > OSQP primal이라 min(dual, primal)로 잘린 경우 — OSQP primal 점이 미세하게 비실행가능하면 엄밀한 상한이 아님.'))
    pol_ok = sum((int(r['polish_success'].split('/')[0]) for r in summary))
    progress.task('2.5', 'done', f"요약 10행. 하한 타당성 위반 {sum((not r['valid_all'] for r in summary))}건, fallback {sum((r['fallback_any'] for r in summary))}건, polish 성공 {pol_ok}/30, primal로 잘림 {sum((r['clip_high_any'] for r in summary))}사례, 총 {time.time() - t_all:.0f}s")
    progress.log('Stage 2 종료')
if __name__ == '__main__':
    main()
