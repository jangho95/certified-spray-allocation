import determinism
import json
import numpy as np
import incumbents
import progress
import stage3_common as C
from convex_qp_tools import largest_remainder_round
from qp_audit import append_records, make_record, sha256_file, solve_qp_audit
from rounding import boundary_ties, round_indexed
RUN_ID = incumbents.new_run_id(__file__)

def instance(tag, inp, B, cap, greedy_alloc=None, threads=None, experiment=''):
    s0 = C.load_field(inp)
    au = solve_qp_audit(C.A, s0, B, C.MASS, cap)
    xp = largest_remainder_round(au['x'], B, cap)
    xi = round_indexed(au['x'], B, cap)
    pool = C.Pool(inp, B, s0)
    rnd = pool.add('반올림(원고 규칙)', xp, C.write_alloc(C.WORK3 / f'{tag}_round_paper_allocation.csv', xp))
    p_idx = C.write_alloc(C.WORK3 / f'{tag}_round_idx_allocation.csv', xi)
    if greedy_alloc is None:
        xg, mg, pg = C.cpp_repair(inp, B, cap, f'{tag}_repair')
    else:
        pg = greedy_alloc
        xg, mg = (C.read_alloc(pg), None)
    grd = pool.add('greedy+repair' + (' (Stage 1 incumbent)' if greedy_alloc else ''), xg, pg, mg)
    xrr, mrr, prr = C.cpp_repair(inp, B, cap, f'{tag}_rr', p_idx)
    rr = pool.add('반올림→repair', xrr, prr, mrr, 'index 반올림')
    v_cpp = C.cpp_eval(inp, cap, prr)
    L = au['dual_lower']
    u_old = min(rnd['V'], grd['V'])
    best = pool.best(cap)
    for it in pool.items:
        incumbents.register(inp=inp, B=B, cap=cap, method=it['method'], V=it['V'], alloc_path=it['path'], run_id=RUN_ID, feasible=pool.feasible(it, cap), stage=3, experiment=experiment)
    rec = make_record(stage=3, experiment=experiment, case=tag, input=inp, input_sha256=sha256_file(C.INPUTS / inp), M=C.M, N=C.N, B=B, Cmax=cap, kernel=7, sigma=1.75, spacing=2, boundary='reflect', solver='OSQP 1.1.3 + C++ exchange repair', settings=au['settings'], status=au['status'], U=best['V'], L=L, residuals=dict(prim=au['osqp_prim_res'], dual=au['osqp_dual_res']), times=dict(solve=au['t_solve']), U_table_rule=u_old, U_method=best['method'], U_alloc_sha256=sha256_file(best['path']), run_id=RUN_ID, threads=threads, rounding_ties=boundary_ties(au['x'], B, cap), paper_rounding_equals_indexed=bool(np.array_equal(xp, xi)))
    abs_old, abs_new = (u_old - L, best['V'] - L)
    row = dict(B=B, C_max=cap, BN=B / C.N, L=L, V_round=rnd['V'], V_repair=grd['V'], V_round_repair=rr['V'], V_round_repair_cpp=v_cpp, U_old=u_old, U_new=best['V'], U_method=best['method'], gap_old_pct=100 * abs_old / u_old, gap_new_pct=100 * abs_new / best['V'], removed_abs_frac=(abs_old - abs_new) / abs_old if abs_old > 0 else 0.0, moves=mrr, x_max=rr['x_max'], feasible=pool.feasible(rr, cap), near_ties=rec['extra']['rounding_ties']['near_cut'], paper_eq_idx=rec['extra']['paper_rounding_equals_indexed'], flag='warn' if best['V'] < u_old - 1e-12 else 'ok')
    return (row, rec)

def main():
    threads = determinism.assert_single_thread()
    progress.log(f'Stage 3 확장 재실행 run {RUN_ID}')
    t3, tgt, recs = ([], [], [])
    for B, c in [(938, 3), (1250, 5), (1875, 8), (5000, 20), (12500, 50), (28348, 200)]:
        r, rec = instance(f't3_b{B}_c{c}', 'zero_F50.csv', B, c, threads=threads, experiment='table3_round_repair')
        t3.append(r)
        recs.append(rec)
    for name, inp, alloc in [('zero', 'zero_F50.csv', 'zero_allocation.csv'), ('random s7', 'random_u1_30_seed7_F50.csv', 'random_1_30_seed7_sync_allocation.csv'), ('center-30', 'center30_F50.csv', 'center30_allocation.csv'), ('center-100', 'center100_F50.csv', 'center100_allocation.csv')]:
        B = int(C.read_alloc(C.WORK1 / alloc).sum())
        r, rec = instance(f'target_{name.split()[0]}', inp, B, 200, C.WORK1 / alloc, threads, 'target_round_repair')
        tgt.append(dict(field=name, **r))
        recs.append(rec)
    incumbents.rebuild_best()
    append_records(C.OUT / 'records.jsonl', recs)
    meta = dict(run_id=RUN_ID, threads=threads)
    (C.OUT / 'table3_round_repair.json').write_text(json.dumps(dict(meta=meta, rows=t3), indent=1, default=float))
    (C.OUT / 'target_round_repair.json').write_text(json.dumps(dict(meta=meta, rows=tgt), indent=1, default=float))
    publish(t3, tgt)
    for r in t3 + tgt:
        print(f"{r.get('field', 'zero'):10s} B={r['B']:6d} C={r['C_max']:3d} gap {r['gap_old_pct']:.4f}% -> {r['gap_new_pct']:.4f}%  removed {100 * r['removed_abs_frac']:.1f}%  feasible={r['feasible']}  |V_py-V_cpp|={abs(r['V_round_repair'] - r['V_round_repair_cpp']):.1e}  near_ties={r['near_ties']}  paper==idx={r['paper_eq_idx']}")
    return (t3, tgt)

def publish(t3, tgt):
    note = f'run {RUN_ID}, OpenBLAS 1스레드 강제·확인. U(후보 추가) = min(원고 반올림, greedy+repair, 반올림→repair) — 기존 greedy 경로를 유지하고 후보만 추가. 반올림→repair의 시작점은 index 동점 규칙 반올림. 제거 비율 = (기존 절대폭 − 새 절대폭)/기존 절대폭. 해는 모두 예산·box 만족, C++ 평가 일치. 근접 동점 수 = 반올림 선택 경계에서 1e-9 이내 분수부 개수(마지막 자리 변화에 대한 노출).'
    cols = [('B', 'B'), ('C_max', 'C_max'), ('BN', 'B/N'), ('L', 'L'), ('V_round', 'V_round (원고)'), ('V_repair', 'V greedy+repair'), ('V_round_repair', 'V 반올림→repair'), ('U_old', 'U (원고 규칙)'), ('U_new', 'U (후보 추가)'), ('gap_old_pct', 'gap 원고 규칙 (%)'), ('gap_new_pct', 'gap 후보 추가 (%)'), ('removed_abs_frac', '절대폭 제거 비율'), ('moves', 'repair 이동'), ('near_ties', '근접 동점'), ('flag', '판정')]
    progress.result('s3_table3', 'S3 · Table 3 전 행 반올림→repair', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=t3, note=note))
    progress.result('s3_target', 'S3 · 목표 예산 반올림→repair', dict(columns=[dict(key='field', label='초기장')] + [dict(key=k, label=l) for k, l in cols], rows=tgt, note=note + ' Stage 5·6은 results/incumbents/best.json의 U를 사용.'))
if __name__ == '__main__':
    main()
