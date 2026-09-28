import determinism
import json
import numpy as np
import incumbents
import progress
import stage3_common as C
from convex_qp_tools import largest_remainder_round
from qp_audit import append_records, make_record, sha256_file, solve_qp_audit
from rounding import boundary_ties, round_indexed
PAIRS = [(938, 3), (1250, 5), (1875, 8)]
INP = 'zero_F50.csv'
RUN_ID = incumbents.new_run_id(__file__)

def main():
    threads = determinism.assert_single_thread()
    s0 = C.load_field(INP)
    rows, summary, records = ([], [], [])
    progress.task('3.1', 'running', f'재실행(run {RUN_ID}): 단일 스레드 강제·기록, index 동점 규칙, 후보 통합')
    for k, (B, c0) in enumerate(PAIRS, 1):
        progress.detail('3.1', f'{k}/3 · B={B} (B/N={B / C.N:.2f}), C_max {c0} vs 200')
        pool = C.Pool(INP, B, s0)
        info = {}
        for cap in (c0, 200):
            au = solve_qp_audit(C.A, s0, B, C.MASS, cap)
            xp = largest_remainder_round(au['x'], B, cap)
            xi = round_indexed(au['x'], B, cap)
            ties = boundary_ties(au['x'], B, cap)
            p_paper = C.write_alloc(C.WORK3 / f'round_paper_b{B}_c{cap}_allocation.csv', xp)
            p_idx = C.write_alloc(C.WORK3 / f'round_idx_b{B}_c{cap}_allocation.csv', xi)
            xg, mg, pg = C.cpp_repair(INP, B, cap, f'repair_b{B}_c{cap}')
            xrr, mrr, prr = C.cpp_repair(INP, B, cap, f'rr_b{B}_c{cap}', p_idx)
            info[cap] = dict(au=au, ties=ties, paper_eq_idx=bool(np.array_equal(xp, xi)), p_idx=p_idx, round=pool.add(f'반올림(원고 규칙, 상한 {cap})', xp, p_paper, cap_run=cap), repair=pool.add(f'greedy+repair(상한 {cap})', xg, pg, mg, 'x=1에서 greedy', cap), rr=pool.add(f'반올림→repair(상한 {cap})', xrr, prr, mrr, f'index 반올림(상한 {cap})', cap))
            if not info[cap]['paper_eq_idx']:
                pool.add(f'index 반올림(상한 {cap})', xi, p_idx, cap_run=cap)
        xa, ma, pa = C.cpp_repair(INP, B, 200, f'cross_r{c0}_cap200_b{B}', info[c0]['p_idx'])
        pool.add(f'교차: 반올림(상한 {c0})→repair(상한 200)', xa, pa, ma, f'index 반올림(상한 {c0})', 200)
        x200 = C.read_alloc(info[200]['p_idx'])
        if x200.max() <= c0:
            xb, mb, pb = C.cpp_repair(INP, B, c0, f'cross_r200_cap{c0}_b{B}', info[200]['p_idx'])
            pool.add(f'교차: 반올림(상한 200)→repair(상한 {c0})', xb, pb, mb, 'index 반올림(상한 200)', c0)
        old = min((info[c0]['round'], info[c0]['repair']), key=lambda it: it['V'])
        xw, mw, pw = C.cpp_repair(INP, B, 200, f'warm_b{B}', old['path'])
        pool.add(f'warm: 기존 보고 해→repair(상한 200)', xw, pw, mw, old['method'], 200)
        U_old = {cap: min(info[cap]['round']['V'], info[cap]['repair']['V']) for cap in (c0, 200)}
        best = {cap: pool.best(cap) for cap in (c0, 200)}
        U = {cap: best[cap]['V'] for cap in (c0, 200)}
        L = {cap: info[cap]['au']['dual_lower'] for cap in (c0, 200)}
        for it in pool.items:
            rows.append(dict(B=B, BN=B / C.N, C_orig=c0, method=it['method'], cap_run=it['cap_run'], start=it['start'], V=it['V'], x_max=it['x_max'], moves=it['moves'], feasible_orig='예' if pool.feasible(it, c0) else '아니오', n_exceed_orig=int(np.sum(it['x'] > c0)), flag='ok' if it['V'] >= U_old[c0] - 1e-12 else 'warn'))
            for cap in (c0, 200):
                if pool.feasible(it, cap):
                    incumbents.register(inp=INP, B=B, cap=cap, method=it['method'], V=it['V'], alloc_path=it['path'], run_id=RUN_ID, feasible=True, stage=3, experiment='cmax_contrast')
        for cap in (c0, 200):
            au = info[cap]['au']
            b = best[cap]
            records.append(make_record(stage=3, experiment='cmax_contrast', case=f'B={B} cap={cap}', input=INP, input_sha256=sha256_file(C.INPUTS / INP), M=C.M, N=C.N, B=B, Cmax=cap, kernel=7, sigma=1.75, spacing=2, boundary='reflect', solver='OSQP 1.1.3 + C++ exchange repair', settings=au['settings'], status=au['status'], U=U[cap], L=L[cap], residuals=dict(prim=au['osqp_prim_res'], dual=au['osqp_dual_res'], Hz_v=au['range_rel_resid']), times=dict(solve=au['t_solve']), U_table3_rule=U_old[cap], U_method=b['method'], U_alloc_sha256=sha256_file(b['path']), run_id=RUN_ID, threads=threads, rounding_ties=info[cap]['ties'], paper_rounding_equals_indexed=info[cap]['paper_eq_idx']))
        cross = {it['method']: it['V'] for it in pool.items if it['method'].startswith('교차')}
        summary.append(dict(B=B, BN=B / C.N, C_orig=c0, relax_diff=info[200]['au']['primal'] - info[c0]['au']['primal'], relax_xmax=float(info[c0]['au']['x'].max()), relax_upper_active=info[c0]['au']['n_upper_active'], L_orig=L[c0], L_200=L[200], U_old=U_old[c0], U_orig=U[c0], U_200=U[200], U_orig_method=best[c0]['method'], U_200_method=best[200]['method'], cap_effect=U[200] - U[c0], start_effect=U[c0] - U_old[c0], rr_orig=info[c0]['rr']['V'], rr_200=info[200]['rr']['V'], cross=' / '.join((f'{v:.6f}' for v in cross.values())), gap_old_pct=100 * (U_old[c0] - L[c0]) / U_old[c0], gap_orig_pct=100 * (U[c0] - L[c0]) / U[c0], gap_200_pct=100 * (U[200] - L[200]) / U[200], ties_orig=info[c0]['ties']['near_cut'], paper_eq_idx=info[c0]['paper_eq_idx'] and info[200]['paper_eq_idx'], warm_moves=next((it['moves'] for it in pool.items if it['method'].startswith('warm'))), attribution='변화 없음' if abs(U[c0] - U_old[c0]) < 1e-12 and abs(U[200] - U[c0]) < 1e-12 else '시작점 효과(기존 상한에서 같은 개선), 선택된 U의 상한 완화 이득 없음' if abs(U[200] - U[c0]) < 1e-12 else '선택된 U에 상한 완화 이득 있음', flag='ok' if abs(U[c0] - U_old[c0]) < 1e-12 and abs(U[200] - U[c0]) < 1e-12 else 'warn'))
        progress.log(f"Stage 3 B={B}: U_old={U_old[c0]:.6f}, U({c0})={U[c0]:.6f}, U(200)={U[200]:.6f} [{best[200]['method']}]")
    incumbents.rebuild_best()
    append_records(C.OUT / 'records.jsonl', records)
    meta = dict(run_id=RUN_ID, threads=threads)
    (C.OUT / 'cmax_rows.json').write_text(json.dumps(dict(meta=meta, rows=rows), indent=1, default=float, ensure_ascii=False))
    (C.OUT / 'cmax_summary.json').write_text(json.dumps(dict(meta=meta, rows=summary), indent=1, default=float, ensure_ascii=False))
    progress.task('3.1', 'done', f"""run {RUN_ID}, OpenBLAS 1스레드 확인. 연속완화 값 차이(C_orig vs 200) 최대 {max((abs(s['relax_diff']) for s in summary)):.1e}, 연속해 상한 활성 0. Table 3 규칙 U 재현 ({', '.join((f"{s['U_old']:.4f}" for s in summary))})""")
    progress.task('3.2', 'done', 'warm repair(기존 보고 해→상한 200) 이동 ' + ', '.join((str(s['warm_moves']) for s in summary)) + '회. 교차 대조(시작점 고정, repair 상한만 교환): ' + '; '.join((f"B={s['B']} {s['cross']}" for s in summary)))
    progress.task('3.3', 'done', 'U(cap)=해당 상한에서 실행가능한 모든 후보(기존 상한의 새 후보 포함) 중 최솟값. ' + ' · '.join((f"B={s['B']}: {s['U_old']:.4f}→U({s['C_orig']})={s['U_orig']:.4f}, U(200)={s['U_200']:.4f} — {s['attribution']}" for s in summary)))
    progress.task('3.4', 'done', '후보별 목적값·max x·기존 상한 초과·이동 수 기록(S3 · C_max 대조), 최종 U는 records.jsonl과 results/incumbents/best.json에 할당 sha256·run id와 함께 기록')
    cols = [('B', 'B'), ('C_orig', '기존 C_max'), ('method', '후보'), ('cap_run', 'repair 상한'), ('start', '시작점'), ('V', '목적값 V'), ('x_max', 'max x'), ('n_exceed_orig', '기존 상한 초과'), ('feasible_orig', '기존 상한 실행가능'), ('moves', 'repair 이동'), ('flag', '판정')]
    progress.result('s3_rows', 'S3 · C_max 대조', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=rows, note=f'run {RUN_ID}. 판정 warn = Table 3 규칙의 U(min(원고 반올림, greedy+repair))보다 좋은 후보. 교차 대조는 같은 시작점에서 repair 상한만 바꾼 것.'))
    scols = [('B', 'B'), ('BN', 'B/N'), ('C_orig', '기존 C_max'), ('relax_diff', 'V_QP 차이(200−기존)'), ('relax_upper_active', '연속해 상한 활성'), ('L_orig', 'L'), ('U_old', 'U (Table 3 규칙)'), ('U_orig', 'U (기존 상한, 후보 추가)'), ('U_200', 'U (상한 200)'), ('U_200_method', 'U(200) 해'), ('cap_effect', '상한 완화 효과 ΔU'), ('start_effect', '시작점 효과 ΔU'), ('rr_orig', '반올림→repair (기존)'), ('rr_200', '반올림→repair (200)'), ('cross', '교차 대조 V'), ('gap_old_pct', 'gap Table 3 (%)'), ('gap_orig_pct', 'gap 후보 추가 (%)'), ('warm_moves', 'warm 이동'), ('attribution', '원인 분리'), ('flag', '판정')]
    progress.result('s3_summary', 'S3 · 요약', dict(columns=[dict(key=k, label=l) for k, l in scols], rows=summary, note='판정 warn = 새 후보가 Table 3 규칙의 U보다 좋음. 원인 분리는 선택된 U 기준이다: 후보별로는 시작점이 달라 상한에 따라 값이 다를 수 있다(예: B=938 반올림→repair). 교차 대조가 같으면 그 차이는 시작점 차이다. heuristic 결과는 정수 최적값에 대한 결론이 아니다(Stage 6).'))
    return summary
if __name__ == '__main__':
    main()
