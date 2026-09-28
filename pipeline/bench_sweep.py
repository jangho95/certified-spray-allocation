import determinism
import json
import sys
import time
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP / 'work' / 'stage1')]
import incumbents
import progress
import stage9_e4 as s9
import stage9_rigor as sr
from qp_audit import append_records
OUT = EXP / 'results' / 'bench_sweep'
FIELDS = [('zero', 'zero_F50.csv'), ('random s7', 'random_u1_30_seed7_F50.csv'), ('center-30', 'center30_F50.csv'), ('center-100', 'center100_F50.csv')]

def main():
    threads = determinism.assert_single_thread()
    OUT.mkdir(parents=True, exist_ok=True)
    sr.OUT = OUT
    progress.task('F.2', 'running', 'benchmark 초기장 4개 × 21 예산: 세 후보 규칙의 sweep과 엄밀 검증(Table 5 갱신용)')
    jobs = [dict(set='bench', name=n, input=inp) for n, inp in FIELDS]
    t0 = time.perf_counter()
    res = s9.run_set(jobs, 'benchmark 4')
    run_id = incumbents.new_run_id(__file__)
    s9.register_candidates(res, 'bench_sweep', run_id)
    append_records(OUT / 'records.jsonl', s9.records(res, 'bench_sweep', run_id, threads))
    fields = [s9.per_field(r) for r in res]
    t_sweep = time.perf_counter() - t0
    import numpy as np
    g = sr.SimpleNamespace(field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
    st = sr.rp.build_stamps(g, sr.rp.make_waypoints(g))
    np.savez(OUT / 'operator.npz', col_ptr=np.cumsum([0] + [len(t.cells) for t in st]), cells=np.concatenate([np.asarray(t.cells, dtype=np.int64) for t in st]), weights=np.concatenate([np.asarray(t.weights, dtype=np.float64) for t in st]))
    t0 = time.perf_counter()
    rig = []
    for k, r in enumerate(res, 1):
        progress.detail('F.2', f'benchmark sweep 엄밀 검증: 초기장 {k}/4')
        rig.append(sr.field_job(r))
    ver = dict((sr.verify(fr) for fr in rig))
    t_rig = time.perf_counter() - t0
    rows = []
    for f, fr in zip(fields, rig):
        rs = fr['rows']
        tgt = next((x for x in rs if x['is_target']))
        rows.append(dict(field=f['name'], input=f['input'], bstar=f['bstar'], cpp_bstar=f['cpp_bstar'], T1=f['T1'], T2=f['T2'], T1_paper=f['T1_paper'], T2_paper=f['T2_paper'], T1_rig=tgt['W_rig_pct_up'], T2_rig=max((x['W_rig_pct_up'] for x in rs)), T2_B=f['T2_B'], U_target=f['U_target'], L_target=f['L_target'], U_method_target=f['U_method_target'], n_float_above_rig=sum((x['L_osqp_minus_rig'] > 0 for x in rs)), n_extra=sum((x['extra_points'] for x in rs)), verified=all(ver[fr['name']]), flag='ok' if all(ver[fr['name']]) else 'warn'))
    (OUT / 'runs.json').write_text(json.dumps(dict(run_id=run_id, threads=threads, results=res), indent=1, default=float, ensure_ascii=False))
    (OUT / 'rigor.json').write_text(json.dumps(dict(fields=rig, t_sweep_s=t_sweep, t_rigor_s=t_rig), indent=1, default=float, ensure_ascii=False))
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=1, default=float, ensure_ascii=False))
    cols = [('field', '초기장'), ('bstar', 'B*'), ('T1_paper', 'B* 폭, 두 후보 (%)'), ('T1', 'B* 폭, 세 후보 (%)'), ('T1_rig', 'B* 폭, 세 후보 엄밀(올림)'), ('T2_paper', 'sweep 최대, 두 후보'), ('T2', 'sweep 최대, 세 후보'), ('T2_rig', 'sweep 최대, 세 후보 엄밀(올림)'), ('T2_B', '최대 위치 B'), ('U_method_target', 'B*의 U 방법'), ('n_float_above_rig', 'float > 엄밀'), ('n_extra', '추가 기준점'), ('verified', '재계산 일치'), ('flag', '판정')]
    progress.result('s_bench_sweep', 'Table 5 갱신 · benchmark 4개 sweep', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=rows, note=f'원고 Table 5의 네 초기장을 Stage 9 파이프라인(B*, 21점 격자 0.85–1.05 B*, 세 후보)으로 다시 계산하고 Stage 9 엄밀 절차로 검증. 두 후보 = 제출본 규칙(반올림, greedy+repair), 세 후보 = 반올림→repair 추가. 엄밀 폭은 고정 float64 입력에 대한 볼록성 하한과 정확한 U에서 올림. sweep {t_sweep:.0f}s, 엄밀 검증 {t_rig:.0f}s(단일 진단 측정).'))
    progress.task('F.2', 'partial', 'benchmark 4개 sweep(세 후보) 완료: B* 폭 ' + ', '.join((f"{r['field']} {r['T1_rig']:.4f}%" for r in rows)) + '; sweep 최대(엄밀) ' + ', '.join((f"{r['T2_rig']:.4f}%" for r in rows)) + '; 두 후보 sweep 최대 ' + ', '.join((f"{r['T2_paper']:.3f}%" for r in rows)) + f". float > 엄밀 {sum((r['n_float_above_rig'] for r in rows))}건")
    progress.log('benchmark 4개 sweep(세 후보)·엄밀 검증 완료: results/bench_sweep')
if __name__ == '__main__':
    main()
