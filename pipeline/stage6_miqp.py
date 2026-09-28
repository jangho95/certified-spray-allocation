from package_paths import load_json
import determinism
import concurrent.futures as cf
import hashlib
import json
import subprocess
import sys
import time
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
from scip_compare import build_stamps, make_initial, make_waypoints
PY = sys.executable
BIN = EXP / 'build' / 'pufoam_solver'
INPUTS = EXP / 'inputs'
WORK = EXP / 'work' / 'stage6'
OUT = EXP / 'results' / 'stage6'
TIME_LIMIT, MEM_LIMIT, PARALLEL = (300, 3000, 6)
FORMS = ('expanded', 'soc', 'soc_cont')
RUN_ID = incumbents.new_run_id(__file__)

def ensure_input(name, F, field, height=30.0, side=None):
    path = INPUTS / name
    if not path.exists():
        a = SimpleNamespace(field=field, seed=1, random_low=1, random_high=30, center_height=height, center_size=side or 16, field_size=F, initial_field=None)
        v = make_initial(a).reshape(F, F)
        path.write_text('\n'.join((','.join((repr(float(x)) for x in r)) for r in v)) + '\n')
        man = load_json((INPUTS / 'manifest.json').read_text())
        man[name] = dict(description=f'{field} field F={F}' + (f', height {height}, side {side}' if side else '') + ' (Stage 6 series)', field_size=F, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mean=float(v.mean()))
        (INPUTS / 'manifest.json').write_text(json.dumps(man, indent=1, ensure_ascii=False))
    return name

def instances():
    out = []
    for field, inp in [('zero', 'small_zero_F12.csv'), ('random', 'small_random_pcg64_seed7_F12.csv'), ('center-30', 'small_center30_c4_F12.csv')]:
        out.append(dict(id=f'N16_{field}', series='N16 (Table 6)', field=field, input=inp, F=12, s=3, rows=4, cmax=8, B=64, rho=None))
    for F in (12, 16):
        rows = F // 2
        side = round(F / 3)
        fields = [('zero', ensure_input('small_zero_F12.csv' if F == 12 else f'zero_F{F}.csv', F, 'zero')), ('center-30', ensure_input('small_center30_c4_F12.csv' if F == 12 else f'center30_c{side}_F{F}.csv', F, 'center', 30.0, side))]
        for field, inp in fields:
            for rho in (1.5, 3, 45):
                B = int(round(rho * rows * rows))
                out.append(dict(id=f'N{rows * rows}_{field}_rho{rho:g}', series=f'N={rows * rows}', field=field, input=inp, F=F, s=2, rows=rows, cmax=200, B=B, rho=rho))
    return out

def load_field(name):
    return np.array([float(t) for t in (INPUTS / name).read_text().replace('\n', ',').split(',') if t.strip()])

def cpp(inst, prefix, init=None):
    cmd = [str(BIN), '--initial-field', str(INPUTS / inst['input']), '--field-size', str(inst['F']), '--waypoint-rows', str(inst['rows']), '--waypoint-cols', str(inst['rows']), '--spray-interval', str(inst['s']), '--budget', str(inst['B']), '--cmax', str(inst['cmax']), '--quiet', '--out-prefix', str(prefix)]
    subprocess.run(cmd + (['--init-allocation', str(init)] if init else []), check=True, capture_output=True)
    return Path(f'{prefix}_allocation.csv')

def read_alloc(path, n):
    x = np.zeros(n, dtype=int)
    for line in Path(path).read_text().splitlines()[1:]:
        f = line.split(',')
        x[int(f[0])] = int(f[-1])
    return x

def write_alloc(path, x):
    Path(path).write_text('waypoint,shot_count\n' + ''.join((f'{j},{int(v)}\n' for j, v in enumerate(x))))
    return Path(path)

def prepare(inst):
    g = SimpleNamespace(field_size=inst['F'], waypoint_rows=inst['rows'], waypoint_cols=inst['rows'], spray_interval=inst['s'], kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
    st = build_stamps(g, make_waypoints(g))
    n, m = (len(st), inst['F'] ** 2)
    A = dense_A(st, m, n)
    s0 = load_field(inst['input'])
    B, cmax = (inst['B'], inst['cmax'])
    au = solve_qp_audit(A, s0, B, st[0].mass, cmax)
    tag = WORK / inst['id']
    cands = {'반올림(원고 규칙)': write_alloc(f'{tag}_round_paper.csv', largest_remainder_round(au['x'], B, cmax)), 'greedy+repair': cpp(inst, f'{tag}_greedy')}
    ri = write_alloc(f'{tag}_round_idx.csv', round_indexed(au['x'], B, cmax))
    cands['반올림→repair'] = cpp(inst, f'{tag}_rr', ri)
    vals = {}
    for name, path in cands.items():
        x = read_alloc(path, n)
        f = A @ x + s0
        vals[name] = float(np.mean((f - f.mean()) ** 2))
        incumbents.register(inp=inst['input'], B=B, cap=cmax, method=name, V=vals[name], alloc_path=path, run_id=RUN_ID, feasible=bool(x.sum() == B and x.min() >= 1 and (x.max() <= cmax)), F=inst['F'], spacing=inst['s'], stage=6, experiment='miqp_pilot')
    best = min(vals, key=vals.get)
    inst.update(N=n, L=au['dual_lower'], V_QP=au['primal'], cands=vals, U_pipeline=vals[best], U_pipeline_method=best, warm_alloc=str(cands[best]), input_path=str(INPUTS / inst['input']), time_limit=TIME_LIMIT, mem_limit=MEM_LIMIT, qp_status=au['status'])
    (WORK / f"{inst['id']}.json").write_text(json.dumps(inst, indent=1, default=str))
    return inst

def run_job(inst, form):
    out = WORK / f"{inst['id']}_{form}.out.json"
    t0 = time.time()
    try:
        p = subprocess.run([PY, str(EXP / 'pipeline' / 'stage6_scip_worker.py'), str(WORK / f"{inst['id']}.json"), form, str(out)], capture_output=True, text=True, timeout=TIME_LIMIT + 180)
        if p.returncode != 0 or not out.exists():
            return dict(instance=inst['id'], form=form, status='crash', returncode=p.returncode, stderr=(p.stderr or '')[-300:], wall_s=time.time() - t0)
        return load_json(out.read_text()) | dict(wall_s=time.time() - t0)
    except subprocess.TimeoutExpired:
        return dict(instance=inst['id'], form=form, status='external-timeout', wall_s=time.time() - t0)

def main():
    threads = determinism.assert_single_thread()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    insts = instances()
    progress.task('6.1', 'running', f'인스턴스 {len(insts)}개 준비(L, 후보, 공통 초기해)')
    for k, inst in enumerate(insts, 1):
        progress.detail('6.1', f"{k}/{len(insts)} · {inst['id']} · B={inst['B']}")
        prepare(inst)
    incumbents.rebuild_best()
    progress.task('6.1', 'done', 'B/N≈1.5·3·45(실제 정수 B 기록): ' + ', '.join((f"{i['id']} B={i['B']} (B/N={i['B'] / i['N']:.2f})" for i in insts if i['series'] != 'N16 (Table 6)')))
    progress.task('6.2', 'done', '새 계열: s=2, 7×7 kernel, σ=1.75, reflect, C_max=200, 초기장 zero·center-30(한 변 round(F/3)), B=round(ρN). 기존 N=16(Table 6)은 s=3, 4×4 waypoint, C_max=8, B=64, random 초기장 포함 — 새 계열과 설정이 다름')
    progress.task('6.4', 'skipped', '다른 convex MIQP solver 없음: Gurobi 라이선스 없음, HiGHS는 정수 QP 미지원. 같은 SCIP의 두 정식화 비교로 보고(독립 solver 검증 아님)')
    jobs = [(i, f) for i in insts for f in FORMS]
    progress.task('6.3', 'running', f'SCIP {len(jobs)}건(정식화 3 × 인스턴스 {len(insts)}), 동시 {PARALLEL}개, SCIP 제한 {TIME_LIMIT}s·{MEM_LIMIT}MB, 외부 timeout {TIME_LIMIT + 180}s')
    results = []
    with cf.ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(run_job, i, f): (i, f) for i, f in jobs}
        for k, fu in enumerate(cf.as_completed(futs), 1):
            r = fu.result()
            results.append(r)
            progress.detail('6.3', f"{k}/{len(jobs)} 완료 · 최근 {r['instance']} {r['form']}: {r.get('status')}")
    by = {(r['instance'], r['form']): r for r in results}
    rows, records = ([], [])
    for inst in insts:
        for form in ('expanded', 'soc'):
            r = by[inst['id'], form]
            if r.get('alloc') and r.get('feasible'):
                incumbents.register(inp=inst['input'], B=inst['B'], cap=inst['cmax'], method=f'SCIP {form}', V=r['V_eval'], alloc_path=r['alloc'], run_id=RUN_ID, feasible=True, F=inst['F'], spacing=inst['s'], stage=6, experiment='miqp_pilot')
    best = incumbents.rebuild_best()
    for inst in insts:
        e, so, sc = (by[inst['id'], 'expanded'], by[inst['id'], 'soc'], by[inst['id'], 'soc_cont'])
        key = incumbents.key(inst['input'], inst['B'], inst['cmax'], F=inst['F'], spacing=inst['s'])
        U = best[key]['V']
        duals = [r['dual_bound'] for r in (e, so) if isinstance(r.get('dual_bound'), float)]
        D = max(duals) if duals else None
        optimal = any((r.get('status') == 'optimal' for r in (e, so)))
        vz = min((r['V_eval'] for r in (e, so) if r.get('status') == 'optimal' and r.get('V_eval') is not None)) if optimal else None
        width = U - inst['L']
        row = dict(id=inst['id'], series=inst['series'], field=inst['field'], N=inst['N'], B=inst['B'], BN=inst['B'] / inst['N'], cmax=inst['cmax'], L=inst['L'], V_QP=inst['V_QP'], U_pipeline=inst['U_pipeline'], U_pipeline_method=inst['U_pipeline_method'], U_final=U, U_final_method=best[key]['method'], exp_status=e.get('status'), exp_primal=e.get('V_eval'), exp_dual=e.get('dual_bound'), exp_gap=e.get('gap'), exp_nodes=e.get('nodes'), exp_time=e.get('scip_time_s'), exp_warm=e.get('warm_accepted'), soc_status=so.get('status'), soc_primal=so.get('V_eval'), soc_dual=so.get('dual_bound'), soc_gap=so.get('gap'), soc_nodes=so.get('nodes'), soc_time=so.get('scip_time_s'), soc_warm=so.get('warm_accepted'), cont_status=sc.get('status'), cont_dual=sc.get('dual_bound'), cont_primal=sc.get('primal_bound'), D=D, V_Z=vz, optimal=optimal, integrality_gap=vz - inst['V_QP'] if vz is not None else None, inc_subopt_pipeline=inst['U_pipeline'] - vz if vz is not None else None, proven_share=(D - inst['L']) / width if D is not None and width > 0 else None, rel_width_pct=100 * width / U, rel_open_pct=100 * (U - D) / U if D is not None else None)
        row['flag'] = 'ok' if optimal else 'warn' if D is not None else 'fail'
        rows.append(row)
        records.append(make_record(stage=6, experiment='miqp_pilot', case=inst['id'], input=inst['input'], input_sha256=sha256_file(INPUTS / inst['input']), M=inst['F'] ** 2, N=inst['N'], B=inst['B'], Cmax=inst['cmax'], kernel=7, sigma=1.75, spacing=inst['s'], boundary='reflect', solver='SCIP 10.0 (PySCIPOpt 6.2.1) + OSQP 1.1.3', settings=dict(time_limit=TIME_LIMIT, mem_limit=MEM_LIMIT, gap=0), status=f"expanded={e.get('status')}, soc={so.get('status')}", U=U, L=inst['L'], D=D, V_Z=vz, U_method=best[key]['method'], U_alloc_sha256=best[key]['alloc_sha256'], run_id=RUN_ID, threads=threads))
    append_records(OUT / 'records.jsonl', records)
    (OUT / 'scip_jobs.json').write_text(json.dumps(results, indent=1, default=float, ensure_ascii=False))
    (OUT / 'miqp_summary.json').write_text(json.dumps(dict(run_id=RUN_ID, time_limit=TIME_LIMIT, mem_limit=MEM_LIMIT, rows=rows), indent=1, default=float, ensure_ascii=False))
    publish(rows, results)
    n_opt = sum((r['optimal'] for r in rows))
    progress.task('6.3', 'done', f'두 정식화 결과: 최적성 증명 {n_opt}/{len(rows)} 인스턴스. 전개식 상태 ' + ', '.join(sorted({str(r['exp_status']) for r in rows})) + ' · SOC 상태 ' + ', '.join(sorted({str(r['soc_status']) for r in rows})))
    progress.task('6.5', 'done', f"공통 초기해 = 파이프라인 최선 해, 전개식·SOC 모두 checkSol 후 addSol. 수용 전개식 {sum((bool(r['exp_warm']) for r in rows))}/{len(rows)}, SOC {sum((bool(r['soc_warm']) for r in rows))}/{len(rows)}. SCIP {TIME_LIMIT}s·{MEM_LIMIT}MB, 외부 timeout {TIME_LIMIT + 180}s")
    progress.task('6.6', 'done', '종료 상태·incumbent·정수 하한 D·연속완화 값(OSQP V_QP, L, SCIP 연속 SOC)·원래 U·MIP gap·node 수를 results/stage6/miqp_summary.json·scip_jobs.json·records.jsonl에 저장')
    return rows

def publish(rows, results):
    cols = [('id', '인스턴스'), ('N', 'N'), ('B', 'B'), ('BN', 'B/N'), ('L', 'L'), ('V_QP', 'V_QP'), ('U_pipeline', 'U 파이프라인'), ('U_final', 'U 최종'), ('U_final_method', 'U 해'), ('D', '정수 하한 D'), ('V_Z', 'V_Z* (증명)'), ('integrality_gap', '정수화 갭 V_Z*−V_QP'), ('inc_subopt_pipeline', '파이프라인 incumbent 준최적'), ('proven_share', '(D−L)/(U−L)'), ('rel_width_pct', '인증폭 (U−L)/U %'), ('rel_open_pct', '남은 폭 (U−D)/U %'), ('exp_status', '전개식 상태'), ('exp_gap', '전개식 gap'), ('exp_nodes', '전개식 node'), ('exp_time', '전개식 시간'), ('soc_status', 'SOC 상태'), ('soc_gap', 'SOC gap'), ('soc_nodes', 'SOC node'), ('soc_time', 'SOC 시간'), ('cont_status', '연속 SOC 상태'), ('cont_dual', '연속 SOC 하한'), ('flag', '판정')]
    progress.result('s6_summary', 'S6 · 정수 최적화 파일럿', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=rows, note=f'SCIP 10.0, 정식화 2종(전개된 이차식, 셀별 잔차 제곱 SOC) + 연속 SOC. 제한 {TIME_LIMIT}s·{MEM_LIMIT}MB, 공통 초기해. 판정 ok = 최적성 증명(정수화 갭과 incumbent 준최적성 분리 가능), warn = 정수 하한 D만 있음. (D−L)/(U−L)은 인증폭 중 정수 하한으로 설명된 비율. 같은 SCIP의 두 정식화 비교이며 독립 solver 검증이 아님. 작은 문제의 결론을 N=625에 자동 일반화하지 않는다.'))
    jc = list(dict.fromkeys((k for r in results for k in r if k not in ('alloc',))))
    progress.result('s6_jobs', 'S6 · SCIP 실행별', dict(columns=[dict(key=k, label=k) for k in jc], rows=[{**r, 'flag': 'ok' if r.get('status') == 'optimal' else 'fail' if r.get('status') in ('crash', 'external-timeout') else 'warn'} for r in results], note='각 SCIP 실행(별도 프로세스). crash/external-timeout은 프로세스 단위 실패.'))
if __name__ == '__main__':
    main()
