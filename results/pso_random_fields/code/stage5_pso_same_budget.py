from package_paths import load_json
import determinism
import concurrent.futures as cf
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP / 'work' / 'stage1')]
import incumbents
import progress
from convex_qp_tools import dense_A, largest_remainder_round
from qp_audit import append_records, make_record, sha256_file, solve_qp_audit
from rounding import round_indexed
from scip_compare import build_stamps, make_waypoints
from stage4_worker import cpp_cmd, geometry, load_field, read_alloc, variance, write_alloc
INPUTS = EXP / 'inputs'
ARCH = EXP / 'reference' / 'archive'
WORK = EXP / 'work' / 'stage5'
OUT = EXP / 'results' / 'stage5'
RUN_ID = incumbents.new_run_id(__file__)
FIELDS = [('zero', 'zero_F50.csv', dict(field='zero'), 'zero'), ('random_1_30_seed7', 'random_u1_30_seed7_F50.csv', dict(field='random'), 'random'), ('center30', 'center30_F50.csv', dict(field='center', center_height=30.0), 'center30'), ('center100', 'center100_F50.csv', dict(field='center', center_height=100.0), 'center100')]
B_TARGET = dict(zero=28348, random_1_30_seed7=26120, center30=27912, center100=26896)
G, CNT = geometry(50)
ST = build_stamps(G, make_waypoints(G))
N, M = (len(ST), 2500)
A = dense_A(ST, M, N)
MASS = ST[0].mass

def pso_job(job):
    name, inp, over, seed, iters, log_every = job
    import pso_repeated_runs as P
    from paper_pso import run_pso
    a = P.make_args(dict(over, initial_field=INPUTS / inp, iters=iters, log_every=log_every), seed)
    sim, gbest, met, hist, _ = run_pso(a)
    tag = f'pso_{name}_s{seed}_it{iters}'
    apath = WORK / f'{tag}_allocation.csv'
    write_alloc(apath, gbest)
    s0sum = float(sim.initial.sum())
    with open(WORK / f'{tag}_history.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['iter', 'gbest_variance', 'gbest_mean', 'gbest_budget', 'gbest_fit'])
        for h in hist:
            w.writerow([h['iter'], h['gbest_variance'], h['gbest_mean'], int(round((h['gbest_mean'] * sim.m - s0sum) / sim.mass)), h['gbest_fit']])
    return dict(field=name, input=inp, seed=seed, iters=iters, P=met['variance'], mean=met['mean'], B=int(gbest.sum()), alloc=str(apath), history=str(WORK / f'{tag}_history.csv'))

def budget_candidates(inp, B):
    s0 = load_field(inp)
    au = solve_qp_audit(A, s0, B, MASS, 200)
    xp = largest_remainder_round(au['x'], B, 200)
    xi = round_indexed(au['x'], B, 200)
    tag = f'{inp[:-4]}_B{B}'
    pp, pi = (WORK / f'{tag}_round_paper.csv', WORK / f'{tag}_round_idx.csv')
    write_alloc(pp, xp)
    write_alloc(pi, xi)
    subprocess.run(cpp_cmd(inp, 50, CNT, B, WORK / f'{tag}_greedy'), check=True, capture_output=True)
    subprocess.run(cpp_cmd(inp, 50, CNT, B, WORK / f'{tag}_rr', pi), check=True, capture_output=True)
    cands = {'반올림(원고 규칙)': pp, 'greedy+repair': WORK / f'{tag}_greedy_allocation.csv', '반올림→repair': WORK / f'{tag}_rr_allocation.csv'}
    vals = {}
    for name, path in cands.items():
        x = read_alloc(path, N)
        ok = bool(x.sum() == B and x.min() >= 1 and (x.max() <= 200))
        vals[name] = variance(A, x, s0)
        incumbents.register(inp=inp, B=B, cap=200, method=name, V=vals[name], alloc_path=path, run_id=RUN_ID, feasible=ok, stage=5, experiment='pso_same_budget')
    return dict(L=au['dual_lower'], status=au['status'], V_QP=au['primal'], cands=vals, audit=au)

def main():
    threads = determinism.assert_single_thread()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(n, i, o, s, 500, 1) for n, i, o, _ in FIELDS for s in range(1, 26)] + [(n, i, o, 1, 5000, 10) for n, i, o, _ in FIELDS]
    progress.task('5.1', 'running', f'PSO {len(jobs)}회 재실행(500회 100개 + 5000회 4개), 할당·이력 저장')
    runs = []
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        for k, r in enumerate(ex.map(pso_job, jobs), 1):
            runs.append(r)
            if k % 8 == 0 or k == len(jobs):
                progress.detail('5.1', f'PSO 재실행 {k}/{len(jobs)}')
    arch = {(r['instance'], int(r['run'])): r for r in csv.DictReader(open(ARCH / 'pso_repeat_runs.csv'))}
    diffs, alloc_checks = ([], [])
    for r in runs:
        x = read_alloc(r['alloc'], N)
        r['V_eval'] = variance(A, x, load_field(r['input']))
        r['feasible'] = bool(x.sum() == r['B'] and x.min() >= 1 and (x.max() <= 200))
        if r['iters'] == 500:
            a = arch[r['field'], r['seed']]
            diffs.append(max(abs(r['P'] - float(a['variance'])) / float(a['variance']), abs(r['mean'] - float(a['mean'])) / float(a['mean']), abs(r['B'] - int(a['total_shots']))))
        else:
            short = dict(((f[0], f[3]) for f in FIELDS))[r['field']]
            ap = ARCH / f'pso_long_{short}_seed1_allocation.csv'
            if ap.exists():
                xa = read_alloc(ap, N)
                alloc_checks.append(dict(field=r['field'], identical=bool(np.array_equal(xa, x)), V_archived_alloc=variance(A, xa, load_field(r['input'])), V_rerun=r['V_eval']))
    max_diff = max(diffs)
    progress.task('5.1', 'done' if max_diff < 1e-09 else 'warn', f"500회 100개: 아카이브의 P·평균·B와 최대 상대차 {max_diff:.1e}. 모든 해 실행가능 {sum((r['feasible'] for r in runs))}/{len(runs)}, 독립 재평가 |V−P| 최대 {max((abs(r['V_eval'] - r['P']) for r in runs)):.1e}. 5000회 seed 1: 아카이브 할당과 동일 {sum((c['identical'] for c in alloc_checks))}/{len(alloc_checks)}")
    budgets = sorted({(r['input'], r['B']) for r in runs})
    progress.task('5.2', 'running', f'고유 (초기장, B) {len(budgets)}개에서 L과 후보 계산')
    info = {}
    for k, (inp, B) in enumerate(budgets, 1):
        progress.detail('5.2', f'{k}/{len(budgets)} · {inp} · B={B}')
        info[inp, B] = budget_candidates(inp, B)
    for r in runs:
        incumbents.register(inp=r['input'], B=r['B'], cap=200, method=f"PSO seed {r['seed']} ({r['iters']} it)", V=r['V_eval'], alloc_path=r['alloc'], run_id=RUN_ID, feasible=r['feasible'], stage=5, experiment='pso_same_budget')
    best = incumbents.rebuild_best()
    reg = [load_json(l) for l in incumbents.REG.read_text().splitlines()]
    brow, records = ([], [])
    for (inp, B), d in info.items():
        key = incumbents.key(inp, B, 200)
        nopso = min((r['V'] for r in reg if r['key'] == key and r['feasible'] and (not r['method'].startswith('PSO'))))
        d.update(U=best[key]['V'], U_method=best[key]['method'], U_nopso=nopso)
        brow.append(dict(input=inp, B=B, L=d['L'], **{f'V {k}': v for k, v in d['cands'].items()}, U=d['U'], U_method=d['U_method'], U_without_pso=nopso, gap_pct=100 * (d['U'] - d['L']) / d['U'], pso_is_best=best[key]['method'].startswith('PSO'), flag='warn' if best[key]['method'].startswith('PSO') else 'ok'))
        au = d['audit']
        records.append(make_record(stage=5, experiment='pso_same_budget', case=f'{inp} B={B}', input=inp, input_sha256=sha256_file(INPUTS / inp), M=M, N=N, B=B, Cmax=200, kernel=7, sigma=1.75, spacing=2, boundary='reflect', solver='OSQP 1.1.3 + C++ exchange repair', settings=au['settings'], status=au['status'], U=d['U'], L=d['L'], residuals=dict(prim=au['osqp_prim_res'], dual=au['osqp_dual_res'], Hz_v=au['range_rel_resid']), U_method=d['U_method'], U_alloc_sha256=best[key]['alloc_sha256'], run_id=RUN_ID, threads=threads))
    append_records(OUT / 'records.jsonl', records)
    progress.task('5.2', 'done', f"{len(budgets)}개 (초기장, B)의 L과 후보 3종 등록, PSO 해 {len(runs)}개도 후보로 등록. PSO가 최선인 예산 {sum((b['pso_is_best'] for b in brow))}개")
    rows = []
    for r in runs:
        d = info[r['input'], r['B']]
        L, U, P = (d['L'], d['U'], r['V_eval'])
        rows.append(dict(field=r['field'], seed=r['seed'], iters=r['iters'], B=r['B'], P=P, L=L, U=U, U_method=d['U_method'], P_minus_U=P - U, P_minus_L=P - L, rel_low=P / U - 1, rel_high=P / L - 1 if L > 0 else None, cert_gap_pct=100 * (U - L) / U, flag='ok' if L > 0 else 'warn'))
    summ = summarize(rows)
    progress.task('5.3', 'done', ' · '.join((f"{s['field']}: P/V*−1 ∈ [{fmt(s['rel_low_med'])}, {fmt(s['rel_high_med'])}] (중앙값), 원고 S7 비율 {s['s7_ratio']:.3f} → 같은 예산 P/U 평균 {s['same_budget_ratio_mean']:.3f}" for s in summ)))
    nz = sum((s['L_zero'] for s in summ))
    progress.task('5.4', 'done', f"L=0인 실행 {nz}개 — {('모든 실행에 유한한 상한 구간 제시' if nz == 0 else 'L=0 실행은 상한 미제시')}")
    progress.task('5.5', 'done', f'아카이브에는 실행별 P·평균·B만 있고 할당은 seed 1(500·5000회)뿐. 같은 seed의 결정적 재실행으로 100개 할당을 복원했고 P·평균·B가 아카이브와 일치(최대 상대차 {max_diff:.1e}). 원래 실행의 할당 자체를 검증한 것은 아니며, 재실행 할당으로 검증했다는 범위를 명시')
    progress.task('5.6', 'done', f'500회(매 반복)와 5000회(10회마다) 이력에 gbest 분산·평균·예산을 저장 (work/stage5/*_history.csv), 할당 {len(runs)}개 저장·레지스트리 등록')
    (OUT / 'pso_runs.json').write_text(json.dumps(dict(run_id=RUN_ID, threads=threads, rows=rows, alloc_checks=alloc_checks, max_rel_diff_vs_archive=max_diff), indent=1, default=float, ensure_ascii=False))
    (OUT / 'pso_budgets.json').write_text(json.dumps(brow, indent=1, default=float, ensure_ascii=False))
    (OUT / 'pso_summary.json').write_text(json.dumps(summ, indent=1, default=float, ensure_ascii=False))
    publish(rows, brow, summ)
    return (rows, brow, summ)

def summarize(rows):
    arch_sum = {r['instance']: r for r in csv.DictReader(open(ARCH / 'pso_repeat_summary.csv'))}
    reg = [load_json(l) for l in incumbents.REG.read_text().splitlines()]
    old_rule = {}
    for r in reg:
        if r.get('experiment') == 'pso_same_budget' and r['method'] in ('반올림(원고 규칙)', 'greedy+repair'):
            old_rule[r['key']] = min(old_rule.get(r['key'], float('inf')), r['V'])
    summ = []
    for name, inp, _, _ in FIELDS:
        rs = [x for x in rows if x['field'] == name and x['iters'] == 500]
        lo = np.array([x['rel_low'] for x in rs])
        hi = [x['rel_high'] if x['rel_high'] is not None else float('inf') for x in rs]
        lg = [x for x in rows if x['field'] == name and x['iters'] == 5000]
        u_old = [old_rule[incumbents.key(inp, x['B'], 200)] for x in rs]
        r_s7 = float(arch_sum[name]['variance_over_iqp'])
        r_budget = float(np.mean([x['P'] / u for x, u in zip(rs, u_old)]))
        r_new = float(np.mean([x['P'] / x['U'] for x in rs]))
        r_new_bstar = float(np.mean([x['P'] for x in rs]) / incumbents.best_for(inp, B_TARGET[name], 200)['V'])
        summ.append(dict(field=name, runs=len(rs), budgets=len({x['B'] for x in rs}), at_target=sum((1 for x in rs if x['B'] == B_TARGET[name])), rel_low_med=float(np.median(lo)), rel_low_min=float(lo.min()), rel_low_max=float(lo.max()), rel_high_med=_json_num(float(np.median(hi))), rel_high_max=_json_num(float(max(hi))), n_upper_finite=sum((1 for v in hi if v != float('inf'))), s7_ratio=r_s7, ratio_budget_only=r_budget, same_budget_ratio_mean=r_new, budget_effect=r_budget - r_s7, incumbent_effect=r_new - r_budget, total_change=r_new - r_s7, ratio_new_at_Bstar=r_new_bstar, incumbent_effect_at_Bstar=r_new_bstar - r_s7, budget_effect_new_rule=r_new - r_new_bstar, cert_gap_med=float(np.median([x['cert_gap_pct'] for x in rs])), long_rel_low=lg[0]['rel_low'] if lg else None, long_rel_high=lg[0]['rel_high'] if lg else None, L_zero=sum((1 for x in rs if x['L'] <= 0)), flag='ok'))
    return summ

def _json_num(v):
    return 'inf' if v == float('inf') else v

def fmt(v, spec='.3f'):
    if v is None:
        return '—'
    if v == 'inf' or v == float('inf'):
        return '∞'
    return format(v, spec)

def summary_only():
    rows = load_json((OUT / 'pso_runs.json').read_text())['rows']
    brow = load_json((OUT / 'pso_budgets.json').read_text())
    summ = summarize(rows)
    (OUT / 'pso_summary.json').write_text(json.dumps(summ, indent=1, default=float, ensure_ascii=False))
    publish(rows, brow, summ)
    progress.task('5.3', 'done', ' · '.join((f"{s['field']}: P/V*−1 중앙값 구간 [{fmt(s['rel_low_med'])}, {fmt(s['rel_high_med'])}]; S7 비율 {s['s7_ratio']:.4f} → {s['same_budget_ratio_mean']:.4f} (전체 {s['total_change']:+.4f}; 예산 보정 {s['budget_effect']:+.1e}(A)/{s['budget_effect_new_rule']:+.1e}(B), 나머지는 incumbent 개선)" for s in summ)))
    nz = sum((s['L_zero'] for s in summ))
    progress.task('5.4', 'done', f'L=0인 실행 {nz}개. 상한이 없는 실행은 P/L−1=+∞로 전체 통계에 포함(중앙값·최댓값이 ∞가 될 수 있음), 유한 상한 실행 수를 n_upper_finite로 따로 보고, 출력은 ∞/— 문자열 처리. 합성 입력(1개·전부 L=0)으로 확인')
    return summ

def publish(rows, brow, summ):
    rc = [('field', '초기장'), ('seed', 'seed'), ('iters', '반복'), ('B', 'B'), ('P', 'P (PSO 분산)'), ('L', 'L(B)'), ('U', 'U(B)'), ('U_method', 'U 해'), ('P_minus_U', 'P−U'), ('P_minus_L', 'P−L'), ('rel_low', 'P/U−1 (하한)'), ('rel_high', 'P/L−1 (상한)'), ('cert_gap_pct', '인증폭 (U−L)/U %'), ('flag', '판정')]
    progress.result('s5_runs', 'S5 · PSO 실행별 같은 예산 비교', dict(columns=[dict(key=k, label=l) for k, l in rc], rows=rows, note='P는 PSO 최종 해의 분산(독립 재평가). 같은 예산 B에서 P/U−1 ≤ P/V*−1 ≤ P/L−1 (L>0). U는 best.json(원고 반올림·greedy+repair·반올림→repair·PSO 해를 포함한 최선). 상대 초과분의 분모는 V*이며 원고 인증폭 (U−L)/U와 다른 지표.'))
    bc = list(dict.fromkeys((k for r in brow for k in r)))
    progress.result('s5_budgets', 'S5 · 예산별 L·U', dict(columns=[dict(key=k, label=k) for k in bc], rows=brow, note='고유 (초기장, B)마다 L과 후보. U_without_pso는 PSO 해를 뺀 최선. 판정 warn = PSO 해가 이 예산의 최선.'))
    sc = [('field', '초기장'), ('runs', '실행'), ('budgets', '고유 예산'), ('at_target', 'B*와 같은 실행'), ('rel_low_med', 'P/U−1 중앙값'), ('rel_low_min', 'P/U−1 최소'), ('rel_low_max', 'P/U−1 최대'), ('rel_high_med', 'P/L−1 중앙값'), ('rel_high_max', 'P/L−1 최대'), ('s7_ratio', '원고 S7 PSO/V_UB (B* 기준)'), ('ratio_budget_only', '같은 예산, 원고 incumbent 규칙'), ('same_budget_ratio_mean', '같은 예산 P/U 평균 (최선 U)'), ('budget_effect', '예산 보정 효과 (A: 원고 규칙 고정)'), ('incumbent_effect', 'incumbent 개선 효과 (A)'), ('incumbent_effect_at_Bstar', 'incumbent 개선 효과 (B: B*에서 먼저)'), ('budget_effect_new_rule', '예산 보정 효과 (B: 새 규칙 고정)'), ('total_change', '전체 변화'), ('cert_gap_med', '인증폭 중앙값 %'), ('long_rel_low', '5000회 seed1 P/U−1'), ('long_rel_high', '5000회 seed1 P/L−1'), ('L_zero', 'L=0 실행'), ('n_upper_finite', '유한 상한 실행 수'), ('flag', '판정')]
    progress.result('s5_summary', 'S5 · 요약', dict(columns=[dict(key=k, label=l) for k, l in sc], rows=summ, note='500회 반복 25개 seed. 원고 S7은 PSO 평균 분산을 목표 예산 B*의 V_UB와 비교했으나 78%의 실행이 다른 예산을 썼다. 여기서는 각 실행의 실제 예산에서 L·U와 비교. P/V*−1은 PSO 해가 같은 예산의 정수 최적값보다 몇 배 나쁜지의 구간.'))
if __name__ == '__main__':
    summary_only() if '--summary-only' in sys.argv else main()
