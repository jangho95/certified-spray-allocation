import determinism
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / 'work/stage1'))
import progress
import incumbents
import rigor_pilot as rp
from stage4_worker import geometry, load_field
from scip_compare import build_stamps, make_waypoints
from stage5_long_runs import OUT, FIELDS, allocation, atomic, digest

def quantile(values, probability):
    a = sorted(values)
    pos = (len(a) - 1) * Fraction(probability)
    i = pos.numerator // pos.denominator
    return a[i] if i == len(a) - 1 else a[i] + (pos - i) * (a[i + 1] - a[i])

def percent(x, upward=False, digits=1):
    scale = 10 ** digits
    y = Fraction(x) * 100 * scale
    n = math.ceil(y) if upward else math.floor(y)
    sign = '-' if n < 0 else ''
    n = abs(n)
    return f'{sign}{n // scale}.{n % scale:0{digits}d}'

def interval(lo, hi, digits=1):
    return f'[{percent(lo, False, digits)}, {percent(hi, True, digits)}]'

def write_operator(stamps):
    ptr, cells, weights = ([0], [], [])
    for st in stamps:
        cells.extend((int(c) for c in st.cells))
        weights.extend((float(v) for v in st.weights))
        ptr.append(len(cells))
    np.savez_compressed(OUT / 'operator.npz', ptr=np.array(ptr), cells=np.array(cells), weights=np.array(weights), M=np.array(2500), N=np.array(625))
    atomic(OUT / 'operator.json', dict(M=2500, N=625, columns=[[[int(c), float(w).hex()] for c, w in zip(st.cells, st.weights)] for st in stamps]))

def main():
    determinism.assert_single_thread()
    long = json.loads((OUT / 'runs.json').read_text())
    assert not long['failures'] and len(long['rows']) == 100
    short = json.loads((EXP / 'results/stage5/pso_runs.json').read_text())['rows']
    old = json.loads((EXP / 'results/stage5_rigor/rows.json').read_text())['instances']
    old_map = {(r['input'], r['B']): r for r in old}
    fmap = {f: inp for f, inp, _ in FIELDS}
    targets = dict(zero=28348, random_1_30_seed7=26120, center30=27912, center100=26896)
    runs = list(long['rows'])
    for r in short:
        if r['iters'] != 500:
            continue
        ap = OUT / 'alloc' / f"pso_{r['field']}_s{r['seed']}_it500.csv"
        hp = OUT / 'history' / f"pso_{r['field']}_s{r['seed']}_it500.csv"
        stem = f"pso_{r['field']}_s{r['seed']}_it500"
        shutil.copy2(EXP / 'work/stage5' / f'{stem}_allocation.csv', ap)
        shutil.copy2(EXP / 'work/stage5' / f'{stem}_history.csv', hp)
        runs.append(dict(field=r['field'], input=fmap[r['field']], seed=r['seed'], iters=500, B=r['B'], P=r['P'], alloc=str(ap.relative_to(OUT)), history=str(hp.relative_to(OUT)), alloc_sha256=digest(ap), history_sha256=digest(hp), source='preserved Stage 5 500-iteration run'))
    assert len(runs) == 200 and len({(r['field'], r['seed'], r['iters']) for r in runs}) == 200
    pairs = sorted({(r['input'], r['B']) for r in runs})
    assert set(pairs) <= set(old_map), 'New budget requires an additional rigorous lower-bound solve'
    records = [json.loads(line) for line in (EXP / 'results/stage5/records.jsonl').read_text().splitlines()]
    old_sha = {(r['input'], r['B']): r['extra']['U_alloc_sha256'] for r in records}
    best = json.loads(incumbents.BEST.read_text())
    atomic(OUT / 'registry_snapshot.json', {incumbents.key(inp, B, 200): best[incumbents.key(inp, B, 200)] for inp, B in pairs})
    atomic(OUT / 'analysis_protocol.json', dict(status='post hoc extension analysis; run list fixed before execution', lower_bounds='Re-evaluate the stored supporting-plane witness exactly for every reached budget. Preserve the original rigorous lower bound only after equality is checked.', upper_bounds='Use the saved incumbent registry snapshot. Check integer allocations, budget and box; evaluate exact variance. Include each PSO allocation when checking whether the incumbent is best.', uncertainty='Separate exact optimization brackets from seed-sampling uncertainty. For each fixed field and iteration count, a marginal distribution-free median confidence interval uses lower-end order statistic 8 and upper-end order statistic 18 of n=25. Coverage >=1-2*sum(comb(25,j),j=0..7)/2**25 under independent-run sampling; no simultaneous coverage claim.', quantiles='Descriptive sample median and IQR use exact linear interpolation, with interval endpoints rounded outward.', n_instances=len(pairs), n_runs=len(runs), code_sha256=digest(__file__)))
    for folder in ['instances', 'witness', 'inputs']:
        (OUT / folder).mkdir(exist_ok=True)
    for _, inp, _ in FIELDS:
        shutil.copy2(EXP / 'inputs' / inp, OUT / 'inputs' / inp)
    geo, _ = geometry(50)
    stamps = build_stamps(geo, make_waypoints(geo))
    write_operator(stamps)
    all_rows, instances = ([], [])
    t0 = time.perf_counter()
    for count, (inp, B) in enumerate(pairs, 1):
        prior = old_map[inp, B]
        ex = rp.Exact(stamps, load_field(inp), B, 200)
        src = EXP / 'results/stage5_rigor/witness' / f'{Path(inp).stem}_B{B}.npz'
        xb = np.load(src)['xbar']
        L = max(ex.bound(xb)[0], Fraction(0))
        assert L == Fraction(prior['L_exact'])
        incumbent = best[incumbents.key(inp, B, 200)]
        assert incumbent['alloc_sha256'] == old_sha[inp, B], 'Registry changed since the original evaluation'
        ua = OUT / 'alloc' / f'U_{Path(inp).stem}_B{B}.csv'
        shutil.copy2(EXP / incumbent['alloc_file'], ua)
        xu = allocation(ua)
        assert all(rp.check_int(xu, B, 200).values())
        U = ex.value(xu)
        assert U == Fraction(prior['U_exact'])
        members = [r for r in runs if r['input'] == inp and r['B'] == B]
        checked = []
        for r in members:
            xp = allocation(OUT / r['alloc'])
            assert all(rp.check_int(xp, B, 200).values())
            assert digest(OUT / r['alloc']) == r['alloc_sha256']
            P = ex.value(xp)
            checked.append((r, P))
        assert all((P > U for _, P in checked)), 'A PSO candidate improves the reference incumbent; update the comparison explicitly'
        item = dict(input=inp, B=B, Cmax=200, L_exact=rp.frs(L), U_exact=rp.frs(U), U_alloc=str(ua.relative_to(OUT)), U_sha256=digest(ua), W_pct_up=rp.up(100 * (U - L) / U), lower_source='reverified Stage 5 rigorous witness', L_float_historical=prior['L_float'], n_runs=len(members))
        witness = dict(item, s0_hex=[float(v).hex() for v in load_field(inp)], xbar_hex=[float(v).hex() for v in xb], integer_alloc=xu.tolist(), pso=[dict(alloc=r['alloc'], P_exact=rp.frs(P)) for r, P in checked])
        wp = OUT / 'witness' / f'{Path(inp).stem}_B{B}.json'
        atomic(wp, witness)
        item['witness'] = str(wp.relative_to(OUT))
        atomic(OUT / 'instances' / f'{Path(inp).stem}_B{B}.json', item)
        instances.append(item)
        for r, P in checked:
            lo, hi = (P / U - 1, P / L - 1)
            all_rows.append(dict(r, P_exact=rp.frs(P), L_exact=rp.frs(L), U_exact=rp.frs(U), excess_low_exact=rp.frs(lo), excess_high_exact=rp.frs(hi), excess_low_pct=rp.down(100 * lo), excess_high_pct=rp.up(100 * hi), excess_interval_pct=interval(lo, hi), W_pct_up=item['W_pct_up']))
        progress.detail('5.8', f'PSO 100/100 완료 · 엄밀 검증 {count}/{len(pairs)} 예산 · 배분 {len(all_rows)}/200개')
    summary = []
    for field, inp, _ in FIELDS:
        for iters in [500, 5000]:
            rs = [r for r in all_rows if r['field'] == field and r['iters'] == iters]
            assert sorted((r['seed'] for r in rs)) == list(range(1, 26))
            lo = [Fraction(r['excess_low_exact']) for r in rs]
            hi = [Fraction(r['excess_high_exact']) for r in rs]
            medlo, medhi = (quantile(lo, Fraction(1, 2)), quantile(hi, Fraction(1, 2)))
            q1 = (quantile(lo, Fraction(1, 4)), quantile(hi, Fraction(1, 4)))
            q3 = (quantile(lo, Fraction(3, 4)), quantile(hi, Fraction(3, 4)))
            cilo, cihi = (sorted(lo)[7], sorted(hi)[17])
            ps = np.array([float(Fraction(r['P_exact'])) for r in rs])
            summary.append(dict(field=field, iters=iters, runs=25, B_min=min((r['B'] for r in rs)), B_max=max((r['B'] for r in rs)), n_budgets=len({r['B'] for r in rs}), at_target=sum((r['B'] == targets[field] for r in rs)), P_mean=float(ps.mean()), P_sd=float(ps.std(ddof=1)), median=interval(medlo, medhi), q1=interval(*q1), q3=interval(*q3), range=interval(min(lo), max(hi)), median_ci=interval(cilo, cihi), median_low_pct=rp.down(100 * medlo), median_high_pct=rp.up(100 * medhi), median_ci_low_pct=rp.down(100 * cilo), median_ci_high_pct=rp.up(100 * cihi), min_gap_pct=rp.down(100 * min(lo)), max_gap_pct=rp.up(100 * max(hi)), min_W_pct=min((r['W_pct_up'] for r in rs)), max_W_pct=max((r['W_pct_up'] for r in rs))))
    paired = []
    for field, _, _ in FIELDS:
        changes = []
        for seed in range(1, 26):
            a = next((r for r in all_rows if (r['field'], r['seed'], r['iters']) == (field, seed, 500)))
            b = next((r for r in all_rows if (r['field'], r['seed'], r['iters']) == (field, seed, 5000)))
            changes.append((Fraction(a['excess_low_exact']) - Fraction(b['excess_high_exact']), Fraction(a['excess_high_exact']) - Fraction(b['excess_low_exact'])))
        paired.append(dict(field=field, n_certifiably_smaller=sum((lo > 0 for lo, _ in changes)), median_reduction_pp=interval(quantile([a for a, _ in changes], Fraction(1, 2)), quantile([b for _, b in changes], Fraction(1, 2)))))
    meta = dict(n_runs=200, n_long=100, n_new_long=96, reused_long=4, n_instances=len(pairs), failures=0, excluded=0, analysis_wall_s=time.perf_counter() - t0, max_float_P_error=max((abs(r['P'] - float(Fraction(r['P_exact']))) for r in all_rows)), median_ci_coverage=1 - 2 * sum((math.comb(25, j) for j in range(8))) / 2 ** 25, pso_best_budgets=0)
    atomic(OUT / 'analysis.json', dict(meta=meta, summary=summary, paired=paired, rows=all_rows, instances=instances))
    run_id = incumbents.new_run_id(__file__)
    if not (OUT / 'registry_added.json').exists():
        for r in all_rows:
            if r['iters'] == 5000 and r['seed'] != 1:
                incumbents.register(inp=r['input'], B=r['B'], cap=200, method=f"PSO seed {r['seed']} (5000 it)", V=float(Fraction(r['P_exact'])), alloc_path=OUT / r['alloc'], run_id=run_id, feasible=True, stage=5, experiment='pso_long25')
        incumbents.rebuild_best()
        atomic(OUT / 'registry_added.json', dict(run_id=run_id, new_candidates=96))
    columns = [('field', '초기장'), ('iters', 'iteration'), ('runs', 'seed 수'), ('median', '표본 중앙값의 엄밀 포함구간 (%)'), ('q1', 'Q1 구간 (%)'), ('q3', 'Q3 구간 (%)'), ('range', '실행 범위 (%)'), ('median_ci', '모집단 중앙값 신뢰구간 (%)'), ('at_target', '목표 예산 도달 수'), ('B_min', '최소 B'), ('B_max', '최대 B')]
    progress.result('s5_long_summary', 'S5 · 5,000회 25 seed · 검증 요약', dict(columns=[dict(key=k, label=v) for k, v in columns], rows=summary, note='500회와 5000회는 각각 초기장별 동일한 seed 1–25입니다. 모든 초과분은 각 실행의 최종 예산에서 정확 P와 검증된 L/U로 계산했습니다. 중앙값·Q1·Q3 구간은 유한 표본 통계량의 최적값 불확실성입니다. 별도 중앙값 신뢰구간은 독립 seed 가정 아래 순서통계량(8번째 하단, 18번째 상단)으로 계산한 모집단 중앙값의 주변적 95.67% 구간이며, 여러 환경에 대한 동시 보장이 아닙니다.'))
    progress.result('s5_long_runs', 'S5 · 5,000회 25 seed · 실행별 검증', dict(columns=[dict(key=k, label=v) for k, v in [('field', '초기장'), ('seed', 'seed'), ('iters', 'iteration'), ('B', '최종 B'), ('P', '분산 float'), ('excess_interval_pct', '엄밀 초과분 (%)'), ('W_pct_up', '검증 폭 (%)')]], rows=all_rows, note='200개 배분 모두 정수성·예산·box 검사 및 정확 목적값 평가 통과. 추가된 96개 배분은 레지스트리에 등록. 기존 Stage 5 결과는 보존.'))
    print(json.dumps(dict(meta=meta, summary=summary, paired=paired), ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
