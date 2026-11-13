import determinism
import concurrent.futures as cf
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / 'work/stage1'))
import pso_repeated_runs as repeated
from paper_pso import run_pso
import progress
OUT = EXP / 'results/stage5_long25'
FIELDS = [('zero', 'zero_F50.csv', {'field': 'zero'}), ('random_1_30_seed7', 'random_u1_30_seed7_F50.csv', {'field': 'random'}), ('center30', 'center30_F50.csv', {'field': 'center', 'center_height': 30.0}), ('center100', 'center100_F50.csv', {'field': 'center', 'center_height': 100.0})]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)

def allocation(path):
    rows = list(csv.DictReader(Path(path).open()))
    ids = [int(r['waypoint']) for r in rows]
    assert sorted(ids) == list(range(625))
    x = np.zeros(625, dtype=np.int64)
    for r in rows:
        value = float(r['shot_count'])
        assert value.is_integer()
        x[int(r['waypoint'])] = int(value)
    assert np.all((x >= 1) & (x <= 200))
    return x

def job(args):
    field, inp, over, seed = args
    threads = determinism.assert_single_thread()
    a = repeated.make_args(dict(over, initial_field=EXP / 'inputs' / inp, iters=5000, log_every=10), seed)
    start = time.perf_counter()
    sim, x, met, hist, _ = run_pso(a)
    tag = f'pso_{field}_s{seed}_it5000'
    ap = OUT / 'alloc' / f'{tag}.csv'
    hp = OUT / 'history' / f'{tag}.csv'
    assert np.all(x == np.rint(x)) and np.all((x >= 1) & (x <= 200))
    with ap.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['waypoint', 'shot_count'])
        writer.writerows(enumerate(x.astype(int)))
    s0sum = float(sim.initial.sum())
    with hp.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['iter', 'gbest_variance', 'gbest_mean', 'gbest_budget', 'gbest_fit'])
        for h in hist:
            writer.writerow([h['iter'], h['gbest_variance'], h['gbest_mean'], int(round((h['gbest_mean'] * sim.m - s0sum) / sim.mass)), h['gbest_fit']])
    field_value = sim.A @ x + sim.initial
    p = float(np.mean((field_value - field_value.mean()) ** 2))
    assert abs(p - met['variance']) <= 1e-08 * max(1, abs(p))
    result = dict(field=field, input=inp, seed=seed, iters=5000, B=int(x.sum()), P=p, P_reported=met['variance'], mean=float(field_value.mean()), alloc=str(ap.relative_to(OUT)), history=str(hp.relative_to(OUT)), alloc_sha256=digest(ap), history_sha256=digest(hp), elapsed_s=time.perf_counter() - start, optimizer_s=met['elapsed_s'], source='new execution', threads=threads)
    atomic(OUT / 'runs' / f'{field}_s{seed}.json', result)
    return result

def initialize():
    for name in ['alloc', 'history', 'runs']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    code_names = ['pipeline/stage5_long_runs.py', 'pipeline/determinism.py', 'work/stage1/paper_pso.py', 'work/stage1/pso_repeated_runs.py', 'work/stage1/scip_compare.py']
    hashes = {n: digest(EXP / n) for n in code_names}
    for _, inp, _ in FIELDS:
        hashes[f'inputs/{inp}'] = digest(EXP / 'inputs' / inp)
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['sha256'] == hashes, 'Frozen code/input changed'
    else:
        assert not list((OUT / 'runs').glob('*.json'))
        atomic(path, dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), status='follow-up extension specified after the original Stage 5 results', design='Four fixed canonical fields; PSO seeds 1..25; 100 particles; 5000 iterations; all other settings unchanged', reuse='Four previously verified seed-1 runs; 96 new runs for seeds 2..25', reporting='All 25 seeds per field. Descriptive median/IQR/range and exact same-budget excess bounds. Optional median confidence intervals use exact binomial order-statistic coverage >=95%, marginal per field.', failures='Keep every planned seed. Save exceptions. Do not drop failures; require completion or explicitly report unresolved runs. No outcome-dependent stopping or extra seeds.', histories='Iteration 1 and every 10 iterations; compare iteration-500 P, mean and budget with the existing 500-iteration run of the same seed', sha256=hashes, workers=8, blas_threads=1, defaults={k: str(v) if isinstance(v, Path) else v for k, v in repeated.DEFAULTS.items()}))

    def add_task(state):
        stage = next((s for s in state['stages'] if s['id'] == '5'))
        if not any((t['id'] == '5.8' for t in stage['tasks'])):
            stage['tasks'].append(dict(id='5.8', title='PSO 5000회 × 초기장별 25 seed 추가 검증', status='pending'))
    progress._update(add_task)
    old = json.loads((EXP / 'results/stage5/pso_runs.json').read_text())['rows']
    for field, inp, _ in FIELDS:
        path = OUT / 'runs' / f'{field}_s1.json'
        if path.exists():
            continue
        r = next((r for r in old if r['field'] == field and r['iters'] == 5000 and (r['seed'] == 1)))
        stem = f'pso_{field}_s1_it5000'
        ap, hp = (OUT / 'alloc' / f'{stem}.csv', OUT / 'history' / f'{stem}.csv')
        shutil.copy2(EXP / 'work/stage5' / f'{stem}_allocation.csv', ap)
        shutil.copy2(EXP / 'work/stage5' / f'{stem}_history.csv', hp)
        x = allocation(ap)
        assert int(x.sum()) == r['B']
        atomic(path, dict(field=field, input=inp, seed=1, iters=5000, B=r['B'], P=r['P'], alloc=str(ap.relative_to(OUT)), history=str(hp.relative_to(OUT)), alloc_sha256=digest(ap), history_sha256=digest(hp), source='preserved Stage 5 seed-1 run', elapsed_s=None))

def main():
    initialize()
    progress.task('5.8', 'running', '기존 seed 1 네 건 보존, seed 2–25의 96건 실행')
    planned = [(f, inp, over, seed) for seed in range(1, 26) for f, inp, over in FIELDS]
    jobs = [j for j in planned if not (OUT / 'runs' / f'{j[0]}_s{j[3]}.json').exists()]
    for p in (OUT / 'runs').glob('*.json'):
        r = json.loads(p.read_text())
        assert digest(OUT / r['alloc']) == r['alloc_sha256']
        assert digest(OUT / r['history']) == r['history_sha256']
    errors = []
    start = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        pending = {executor.submit(job, j): j for j in jobs}
        while pending:
            done, _ = cf.wait(pending, timeout=20, return_when=cf.FIRST_COMPLETED)
            for future in done:
                j = pending.pop(future)
                try:
                    result = future.result()
                    print(f"Done {result['field']} seed {result['seed']}: B={result['B']}, P={result['P']:.8f}", flush=True)
                except Exception as exc:
                    errors.append(dict(field=j[0], seed=j[3], error=repr(exc)))
                    atomic(OUT / 'failures.json', errors)
            count = len(list((OUT / 'runs').glob('*.json')))
            note = f'PSO 5000회 {count}/100 완료 (seed 1 보존 4건 포함), 실패 {len(errors)}건, 경과 {time.perf_counter() - start:.0f}s'
            progress.detail('5.8', note)
            print(note, flush=True)
    rows = [json.loads(p.read_text()) for p in sorted((OUT / 'runs').glob('*.json'))]
    atomic(OUT / 'runs.json', dict(rows=rows, failures=errors, session_wall_s=time.perf_counter() - start))
    if errors or len(rows) != 100:
        progress.task('5.8', 'warn', f'완료 {len(rows)}/100, 실패 {len(errors)} — 실패를 제외한 집계는 하지 않음')
        raise RuntimeError('Incomplete planned sample')
    old = json.loads((EXP / 'results/stage5/pso_runs.json').read_text())['rows']
    checks = []
    for r in rows:
        h = next((h for h in csv.DictReader((OUT / r['history']).open()) if int(h['iter']) == 500))
        ref = next((x for x in old if x['field'] == r['field'] and x['seed'] == r['seed'] and (x['iters'] == 500)))
        rel = abs(float(h['gbest_variance']) - ref['P']) / max(1, abs(ref['P']))
        same_budget = int(h['gbest_budget']) == ref['B']
        checks.append(dict(field=r['field'], seed=r['seed'], relative_P_difference=rel, same_budget=same_budget))
    atomic(OUT / 'prefix500_checks.json', checks)
    assert all((c['same_budget'] and c['relative_P_difference'] < 1e-09 for c in checks))
    progress.detail('5.8', '100/100 완료, 500회 시점의 예산·분산 대조 통과. 예산별 엄밀 검증 대기')
    print('Completed all 100 long runs; all iteration-500 prefix checks passed.', flush=True)
if __name__ == '__main__':
    main()
