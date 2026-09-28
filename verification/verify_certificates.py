import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import subprocess
import time

from exact_oracle import Oracle, Q, checked_alloc, down, np, up

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ensure(value, message):
    if not value:
        raise AssertionError(message)

def field(inp):
    return np.array([float(v) for v in (ROOT / 'inputs' / inp).read_text().replace('\n', ',').split(',') if v.strip()])

def allocation(path, n):
    values = {}
    with path.open(newline='') as f:
        rows = csv.reader(f)
        next(rows)
        for row in rows:
            j, v = int(row[0]), Q(row[-1])
            ensure(j not in values and v.denominator == 1, str(path))
            values[j] = int(v)
    ensure(set(values) == set(range(n)), str(path))
    return np.array([values[j] for j in range(n)], dtype=np.int64)

def stored_path(name):
    if '/experiments/' in name:
        return ROOT / name.split('/experiments/', 1)[1]
    p = Path(name)
    return p if p.is_absolute() else ROOT / p

def stage9_field(fr):
    out = ROOT / 'results/stage9_rigor'
    eps = Q(read(out / 'rows.json')['meta']['eps_ideal'])
    with np.load(out / 'operator.npz') as z:
        op = dict(z)
    with np.load(out / 'witness' / (fr['name'] + '.npz')) as z:
        w = dict(z)
    Bvals, C = w['B'], int(w['Cmax'])
    ensure(np.array_equal(w['s0'], field(fr['input'])), fr['name'] + ' input')
    ensure(sha(ROOT / 'inputs' / fr['input']) == str(w['input_sha256']), fr['name'] + ' hash')
    ensure(len(fr['rows']) == len(Bvals) == 21, fr['name'] + ' budgets')
    ora = Oracle(op, w['s0'])
    widths, target = [], None
    for k, r in enumerate(fr['rows']):
        B = int(Bvals[k])
        ensure(B == r['B'], 'budget')
        checked_alloc(w['alloc'][k], B, C)
        L, _ = ora.evaluate(w['xbar'][k], B, C)
        U = ora.evaluate(w['alloc'][k])
        ensure(L == Q(r['L_exact']) and U == Q(r['U_exact']), fr['name'] + ' exact bounds')
        ensure(Q(r['L_osqp']) <= L <= U, fr['name'] + ' lower bound')
        ensure(r['L_rig'] == down(L) and r['U_rig'] == up(U), 'directed bounds')
        width = up(100 * (U-L)/U)
        ensure(width == r['W_rig_pct_up'], 'directed width')
        slo, shi = math.sqrt(down(L)), math.sqrt(up(U))
        while Q(slo)**2 > L:
            slo = math.nextafter(slo, -math.inf)
        while Q(shi)**2 < U:
            shi = math.nextafter(shi, math.inf)
        lower = Q(down(max(Q(0), Q(slo)-eps)**2))
        upper = Q(up((Q(shi)+eps)**2))
        ensure(up(100*(upper-lower)/upper) == r['W_ideal_pct_up'], 'Gaussian directed width')
        widths.append(width)
        if r['is_target']:
            target = width
    return dict(name=fr['name'], count=len(widths), T1=target, T2=max(widths))

def gaussian_error_squared(op, spacing, cap):
    intervals = {}
    for exponent in (0, 1, 2, 4, 5, 8, 9, 10, 13, 18):
        a = Q(exponent) / (2 * Q(7, 4)**2)
        term, total = Q(1), Q(1)
        for k in range(1, 151):
            term *= -a/k
            total += term
        intervals[exponent] = (total + term*(-a)/151, total)
    pts = [(r, c) for r in range(0, 50, spacing) for c in range(0, 50, spacing)]
    ensure(len(pts)+1 == len(op['col_ptr']), 'Gaussian geometry')
    delta = [Q(0)] * 2500
    for j, (r, c) in enumerate(pts):
        expected = {}
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                rr, cc = r+dr, c+dc
                rr = -rr-1 if rr < 0 else 99-rr if rr >= 50 else rr
                cc = -cc-1 if cc < 0 else 99-cc if cc >= 50 else cc
                i = 50*rr+cc
                lo, hi = intervals[dr*dr+dc*dc]
                a, b = expected.get(i, (Q(0), Q(0)))
                expected[i] = a+lo, b+hi
        start, stop = op['col_ptr'][j:j+2]
        ensure(set(map(int, op['cells'][start:stop])) == set(expected), 'Gaussian support')
        for i, v in zip(op['cells'][start:stop], op['weights'][start:stop]):
            lo, hi = expected[int(i)]
            delta[int(i)] += max(abs(Q(float(v))-lo), abs(Q(float(v))-hi))
    return sum((cap*d)**2 for d in delta) / 2500

def stage9(workers):
    out = ROOT / 'results/stage9_rigor'
    obj = read(out / 'rows.json')
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(stage9_field, obj['fields']))
    ensure(len(rows) == 59 and sum(r['count'] for r in rows) == 1239, 'sample size')
    summary = read(out / 'summary.json')['summary']
    for key in ('T1', 'T2'):
        ensure(max(r[key] for r in rows) == summary[key]['max'], key)
    with np.load(out / 'operator.npz') as z:
        op = dict(z)
    ensure(Q(obj['meta']['eps_ideal'])**2 >= gaussian_error_squared(op, 2, 200), 'Stage 9 Gaussian correction')
    old = read(ROOT / 'results/stage9/e4_runs.json')['results']
    indexed = {f['name']: f for f in old}
    two_candidate = []
    for fr in obj['fields']:
        with np.load(out / 'witness' / (fr['name'] + '.npz')) as w:
            ora = Oracle(op, w['s0'])
            widths = []
            for k, r in enumerate(indexed[fr['name']]['rows']):
                path = stored_path(r['cands'][r['U_method']]['path'])
                ensure(np.array_equal(allocation(path, len(w['alloc'][k])), w['alloc'][k]), 'Stage 9 original allocation')
                candidates = [r['cands'][key] for key in ('반올림(원고 규칙)', 'greedy+repair')]
                selected = min(candidates, key=lambda c: c['V'])
                ensure(selected['V'] == r['U_paper'], 'Stage 9 submitted-rule selection')
                x = allocation(stored_path(selected['path']), len(w['alloc'][k]))
                checked_alloc(x, r['B'], int(w['Cmax']))
                U = ora.evaluate(x)
                L = Q(fr['rows'][k]['L_exact'])
                widths.append(up(100*(U-L)/U))
            two_candidate.append(dict(field=fr['name'], T2=max(widths)))
    expected = read(out / 'two_candidate_check.json')
    ensure(two_candidate == expected['fields'], 'Stage 9 submitted-rule verified widths')
    ensure(max(r['T2'] for r in two_candidate) == expected['max'], 'Stage 9 submitted-rule maximum')
    return dict(fields=59, instances=1239, exact_bounds=True, feasible_integer_allocations=True,
                gaussian_error_bound=True, T2_max_pct=summary['T2']['max'],
                two_candidate_T2_max_pct=expected['max'])

def feasible_point(xbar, B, cap):
    xf = [Q(float(min(max(v, 1), cap))) for v in xbar]
    residual = Q(B)-sum(xf)
    for j in sorted(range(len(xf)), key=lambda j: -min(xf[j]-1, cap-xf[j])):
        if 1 <= xf[j]+residual <= cap:
            xf[j] += residual
            ensure(sum(xf) == B and all(1 <= v <= cap for v in xf), 'continuous feasibility')
            return xf
    raise AssertionError('Unable to reconstruct feasible point')

def benchmark():
    out = ROOT / 'results/bench_sweep'
    runs = {r['name']: r for r in read(out / 'runs.json')['results']}
    summaries = {r['field']: r for r in read(out / 'summary.json')}
    with np.load(out / 'operator.npz') as z:
        op = dict(z)
    done = []
    for fr in read(out / 'rigor.json')['fields']:
        run = runs[fr['name']]
        with np.load(out / 'witness' / (fr['name'] + '.npz')) as z:
            w = dict(z)
        cap = int(w['Cmax'])
        ensure(np.array_equal(w['s0'], field(fr['input'])), 'benchmark input')
        ensure(sha(ROOT / 'inputs' / fr['input']) == run['input_sha256'], 'benchmark input hash')
        ensure(len(fr['rows']) == len(run['rows']) == len(w['B']) == 21, 'benchmark budgets')
        ora = Oracle(op, w['s0'])
        widths, violations, ncands, target = [], [], 0, None
        for k, (r, original) in enumerate(zip(fr['rows'], run['rows'])):
            B = int(w['B'][k])
            ensure(B == r['B'] == original['B'], 'benchmark budget')
            checked_alloc(w['alloc'][k], B, cap)
            L, _ = ora.evaluate(w['xbar'][k], B, cap)
            U = ora.evaluate(w['alloc'][k])
            Vf = ora.evaluate(feasible_point(w['xbar'][k], B, cap))
            ensure(L == Q(r['L_exact']) and U == Q(r['U_exact']), 'benchmark exact bounds')
            ensure(L <= Vf <= U, 'benchmark continuous enclosure')
            ensure(down(L) == r['L_rig'] and up(U) == r['U_rig'], 'benchmark directed bounds')
            width = up(100*(U-L)/U)
            ensure(width == r['W_rig_pct_up'], 'benchmark directed width')
            selected = stored_path(original['cands'][original['U_method']]['path'])
            ensure(np.array_equal(allocation(selected, len(w['alloc'][k])), w['alloc'][k]), 'benchmark original allocation')
            values = []
            for candidate in original['cands'].values():
                x = allocation(stored_path(candidate['path']), len(w['alloc'][k]))
                checked_alloc(x, B, cap)
                values.append(ora.evaluate(x))
                ncands += 1
            ensure(U == min(values), 'benchmark best of three candidates')
            if Q(r['L_osqp']) > Vf:
                violations.append(B)
            widths.append(width)
            if r['is_target']:
                target = width
        summary = summaries[fr['name']]
        ensure(target == summary['T1_rig'] and max(widths) == summary['T2_rig'], 'benchmark summary')
        ensure(len(violations) == (10 if fr['name'] == 'zero' else 0), 'benchmark float violations')
        done.append(dict(field=fr['name'], instances=len(widths), candidates=ncands,
                         T1_pct=target, T2_pct=max(widths), float_bound_violation_budgets=violations))
    ensure(len(done) == 4 and sum(r['instances'] for r in done) == 84, 'benchmark count')
    ensure(sum(r['candidates'] for r in done) == 252, 'benchmark candidates')
    return dict(fields=done, instances=84, candidates=252, exact_bounds=True)

def stage5():
    out = ROOT / 'results/stage5_rigor'
    obj = read(out / 'rows.json')
    refs = {}
    for line in (ROOT / 'results/stage5/records.jsonl').read_text().splitlines():
        r = json.loads(line)
        refs[r['input'], r['B']] = r
    with np.load(ROOT / 'results/stage9_rigor/operator.npz') as z:
        op = dict(z)
    n = len(op['col_ptr'])-1
    violations, pso_rows = [], []
    for r in obj['instances']:
        B, inp = r['B'], r['input']
        with np.load(out / 'witness' / f'{Path(inp).stem}_B{B}.npz') as z:
            w = dict(z)
        cap = int(w['Cmax'])
        ensure(int(w['B']) == B and np.array_equal(w['s0'], field(inp)), 'Stage 5 input')
        ora = Oracle(op, w['s0'])
        L, _ = ora.evaluate(w['xbar'], B, cap)
        digest = refs[inp, B]['extra']['U_alloc_sha256']
        path = ROOT / 'results/incumbents/alloc' / (digest+'.csv')
        ensure(sha(path) == digest, 'incumbent hash')
        x = allocation(path, n)
        checked_alloc(x, B, cap)
        U = ora.evaluate(x)
        Vf = ora.evaluate(feasible_point(w['xbar'], B, cap))
        ensure(L == Q(r['L_exact']) and U == Q(r['U_exact']), 'Stage 5 exact bounds')
        ensure(L <= Vf <= U and up(Vf) == r['V_feas_up'], 'Stage 5 continuous upper bound')
        ensure(down(L) == r['L_rig'] and up(U) == r['U_rig'], 'Stage 5 rounding')
        ensure(up(100*(U-L)/U) == r['W_rig_pct_up'], 'Stage 5 width')
        if Q(r['L_float']) > Vf:
            violations.append(B)
        for p in r['pso']:
            path = ROOT / 'work/stage5' / f"pso_{p['field']}_s{p['seed']}_it{p['iters']}_allocation.csv"
            x = allocation(path, n)
            checked_alloc(x, B, cap)
            P = ora.evaluate(x)
            ensure(P == Q(p['P_exact']), 'PSO exact objective')
            lo, hi = P/U-1, P/L-1
            ensure(down(lo) == p['excess_low_down'] and up(hi) == p['excess_high_up'], 'PSO bounds')
            pso_rows.append(dict(field=p['field'], iters=p['iters'], lo=lo, hi=hi))
    ensure(len(obj['instances']) == 40 and len(pso_rows) == 104, 'Stage 5 counts')
    ensure(sorted(violations) == [28338, 28341, 28344, 28347, 28349], 'Stage 5 floating-point violations')
    for s in read(out / 'summary.json')['summary']:
        selected = [p for p in pso_rows if p['field'] == s['field'] and p['iters'] == 500]
        ensure(len(selected) == 25, 'PSO count')
        lo = sorted(p['lo'] for p in selected)[12]
        hi = sorted(p['hi'] for p in selected)[12]
        ensure(down(lo) == s['excess_low_med_rig'] and up(hi) == s['excess_high_med_rig'], 'PSO median')
    return dict(instances=40, pso_allocations=104, exact_bounds=True, float_bound_violation_budgets=sorted(violations))

def pilot():
    out = ROOT / 'results/rigor_pilot'
    done = []
    cached = {}
    for row in read(out / 'pilot_rows.json')['rows']:
        folder = ROOT / row['witness']
        w = read(folder / 'witness.json')
        with np.load(folder / 'operator.npz') as z:
            op = dict(z)
        with np.load(folder / 'points.npz') as z:
            pt = dict(z)
        ora = Oracle(op, op['s0'])
        L, _ = ora.evaluate(pt['xbar'], w['B'], w['Cmax'])
        checked_alloc(pt['alloc'], w['B'], w['Cmax'])
        U = ora.evaluate(pt['alloc'])
        xf = [Q(float(v)) for v in pt['x_feas_base']]
        for j, v in w['x_feas']['adjusted'].items():
            xf[int(j)] = Q(v)
        ensure(sum(xf) == w['B'] and all(1 <= v <= w['Cmax'] for v in xf), 'pilot feasibility')
        Vf = ora.evaluate(xf)
        ensure([L, U, Vf] == [Q(w[k]) for k in ('L_exact', 'U_exact', 'V_feas_exact')], 'pilot exact values')
        ensure(up(100*(U-L)/U) == row['W_rig_pct_up'], 'pilot width')
        key = row['spacing'], w['Cmax']
        if key not in cached:
            cached[key] = gaussian_error_squared(op, *key)
        eps = Q(row['eps_ideal'])
        ensure(eps**2 >= cached[key], 'pilot Gaussian correction')
        lower, upper = Q(row['L_ideal']), Q(row['U_ideal'])
        margin = L-lower-eps**2
        ensure(lower == 0 or (margin >= 0 and margin**2 >= 4*eps**2*lower), 'ideal lower')
        margin = upper-U-eps**2
        ensure(margin >= 0 and margin**2 >= 4*eps**2*U, 'ideal upper')
        ensure(row['W_ideal_pct_up'] == up(100*(upper-lower)/upper), 'ideal width')
        done.append(row['id'])
    ensure(len(done) == 6, 'pilot count')
    return dict(instances=6, exact_bounds=True, feasible_continuous_upper_bounds=True, gaussian_error_bounds=True)

def stage5_long25():
    run = subprocess.run([sys.executable, str(ROOT / 'pipeline/verify_stage5_long25.py')], check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    ensure(result['status'] == 'passed', 'long PSO verification')
    return result

def sensitivity(workers):
    from verify_sensitivity import verify
    return verify(workers)

def main():
    if not __debug__:
        raise SystemExit('Run without Python -O: assertion checks are required.')
    parser = argparse.ArgumentParser(description='Independent exact verification without optimization solvers.')
    parser.add_argument('--suite', choices=('all', 'pilot', 'stage5', 'stage9', 'benchmark', 'stage5_long25', 'sensitivity'), default='all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ensure(args.workers >= 1, 'workers must be positive')
    start = time.perf_counter()
    result = {}
    for name, fn in [('pilot', pilot), ('stage5', stage5), ('stage9', lambda: stage9(args.workers)), ('benchmark', benchmark), ('stage5_long25', stage5_long25), ('sensitivity', lambda: sensitivity(args.workers))]:
        if args.suite in ('all', name):
            print('Verifying ' + name, flush=True)
            result[name] = fn()
    result.update(passed=True, elapsed_s=time.perf_counter()-start)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n')
    print(text)

if __name__ == '__main__':
    main()
