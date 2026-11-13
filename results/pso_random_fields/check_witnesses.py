import os, sys
sys.dont_write_bytecode = True
import concurrent.futures as cf
import csv, json, hashlib, time
from fractions import Fraction as F
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def value_and_gradient(columns, s0, x, gradient=False):
    weights = [(j, int(i), F(float.fromhex(w))) for j, c in enumerate(columns) for i, w in c]
    ea = max((v.denominator.bit_length() - 1 for _, _, v in weights))
    es = max((v.denominator.bit_length() - 1 for v in s0))
    ex = max((v.denominator.bit_length() - 1 for v in x))
    e = max(ea + ex, es)
    X = [int(v * (1 << ex)) for v in x]
    cells = [int(v * (1 << e)) for v in s0]
    ints = [(j, i, int(v * (1 << ea))) for j, i, v in weights]
    shift = e - ea - ex
    for j, i, a in ints:
        cells[i] += a * X[j] << shift
    m = len(cells)
    total = sum(cells)
    V = F(m * sum((v * v for v in cells)) - total * total, m * m * (1 << 2 * e))
    if not gradient:
        return V
    g = [0] * len(x)
    for j, i, a in ints:
        g[j] += a * (m * cells[i] - total)
    factor = F(2, m * m * (1 << ea + e))
    return (V, [factor * v for v in g])

def read_allocation(path, B, C=200):
    rows = list((ROOT / path).open())
    d = list(csv.DictReader(rows))
    assert len(d) == 625
    assert sorted((int(r['waypoint']) for r in d)) == list(range(625))
    x = [None] * 625
    for r in d:
        x[int(r['waypoint'])] = F(r['shot_count'])
    assert all((v.denominator == 1 and 1 <= v <= C for v in x)) and sum(x) == B
    return x

def one(path):
    w = json.loads((ROOT / path).read_text())
    op = json.loads((ROOT / 'operator.json').read_text())
    raw = (ROOT / w['input']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == w['input_sha256']
    s0 = [F(float(t)) for t in raw.decode().replace('\n', ',').split(',') if t.strip()]
    xb = [F(float.fromhex(t)) for t in w['xbar_hex']]
    V, g = value_and_gradient(op['columns'], s0, xb, True)
    z = [1] * 625
    rest = w['B'] - 625
    for j in sorted(range(625), key=lambda j: g[j]):
        take = min(199, rest)
        z[j] += take
        rest -= take
        if not rest:
            break
    assert rest == 0
    L = max(F(0), V + sum((gi * (F(zi) - xi) for gi, zi, xi in zip(g, z, xb))))
    assert L == F(w['L_exact'])
    values = []
    three = []
    for c in w['candidates']:
        raw = (ROOT / c['allocation']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == c['sha256']
        x = read_allocation(c['allocation'], w['B'])
        v = value_and_gradient(op['columns'], s0, x)
        assert v == F(c['V_exact'])
        values.append(v)
        if c['method'] in ['rounding', 'greedy_exchange', 'rounded_exchange']:
            three.append(v)
    assert len(three) == 3 and min(three) == F(w['U_three_exact'])
    assert min(values) == F(w['U_exact']) and L <= min(values)
    xf = [F(v) for v in w['continuous_feasible']]
    assert sum(xf) == w['B'] and all((1 <= v <= 200 for v in xf))
    assert value_and_gradient(op['columns'], s0, xf) == F(w['continuous_value_exact'])
    return dict(field=w['field'], B=w['B'], passed=True, n_candidates=len(values))

def main():
    t0 = time.perf_counter()
    d = json.loads((ROOT / 'verified_instances.json').read_text())
    assert not d['failures'] and len(d['instances']) == d['planned_pairs']
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        checked = list(ex.map(one, [r['witness'] for r in d['instances']]))
    count = 0
    for i in d['instances']:
        L, U = (F(i['L_exact']), F(i['U_exact']))
        for r in i['runs']:
            P = F(r['P_exact'])
            assert F(r['excess_low_exact']) == P / U - 1
            assert r['excess_high_exact'] == '+Infinity' if L == 0 else F(r['excess_high_exact']) == P / L - 1
            assert F(r['excess_low_pct']) <= 100 * (P / U - 1)
            if L > 0:
                assert F(r['excess_high_pct']) >= 100 * (P / L - 1)
            count += 1
    out = dict(passed=all((r['passed'] for r in checked)), n_pairs=len(checked), n_runs=count, wall_s=time.perf_counter() - t0, instances=checked)
    (ROOT / 'independent_check.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'instances'}), flush=True)
if __name__ == '__main__':
    main()
