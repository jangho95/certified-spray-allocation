from pathlib import Path
import concurrent.futures as cf
import csv, hashlib, importlib.util, io, json, math, sys, tarfile
from fractions import Fraction as F
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/pso_random_fields'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def quantile(a, p):
    a = sorted(a)
    h = (len(a) - 1) * F(p)
    i = h.numerator // h.denominator
    return a[i] if h == i or i == len(a) - 1 else a[i] + (h - i) * (a[i + 1] - a[i])

def main():
    sys.path.insert(0, str(ROOT))
    import check_witnesses as module
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    assert sha(ROOT / 'protocol.json') == (ROOT / 'protocol.sha256').read_text().split()[0]
    for item in protocol['local_hashes']:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    data = json.loads((ROOT / 'verified_instances.json').read_text())
    instances = data['instances']
    assert len(instances) == 88 and (not data['failures'])
    with cf.ProcessPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(module.one, [i['witness'] for i in instances]))
    assert all((i['passed'] for i in checked))
    runs = [r for i in instances for r in i['runs']]
    assert len(runs) == 1475 and len({r['seed'] for r in runs}) == 1475
    summary = json.loads((ROOT / 'summary.json').read_text())
    lo = []
    hi = []
    for field in protocol['fields']:
        members = [r for r in runs if r['field'] == field['field']]
        assert sorted((r['seed'] for r in members)) == field['seeds']
        lower = []
        upper = []
        for r in members:
            P, U, L = map(F, [r['P_exact'], r['U_exact'], r['L_exact']])
            a, b = (P / U - 1, P / L - 1)
            assert F(r['excess_low_exact']) == a and F(r['excess_high_exact']) == b
            assert F(r['excess_low_pct']) <= 100 * a and F(r['excess_high_pct']) >= 100 * b
            lower.append(a)
            upper.append(b)
        a, b = (quantile(lower, F(1, 2)), quantile(upper, F(1, 2)))
        lo.append(a)
        hi.append(b)
        f = json.loads((ROOT / 'fields' / field['field'] / 'summary.json').read_text())['median_excess']
        assert F(f['low_exact']) == a and F(f['high_exact']) == b
    for name, p in [('minimum', 0), ('Q1', F(1, 4)), ('median', F(1, 2)), ('Q3', F(3, 4)), ('maximum', 1)]:
        item = summary['field_median_excess_summary'][name]
        a, b = (quantile(lo, p), quantile(hi, p))
        assert F(item['low_exact']) == a and F(item['high_exact']) == b
        assert F(item['low_pct']) <= 100 * a and F(item['high_pct']) >= 100 * b
    assert F(summary['population_5th_percentile_lower_bound']['lower_bound_exact']) == min(lo)
    assert F(summary['population_5th_percentile_lower_bound']['confidence_exact']) == 1 - F(19, 20) ** 59
    metadata = json.loads((ROOT / 'history_archive.json').read_text())
    assert sha(ROOT / metadata['path']) == metadata['sha256']
    less = equal = higher = 0
    members = {r['history']: r for r in runs}
    seen = set()
    with tarfile.open(ROOT / metadata['path'], 'r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            assert member.name in members and member.name not in seen
            r = members[member.name]
            seen.add(member.name)
            raw = archive.extractfile(member).read()
            assert hashlib.sha256(raw).hexdigest() == r['history_sha256']
            h = {int(t['iter']): float(t['gbest_variance']) for t in csv.DictReader(io.StringIO(raw.decode()))}
            less += h[5000] < h[4500]
            equal += h[5000] == h[4500]
            higher += h[5000] > h[4500]
    assert len(seen) == 1475
    assert (less, equal, higher) == (1445, 30, 0)
    assert less == summary['n_variance_lower_at_5000_than_4500']
    comparison = summary['benchmark_random_position']
    assert sum((v > F(268, 1000) for v in lo)) == comparison['n_fields_definitely_above'] == 43
    assert sum((v < F(266, 1000) for v in hi)) == comparison['n_fields_definitely_below'] == 13
    assert comparison['n_fields_overlapping'] == 3
    assert not any((r['P_better_than_three'] for r in runs))
    print(json.dumps(dict(passed=True, fields=59, runs=1475, verified_pairs=88, histories=1475, history_comparison=[less, equal, higher], planned_summaries_checked=True), indent=2))
if __name__ == '__main__':
    main()
