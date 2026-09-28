import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from exact_oracle import Oracle, Q, checked_alloc, down, np, up

ROOT = Path(__file__).resolve().parents[1]


def ensure(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text())


@lru_cache(maxsize=8)
def operator(version, kind, F):
    name = 'sensitivity_extension' + ('_v2' if version == 2 else '')
    with np.load(ROOT / 'results' / name / f'operator_{kind}_{F}.npz', allow_pickle=False) as z:
        op = dict(z)
    ensure(int(op['M']) == F * F and int(op['N']) == 625, 'operator dimensions')
    ensure(len(op['col_ptr']) == 626 and op['col_ptr'][0] == 0, 'column pointers')
    ensure(op['col_ptr'][-1] == len(op['cells']) == len(op['weights']), 'operator entries')
    ensure(np.all(np.diff(op['col_ptr']) >= 0), 'column order')
    ensure(np.all((op['cells'] >= 0) & (op['cells'] < F * F)), 'cell indices')
    ensure(np.all(np.isfinite(op['weights'])), 'finite weights')
    return op


def check_row(row, version):
    cid = row['id']
    ensure(not row['failed'], cid + ' failed record')
    directory = ROOT / 'results' / ('sensitivity_extension' + ('_v2' if version == 2 else ''))
    path = ROOT / row['input']
    ensure(hashlib.sha256(path.read_bytes()).hexdigest() == row['input_sha256'], cid + ' input hash')
    s0 = np.loadtxt(path, delimiter=',').reshape(50, 50)
    factor = row['F'] // 50
    s0 = np.repeat(np.repeat(s0, factor, axis=0), factor, axis=1).ravel()
    with np.load(directory / row['witness'], allow_pickle=False) as z:
        w = dict(z)
    B, C = int(w['B']), int(w['Cmax'])
    op = operator(version, row['kind'], row['F'])
    ensure(B == row['B'] and C == 200 and int(w['M']) == row['M'] == len(s0), cid + ' problem')
    ensure(np.array_equal(s0, w['s0']), cid + ' stored field')
    ensure(len(w['allocation']) == row['N'] == int(op['N']), cid + ' allocation dimension')
    checked_alloc(w['allocation'], B, C)
    ora = Oracle(op, s0)
    L, _ = ora.evaluate(w['xbar'], B, C)
    U = ora.evaluate(w['allocation'])
    xf = [Q(str(v)) for v in w['feasible_relaxation']]
    ensure(len(xf) == row['N'] and sum(xf) == B and all(1 <= v <= C for v in xf), cid + ' continuous feasibility')
    Vf = ora.evaluate(xf)
    ensure([L, U, Vf] == [Q(row[k]) for k in ['L_exact', 'U_exact', 'Vfeas_exact']], cid + ' exact values')
    ensure(L <= Vf and L <= U, cid + ' enclosure')
    if version == 2:
        ensure(Vf <= U, cid + ' v2 continuous upper')
    ensure(down(L) == row['L'] and up(U) == row['U'], cid + ' directed endpoints')
    ensure(up(U - L) == row['abs_width'], cid + ' absolute width')
    ensure(up(100 * (U - L) / U) == row['width_pct'], cid + ' relative width')
    ensure(up(Vf - L) == row['continuous_interval_width'], cid + ' continuous width')
    values = {}
    ensure(len(w['candidate_names']) == len(w['candidates']) == 3, cid + ' candidate count')
    for name, x in zip(w['candidate_names'], w['candidates']):
        name = str(name)
        checked_alloc(x, B, C)
        values[name] = ora.evaluate(x)
        ensure(up(values[name]) == row['candidate_U'][name], cid + ' candidate objective')
    ensure(set(values) == set(row['candidate_U']), cid + ' candidate names')
    ensure(values[row['selected']] == U == min(values.values()), cid + ' best candidate')
    violation = Q(row['L_float']) > Vf
    ensure(violation == row['L_float_above_feasible_exact'], cid + ' float-bound comparison')
    if version == 2:
        ensure(np.array_equal(w['lower_reference'], w['xbar']), cid + ' lower reference')
    return L, U, Vf, w, violation


def check_pair(pair):
    old, new = pair
    L1, U1, Vf1, w1, _ = check_row(old, 1)
    L2, U2, Vf2, w2, violation = check_row(new, 2)
    ensure(old['id'] == new['id'], 'case identity')
    ensure(L2 >= L1 and U2 == U1 and Vf2 <= Vf1, new['id'] + ' v2 preservation')
    ensure(Q(new['v1_L_exact']) == L1 and Q(new['v1_Vfeas_exact']) == Vf1, 'archived bounds')
    ensure(np.array_equal(w1['allocation'], w2['allocation']), 'preserved integer allocation')
    ensure(np.array_equal(w1['candidates'], w2['candidates']), 'preserved candidates')
    ensure(np.array_equal(w1['candidate_names'], w2['candidate_names']), 'preserved candidate names')
    return dict(id=new['id'], group=new['group'], width_pct=up(100 * (U2 - L2) / U2),
                continuous_width=up(Vf2 - L2), float_exceedance=violation)


def verify(workers=4):
    old = read(ROOT / 'results/sensitivity_extension/rows.json')
    new = read(ROOT / 'results/sensitivity_extension_v2/rows.json')
    before = {r['id']: r for r in old}
    ensure(len(before) == len(old) == len(new) == 156, 'sensitivity count')
    ensure(set(before) == {r['id'] for r in new}, 'sensitivity cases')
    for kind, F in [('benchmark', 50), ('cell_average', 50), ('cell_average', 100), ('cell_average', 200)]:
        a, b = operator(1, kind, F), operator(2, kind, F)
        ensure(set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a), 'preserved operator')
    with ProcessPoolExecutor(max_workers=workers) as executor:
        checked = list(executor.map(check_pair, [(before[r['id']], r) for r in new]))
    groups = {g: sum(r['group'] == g for r in checked) for g in ['target', 'distribution', 'mesh']}
    ensure(groups == dict(target=24, distribution=120, mesh=12), 'sensitivity groups')
    summary = read(ROOT / 'results/sensitivity_extension_v2/summary.json')
    ensure(max(r['continuous_width'] for r in checked) == summary['max_continuous_interval_width'], 'enclosure summary')
    for g, count in groups.items():
        widths = [r['width_pct'] for r in checked if r['group'] == g]
        ensure(summary['groups'][g] == dict(n=count, width_min=min(widths), width_max=max(widths)), 'width summary')
    violations = sorted(r['id'] for r in checked if r['float_exceedance'])
    ensure(violations == ['mesh_zero_F200', 'mesh_zero_F50', 'target_zero_T150', 'target_zero_T300'], 'float violations')
    return dict(instances=156, archived_v1_instances=156, candidate_evaluations=936,
                groups=groups, exact_bounds=True, feasible_continuous_upper_bounds=True,
                preserved_integer_allocations=True, input_hashes=True,
                float_bound_violation_cases=violations,
                max_continuous_interval_width=max(r['continuous_width'] for r in checked))
