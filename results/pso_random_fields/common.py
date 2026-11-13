import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import ast
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))

def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=True, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def error_text(exc):
    return str(exc).replace(str(ROOT), '[package]').replace(str(ROOT.parents[2]), '[workspace]')

def threads():
    from threadpoolctl import threadpool_info
    info = [{k: p.get(k) for k in ['user_api', 'internal_api', 'num_threads', 'version']} for p in threadpool_info()]
    assert all((p['num_threads'] == 1 for p in info)), info
    return info

def read_alloc(path, n=625):
    import numpy as np
    rows = list(csv.DictReader(Path(path).open()))
    assert len(rows) == n
    ids = [int(r['waypoint']) for r in rows]
    assert sorted(ids) == list(range(n))
    x = np.zeros(n, dtype=int)
    for r in rows:
        v = Fraction(r['shot_count'])
        assert v.denominator == 1
        x[int(r['waypoint'])] = int(v)
    return x

def write_alloc(path, x):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['waypoint', 'shot_count'])
        w.writerows(enumerate(map(int, x)))

def field_values(path):
    import numpy as np
    a = np.array([float(t) for t in Path(path).read_text().replace('\n', ',').split(',') if t.strip()])
    assert a.shape == (2500,) and np.isfinite(a).all()
    return a

def geometry():
    return SimpleNamespace(field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')

def load_math():
    import clarabel
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    selected = {'dy', 'down', 'up', 'Exact', 'lifted', 'clarabel_x', 'kkt_solve', 'refine', 'feasible_point', 'feasible_upper', 'check_int', 'frs'}
    tree = ast.parse((ROOT / 'code/rigor_pilot.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected]
    assert {n.name for n in nodes} == selected
    env = dict(np=np, sp=sp, spla=spla, clarabel=clarabel, Fraction=Fraction, math=math, time=time, M=2500)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'code/rigor_pilot.py', 'exec'), env)
    return SimpleNamespace(**{n: env[n] for n in selected})

def check_protocol():
    p = json.loads((ROOT / 'protocol.json').read_text())
    assert digest(ROOT / 'protocol.json') == (ROOT / 'protocol.sha256').read_text().split()[0]
    for entry in p['local_hashes']:
        assert digest(ROOT / entry['path']) == entry['sha256'], entry['path']
    return p

def pct(fr, up=False, digits=6):
    if fr is None:
        return 'undefined'
    if fr == math.inf:
        return '+Infinity'
    y = Fraction(fr) * 100 * 10 ** digits
    n = math.ceil(y) if up else math.floor(y)
    sign = '-' if n < 0 else ''
    n = abs(n)
    return f'{sign}{n // 10 ** digits}.{n % 10 ** digits:0{digits}d}'

def quantile(values, p):
    a = sorted(values)
    pos = (len(a) - 1) * Fraction(p)
    i = pos.numerator // pos.denominator
    if i == len(a) - 1 or pos == i:
        return a[i]
    if a[i + 1] == math.inf:
        return math.inf
    return a[i] + (pos - i) * (a[i + 1] - a[i])
