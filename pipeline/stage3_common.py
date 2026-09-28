from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
EXP = Path(__file__).resolve().parents[1]
WORK1, WORK3 = (EXP / 'work' / 'stage1', EXP / 'work' / 'stage3')
OUT = EXP / 'results' / 'stage3'
INPUTS = EXP / 'inputs'
BIN = EXP / 'build' / 'pufoam_solver'
for _d in (WORK3, OUT):
    _d.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(WORK1), str(EXP / 'pipeline')]
from convex_qp_tools import dense_A
from scip_compare import build_stamps, load_warm_start, make_waypoints
G = SimpleNamespace(field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
STAMPS = build_stamps(G, make_waypoints(G))
N, M = (len(STAMPS), 2500)
A = dense_A(STAMPS, M, N)
MASS = STAMPS[0].mass

def load_field(name):
    return np.array([float(t) for t in (INPUTS / name).read_text().replace('\n', ',').split(',') if t.strip()])

def variance(x, s0):
    f = A @ np.asarray(x, float) + s0
    return float(np.mean((f - f.mean()) ** 2))

def write_alloc(path, x):
    Path(path).write_text('waypoint,shot_count\n' + ''.join((f'{j},{int(v)}\n' for j, v in enumerate(x))))
    return Path(path)

def read_alloc(path):
    return np.array(load_warm_start(Path(path), N))

def cpp_repair(inp, B, cap, prefix, init=None):
    cmd = [str(BIN), '--initial-field', str(INPUTS / inp), '--budget', str(B), '--cmax', str(cap), '--quiet', '--full-precision', '--out-prefix', str(WORK3 / prefix)]
    if init is not None:
        cmd += ['--init-allocation', str(init)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    path = WORK3 / f'{prefix}_allocation.csv'
    return (read_alloc(path), int(re.search('exchange_moves=(\\d+)', out).group(1)), path)

def cpp_eval(inp, cap, alloc_path):
    out = subprocess.run([str(BIN), '--initial-field', str(INPUTS / inp), '--cmax', str(cap), '--eval-allocation', str(alloc_path), '--quiet'], check=True, capture_output=True, text=True).stdout
    return float(re.search('variance=([0-9.eE+-]+)', out).group(1))

class Pool:

    def __init__(self, inp, B, s0):
        self.inp, self.B, self.s0, self.items = (inp, B, s0, [])

    def add(self, name, x, path, moves=None, start=None, cap_run=None):
        x = np.asarray(x)
        item = dict(method=name, V=variance(x, self.s0), x_max=int(x.max()), x_min=int(x.min()), budget_ok=int(x.sum()) == self.B, moves=moves, start=start, cap_run=cap_run, path=str(path), x=x)
        self.items.append(item)
        return item

    def feasible(self, it, cap):
        return it['budget_ok'] and it['x_min'] >= 1 and (it['x_max'] <= cap)

    def best(self, cap):
        pool = [it for it in self.items if self.feasible(it, cap)]
        return min(pool, key=lambda it: it['V'])
