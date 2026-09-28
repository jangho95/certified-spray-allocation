import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / 'src'))
sys.path.insert(0, str(EXP / 'pipeline'))
from convex_qp_tools import dense_A
from scip_compare import build_stamps, evaluate, load_warm_start, make_waypoints
import progress
BIN = EXP / 'build' / 'pufoam_solver'
INP = EXP / 'inputs'
WORK = EXP / 'work' / 'stage0'

def geom(F=50, rows=25, cols=25, s=2, kernel=7, boundary='reflect'):
    return SimpleNamespace(field_size=F, waypoint_rows=rows, waypoint_cols=cols, spray_interval=s, kernel_size=kernel, sigma_x=1.75, sigma_y=1.75, boundary=boundary)
CASES = [('zero target', 'zero_F50.csv', geom(), 200, None, []), ('random s7 target', 'random_u1_30_seed7_F50.csv', geom(), 200, None, []), ('center-30 target', 'center30_F50.csv', geom(), 200, None, []), ('center-100 target', 'center100_F50.csv', geom(), 200, None, []), ('zero low-count Cmax=3', 'zero_F50.csv', geom(), 3, 938, []), ('zero truncation', 'zero_F50.csv', geom(boundary='truncate'), 200, 28348, ['--boundary', 'truncate']), ('reduced F12 random', 'small_random_pcg64_seed7_F12.csv', geom(12, 4, 4, 3), 8, 64, [])]

def load_field(path):
    return np.array([float(t) for t in path.read_text().replace('\n', ',').split(',') if t.strip()])
rows = []
for name, inp, g, cmax, budget, extra in CASES:
    tag = re.sub('[^a-z0-9]+', '_', name.lower())
    cmd = [str(BIN), '--initial-field', str(INP / inp), '--field-size', str(g.field_size), '--waypoint-rows', str(g.waypoint_rows), '--waypoint-cols', str(g.waypoint_cols), '--spray-interval', str(g.spray_interval), '--cmax', str(cmax), '--quiet', '--full-precision', '--out-prefix', str(WORK / f'cons_{tag}'), *extra]
    if budget is not None:
        cmd += ['--budget', str(budget)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    cpp_var = float(re.search('final: .*variance=([0-9.eE+-]+)', out).group(1))
    cpp_B = int(re.search('total_shots=(\\d+)', out).group(1))
    stamps = build_stamps(g, make_waypoints(g))
    n, m = (len(stamps), g.field_size ** 2)
    x = np.array(load_warm_start(WORK / f'cons_{tag}_allocation.csv', n))
    s0 = load_field(INP / inp)
    f_cpp = load_field(WORK / f'cons_{tag}_field.csv')
    f_op = dense_A(stamps, m, n) @ x + s0
    _, py_var, _, _ = evaluate(list(x), s0, stamps, g.field_size)
    op_var = float(np.mean((f_op - f_op.mean()) ** 2))
    scale = float(np.max(np.abs(f_cpp)))
    r = dict(case=name, N=n, M=m, B=int(x.sum()), Cmax=cmax, budget_ok=bool(int(x.sum()) == cpp_B and (budget is None or cpp_B == budget)), box_ok=bool(x.min() >= 1 and x.max() <= cmax), field_rel_diff_cpp_vs_operator=float(np.max(np.abs(f_cpp - f_op))) / scale, V_cpp=cpp_var, V_python_loop=py_var, V_operator=op_var, V_rel_diff_max=max(abs(cpp_var - py_var), abs(cpp_var - op_var)) / cpp_var)
    r['flag'] = 'ok' if r['budget_ok'] and r['box_ok'] and (r['field_rel_diff_cpp_vs_operator'] < 1e-13) and (r['V_rel_diff_max'] < 1e-12) else 'fail'
    rows.append(r)
    print(r, flush=True)
(EXP / 'results' / 'stage0' / 'consistency.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
progress.result('s0_consistency', 'S0 · Python·C++ 일치', dict(columns=[dict(key='case', label='사례'), dict(key='N', label='N'), dict(key='B', label='B'), dict(key='Cmax', label='Cmax'), dict(key='budget_ok', label='예산'), dict(key='box_ok', label='box'), dict(key='field_rel_diff_cpp_vs_operator', label='두께장 상대차 (C++ vs Ax+s0)'), dict(key='V_cpp', label='V (C++)'), dict(key='V_operator', label='V (Ax+s0)'), dict(key='V_rel_diff_max', label='V 상대차 최대'), dict(key='flag', label='판정')], rows=[{**r, 'budget_ok': 'OK' if r['budget_ok'] else 'FAIL', 'box_ok': 'OK' if r['box_ok'] else 'FAIL'} for r in rows], note='같은 정본 입력 CSV와 같은 정수해를 C++ 출력 두께장, Python stamp 루프, 인증용 연산자 A x + s0로 각각 평가. 판정 기준: 두께장 상대차 < 1e-13, 분산 상대차 < 1e-12, 예산·box 정확히 만족.'))
print('all ok:', all((r['flag'] == 'ok' for r in rows)))
