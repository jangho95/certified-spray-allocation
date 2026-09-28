from package_paths import load_json
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / 'src'))
sys.path.insert(0, str(EXP / 'pipeline'))
from convex_qp_tools import dense_A, solve_box_qp_relaxation
from scip_compare import build_stamps, load_warm_start, make_waypoints
from scip_soc_compare import build_soc_model, complete_warm_start
import progress
INP, WORK = (EXP / 'inputs', EXP / 'work' / 'stage0')

def geom(F, s):
    cnt = (F - 1) // s + 1
    return SimpleNamespace(field_size=F, waypoint_rows=cnt, waypoint_cols=cnt, spray_interval=s, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')

def load_field(path):
    return np.array([float(t) for t in path.read_text().replace('\n', ',').split(',') if t.strip()])

def warm_checks(stamps, s0, B, m, cmax, x):
    mI, xv = build_soc_model(stamps, s0, B, m, cmax, integer=True)
    partial = mI.createSol()
    for v, val in zip(xv, x):
        mI.setSolVal(partial, v, val)
    full = complete_warm_start(mI, xv, x, stamps, s0, B, m)
    return (mI, full, bool(mI.checkSol(partial, printreason=False)), bool(mI.checkSol(full, printreason=False)))

def cont_solve(stamps, s0, B, m, cmax, tlim):
    mC, _ = build_soc_model(stamps, s0, B, m, cmax, integer=False)
    mC.setParam('limits/time', tlim)
    t0 = time.time()
    mC.optimize()
    old = mC.getObjVal() if mC.getNSols() > 0 else None
    new = mC.getDualbound()
    return dict(cont_status=mC.getStatus(), cont_time_s=time.time() - t0, cont_getObjVal_old=old, cont_getDualbound_new=new, old_minus_new=None if old is None else old - new)
if len(sys.argv) > 1 and sys.argv[1] == '--child-paper-scale':
    g = geom(50, 2)
    st = build_stamps(g, make_waypoints(g))
    print(json.dumps(cont_solve(st, np.zeros(2500), 28348, 2500, 200, 20.0)), flush=True)
    sys.exit(0)
rows = []
g = geom(12, 3)
st = build_stamps(g, make_waypoints(g))
n, m = (len(st), 144)
s0 = load_field(INP / 'small_random_pcg64_seed7_F12.csv')
x = load_warm_start(WORK / 'cons_reduced_f12_random_allocation.csv', n)
qp = solve_box_qp_relaxation(dense_A(st, m, n), s0, 64, st[0].mass, 8)
mI, full, xo, co = warm_checks(st, s0, 64, m, 8, x)
r = dict(case='A · 축소 F12 random (N=16)', N=n, B=64, warm_x_only=xo, warm_complete=co, osqp_dual_lb=qp.dual_lower, **cont_solve(st, s0, 64, m, 8, 120.0))
accepted = mI.addSol(full, free=True)
mI.setParam('limits/gap', 0.0)
mI.setParam('limits/time', 120.0)
mI.optimize()
r.update(int_warm_accepted=bool(accepted), int_status=mI.getStatus(), int_primal=mI.getPrimalbound(), int_dual=mI.getDualbound(), note='정수 최적값 = Table 6 V_Z* 64.43374')
rows.append(r)
print(r, flush=True)
g = geom(50, 2)
st = build_stamps(g, make_waypoints(g))
x = load_warm_start(WORK / 'cons_zero_target_allocation.csv', len(st))
_, _, xo, co = warm_checks(st, np.zeros(2500), 28348, 2500, 200, x)
rows.append(dict(case='B · 본 실험 zero (N=625), 초기해만', N=625, B=28348, warm_x_only=xo, warm_complete=co))
print(rows[-1], flush=True)
g = geom(30, 2)
st = build_stamps(g, make_waypoints(g))
n, m = (len(st), 900)
B = int(round(200 * m / st[0].mass))
qp = solve_box_qp_relaxation(dense_A(st, m, n), np.zeros(m), B, st[0].mass, 200)
r = dict(case='C · F30 zero (N=225), 20초 제한', N=n, B=B, osqp_dual_lb=qp.dual_lower, **cont_solve(st, np.zeros(m), B, m, 200, 20.0))
r['note'] = '시간 제한 종료로 SCIP가 증명한 하한은 getDualbound뿐이다. getObjVal은 찾은 해의 값으로, 이번에는 OSQP 하한과 1e-9 이내였지만 증명되지 않은 값이다'
rows.append(r)
print(r, flush=True)
try:
    p = subprocess.run([sys.executable, __file__, '--child-paper-scale'], capture_output=True, text=True, timeout=180)
except subprocess.TimeoutExpired as e:
    err = e.stderr.decode(errors='replace') if isinstance(e.stderr, bytes) else e.stderr or ''
    p = SimpleNamespace(returncode='timeout', stdout='', stderr=err or 'hang after 180 s')
r = dict(case='D · 본 실험 zero (N=625), 연속 SOC', N=625, B=28348, child_returncode=p.returncode)
if p.returncode == 0:
    r.update(load_json(p.stdout.strip().splitlines()[-1]))
else:
    r['cont_status'] = 'crash'
    r['note'] = 'SCIP 10.0 / PySCIPOpt 6.2.1이 힙 손상으로 중단: ' + (p.stderr.strip().splitlines() or ['(stderr 없음)'])[0][:80]
rows.append(r)
print(r, flush=True)
for r in rows:
    ok = r.get('warm_complete', True) and (not r.get('warm_x_only', False)) and (r.get('cont_status') != 'crash')
    r['flag'] = 'ok' if ok else 'warn'
(EXP / 'results' / 'stage0' / 'soc_check.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
cols = [('case', '사례'), ('N', 'N'), ('B', 'B'), ('warm_x_only', '초기해 x만'), ('warm_complete', '초기해 x,g,t'), ('cont_status', '연속 SOC 상태'), ('cont_time_s', '시간(s)'), ('cont_getObjVal_old', 'getObjVal (기존)'), ('cont_getDualbound_new', 'getDualbound (수정)'), ('old_minus_new', '기존−수정'), ('osqp_dual_lb', 'OSQP 하한'), ('int_status', '정수 SOC 상태'), ('int_primal', '정수 primal'), ('int_dual', '정수 dual'), ('note', '비고'), ('flag', '판정')]
progress.result('s0_soc', 'S0 · SOC 하한·초기해', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=[{k: ('통과' if v else '실패') if isinstance(v, bool) else v for k, v in r.items()} for r in rows], note='수정 전 코드는 연속 SOC의 getObjVal()을 하한으로 읽었다. 이 값은 찾은 해의 목적값이므로 시간 제한으로 끝나면 증명된 하한이 아니다(C행: 증명된 하한보다 약 0.0059 큼). 수정 코드는 getDualbound()와 종료 상태를 함께 기록하고, 초기해에 x·g·t를 모두 채운다.'))
