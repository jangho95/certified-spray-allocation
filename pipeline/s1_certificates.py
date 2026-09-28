from __future__ import annotations
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
EXP = Path(__file__).resolve().parents[1]
WORK1 = EXP / 'work' / 'stage1'
sys.path[:0] = [str(WORK1), str(EXP / 'pipeline')]
from convex_qp_tools import dense_A, largest_remainder_round
from scip_compare import build_stamps, evaluate, load_warm_start, make_waypoints
from qp_audit import append_records, make_record, sha256_file, solve_qp_audit
INP = EXP / 'inputs'
OUT = EXP / 'results' / 'stage1'
OUT.mkdir(parents=True, exist_ok=True)
G = SimpleNamespace(field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
STAMPS = build_stamps(G, make_waypoints(G))
M, N = (2500, len(STAMPS))
A = dense_A(STAMPS, M, N)
MASS = STAMPS[0].mass
TARGETS = [('Zero', 'zero_F50.csv', 'zero_allocation.csv'), ('Random', 'random_u1_30_seed7_F50.csv', 'random_1_30_seed7_sync_allocation.csv'), ('Center-30', 'center30_F50.csv', 'center30_allocation.csv'), ('Center-100', 'center100_F50.csv', 'center100_allocation.csv')]
LOWCOUNT = [(3, 938), (5, 1250), (8, 1875), (20, 5000), (50, 12500), (200, 28348)]

def load_field(name):
    return np.array([float(t) for t in (INP / name).read_text().replace('\n', ',').split(',') if t.strip()])

def certificate(case, inp, alloc_file, cmax, budget=None):
    s0 = load_field(inp)
    x_rep = np.array(load_warm_start(WORK1 / alloc_file, N))
    B = int(x_rep.sum()) if budget is None else budget
    au = solve_qp_audit(A, s0, B, MASS, cmax)
    x_rnd = largest_remainder_round(au['x'], B, cmax)
    v_rep = evaluate(list(x_rep), s0, STAMPS, 50)[1]
    v_rnd = evaluate(list(x_rnd), s0, STAMPS, 50)[1]
    x_ub = x_rep if v_rep <= v_rnd else x_rnd
    U = min(v_rep, v_rnd)
    inc_viol = max(abs(int(x_ub.sum()) - B), max(0, 1 - int(x_ub.min())), max(0, int(x_ub.max()) - cmax))
    row = dict(case=case, B=B, Cmax=cmax, status=au['status'], V_QP=au['primal'], L=au['dual_lower'], V_round=v_rnd, V_repair=v_rep, U=U, gap_pct=100 * (U - au['dual_lower']) / U, prim_res=au['osqp_prim_res'], dual_res=au['osqp_dual_res'], Hz_v=au['range_rel_resid'], fallback=au['fallback'], clip=au['clip_low'] or au['clip_high'], inc_viol=inc_viol, primal_minus_L=au['primal'] - au['dual_lower'], iters=au['iter'], t_gram=au['t_gram'], t_setup=au['t_setup'], t_solve=au['t_solve'], t_dual=au['t_dual'], incumbent='repair' if v_rep <= v_rnd else 'round')
    rec = make_record(stage=1, experiment='table5_s8', case=case, input=inp, input_sha256=sha256_file(INP / inp), M=M, N=N, B=B, Cmax=cmax, kernel=7, sigma=1.75, spacing=2, boundary='reflect', solver='OSQP 1.1.3', settings=au['settings'], status=au['status'], U=U, L=au['dual_lower'], residuals=dict(prim=au['osqp_prim_res'], dual=au['osqp_dual_res'], Hz_v=au['range_rel_resid'], kkt_inf=au['kkt_stationarity_inf'], inc_viol=inc_viol), times=dict(gram=au['t_gram'], setup=au['t_setup'], solve=au['t_solve'], dual=au['t_dual']), V_QP=au['primal'], V_round=v_rnd, V_repair=v_rep)
    return (row, rec, dict(s0=s0, B=B, x_R=au['x'], x_ub=x_ub, x_rnd=x_rnd, U=U, L=au['dual_lower'], V_QP=au['primal']))

def table5():
    rows, recs = ([], [])
    for case, inp, alloc in TARGETS:
        r, rec, _ = certificate(case, inp, alloc, 200)
        rows.append(r)
        recs.append(rec)
    for cmax, B in LOWCOUNT:
        r, rec, _ = certificate(f'Cmax={cmax}', 'zero_F50.csv', f'cmax_zero_c{cmax}_b{B}_allocation.csv', cmax, B)
        rows.append(r)
        recs.append(rec)
    (OUT / 'table5_s8.json').write_text(json.dumps(rows, indent=1, default=float))
    append_records(OUT / 'records.jsonl', recs)
    for r in rows:
        print(f"{r['case']:11s} B={r['B']:6d} L={r['L']:.6f} Vqp={r['V_QP']:.6f} U={r['U']:.6f} gap={r['gap_pct']:.3f}% prim={r['prim_res']:.1e} dual={r['dual_res']:.1e} Hz={r['Hz_v']:.1e}")

def table_s6():
    Acen = A - A.mean(axis=0, keepdims=True)
    c = Acen @ np.ones(N) / N
    AP = Acen @ (np.eye(N) - np.ones((N, N)) / N)
    lam_max = float(np.linalg.eigvalsh(A.T @ A / M)[-1])
    yz, *_ = np.linalg.lstsq(AP, c, rcond=None)
    gamma = float((c - AP @ yz) @ (c - AP @ yz) / M)

    def phi(B, s0):
        w = s0 - s0.mean() + B * c
        y, *_ = np.linalg.lstsq(AP, -w, rcond=None)
        r = w + AP @ y
        return float(r @ r / M)
    rows = []
    for case, inp, alloc in [TARGETS[0], TARGETS[2], TARGETS[3]]:
        _, _, d = certificate(case, inp, alloc, 200)
        rows.append(dict(case=f'Target {case.lower()}', B=d['B'], phi=phi(d['B'], d['s0']), V_QP=d['V_QP']))
    for cmax, B in LOWCOUNT:
        _, _, d = certificate(f'Cmax={cmax}', 'zero_F50.csv', f'cmax_zero_c{cmax}_b{B}_allocation.csv', cmax, B)
        ph = phi(B, d['s0'])
        r_B = d['s0'] - (d['s0'].sum() + B * MASS) / M
        grad = 2.0 / M * (A.T @ (A @ d['x_R'] + r_B))
        delta = d['x_ub'] - d['x_R']
        first, curv = (float(grad @ delta), float(A @ delta @ (A @ delta) / M))
        dr = d['x_rnd'] - d['x_R']
        rows.append(dict(case=f'Cmax={cmax}', B=B, phi=ph, V_QP=d['V_QP'], cert_over_phi=100 * (d['U'] - d['L']) / ph, first_over_phi=100 * first / ph, curv_over_phi=100 * curv / ph, round_over_phi=100 * float(A @ dr @ (A @ dr) / M) / ph, spect_over_phi=100 * lam_max * float(dr @ dr) / ph, split_check=d['U'] - d['V_QP'] - (first + curv)))
    out = dict(lambda_max_Q=lam_max, gamma=gamma, rows=rows)
    (OUT / 'table_s6_full.json').write_text(json.dumps(out, indent=1, default=float))
    print(f'lambda_max(Q)={lam_max:.6e} gamma={gamma:.6e}')
    for r in rows:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
if __name__ == '__main__':
    {'table5': table5, 's6': table_s6}[sys.argv[1]]()
