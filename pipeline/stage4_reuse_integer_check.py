import determinism
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import osqp
import scipy.sparse as sp
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP / 'work' / 'stage1')]
import incumbents
import progress
from convex_qp_tools import dense_A, largest_remainder_round
from qp_audit import solve_qp_audit
from rounding import round_indexed
from stage4_worker import cpp_cmd, geometry, load_field, read_alloc, target_budget, variance, write_alloc
from scip_compare import build_stamps, make_waypoints
WORK = EXP / 'work' / 'stage4'
OUT = EXP / 'results' / 'stage4'
RUN_ID = incumbents.new_run_id(__file__)

def main():
    threads = determinism.assert_single_thread()
    rows = []
    for F in (25, 50, 100):
        g, cnt = geometry(F)
        st = build_stamps(g, make_waypoints(g))
        n, m = (len(st), F * F)
        A = dense_A(st, m, n)
        inp = f'zero_F{F}.csv'
        s0 = load_field(inp)
        mass = st[0].mass
        B0 = target_budget(s0, m, mass, n)
        budgets = sorted({int(round(b)) for b in np.linspace(0.85 * B0, 1.05 * B0, 21)})
        H = 2.0 / m * (A.T @ A)
        As0, kappa = (A.T @ s0, A.T @ np.ones(m))
        G = sp.vstack([sp.csc_matrix(np.ones((1, n))), sp.identity(n)]).tocsc()
        prob = osqp.OSQP()

        def vecs(B):
            mean = (float(s0.sum()) + B * mass) / m
            return (2.0 / m * (As0 - mean * kappa), np.concatenate([[float(B)], np.full(n, 1.0)]), np.concatenate([[float(B)], np.full(n, 200.0)]))
        q, lo, hi = vecs(budgets[0])
        prob.setup(sp.csc_matrix(H), q, G, lo, hi, eps_abs=1e-08, eps_rel=1e-08, max_iter=400000, polishing=True, warm_starting=False, verbose=False)
        diff_paper = diff_idx = 0
        rho_updates = []
        x_reuse_target = None
        for B in budgets:
            progress.detail('4.5', f'재사용 경로 정수해 점검 · F={F} · B={B}')
            q, lo, hi = vecs(B)
            prob.update(q=q, l=lo, u=hi)
            r = prob.solve()
            rho_updates.append(int(r.info.rho_updates))
            xu = np.asarray(r.x)
            xi = solve_qp_audit(A, s0, B, mass, 200)['x']
            diff_paper += not np.array_equal(largest_remainder_round(xu, B, 200), largest_remainder_round(xi, B, 200))
            diff_idx += not np.array_equal(round_indexed(xu, B, 200), round_indexed(xi, B, 200))
            if B == min(budgets, key=lambda b: abs(b - B0)):
                x_reuse_target, B_t = (xu, B)
        rpath = WORK / f'reusecheck_F{F}_round.csv'
        write_alloc(rpath, round_indexed(x_reuse_target, B_t, 200))
        out = subprocess.run(cpp_cmd(inp, F, cnt, B_t, WORK / f'reusecheck_F{F}_rr', rpath), capture_output=True, text=True, check=True).stdout
        cand = {'반올림(재사용 QP)→repair': WORK / f'reusecheck_F{F}_rr_allocation.csv'}
        if B_t == B0:
            cand['greedy+repair (Stage 4)'] = WORK / f'single_F{F}_zero_r1_greedy_allocation.csv'
            cand['반올림→repair (Stage 4, 현재 구현 QP)'] = WORK / f'single_F{F}_zero_r1_rr_allocation.csv'
        vals = {}
        for name, path in cand.items():
            x = read_alloc(path, n)
            ok = bool(x.sum() == B_t and x.min() >= 1 and (x.max() <= 200))
            vals[name] = variance(A, x, s0)
            incumbents.register(inp=inp, B=B_t, cap=200, method=name, V=vals[name], alloc_path=path, run_id=RUN_ID, feasible=ok, F=F, stage=4, experiment='reuse_integer_check')
        best = incumbents.rebuild_best()
        k = incumbents.key(inp, B_t, 200, F=F)
        rows.append(dict(F=F, N=n, budgets=len(budgets), rounding_diff_paper=diff_paper, rounding_diff_indexed=diff_idx, rho_updates_max=max(rho_updates), B_target=B_t, **{f'V {k2}': v for k2, v in vals.items()}, best_method=best[k]['method'], best_V=best[k]['V'], reuse_moves=int(re.search('exchange_moves=(\\d+)', out).group(1)), flag='warn' if diff_idx else 'ok'))
    (OUT / 'reuse_integer_check.json').write_text(json.dumps(dict(run_id=RUN_ID, threads=threads, rows=rows), indent=1, default=float, ensure_ascii=False))
    keys = list(dict.fromkeys((k for r in rows for k in r)))
    progress.result('s4_reuse_int', 'S4 · 재사용 경로 정수해', dict(columns=[dict(key=k, label=k) for k in keys], rows=rows, note='재사용 경로(setup 한 번 후 q·l·u 갱신)의 연속해로 반올림하면 대부분 예산에서 배분이 달라지고, 목표 예산 repair 결과도 달라진다. 하한이 거의 같아도 정수 후보는 같지 않으므로 재사용 경로의 해는 별도 후보로 등록하고 best.json은 모든 후보 중 최선을 유지한다. rho_updates는 OSQP가 rho를 갱신(재인수분해)한 횟수의 최댓값.'))
    for r in rows:
        print(r)
if __name__ == '__main__':
    main()
