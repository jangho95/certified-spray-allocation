from package_paths import load_json
import determinism
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import clarabel
import numpy as np
import scipy.sparse as sp
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP / 'work' / 'stage1')]
import incumbents
import progress
from convex_qp_tools import dense_A
from qp_audit import append_records, lower_bound_at, make_record, sha256_file, solve_qp_audit
from scip_compare import build_stamps, make_waypoints
from stage7_range import CMAX, FIELDS, F, S, load_field, read_alloc
OUT = EXP / 'results' / 'stage7'
RDIR = OUT / 'refine'
RUN_ID = incumbents.new_run_id(__file__)
INP = dict(FIELDS)
TOL_ACTIVE = 1e-05

def operator(kernel, sigma):
    g = SimpleNamespace(field_size=F, waypoint_rows=25, waypoint_cols=25, spray_interval=S, kernel_size=kernel, sigma_x=sigma, sigma_y=sigma, boundary='reflect')
    st = build_stamps(g, make_waypoints(g))
    return (dense_A(st, F * F, len(st)), st[0].mass)

def qp_data(A, s0, B, mass):
    m, n = A.shape
    r0 = s0 - (float(s0.sum()) + B * mass) / m
    H = 2.0 / m * (A.T @ A)
    return (H, 2.0 / m * (A.T @ r0), float(r0 @ r0) / m, np.concatenate([[float(B)], np.full(n, 1.0)]), np.concatenate([[float(B)], np.full(n, float(CMAX))]))

def variance(A, x, s0):
    f = A @ x + s0
    return float(np.mean((f - f.mean()) ** 2))

def violation(x, B):
    return float(max(abs(float(x.sum()) - B) / B, max(0.0, float(np.max(1.0 - x)), float(np.max(x - CMAX)))))

def lagrangian(H, q, c, lo, hi, y, primal):
    v = q + y[0] + y[1:]
    z = np.linalg.solve(H, v)
    rel = float(np.linalg.norm(H @ z - v)) / (1.0 + float(np.linalg.norm(v)))
    raw = c - 0.5 * float(v @ z) - float(hi @ np.maximum(y, 0.0)) + float(lo @ np.maximum(-y, 0.0))
    return lower_bound_at(dict(dual_raw=raw, primal=primal, range_rel_resid=rel), 1e-10) | dict(dual_raw=raw, range_rel=rel)

def clarabel_path(A, s0, B, mass):
    H, q, c, lo, hi = qp_data(A, s0, B, mass)
    n = A.shape[1]
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = 1e-10
    st.max_iter = 400
    Ac = sp.vstack([sp.csc_matrix(np.ones((1, n))), sp.identity(n), -sp.identity(n)]).tocsc()
    b = np.concatenate([[float(B)], np.full(n, float(CMAX)), -np.ones(n)])
    sol = clarabel.DefaultSolver(sp.triu(sp.csc_matrix(H)).tocsc(), q, Ac, b, [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(2 * n)], st).solve()
    x, z = (np.asarray(sol.x), np.asarray(sol.z))
    y = np.concatenate([[z[0]], z[1:n + 1] - z[n + 1:]])
    primal = float(0.5 * x @ H @ x + q @ x + c)
    lb = lagrangian(H, q, c, lo, hi, y, primal)
    return dict(status=str(sol.status), iters=int(sol.iterations), primal=primal, x=x, y=y, L=lb['dual_lower'], dual_raw=lb['dual_raw'], range_rel=lb['range_rel'], fallback=lb['fallback'], clip=lb['clip_low'] or lb['clip_high'], violation=violation(x, B))

def osqp_path(A, s0, B, mass, eps):
    au = solve_qp_audit(A, s0, B, mass, CMAX, eps_abs=eps, eps_rel=eps)
    return dict(status=au['status'], iters=au['iter'], primal=au['primal'], x=au['x'], y=au['y'], L=au['dual_lower'], dual_raw=au['dual_raw'], range_rel=au['range_rel_resid'], fallback=au['fallback'], clip=au['clip_low'] or au['clip_high'], violation=violation(au['x'], B), prim_res=au['osqp_prim_res'], dual_res=au['osqp_dual_res'])

def split(A, s0, B, mass, x_ref, xU, U, L):
    m = A.shape[0]
    rB = s0 - (s0.sum() + B * mass) / m
    grad = 2.0 / m * (A.T @ (A @ x_ref + rB))
    d = xU - x_ref
    first, curv = (float(grad @ d), float(A @ d @ (A @ d) / m))
    V_ref = variance(A, x_ref, s0)
    W = U - L
    return dict(V_ref=V_ref, first=first, curv=curv, numeric=V_ref - L, abs_width=W, rel_width_pct=100 * W / U, curv_share=curv / W if W > 0 else None, split_check=U - L - (first + curv + V_ref - L), lower_active=int(np.sum(np.abs(x_ref - 1) < TOL_ACTIVE)), upper_active=int(np.sum(np.abs(x_ref - CMAX) < TOL_ACTIVE)), x_max=float(x_ref.max()))

def main():
    threads = determinism.assert_single_thread()
    RDIR.mkdir(parents=True, exist_ok=True)
    base = load_json((OUT / 'range_rows_before_refine.json').read_text())
    rows = base['rows']
    orig_records = {}
    for line in (OUT / 'records.jsonl').read_text().splitlines():
        rec = load_json(line)
        if rec['extra'].get('run_id') == base['run_id']:
            orig_records[rec['case']] = rec
    cond = {}
    ops = {}
    todo = [r for r in rows if r['U'] - r['L'] > 0 and r['V_QP'] - r['L'] > 0.1 * (r['U'] - r['L'])]
    progress.task('7.8', 'running', f'보정 v2: {len(todo)}개 사례의 기준 연속해·하한·분해·활성 제약 재계산')
    records = []
    for k, r in enumerate(rows, 1):
        kern = int(r['kernel'].split('×')[0])
        if (kern, r['sigma']) not in ops:
            ops[kern, r['sigma']] = operator(kern, r['sigma'])
            A_ = ops[kern, r['sigma']][0]
            ev = np.linalg.eigvalsh(A_.T @ A_)
            cond[f"{r['kernel']} σ={r['sigma']:g}"] = float(ev[-1] / ev[0])
        A, mass = ops[kern, r['sigma']]
        s0 = load_field(INP[r['field']])
        xU = read_alloc(EXP / r['U_alloc'], A.shape[1]).astype(float) if 'U_alloc' in r else None
        if xU is None:
            best = incumbents.best_for(INP[r['field']], r['B'], CMAX, F=F, kernel=kern, sigma=(r['sigma'], r['sigma']), spacing=S)
            xU = read_alloc(EXP / best['alloc_file'], A.shape[1]).astype(float)
        case = f"{r['field']} k{kern} sigma{r['sigma']:g} rho{r['rho']:g}"
        for key in ('L', 'V_QP', 'abs_width', 'rel_width_pct', 'first', 'curv', 'numeric', 'curv_share', 'split_check', 'relax_lower_active', 'relax_upper_active', 'relax_x_max'):
            r[f'{key}_orig'] = r[key]
        refine = r in todo
        if refine:
            progress.detail('7.8', f'보정 {sum((1 for x in todo[:todo.index(r) + 1]))}/{len(todo)} · {case}')
            paths = {'osqp_1e-8': osqp_path(A, s0, r['B'], mass, 1e-08), 'osqp_1e-10': osqp_path(A, s0, r['B'], mass, 1e-10), 'clarabel': clarabel_path(A, s0, r['B'], mass)}
            src = max(paths, key=lambda p: paths[p]['L'])
            L = paths[src]['L']
            o, c = (paths['osqp_1e-10'], paths['clarabel'])
            use_c = c['status'] == 'Solved' and c['primal'] < o['primal'] and (c['violation'] < o['violation'])
            xref_src = 'clarabel' if use_c else 'osqp_1e-10'
            x_ref = paths[xref_src]['x']
            np.savez_compressed(RDIR / f"{case.replace(' ', '_')}.npz", x_ref=x_ref, **{f"y_{p.replace('-', '_')}": v['y'] for p, v in paths.items()})
            audit = {p: {kk: vv for kk, vv in v.items() if kk not in ('x', 'y')} for p, v in paths.items()}
        else:
            L, src, xref_src, audit = (r['L'], 'osqp_1e-8', 'osqp_1e-8', None)
            x_ref = np.array(r['x_R']) if 'x_R' in r else None
            if x_ref is None:
                x_ref = osqp_path(A, s0, r['B'], mass, 1e-08)['x']
        sp_ = split(A, s0, r['B'], mass, x_ref, xU, r['U'], L)
        r.update(L_final=L, L_source=src, x_ref_source=xref_src, x_ref_violation=violation(x_ref, r['B']), V_QP_final=sp_['V_ref'], abs_width_final=sp_['abs_width'], rel_width_final_pct=sp_['rel_width_pct'], first_final=sp_['first'], curv_final=sp_['curv'], numeric_final=sp_['numeric'], curv_share_final=sp_['curv_share'], split_check_final=sp_['split_check'], relax_lower_active_final=sp_['lower_active'], relax_upper_active_final=sp_['upper_active'], relax_x_max_final=sp_['x_max'], refined=refine, gram_cond=cond[f"{r['kernel']} σ={r['sigma']:g}"], bound_audit=audit, final_record_run_id=RUN_ID if refine else base['run_id'])
        r['flag'] = 'ok' if r['rel_width_final_pct'] <= 1 else 'warn' if r['rel_width_final_pct'] <= 5 else 'fail'
        if refine:
            o = orig_records.get(case)
            records.append(make_record(stage=7, experiment=f"range_{r['series']}_final", case=case, input=INP[r['field']], input_sha256=sha256_file(EXP / 'inputs' / INP[r['field']]), M=F * F, N=625, B=r['B'], Cmax=CMAX, kernel=kern, sigma=r['sigma'], spacing=S, boundary='reflect', solver='OSQP 1.1.3 (1e-8, 1e-10) + Clarabel 0.11.1', settings=dict(range_tol=1e-10), status=audit[src]['status'], U=r['U'], L=L, residuals=dict(Hz_v=audit[src]['range_rel'], violation=audit[src]['violation']), L_source=src, bound_audit=audit, x_ref_source=xref_src, V_ref=sp_['V_ref'], decomposition={kk: sp_[kk] for kk in ('first', 'curv', 'numeric', 'split_check')}, upper_active=sp_['upper_active'], gram_cond=cond[f"{r['kernel']} σ={r['sigma']:g}"], run_id=RUN_ID, supersedes_run_id=base['run_id'], threads=threads, refine_arrays=str((RDIR / f"{case.replace(' ', '_')}.npz").relative_to(EXP))))
        r.pop('x_R', None)
    append_records(OUT / 'records.jsonl', records)
    out = dict(base, rows=rows, refine_run_id=RUN_ID, gram_cond=cond, note='*_orig = OSQP eps 1e-8 path; *_final = final bound and precise reference solution')
    (OUT / 'range_rows.json').write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    for r in todo:
        print(f"{r['field']:10s} {r['kernel']:6s} rho={r['rho']:<4} L {r['L_orig']:.6g} -> {r['L_final']:.8g} [{r['L_source']}], x_ref {r['x_ref_source']} viol {r['x_ref_violation']:.1e}, numeric {r['numeric_final']:.3g}, curv share {r['curv_share_final']:.4f}, upper active {r['relax_upper_active_final']}, rel {r['rel_width_final_pct']:.4g}%")
    print('cond', cond)
if __name__ == '__main__':
    main()
