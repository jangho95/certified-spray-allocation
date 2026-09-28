from package_paths import load_json
import determinism
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import clarabel
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP / 'work' / 'stage1')]
import incumbents
import progress
from convex_qp_tools import dense_A
from qp_audit import sha256_file, solve_qp_audit
from scip_compare import build_stamps, load_warm_start, make_waypoints, reflect_index
INPUTS, WORK1 = (EXP / 'inputs', EXP / 'work' / 'stage1')
OUT = EXP / 'results' / 'rigor_pilot'
F, M, K, SIGMA = (50, 2500, 7, Fraction(7, 4))
CASES = [('T-zero', '목표 · zero', 2, 'zero_F50.csv', 200, 'zero_allocation.csv'), ('T-random', '목표 · random s7', 2, 'random_u1_30_seed7_F50.csv', 200, 'random_1_30_seed7_sync_allocation.csv'), ('T-c30', '목표 · center-30', 2, 'center30_F50.csv', 200, 'center30_allocation.csv'), ('T-c100', '목표 · center-100', 2, 'center100_F50.csv', 200, 'center100_allocation.csv'), ('V-s1c30', '어려운 사례 · s=1 center-30 (cond ≈ 4.2e11)', 1, 'center30_F50.csv', 200, 'ablation_center30_s1_allocation.csv'), ('V-s1zero', '어려운 사례 · s=1 zero (원고 L = 0 fallback)', 1, 'zero_F50.csv', 200, 'ablation_zero_s1_allocation.csv')]

def dy(v):
    fr = v if isinstance(v, Fraction) else Fraction(float(v))
    d = fr.denominator
    assert d & d - 1 == 0, 'not dyadic'
    return (fr.numerator, -(d.bit_length() - 1))

def down(fr):
    v = float(fr)
    return v if Fraction(v) <= fr else math.nextafter(v, -math.inf)

def up(fr):
    v = float(fr)
    return v if Fraction(v) >= fr else math.nextafter(v, math.inf)

class Exact:

    def __init__(self, stamps, s0, B, C):
        self.n, self.B, self.C = (len(stamps), B, C)
        ents = [(j, i, *dy(w)) for j, st in enumerate(stamps) for i, w in zip(st.cells, st.weights)]
        self.Ea = min((e for *_, e in ents))
        self.cols = [[] for _ in range(self.n)]
        for j, i, m, e in ents:
            self.cols[j].append((i, m << e - self.Ea))
        s = [dy(v) for v in s0]
        self.Es = min((e for m, e in s if m)) if any((m for m, _ in s)) else 0
        self.S0 = [m << e - self.Es if m else 0 for m, e in s]

    def field(self, x):
        xs = [dy(v) for v in x]
        Ex = min((e for m, e in xs if m))
        X = [m << e - Ex if m else 0 for m, e in xs]
        Fa = [0] * M
        for j, col in enumerate(self.cols):
            xj = X[j]
            if xj:
                for i, a in col:
                    Fa[i] += a * xj
        E = min(self.Ea + Ex, self.Es)
        sa, ss = (self.Ea + Ex - E, self.Es - E)
        return ([(fa << sa) + (s0 << ss) for fa, s0 in zip(Fa, self.S0)], E, X, Ex)

    def value(self, x):
        Fi, E, *_ = self.field(x)
        S, Q = (sum(Fi), sum((v * v for v in Fi)))
        return Fraction(M * Q - S * S, M * M) * Fraction(2) ** (2 * E)

    def bound(self, x):
        Fi, E, X, Ex = self.field(x)
        S, Q = (sum(Fi), sum((v * v for v in Fi)))
        V = Fraction(M * Q - S * S, M * M) * Fraction(2) ** (2 * E)
        D = [M * v - S for v in Fi]
        G = [sum((a * D[i] for i, a in col)) for col in self.cols]
        c = Fraction(2) ** (E + self.Ea + 1) / (M * M)
        Gx = Fraction(sum((g * xj for g, xj in zip(G, X)))) * Fraction(2) ** Ex
        rest, Gmin = (self.B - self.n, sum(G))
        for j in sorted(range(self.n), key=G.__getitem__):
            if rest == 0:
                break
            take = min(self.C - 1, rest)
            Gmin += G[j] * take
            rest -= take
        gap = c * (Gx - Gmin)
        return (V - gap, V, gap, G, c)

def exp_neg(a, K_terms=60):
    s, term = (Fraction(0), Fraction(1))
    for k in range(K_terms + 1):
        s += term
        term = term * -a / (k + 1)
    rem = abs(term) / (1 - a / (K_terms + 2))
    q = 1 << 200
    return (Fraction(math.floor((s - rem) * q), q), Fraction(math.ceil((s + rem) * q), q))

def ideal_eps(stamps, pts, s, C):
    r = K // 2
    t = {}
    delta_row = [Fraction(0)] * M
    dmax = Fraction(0)
    for (wr, wc), st in zip(pts, stamps):
        ex = {}
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                idx = reflect_index(wr + dr, F) * F + reflect_index(wc + dc, F)
                ex.setdefault(idx, []).append(dr * dr + dc * dc)
        assert sorted(ex) == list(st.cells)
        for i, w in zip(st.cells, st.weights):
            lo = hi = Fraction(0)
            for e in ex[i]:
                if e not in t:
                    t[e] = exp_neg(Fraction(e) / (2 * SIGMA * SIGMA))
                lo, hi = (lo + t[e][0], hi + t[e][1])
            d = max(Fraction(w) - lo, hi - Fraction(w))
            delta_row[i] += d
            dmax = max(dmax, d)
    eps2 = sum(((C * d) ** 2 for d in delta_row)) / M
    e = up(Fraction(math.sqrt(float(eps2))))
    while Fraction(e) ** 2 < eps2:
        e = math.nextafter(e, math.inf)
    return (e, up(dmax))

def sqrt_down(fr):
    v = math.sqrt(max(0.0, down(fr)))
    while v > 0 and Fraction(v) ** 2 > fr:
        v = math.nextafter(v, -math.inf)
    return v

def sqrt_up(fr):
    v = math.sqrt(up(fr))
    while Fraction(v) ** 2 < fr:
        v = math.nextafter(v, math.inf)
    return v

def lifted(A, s0, B, C):
    As = sp.csc_matrix(A)
    n = A.shape[1]
    one = np.ones((M, 1))
    Z = sp.hstack([As, sp.csc_matrix(-one)]).tocsc()
    H = 2.0 / M * (Z.T @ Z)
    q = 2.0 / M * (Z.T @ s0)
    return (As, Z, H.tocsc(), q, n)

def clarabel_x(A, s0, B, C):
    As, Z, H, q, n = lifted(A, s0, B, C)
    Ac = sp.vstack([sp.hstack([sp.csc_matrix(np.ones((1, n))), sp.csc_matrix((1, 1))]), sp.hstack([sp.identity(n), sp.csc_matrix((n, 1))]), sp.hstack([-sp.identity(n), sp.csc_matrix((n, 1))])]).tocsc()
    b = np.concatenate([[float(B)], np.full(n, float(C)), -np.ones(n)])
    st = clarabel.DefaultSettings()
    st.verbose, st.max_iter = (False, 400)
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = 1e-10
    t0 = time.perf_counter()
    sol = clarabel.DefaultSolver(sp.triu(H).tocsc(), q, Ac, b, [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(2 * n)], st).solve()
    return (np.asarray(sol.x)[:n], str(sol.status), time.perf_counter() - t0)

def kkt_solve(A, s0, B, C, ex, H, n, lower, upper, x0, iters=4):
    fr = np.array(sorted(set(range(n)) - lower - upper))
    x = np.clip(x0, 1.0, float(C)).copy()
    x[list(lower)], x[list(upper)] = (1.0, float(C))
    idx = np.concatenate([fr, [n]])
    k = len(fr)
    a = sp.csc_matrix(np.concatenate([np.ones(k), [0.0]])[:, None])
    lu = spla.splu(sp.bmat([[H[idx][:, idx], a], [a.T, None]]).tocsc())
    best, lam = (None, None)
    for it in range(iters):
        L, V, gap, G, c = ex.bound(x)
        if best is None or L > best[0]:
            best = (L, x.copy())
        g = np.array([float(c * gj) for gj in G])
        if lam is None:
            lam = -float(np.median(g[fr])) if k else 0.0
        f = A @ x + s0
        bud = float(Fraction(B) - sum((Fraction(float(v)) for v in x)))
        d = lu.solve(-np.concatenate([g[fr] + lam, [0.0], [-bud]]))
        x[fr] += d[:k]
        lam += d[k + 1]
    L, *_ = ex.bound(x)
    if L > best[0]:
        best = (L, x.copy())
    return (best, x, fr)

def refine(A, s0, B, C, ex, x0, taus=(1e-09, 1e-07, 1e-06, 1e-05, 0.0001, 0.001), rounds=6, feasible_out=None):

    def keep_feasible_candidate(x):
        if feasible_out is None:
            return
        xf = feasible_point(x, B, C)
        if xf is None:
            return
        xa = np.array([float(v) for v in xf])
        value = float(np.var(A @ xa + s0))
        if value < feasible_out.get('value_float', float('inf')):
            feasible_out.update(point=xf, value_float=value)
    keep_feasible_candidate(x0)
    _, _, H, _, n = lifted(A, s0, B, C)
    L0, V0, gap0, G0, c0 = ex.bound(x0)
    g0 = np.array([float(c0 * gj) for gj in G0])
    inner = (x0 > 1 + 0.001) & (x0 < C - 0.001)
    lam0 = -float(np.median(g0[inner])) if inner.any() else -float(np.median(g0))
    mu0 = g0 + lam0
    best, log = ((L0, x0.copy()), [])
    for tau in taus:
        lower = set(np.where((x0 <= 1 + tau) & (mu0 >= 0))[0])
        upper = set(np.where((x0 >= C - tau) & (mu0 <= 0))[0])
        for rnd in range(rounds):
            try:
                (L, xb), x, fr = kkt_solve(A, s0, B, C, ex, H, n, lower, upper, x0)
            except RuntimeError as e:
                log.append(dict(tau=tau, round=rnd, free=n - len(lower) - len(upper), L=None, error=str(e)))
                break
            log.append(dict(tau=tau, round=rnd, free=len(fr), L=float(L)))
            keep_feasible_candidate(x)
            if L > best[0]:
                best = (L, xb)
            viol_lo, viol_hi = (set(fr[x[fr] < 1.0]), set(fr[x[fr] > C]))
            if not (viol_lo or viol_hi):
                break
            lower, upper = (lower | viol_lo, upper | viol_hi)
    return (best, log)

def feasible_point(x, B, C):
    original = list(x)
    x = np.asarray([float(v) for v in original], dtype=float)
    n = len(x)
    if n == 0 or not np.isfinite(x).all() or (not np.isfinite(C)) or (C < 1):
        return None
    if int(B) != B or int(C) != C or B < n or (B > n * C):
        return None
    exact = [v if isinstance(v, Fraction) else Fraction(float(v)) for v in original]
    if sum(exact) == B and all((1 <= v <= C for v in exact)):
        return exact
    if B == n:
        return [Fraction(1)] * n
    if B == n * C:
        return [Fraction(int(C))] * n
    lo, hi = (1.0 - float(x.max()), float(C) - float(x.min()))
    if not np.isfinite([lo, hi]).all():
        return None
    for _ in range(100):
        t = lo / 2 + hi / 2
        total = math.fsum((float(v) for v in np.clip(x + t, 1, C)))
        if total < B:
            lo = t
        else:
            hi = t
        if np.nextafter(lo, math.inf) >= hi:
            break
    xc = [Fraction(float(v)) for v in np.clip(x + (lo / 2 + hi / 2), 1, C)]
    d = Fraction(int(B)) - sum(xc)
    room = lambda j: Fraction(int(C)) - xc[j] if d > 0 else xc[j] - 1
    for j in sorted(range(n), key=room, reverse=True):
        if d == 0:
            break
        step = min(d, Fraction(int(C)) - xc[j]) if d > 0 else max(d, 1 - xc[j])
        xc[j] += step
        d -= step
    if d != 0 or sum(xc) != B or (not all((1 <= v <= C for v in xc))):
        raise ArithmeticError('Exact budget-box projection failed')
    return xc

def feasible_upper(ex, points, B, C):
    best = None
    for name, x in points.items():
        xf = feasible_point(x, B, C)
        if xf is None:
            continue
        val = ex.value(xf)
        if best is None or val < best[0]:
            best = (val, xf, name)
    if best is None:
        raise ValueError('No feasible point for the continuous upper bound')
    return best

def check_int(x, B, C):
    xi = np.rint(x)
    return dict(integral=bool(np.all(x == xi)), budget=bool(int(xi.sum()) == B), box=bool(xi.min() >= 1 and xi.max() <= C))

def frs(fr):
    return f'{fr.numerator}/{fr.denominator}'

def env_versions():
    import importlib.metadata as md
    import platform
    return dict(python=platform.python_version(), **{k: md.version(k) for k in ('numpy', 'scipy', 'osqp', 'clarabel')})
IDEAL = {}

def save_witness(cid, st, s0, B, C, xbar, src, xf, alloc, L, U, Vf, eps, statuses, guarantee):
    d = OUT / 'witness' / cid
    d.mkdir(parents=True, exist_ok=True)
    ptr = np.cumsum([0] + [len(t.cells) for t in st])
    cells = np.concatenate([np.asarray(t.cells, dtype=np.int64) for t in st])
    wts = np.concatenate([np.asarray(t.weights, dtype=np.float64) for t in st])
    np.savez(d / 'operator.npz', col_ptr=ptr, cells=cells, weights=wts, s0=np.asarray(s0, dtype=np.float64))
    base = np.array([float(min(max(v, 1.0), float(C))) for v in xbar])
    adj = [j for j, (u, v) in enumerate(zip(base, xf)) if Fraction(float(u)) != v]
    np.savez(d / 'points.npz', xbar=np.asarray(xbar, dtype=np.float64), x_feas_base=base, alloc=np.asarray(alloc, dtype=np.int64))
    h = lambda arr: __import__('hashlib').sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    w = dict(case=cid, M=M, N=len(st), B=B, Cmax=C, guarantee=guarantee, xbar_source=src, sha256=dict(col_ptr=h(ptr), cells=h(cells), weights=h(wts), s0=h(np.asarray(s0, dtype=np.float64)), xbar=h(np.asarray(xbar, dtype=np.float64)), alloc=h(np.asarray(alloc, dtype=np.int64)), code=sha256_file(Path(__file__))), x_feas=dict(base='clip(xbar) to [1, C] (points.npz x_feas_base)', adjusted={str(j): frs(xf[j]) for j in adj}), L_exact=frs(L), U_exact=frs(U), V_feas_exact=frs(Vf) if Vf is not None else None, eps_ideal=dict(value=eps, hex=float(eps).hex()), alloc_checks=check_int(np.asarray(alloc, float), B, C), solver_status=statuses, environment=env_versions())
    (d / 'witness.json').write_text(json.dumps(w, indent=1, ensure_ascii=False))
    return d

def verify_witness(d):
    w = load_json((d / 'witness.json').read_text())
    op, pt = (np.load(d / 'operator.npz'), np.load(d / 'points.npz'))
    ptr = op['col_ptr']
    st = [SimpleNamespace(cells=op['cells'][ptr[j]:ptr[j + 1]].tolist(), weights=op['weights'][ptr[j]:ptr[j + 1]].tolist()) for j in range(len(ptr) - 1)]
    ex = Exact(st, op['s0'], w['B'], w['Cmax'])
    L = max(ex.bound(pt['xbar'])[0], Fraction(0))
    U = ex.value(pt['alloc'].astype(float))
    xf = [Fraction(float(v)) for v in pt['x_feas_base']]
    for j, v in w['x_feas']['adjusted'].items():
        xf[int(j)] = Fraction(v)
    ok_feas = sum(xf) == w['B'] and all((1 <= v <= w['Cmax'] for v in xf))
    Vf = ex.value(xf)
    return dict(L=frs(L) == w['L_exact'], U=frs(U) == w['U_exact'], V_feas=frs(Vf) == w['V_feas_exact'], x_feas_feasible=ok_feas, alloc=all(check_int(pt['alloc'].astype(float), w['B'], w['Cmax']).values()))

def run_case(case):
    T = {}
    t_all = time.perf_counter()
    cid, label, s, inp, C, alloc = case
    cnt = (F - 1) // s + 1
    t0 = time.perf_counter()
    g = SimpleNamespace(field_size=F, waypoint_rows=cnt, waypoint_cols=cnt, spray_interval=s, kernel_size=K, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
    pts = make_waypoints(g)
    st = build_stamps(g, pts)
    n = len(st)
    A = dense_A(st, M, n)
    s0 = np.array([float(t) for t in (INPUTS / inp).read_text().replace('\n', ',').split(',') if t.strip()])
    x_s1 = np.array(load_warm_start(WORK1 / alloc, n), dtype=float)
    B = int(x_s1.sum())
    ex = Exact(st, s0, B, C)
    T['setup'] = time.perf_counter() - t0
    row = dict(case=label, id=cid, N=n, B=B, BN=B / n, Cmax=C, spacing=s, input=inp, input_sha256=sha256_file(INPUTS / inp))
    t0 = time.perf_counter()
    a8 = solve_qp_audit(A, s0, B, st[0].mass, C)
    T['osqp_1e-8_float_cert'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    a10 = solve_qp_audit(A, s0, B, st[0].mass, C, eps_abs=1e-10, eps_rel=1e-10)
    T['osqp_1e-10'] = time.perf_counter() - t0
    xc, cstat, tc = clarabel_x(A, s0, B, C)
    T['clarabel'] = tc
    row.update(L_float_1e8=a8['dual_lower'], L_float_1e10=a10['dual_lower'], V_QP_float=a8['primal'], float_fallback=bool(a8['fallback']), float_clip_high=bool(a8['clip_high']), clarabel_status=cstat)
    pts_b = {'OSQP 1e-8': a8['x'], 'OSQP 1e-10': a10['x'], 'Clarabel 1e-10': xc}
    res = {}
    for name, xb in pts_b.items():
        t0 = time.perf_counter()
        L, V, gap, *_ = ex.bound(xb)
        T[f'exact_bound[{name}]'] = time.perf_counter() - t0
        res[name] = dict(L=L, gap=float(gap), x=xb)
    t0 = time.perf_counter()
    lean, llog = refine(A, s0, B, C, ex, a8['x'])
    T['refine_from_1e-8'] = time.perf_counter() - t0
    res['정제(OSQP 1e-8 출발)'] = dict(L=lean[0], gap=float(ex.bound(lean[1])[2]), x=lean[1])
    start = max(pts_b, key=lambda k: res[k]['L'])
    t0 = time.perf_counter()
    full, flog = refine(A, s0, B, C, ex, pts_b[start])
    T['refine_from_best'] = time.perf_counter() - t0
    res[f'정제({start} 출발)'] = dict(L=full[0], gap=float(ex.bound(full[1])[2]), x=full[1])
    src = max(res, key=lambda k: res[k]['L'])
    Lr = max(res[src]['L'], Fraction(0))
    xr = res[src]['x']
    t0 = time.perf_counter()
    xf = feasible_point(xr, B, C)
    Vf = ex.value(xf) if xf is not None else None
    T['feasible_point'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    bk = incumbents.best_for(inp, B, C, F=F, kernel=K, sigma=(1.75, 1.75), spacing=s)
    cand = {'stage-1 greedy+repair': x_s1}
    if bk:
        xb_ = np.zeros(n)
        for line in (EXP / bk['alloc_file']).read_text().splitlines()[1:]:
            f_ = line.split(',')
            xb_[int(f_[0])] = int(f_[-1])
        cand[f"registry best ({bk['method']})"] = xb_
    checks = {k: check_int(v, B, C) for k, v in cand.items()}
    Uex = {k: ex.value(v) for k, v in cand.items() if all(checks[k].values())}
    T['U_check_and_exact'] = time.perf_counter() - t0
    usrc = min(Uex, key=Uex.get)
    U, xu = (Uex[usrc], cand[usrc])
    fu = A @ xu + s0
    U_float = float(np.mean((fu - fu.mean()) ** 2))
    t0 = time.perf_counter()
    if s not in IDEAL:
        IDEAL[s] = ideal_eps(st, pts, s, C)
        T['ideal_eps(operator, once)'] = time.perf_counter() - t0
    eps, dmax = IDEAL[s]
    sL, sU = (sqrt_down(Lr), sqrt_up(U))
    dL = max(Fraction(0), Fraction(sL) - Fraction(eps))
    L_id = down(dL * dL)
    U_id = up((Fraction(sU) + Fraction(eps)) ** 2)
    T['total_run_case'] = time.perf_counter() - t_all
    T['lean_total'] = T['setup'] + T['osqp_1e-8_float_cert'] + T['exact_bound[OSQP 1e-8]'] + T['refine_from_1e-8'] + T['U_check_and_exact']
    L8 = row['L_float_1e8']
    Ll = max(res['정제(OSQP 1e-8 출발)']['L'], res['OSQP 1e-8']['L'], Fraction(0))
    wdir = save_witness(cid, st, s0, B, C, xr, src, xf, xu, Lr, U, Vf, eps, dict(osqp_1e8=a8['status'], osqp_1e10=a10['status'], clarabel=cstat), 'fixed float64 data (A from build_stamps, s0 as stored); ideal Gaussian via eps')
    row.update(L_rig=down(Lr), L_rig_source=src, L_rig_by_point={k: down(max(v['L'], Fraction(0))) for k, v in res.items()}, fw_gap_by_point={k: v['gap'] for k, v in res.items()}, L_lean=down(Ll), lean_equals_full=bool(Ll == Lr), W_lean_pct_up=up(100 * (U - Ll) / U) if U > 0 else None, V_feas_up=up(Vf) if Vf is not None else None, QP_enclosure_width_up=up(Vf - Lr) if Vf is not None else None, U_rig=up(U), U_float=U_float, U_source=usrc, U_checks=checks, U_float_minus_exact=float(Fraction(U_float) - U), W_samefloat_pct=100 * (U_float - L8) / U_float, W_rig_pct_up=up(100 * (U - Lr) / U) if U > 0 else None, W_ideal_pct_up=up(100 * (Fraction(U_id) - Fraction(L_id)) / Fraction(U_id)), eps_ideal=eps, delta_A_max=dmax, L_ideal=L_id, U_ideal=U_id, L8_minus_Lrig=float(Fraction(L8) - Lr), L8_above_Vfeas=bool(Vf is not None and Fraction(L8) > Vf), times=T, refine_log_lean=llog, refine_log_full=flog, witness=str(wdir.relative_to(EXP)), witness_verified=verify_witness(wdir))
    row['dW_pp'] = row['W_rig_pct_up'] - row['W_samefloat_pct'] if row['W_rig_pct_up'] is not None else None
    row['flag'] = 'ok' if row['dW_pp'] is not None and abs(row['dW_pp']) < 0.001 and (not row['L8_above_Vfeas']) and all(row['witness_verified'].values()) else 'warn'
    return row

def main():
    threads = determinism.assert_single_thread()
    OUT.mkdir(parents=True, exist_ok=True)
    progress.task('F.2', 'running', '파일럿 v2: 고정 float 입력에 대한 엄밀 하한과 정확한 U(정수해 검사), 바깥쪽 반올림, 검증 자료 저장, 구성요소별 총비용과 간소화 절차(OSQP 1e-8 출발 정제) 측정 — 목표 4개 + s=1 2개')
    rows = []
    for k, case in enumerate(CASES, 1):
        progress.detail('F.2', f'파일럿 v2 {k}/{len(CASES)}: {case[1]}')
        rows.append(run_case(case))
    (OUT / 'pilot_rows.json').write_text(json.dumps(dict(version=2, threads=threads, environment=env_versions(), rows=rows), indent=1, default=float, ensure_ascii=False))
    return rows
if __name__ == '__main__':
    main()
