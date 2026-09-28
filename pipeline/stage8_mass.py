from package_paths import load_json
import determinism
import json
import math
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from functools import reduce
from pathlib import Path
from types import SimpleNamespace
import numpy as np
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP / 'work' / 'stage1')]
import incumbents
import progress
from convex_qp_tools import dense_A
from qp_audit import sha256_file
from scip_compare import build_stamps, make_waypoints
OUT = EXP / 'results' / 'stage8'
CMAX = 200
A_EFF = Fraction(1, 20)
KV = {}

def vm_hwm_mib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) / 1024
    return None

def masses(F, s, rows, boundary='reflect'):
    g = SimpleNamespace(field_size=F, waypoint_rows=rows, waypoint_cols=rows, spray_interval=s, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary=boundary)
    pts = make_waypoints(g)
    st = build_stamps(g, pts)
    A = dense_A(st, F * F, len(st))
    return (A, np.array(pts, dtype=float))

def efficiency(pts, F, a):
    c = (F - 1) / 2.0
    d2 = ((pts - c) ** 2).sum(axis=1)
    return 1.0 - a * d2 / d2.max()

def scenarios(F=50, s=2, rows=25):
    A_ref, pts = masses(F, s, rows, 'reflect')
    A_tr, _ = masses(F, s, rows, 'truncate')
    out = {'reflect': A_ref, 'truncate': A_tr, 'uniform loss ×0.8': 0.8 * A_ref}
    for a in (0.01, 0.05, 0.2):
        out[f'efficiency −{a:.0%}'] = A_ref * efficiency(pts, F, a)[None, :]
    return out

def extreme_mass(kappa, B, cmax):
    n = len(kappa)
    surplus = B - n
    res = []
    for order in (np.argsort(kappa), np.argsort(-kappa)):
        x = np.ones(n)
        left = surplus
        for j in order:
            add = min(cmax - 1, left)
            x[j] += add
            left -= add
            if left == 0:
                break
        res.append(float(kappa @ x))
    return res

def part_a():
    M = 2500
    sc = scenarios()
    kref = float(sc['reflect'].sum(axis=0)[0])
    allocs = {}
    best = incumbents.best_for('zero_F50.csv', 28348, 200)
    allocs['최선 해 (B=28348, 레지스트리)'] = best['alloc_file']
    allocs['greedy+repair (B=28348, Stage 1)'] = 'work/stage1/zero_allocation.csv'
    b938 = incumbents.best_for('zero_F50.csv', 938, 200)
    allocs['최선 해 (B=938, 레지스트리)'] = b938['alloc_file']

    def read(path):
        x = np.zeros(625)
        for line in (EXP / path).read_text().splitlines()[1:]:
            f = line.split(',')
            x[int(f[0])] = int(f[-1])
        return x
    X = {k: read(v) for k, v in allocs.items()}
    files = {k: dict(source=v, sha256=sha256_file(EXP / v)) for k, v in allocs.items()}
    rng = np.random.default_rng(8)
    for B in (938, 28348):
        x = np.ones(625)
        extra = rng.multinomial(B - 625, np.ones(625) / 625)
        x += np.minimum(extra, CMAX - 1)
        while x.sum() < B:
            j = rng.integers(625)
            if x[j] < CMAX:
                x[j] += 1
        X[f'무작위 실행가능 해 (B={B}, seed 8)'] = x
    (OUT / 'alloc').mkdir(parents=True, exist_ok=True)
    for i, (name, x) in enumerate(X.items()):
        dst = OUT / 'alloc' / f'a{i}_B{int(x.sum())}.csv'
        if name in allocs:
            shutil.copyfile(EXP / allocs[name], dst)
        else:
            dst.write_text('waypoint,shot_count\n' + ''.join((f'{j},{int(v)}\n' for j, v in enumerate(x))))
            files[name] = dict(source='numpy default_rng(8) multinomial, capped, topped up')
        files[name].update(file=str(dst.relative_to(EXP)), sha256=sha256_file(dst))
    rows_k, rows_mu = ([], [])
    for name, A in sc.items():
        k = A.sum(axis=0)
        KV[f'F50|{name}'] = k
        rows_k.append(dict(scenario=name, kappa_min=float(k.min()), kappa_max=float(k.max()), kappa_mean=float(k.mean()), rel_spread=float((k.max() - k.min()) / k.mean()), distinct=int(len(np.unique(np.round(k, 12)))), common_mass=bool(np.ptp(k) <= 1e-12 * k.mean()), flag='ok'))
        for B in (938, 28348):
            lo, hi = extreme_mass(k, B, CMAX)
            pr, pb = (kref * B / M, float(k.mean()) * B / M)
            rows_mu.append(dict(scenario=name, allocation=f'평균 범위 (B={B}, 전체 실행가능 배분)', B=B, mean_min=lo / M, mean_max=hi / M, width=(hi - lo) / M, pred_common_kref=pr, err_kref=max(abs(lo / M - pr), abs(hi / M - pr)), pred_common_kbar=pb, err_kbar=max(abs(lo / M - pb), abs(hi / M - pb)), worst_case=True, flag='ok' if hi - lo <= 1e-09 * hi else 'warn'))
        for aname, x in X.items():
            B = int(x.sum())
            actual = float(k @ x) / M
            rows_mu.append(dict(scenario=name, allocation=aname, B=B, mean_actual=actual, worst_case=False, pred_common_kref=kref * B / M, err_kref=actual - kref * B / M, pred_common_kbar=float(k.mean()) * B / M, err_kbar=actual - float(k.mean()) * B / M, flag='ok' if abs(actual - kref * B / M) <= 1e-09 * max(1.0, actual) else 'warn'))
    return (rows_k, rows_mu, kref, files)

def merged_unique(v, rel_tol=1e-12):
    v = np.sort(v)
    tol = rel_tol * max(1.0, float(np.abs(v).max()))
    keep = np.concatenate([[True], np.diff(v) > tol])
    return v[keep]

def exact_eff_weights(pts, F, a=A_EFF):
    D = [(2 * r - F + 1) ** 2 + (2 * c - F + 1) ** 2 for r, c in pts]
    Dm = max(D)
    num = [a.denominator * Dm - a.numerator * d for d in D]
    g = reduce(math.gcd, num)
    return ([v // g for v in num], Fraction(g, a.denominator * Dm))

def rational_rank(vecs):
    rows = [[Fraction(v) for v in r] for r in vecs]
    rank, col, ncol = (0, 0, len(rows[0]) if rows else 0)
    while rank < len(rows) and col < ncol:
        piv = next((i for i in range(rank, len(rows)) if rows[i][col] != 0), None)
        if piv is None:
            col += 1
            continue
        rows[rank], rows[piv] = (rows[piv], rows[rank])
        for i in range(len(rows)):
            if i != rank and rows[i][col] != 0:
                f = rows[i][col] / rows[rank][col]
                rows[i] = [x - f * y for x, y in zip(rows[i], rows[rank])]
        rank, col = (rank + 1, col + 1)
    return rank

def trunc_structure(pts, F, kappa, radius=3, sigma=Fraction(7, 4)):
    E = sorted({dr * dr + dc * dc for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)})
    vec = []
    for r, c in pts:
        v = dict.fromkeys(E, 0)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if 0 <= r + dr < F and 0 <= c + dc < F:
                    v[dr * dr + dc * dc] += 1
        vec.append(tuple((v[e] for e in E)))
    t = math.exp(-1.0 / float(2 * sigma * sigma))
    poly = np.array([sum((ci * t ** e for ci, e in zip(v, E))) for v in vec])
    types = sorted(set(vec))
    counts = [vec.count(v) for v in types]
    return dict(exponents=E, types=types, counts=counts, rank=rational_rank(types), max_rel_err=float(np.max(np.abs(poly - kappa) / kappa)))

def nearest(R, t, smax):
    w = 64
    while True:
        a = max(0, t - w)
        seg = R >> a & (1 << min(smax, t + w) - a + 1) - 1
        bits = bin(seg)[2:][::-1]
        below = [i for i in range(min(t - a, len(bits) - 1), -1, -1) if bits[i] == '1']
        above = [i for i in range(t - a, len(bits)) if bits[i] == '1']
        if below and above:
            return (a + below[0], a + above[0])
        if w > smax:
            return (a + below[0] if below else t, a + above[0] if above else t)
        w *= 4

def bitset_dp(w, U):
    R = 1
    for wj in w:
        k, rem = (1, U)
        while rem > 0:
            take = min(k, rem)
            R |= R << wj * take
            rem -= take
            k *= 2
    return R

def dp_row(wr, U, unit_mass, base_mass):
    t0 = time.perf_counter()
    R = bitset_dp(wr, U)
    dt = time.perf_counter() - t0
    smax = U * sum(wr)
    levels = R.bit_count()
    mid = smax // 2
    lo, hi = nearest(R, mid, smax)
    assert lo == hi == mid, 'the centre is reachable by construction (U even)'
    win = max(1, smax // 100)
    seg = R >> max(0, mid - win) & (1 << 2 * win + 1) - 1
    holes_win = 2 * win + 1 - seg.bit_count()
    return dict(range_states=smax + 1, reachable=levels, holes=smax + 1 - levels, hole_frac=(smax + 1 - levels) / (smax + 1), target=base_mass + mid * unit_mass, target_gap=(hi - lo) * unit_mass, holes_near_target=holes_win, window_states=2 * win + 1, window_hole_frac=holes_win / (2 * win + 1), dp_time_s=dt, bitset_bytes=(R.bit_length() + 7) // 8, flag='ok' if holes_win == 0 else 'warn')

def part_b():
    rows, exact_rows, struct = ([], [], {})
    cases = [('N=16', 12, 3, 4), ('N=36', 12, 2, 6)]
    for label, F, s, r in cases:
        A_ref, pts = masses(F, s, r, 'reflect')
        A_tr, _ = masses(F, s, r, 'truncate')
        kref = float(A_ref.sum(axis=0)[0])
        pts_i = [(int(a), int(b)) for a, b in pts]
        scen = {'truncate': A_tr.sum(axis=0), 'efficiency −5%': (A_ref * efficiency(pts, F, 0.05)[None, :]).sum(axis=0)}
        we, unit = exact_eff_weights(pts_i, F)
        eff_check = float(np.max(np.abs(kref * float(unit) * np.array(we) - scen['efficiency −5%']) / scen['efficiency −5%']))
        ts = trunc_structure(pts_i, F, scen['truncate'])
        struct[label] = dict(truncate=dict(ts, types=[list(v) for v in ts['types']], t='exp(-8/49)'), efficiency=dict(weights=we, unit=f'kappa_ref*{unit}', kappa_ref=kref, gcd_removed=True, max_rel_err=eff_check))
        KV[f'{label}|kappa_ref'] = np.array([kref])
        for sname, kappa in scen.items():
            KV[f'{label}|{sname}|kappa'] = kappa
            n = len(kappa)
            for cmax in (3, 5):
                U = cmax - 1
                if sname == 'truncate':
                    assert ts['rank'] == len(ts['types']) and ts['max_rel_err'] < 1e-13
                    exact, basis = (math.prod((c * U + 1 for c in ts['counts'])), f"구조식 Π(n_i(C_max−1)+1): 질량 유형 {len(ts['types'])}개, 계수 벡터 유리수 rank {ts['rank']}, t=e^(−8/49) 초월수")
                else:
                    assert eff_check < 1e-13
                    ex = dp_row(we, U, kref * float(unit), float(kappa.sum()))
                    exact, basis = (ex['reachable'], f'정확 정수 가중치 DP: κ_j = κ_ref·({unit})·w_j, w 범위 {min(we)}–{max(we)}')
                    exact_rows.append(dict(case=label, scenario=sname, N=n, Cmax=cmax, model='정확 스케일링 κ = g·w', q=None, gcd=None, weight_min=min(we), weight_max=max(we), err_bound_mass=0.0, **ex))
                merged = None
                alph, counts = np.unique(np.round(kappa, 12), return_counts=True)
                if min(cmax ** n, float(np.prod([c * U + 1 for c in counts]))) <= 5000000.0:
                    S = np.array([0.0])
                    for kj in kappa:
                        S = merged_unique((S[:, None] + kj * np.arange(U + 1)[None, :]).ravel())
                    merged = int(S.size)
                    assert merged == exact, (label, sname, cmax, merged, exact)
                for q in (0.01, 0.001, 0.0001):
                    w = np.rint(kappa / q).astype(np.int64)
                    g = reduce(math.gcd, (int(v) for v in w))
                    wr = [int(v // g) for v in w]
                    KV[f'{label}|{sname}|q{q:g}|w_over_gcd'] = np.array(wr, dtype=np.int64)
                    row = dict(case=label, scenario=sname, N=n, Cmax=cmax, model=f'근사 양자화 q={q:g}', distinct_masses=int(len(alph)), q=q, gcd=g, weight_min=min(wr), weight_max=max(wr), err_bound_mass=n * U * q / 2)
                    row.update(dp_row(wr, U, g * q, float(kappa.sum())))
                    row.update(exact_levels=exact, exact_basis=basis, merged_levels_1e12=merged, quantised_vs_exact=row['reachable'] / exact)
                    rows.append(row)
    return (rows, exact_rows, struct)

def main():
    threads = determinism.assert_single_thread()
    OUT.mkdir(parents=True, exist_ok=True)
    progress.task('8.1', 'running', 'A. 여섯 시나리오의 열 질량 κ_j 계산')
    rk, rmu, kref, files = part_a()
    progress.task('8.1', 'done', '시나리오: ' + ', '.join((f"{r['scenario']} κ {r['kappa_min']:.4f}–{r['kappa_max']:.4f} (고유값 {r['distinct']})" for r in rk)))
    tr = next((r for r in rk if r['scenario'] == 'truncate'))
    wc = {(r['scenario'], r['B']): r for r in rmu if r['worst_case']}
    env = wc['truncate', 28348]
    progress.task('8.2', 'done', '같은 B에서 평균 범위 폭(B=28348): ' + ', '.join((f"{sc} {r['width']:.3g}" for (sc, B), r in wc.items() if B == 28348)) + f". truncation 원고 Table 4 대조: κ {tr['kappa_min']:.6f}–{tr['kappa_max']:.6f}, 고유 {tr['distinct']}, 평균 범위 {env['mean_min']:.6f}–{env['mean_max']:.6f}")
    conc = [r for r in rmu if not r['worst_case'] and r['scenario'] != 'reflect']
    scs = list(dict.fromkeys((r['scenario'] for r in conc)))
    progress.task('8.3', 'done', '시나리오 평균 κ를 쓴 예산 기반 평균 예측의 |오차|, 시험한 배분 5개의 최대 / B=28348 전체 실행가능 배분의 최악: ' + ', '.join((f"{sc} {max((abs(r['err_kbar']) for r in conc if r['scenario'] == sc)):.2g} / {wc[sc, 28348]['err_kbar']:.4g}" for sc in scs)) + '. 반사 경계 κ_ref=17.638을 가정하면 시험한 배분에서도 truncation 9.73, 효율 −20% 13.9. 균일 손실은 자기 κ로 예측하면 모든 배분에서 기계 정밀도 — 예산 지표 유지')
    progress.task('8.4', 'running', 'B. bounded DP: N=16·36, C_max 3·5')
    rb, rx, struct = part_b()
    cost = load_json(subprocess.run([sys.executable, __file__, '--measure-part-b'], check=True, capture_output=True, text=True).stdout.splitlines()[-1])
    meta = dict(part_b_wall_s=cost['wall_s'], vm_hwm_before_part_b_mib=cost['hwm_before'], vm_hwm_after_part_b_mib=cost['hwm_after'], cost_scope='dp_time_s = bitset_dp() call only; bitset_bytes = final bitset minimal binary size (not peak memory); part_b_wall_s and VmHWM (after imports -> after part B) come from one fresh process running part B alone, a diagnostic run, not a repeated benchmark', quantised_model='m~(y) = sum_j kappa_j + g q sum_j w_j y_j, w = rint(kappa/q)/g (ties to even); one-shot baseline exact, surplus shots quantised; |m(x) - m~(y)| <= N (C_max - 1) q / 2', target='centre of the DP state range, reachable by construction (C_max - 1 even, y_j = (C_max - 1)/2); window = +-1% of the whole DP state range, not of the target value', structure=struct, threads=threads)
    progress.task('8.4', 'done', 'N=16(F=12, s=3), N=36(F=12, s=2) × C_max 3·5 × 시나리오 truncate·효율 −5% × q 3종 + 효율 −5% 정확 스케일링')
    progress.task('8.5', 'done', '양자화 모델 m̃(y) = Σκ_j + g·q·Σw_j y_j: 1샷 기준 질량은 원래 값, 잉여 샷 y_j = x_j − 1 ∈ [0, C_max−1]만 w_j = rint(κ_j/q)/g(ties-to-even)로 양자화, 오차 |m − m̃| ≤ N(C_max−1)q/2. q ∈ {1e-2, 1e-3, 1e-4}. target = DP 범위 중앙(U 짝수라 y_j = U/2로 설계상 도달), 창 = 전체 DP 범위의 ±1%')

    def near(case, sc, cm, q):
        r = next((x for x in rb if x['case'] == case and x['scenario'] == sc and (x['Cmax'] == cm) and (x['q'] == q)))
        return f"{r['holes_near_target']}/{r['window_states']}"
    allr = rb + rx
    progress.task('8.6', 'done', f"중앙 ±1% 창의 hole: truncation N=36은 모든 q·C_max에서 0, 효율 −5% N=36 C_max=3은 q=1e-2 {near('N=36', 'efficiency −5%', 3, 0.01)}, q=1e-4 {near('N=36', 'efficiency −5%', 3, 0.0001)}. 비용: DP 핵심 연산(bitset_dp 호출) 최대 {max((r['dp_time_s'] for r in allr)):.3g}s, 최종 bitset 저장량 최대 {max((r['bitset_bytes'] for r in allr)) / 1000000.0:.3g} MB(peak 메모리 아님). 진단 1회: part B 전체 {meta['part_b_wall_s']:.3g}s, 새 프로세스 VmHWM import 후 {meta['vm_hwm_before_part_b_mib']:.0f} → {meta['vm_hwm_after_part_b_mib']:.0f} MiB")
    ex = {(r['case'], r['scenario'], r['Cmax']): r['exact_levels'] for r in rb}
    progress.task('8.7', 'done', f"비양자화 모델의 정확 수준 수(샷은 정수) — 효율 −5%(정확 정수 가중치 DP) N=16: {ex['N=16', 'efficiency −5%', 3]:,}/{ex['N=16', 'efficiency −5%', 5]:,}, N=36: {ex['N=36', 'efficiency −5%', 3]:,}/{ex['N=36', 'efficiency −5%', 5]:,}(C_max 3/5); truncation(구조식) N=16: {ex['N=16', 'truncate', 3]:,}/{ex['N=16', 'truncate', 5]:,}, N=36: {ex['N=36', 'truncate', 3]:,}/{ex['N=36', 'truncate', 5]:,}. N=16은 허용오차 병합 열거와 일치. 근사 양자화/정확 비: truncation {min((r['quantised_vs_exact'] for r in rb if r['scenario'] == 'truncate')):.2g}–{max((r['quantised_vs_exact'] for r in rb if r['scenario'] == 'truncate')):.2g}(병합), 효율 −5% {min((r['quantised_vs_exact'] for r in rb if r['scenario'] != 'truncate')):.3g}–{max((r['quantised_vs_exact'] for r in rb if r['scenario'] != 'truncate')):.3g}(분할)")
    progress.task('8.8', 'done', f"dp_time_s = bitset_dp 호출만, bitset_bytes = 최종 bitset 최소 이진 크기. 전체 처리 {meta['part_b_wall_s']:.3g}s·VmHWM {meta['vm_hwm_after_part_b_mib']:.0f} MiB는 별도 필드(mass_dp_meta.json), 반복 benchmark 아님")
    progress.task('8.9', 'done', '허용오차 병합 수준 수(merged_levels_1e12, N=16만)와 정확 수준 수(exact_levels + 근거 exact_basis)를 분리. 효율은 정확 정수 가중치 DP, truncation은 계수 벡터 유리수 rank(=유형 수)와 곱셈식')
    progress.task('8.10', 'done', '원고 Prop.의 κ = g·w(commensurate)·integer-scaled 가정은 그대로 둔다. 효율 −5%는 정확히 κ = g·w → 정확 스케일링 DP 행 추가(원고 명제의 직접 예제); truncation은 유형이 유리수 독립이라 commensurate가 아니며, 임의 q 양자화는 다른 질량 모델')
    progress.task('8.11', 'done', '기준 질량 원점(1샷은 원래 κ, 잉여만 양자화), np.rint ties-to-even, 오차 상계 N(C_max−1)q/2, 중앙 target의 설계상 도달(assert), 창 = 전체 DP 범위 ±1%를 표 주석·메타에 명시')
    progress.task('8.12', 'done', "평균 표에 '전체 실행가능 배분' 최악 오차 행 추가(LP 극값 기준). B=28348 시나리오 평균 κ: " + ', '.join((f"{sc} {wc[sc, 28348]['err_kbar']:.4g}" for sc in scs)) + ' — 시험한 배분 5개의 최대와 구분')
    np.savez_compressed(OUT / 'kappa_vectors.npz', **{k.replace('|', '__'): v for k, v in KV.items()})
    (OUT / 'mass_structure.json').write_text(json.dumps(dict(kappa=rk, means=rmu, kappa_ref=kref, allocations=files, threads=threads), indent=1, default=float, ensure_ascii=False))
    (OUT / 'mass_dp.json').write_text(json.dumps(rb, indent=1, default=float, ensure_ascii=False))
    (OUT / 'mass_dp_exact.json').write_text(json.dumps(rx, indent=1, default=float, ensure_ascii=False))
    (OUT / 'mass_dp_meta.json').write_text(json.dumps(meta, indent=1, default=float, ensure_ascii=False))
    publish(rk, rmu, rb, rx, meta)

def publish(rk, rmu, rb, rx, meta):
    kc = [('scenario', '시나리오'), ('kappa_min', 'κ 최소'), ('kappa_max', 'κ 최대'), ('kappa_mean', 'κ 평균'), ('rel_spread', '상대 범위'), ('distinct', '고유값 수'), ('common_mass', '공통 질량'), ('flag', '판정')]
    progress.result('s8_kappa', 'S8 · 열 질량', dict(columns=[dict(key=k, label=l) for k, l in kc], rows=rk, note='F=50, s=2, 7×7, σ=1.75. 효율 시나리오 e_j = 1 − a(d_j/d_max)², d_j = 중심까지 거리. 균일 손실은 공통 질량을 유지(κ만 0.8배). truncation 값은 원고 Table 4와 대조. κ 벡터: results/stage8/kappa_vectors.npz'))
    mc = [('scenario', '시나리오'), ('allocation', '할당/항목'), ('B', 'B'), ('mean_min', '평균 최소'), ('mean_max', '평균 최대'), ('width', '평균 범위 폭'), ('mean_actual', '실제 평균'), ('pred_common_kref', '예측(κ_ref)'), ('err_kref', '오차(κ_ref)'), ('pred_common_kbar', '예측(κ 평균)'), ('err_kbar', '오차(κ 평균)'), ('flag', '판정')]
    progress.result('s8_means', 'S8 · 같은 예산의 평균', dict(columns=[dict(key=k, label=l) for k, l in mc], rows=rmu, note="'평균 범위 (전체 실행가능 배분)' 행: 같은 B에서 1 ≤ x ≤ 200을 지키며 잉여 샷을 가장 작은/큰 κ_j부터 채운 극단값(LP로 독립 확인). 이 행의 오차는 모든 실행가능 배분에 대한 최악 |실제−예측|. 나머지 행은 시험한 배분 5개의 부호 있는 오차(배분 파일·sha256은 mass_structure.json의 allocations). 예측(κ_ref)은 반사 경계의 공통 질량 17.638, 예측(κ 평균)은 시나리오의 평균 κ 사용. 판정 warn = 공통 질량 가정의 예측이 실제와 다름."))
    dc = [('case', '사례'), ('scenario', '시나리오'), ('Cmax', 'C_max'), ('model', '질량 모델'), ('gcd', 'gcd'), ('weight_min', 'w 최소'), ('weight_max', 'w 최대'), ('err_bound_mass', '질량 오차 상계'), ('range_states', 'DP 범위 상태 수'), ('reachable', '도달 수준 수'), ('holes', 'hole 수'), ('hole_frac', 'hole 비율'), ('holes_near_target', '중앙 ±1% 창 hole'), ('window_states', '창 상태 수'), ('window_hole_frac', '창 hole 비율'), ('dp_time_s', 'DP 핵심 시간 (s)'), ('bitset_bytes', '최종 bitset (bytes)'), ('exact_levels', '정확 수준 수'), ('merged_levels_1e12', '병합 수준 수(상대 1e-12)'), ('quantised_vs_exact', '양자화/정확 수준 비'), ('flag', '판정')]
    progress.result('s8_dp', 'S8 · bounded DP', dict(columns=[dict(key=k, label=l) for k, l in dc], rows=rx + rb, note=f"정확 스케일링 행: 효율 −5%의 질량은 κ_j = κ_ref(4840 − D_j)/4840(D_j 정수)로 정확히 κ = g·w — 원고 DP 명제의 직접 예제. 근사 양자화 행: m̃(y) = Σκ_j + g·q·Σw_j y_j(1샷 기준 질량은 원래 값, 잉여 샷만 양자화, rint ties-to-even), 오차 상계 N(C_max−1)q/2. 결과는 q에 조건부이며 원래 모델과 다른 질량 모델: 가까운 수준을 합치기도(비 < 1) 하고 같은 질량을 쪼개기도(비 > 1) 한다. 정확 수준 수: 효율은 정확 정수 DP, truncation은 질량 유형의 계수 벡터가 유리수 독립(t = e^(−8/49) 초월수)이라 Π(n_i(C_max−1)+1). 병합 수준 수는 N=16에서만 계산한 교차확인. target = DP 범위 중앙(설계상 도달), 창 = 전체 DP 범위의 ±1%. 비용: DP 핵심 시간 = bitset_dp 호출만, 최종 bitset = 저장 크기(peak 메모리 아님); 진단 1회 part B 전체 {meta['part_b_wall_s']:.2g}s, VmHWM {meta['vm_hwm_after_part_b_mib']:.0f} MiB. 판정 warn = 중앙 ±1% 창에 hole."))

def measure_part_b():
    hwm0, t0 = (vm_hwm_mib(), time.perf_counter())
    part_b()
    print(json.dumps(dict(wall_s=time.perf_counter() - t0, hwm_before=hwm0, hwm_after=vm_hwm_mib())))
if __name__ == '__main__':
    if '--measure-part-b' in sys.argv:
        measure_part_b()
    else:
        main()
