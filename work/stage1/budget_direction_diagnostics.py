from __future__ import annotations
import csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from convex_qp_tools import dense_A, largest_remainder_round, solve_box_qp_relaxation
from scip_compare import build_stamps, make_initial, make_waypoints, choose_budget

def setup(field, s=2, center_height=100.0, cmax=200):
    rows = (50 - 1) // s + 1
    a = SimpleNamespace(field=field, seed=7, random_low=1.0, random_high=30.0, center_height=center_height, center_size=16, target=200.0, field_size=50, waypoint_rows=rows, waypoint_cols=rows, spray_interval=s, kernel_size=7, sigma_x=1.75, sigma_y=1.75, cmax=cmax, budget=None, initial_field=None)
    pts = make_waypoints(a)
    stamps = build_stamps(a, pts)
    s0 = make_initial(a)
    m = a.field_size ** 2
    n = len(stamps)
    A = dense_A(stamps, m, n)
    return (a, A, s0, m, n, stamps[0].mass)

def column_center(Mx):
    return Mx - Mx.mean(axis=0, keepdims=True)

def main():
    a, A, s0, m, n, mass = setup('zero', 2)
    Acen = column_center(A)
    c = Acen @ np.ones(n) / n
    AP = Acen @ (np.eye(n) - np.ones((n, n)) / n)
    lam_max = float(np.linalg.eigvalsh(A.T @ A / m)[-1])
    yz, *_ = np.linalg.lstsq(AP, c, rcond=None)
    gamma = float((c - AP @ yz) @ (c - AP @ yz) / m)
    print(f'lambda_max(Q) = {lam_max:.5e}')
    print(f'gamma         = {gamma:.5e}')

    def phi(B, field):
        w = column_center(field.reshape(-1, 1)).ravel() + B * c
        y, *_ = np.linalg.lstsq(AP, -w, rcond=None)
        r = w + AP @ y
        return float(r @ r / m)
    rows = []
    cmax_rows = {}
    cmax_csv = Path('cmax_count_ablation.csv')
    if cmax_csv.exists():
        with cmax_csv.open() as fh:
            for row in csv.DictReader(fh):
                cmax_rows[int(row['cmax']), int(row['budget'])] = row
    for fld, ch in [('zero', 100.0), ('center30', 30.0), ('center100', 100.0)]:
        base = 'center' if fld.startswith('center') else 'zero'
        aa, AA, ss0, mm, nn, ms = setup(base, 2, center_height=ch)
        Bst = choose_budget(aa, ss0, ms, nn)
        res = solve_box_qp_relaxation(AA, ss0, Bst, ms, aa.cmax)
        rows.append(dict(case=f'target:{fld}', cmax=aa.cmax, B=Bst, phi=phi(Bst, ss0), V_QP=res.primal_value, cert_excess='', cert_ratio_pct='', dQd='', apriori='', rounding_ratio_pct='', apriori_ratio_pct=''))
    for cmax, B in [(3, 938), (20, 5000), (50, 12500), (200, 28348)]:
        res = solve_box_qp_relaxation(A, s0, B, mass, cmax)
        xr = largest_remainder_round(res.x, B, cmax)
        d = xr - res.x
        dQd = float(A @ d @ (A @ d) / m)
        apriori = lam_max * float(d @ d)
        ph = phi(B, s0)
        cert_excess = ''
        cert_ratio = ''
        if (cmax, B) in cmax_rows:
            row = cmax_rows[cmax, B]
            cert_excess = float(row['var_upper']) - float(row['var_lower'])
            cert_ratio = 100 * cert_excess / ph
        rows.append(dict(case='lowcount:zero', cmax=cmax, B=B, phi=ph, V_QP=res.primal_value, cert_excess=cert_excess, cert_ratio_pct=cert_ratio, dQd=dQd, apriori=apriori, rounding_ratio_pct=100 * dQd / ph, apriori_ratio_pct=100 * apriori / ph))
    with open('budget_direction_diagnostics.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['lambda_max_Q', lam_max, 'gamma', gamma])
        w.writerow(['case', 'cmax', 'B', 'phi', 'V_QP', 'cert_excess', 'cert_ratio_pct', 'rounding_dQd', 'apriori', 'rounding_ratio_pct', 'apriori_ratio_pct'])
        for r in rows:
            w.writerow([r['case'], r['cmax'], r['B'], f"{r['phi']:.6f}", f"{r['V_QP']:.6f}", r.get('cert_excess', '') if r.get('cert_excess', '') == '' else f"{r['cert_excess']:.6f}", r.get('cert_ratio_pct', '') if r.get('cert_ratio_pct', '') == '' else f"{r['cert_ratio_pct']:.2f}", r['dQd'] if r['dQd'] == '' else f"{r['dQd']:.6f}", r['apriori'] if r['apriori'] == '' else f"{r['apriori']:.6f}", r['rounding_ratio_pct'] if r['rounding_ratio_pct'] == '' else f"{r['rounding_ratio_pct']:.2f}", r['apriori_ratio_pct'] if r['apriori_ratio_pct'] == '' else f"{r['apriori_ratio_pct']:.2f}"])
    print('\ncase            cmax     B      phi       V_QP     cert%   round%   apriori%')
    for r in rows:
        cr = r.get('cert_ratio_pct', '')
        rr = r.get('rounding_ratio_pct', '')
        ar = r['apriori_ratio_pct']
        print(f"{r['case']:15s}{r['cmax']:5}{r['B']:7} {r['phi']:9.4f} {r['V_QP']:9.4f}  {('' if cr == '' else f'{cr:7.2f}')}  {('' if rr == '' else f'{rr:7.2f}')}  {('' if ar == '' else f'{ar:8.1f}')}")
    print('\nwrote budget_direction_diagnostics.csv')
if __name__ == '__main__':
    main()
