import argparse
import json
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'results' / 'sensitivity_extension_v2'
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports' / 'sensitivity_tables')
OUT = parser.parse_args().output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
rows = json.loads((V2 / 'rows.json').read_text())
assert len(rows) == 156 and (not any((r['failed'] for r in rows)))
verif = json.loads((V2 / 'verification.json').read_text())
assert verif['passed'] == 156 and verif['failed'] == 0
mesh_cmp = json.loads((V2 / 'mesh_comparison.json').read_text())
FIELD = {'zero': 'Zero', 'random': 'Random', 'center30': 'Center-30', 'center100': 'Center-100'}
FAMILY = {'uniform': 'Uniform$[1,30]$', 'beta_half': 'Beta$(\\tfrac12,\\tfrac12)$, scaled', 'trunc_normal': 'Truncated normal', 'correlated_uniform': 'Correlated uniform'}
FAMILY_SHORT = {'uniform': 'Uniform', 'beta_half': 'Beta$(\\tfrac12,\\tfrac12)$', 'trunc_normal': 'Trunc.\\ normal', 'correlated_uniform': 'Correlated'}
TARGETS = [50, 100, 150, 200, 300, 400]

def frac(s):
    return Fraction(s)

def rnd(v, places, mode):
    q = Decimal(1).scaleb(-places)
    if isinstance(v, Fraction):
        d = Decimal(v.numerator) / Decimal(v.denominator)
    else:
        d = Decimal(repr(float(v)))
    return str(d.quantize(q, rounding=mode))

def up(v, places=4):
    return rnd(v, places, ROUND_CEILING)

def down(v, places=4):
    return rnd(v, places, ROUND_FLOOR)

def near(v, places=4):
    return rnd(v, places, ROUND_HALF_EVEN)

def width_pct(r):
    U, L = (frac(r['U_exact']), frac(r['L_exact']))
    return (U - L) / U * 100

def abs_width(r):
    return frac(r['U_exact']) - frac(r['L_exact'])
check = {}
tgt = {(r['field'], r['T']): r for r in rows if r['group'] == 'target'}
assert len(tgt) == 24
bn = {T: (min((tgt[f, T]['BN'] for f in FIELD)), max((tgt[f, T]['BN'] for f in FIELD))) for T in TARGETS}
main_A = []
for f, name in FIELD.items():
    main_A.append(f'{name:11s} & ' + ' & '.join((up(width_pct(tgt[f, T])) for T in TARGETS)) + '\\\\')
main_A_bn = '$B/N$ (range) & ' + ' & '.join((f'{bn[T][0]:.1f}--{bn[T][1]:.1f}' for T in TARGETS)) + '\\\\'
wt = [width_pct(r) for r in tgt.values()]
check['target'] = dict(width_min_up=up(min(wt)), width_max_up=up(max(wt)), width_max_T_ge_150_up=up(max((width_pct(tgt[f, T]) for f in FIELD for T in TARGETS if T >= 150))), abs_width_min_down=down(min((abs_width(r) for r in tgt.values())), 3), abs_width_max_up=up(max((abs_width(r) for r in tgt.values())), 3), zero_T50=dict(width_up=up(width_pct(tgt['zero', 50])), U=near(frac(tgt['zero', 50]['U_exact']), 3), abs_up=up(abs_width(tgt['zero', 50]), 4), BN=round(tgt['zero', 50]['BN'], 2)), center30_T50=dict(width_up=up(width_pct(tgt['center30', 50])), BN=round(tgt['center30', 50]['BN'], 2)), center100_T50=dict(width_up=up(width_pct(tgt['center100', 50])), U=near(frac(tgt['center100', 50]['U_exact']), 2), abs_up=up(abs_width(tgt['center100', 50]), 4)), cap_T400={f: (tgt[f, 400]['allocation_cap_active'], tgt[f, 400]['upper_active']) for f in FIELD}, budget_clipped=any((r['budget_clipped'] for r in tgt.values())), max_target_error=max((abs(r['target_error']) for r in tgt.values())), BN_range=(min((r['BN'] for r in tgt.values())), max((r['BN'] for r in tgt.values()))))
S13 = ['\\begin{table}[!tbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}', '\\caption{Target-thickness series: the four benchmark fields at $T\\in\\{50,100,150,200,300,400\\}$', 'with the benchmark kernel and $C_{\\max}=200$, three-candidate rule, exact-arithmetic verification.', '$B^\\star$ is the target budget, $\\overline V$ the exact variance of the selected allocation', '(rounded up), $\\underline V$ the verified lower bound (rounded down), and the absolute and', 'relative widths are computed from the exact values and rounded up. The footnote gives the number', 'of waypoints at $C_{\\max}$ in the selected integer allocation and at the reference point of the', 'relaxation for the two cases in which the cap was active; no budget was clipped.}', '\\label{tab:target-series-si}', '\\begin{tabular}{lrrrrrrr}', '\\toprule', 'Field & $T$ & $B^\\star$ & $B/N$ & $\\overline V$ & $\\underline V$ & $\\overline V-\\underline V$ & Width (\\%) \\\\', '\\midrule']
for f, name in FIELD.items():
    for T in TARGETS:
        r = tgt[f, T]
        cap = f"{r['allocation_cap_active']} ({r['upper_active']})" if r['allocation_cap_active'] or r['upper_active'] else '0'
        S13.append(f"{name} & {T} & {r['B']} & {r['BN']:.1f} & {up(frac(r['U_exact']))} & {down(frac(r['L_exact']))} & {up(abs_width(r))} & {up(width_pct(r))}{('' if cap == '0' else ' $^{c}$')}\\\\")
    if f != 'center100':
        S13.append('\\addlinespace[2pt]')
S13 += ['\\bottomrule', '\\end{tabular}', '', '\\smallskip\\footnotesize $^{c}$ Cap contacts: Random $T=400$: $3$ waypoints in the integer allocation, $0$ at the relaxation reference; Center-100 $T=400$: $11$ and $5$.', '\\end{table}']
assert check['target']['cap_T400'] == {'zero': (0, 0), 'random': (3, 0), 'center30': (0, 0), 'center100': (11, 5)}
dist = {fam: [r for r in rows if r['group'] == 'distribution' and r['field'] == fam] for fam in FAMILY}
assert all((len(v) == 30 for v in dist.values()))

def quart(vals, places, outward=True):
    arr = np.array([float(v) for v in vals])
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    exact_sorted = sorted(vals)
    lo, hi = (exact_sorted[0], exact_sorted[-1])
    if outward:
        return (down(lo, places), near(q1, places), near(med, places), near(q3, places), up(hi, places))
    return (near(lo, places), near(q1, places), near(med, places), near(q3, places), near(hi, places))
main_B, S14 = ([], [])
check['distribution'] = {}
for fam, name in FAMILY.items():
    rs = dist[fam]
    w = [width_pct(r) for r in rs]
    aw = [abs_width(r) for r in rs]
    U = [frac(r['U_exact']) for r in rs]
    sd = [r['sd_initial'] for r in rs]
    cor = [r['initial_neighbor_corr'] for r in rs]
    BN = [r['BN'] for r in rs]
    qw = quart(w, 4)
    qaw = quart(aw, 4)
    qU = quart(U, 1, outward=False)
    corr_mean = float(np.mean(cor))
    corr_txt = f'{abs(corr_mean):.2f}' if abs(corr_mean) < 0.005 else f'{corr_mean:.2f}'
    main_B.append(f'{FAMILY_SHORT[fam]} & {np.mean(sd):.2f} & {corr_txt} & {qU[2]} & {qw[2]} & {qw[0]}--{qw[4]} & {qaw[4]}\\\\')
    S14.append(f"{name} & width (\\%) & {' & '.join(qw)}\\\\")
    S14.append(f" & $\\overline V-\\underline V$ & {' & '.join(qaw)}\\\\")
    S14.append(f" & $\\overline V$ & {' & '.join(qU)}\\\\")
    S14.append(f" & initial s.d. & {' & '.join(quart(sd, 2, outward=False))}\\\\")
    S14.append(f" & neighbor corr. & {' & '.join(quart(cor, 3, outward=False))}\\\\")
    S14.append(f" & $B/N$ & {' & '.join(quart(BN, 2, outward=False))}\\\\")
    S14.append('\\addlinespace[3pt]')
    check['distribution'][fam] = dict(width_med=qw[2], width_min_down=qw[0], width_max_up=qw[4], U_med=qU[2], sd_mean=round(float(np.mean(sd)), 2), corr_mean=round(float(np.mean(cor)), 3), corr_min=round(min(cor), 3), corr_max=round(max(cor), 3), extra_reference=sum((r['extra_reference'] for r in rs)), mean_initial_range=(round(min((r['mean_initial'] for r in rs)), 2), round(max((r['mean_initial'] for r in rs)), 2)))
S14.pop()
allw = [width_pct(r) for fam in FAMILY for r in dist[fam]]
allaw = [abs_width(r) for fam in FAMILY for r in dist[fam]]
check['distribution']['all'] = dict(width_min_down=down(min(allw)), width_max_up=up(max(allw)), abs_min_down=down(min(allaw), 3), abs_max_up=up(max(allaw), 3), mean_final_range=(min((r['mean_final'] for fam in FAMILY for r in dist[fam])), max((r['mean_final'] for fam in FAMILY for r in dist[fam]))))
S14 = ['\\begin{table}[!tbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}', '\\caption{Initial-field distribution series: $30$ independent $50\\times50$ fields per family at $T=200$', '(benchmark kernel, $C_{\\max}=200$, three-candidate rule, exact-arithmetic verification). All families have', 'cell values in $[1,30]$ and population mean $15.5$ but are not variance matched. Sample minima of widths', 'are rounded down, sample maxima up, quartiles and medians to nearest. The neighbor correlation is the', 'sample correlation between horizontally adjacent cells (no wrap). These are descriptive statistics;', 'no percentile guarantee is derived from this series.}', '\\label{tab:distribution-series-si}', '\\begin{tabular}{llrrrrr}', '\\toprule', 'Family & Quantity & Min & Q1 & Median & Q3 & Max\\\\', '\\midrule'] + S14 + ['\\bottomrule', '\\end{tabular}', '\\end{table}']
mesh = {(r['field'], r['F']): r for r in rows if r['group'] == 'mesh'}
mc = {(m['field'], m['F']): m for m in mesh_cmp}
assert len(mesh) == 12 and len(mc) == 12
main_C = []
for f, name in FIELD.items():
    main_C.append(f'{name:11s} & ' + ' & '.join((up(width_pct(mesh[f, F])) for F in (50, 100, 200))) + ' & ' + ' & '.join((near(mc[f, F]['own_allocation_quadrature_error_pct'], 2) for F in (50, 100, 200))) + '\\\\')
S15 = ['\\begin{table}[!tbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}', '\\caption{Grid-resolution series on the cell-average Gaussian model defined in this section:', 'the same $50\\times50$ domain, $25\\times25$ waypoints, per-shot mass and piecewise-constant initial', 'field discretized with cells of side $h$. Each grid is a separate discrete problem with its own verified', 'certificate ($\\underline V$ rounded down, $\\overline V$ rounded up, width computed from the exact values and', 'rounded up). The remaining columns are floating-point analytic-integral evaluations, rounded to nearest,', 'with $V_h(\\cdot)$ the grid objective and $V_c(\\cdot)$ the continuous variance: $V_c(\\mathbf{x}_h)$ is', 'the continuous variance of the allocation $\\mathbf{x}_h$ selected on grid $h$;', '$\\delta_{\\mathrm{own}}=1-V_h(\\mathbf{x}_h)/V_c(\\mathbf{x}_h)$ is the quadrature error of the grid', 'objective for its own allocation, relative to the continuous value;', '$\\delta_{\\mathrm{fine}}=1-V_h(\\mathbf{x}_{0.25})/V_c(\\mathbf{x}_{0.25})$ is the same error for the', 'allocation selected on the finest grid, evaluated on grid $h$; $\\Delta_c=V_c(\\mathbf{x}_h)/V_c(\\mathbf{x}_{0.25})-1$;', '$\\|\\mathbf{x}_h-\\mathbf{x}_{0.25}\\|_1$ is the shot-count distance. All three are shown in percent.}', '\\label{tab:mesh-series-si}', '\\begin{tabular}{lrrrrrrrrrr}', '\\toprule', 'Field & $h$ & $M$ & $B^\\star$ & $\\underline V$ & $\\overline V$ & Width (\\%) & $V_c(\\mathbf{x}_h)$ & $\\delta_{\\mathrm{own}}$ (\\%) & $\\delta_{\\mathrm{fine}}$ (\\%) & $\\Delta_c$ (\\%) / $\\ell_1$\\\\', '\\midrule']
for f, name in FIELD.items():
    for F in (50, 100, 200):
        r, m = (mesh[f, F], mc[f, F])
        assert r['B'] == m['B'] and abs(r['U'] - m['U']) < 1e-09
        S15.append(f"{name} & {50 / F:g} & {r['M']} & {r['B']} & {down(frac(r['L_exact']))} & {up(frac(r['U_exact']))} & {up(width_pct(r))} & {m['continuous_evaluation']:.3f} & {m['own_allocation_quadrature_error_pct']:.2f} & {m['fixed_allocation_quadrature_error_pct']:.2f} & {m['continuous_relative_to_finest_pct']:.2f} / {m['allocation_L1_from_finest']}\\\\")
    if f != 'center100':
        S15.append('\\addlinespace[2pt]')
S15 += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
wm = [width_pct(r) for r in mesh.values()]
own = {F: [mc[f, F]['own_allocation_quadrature_error_pct'] for f in FIELD] for F in (50, 100, 200)}
fine = {F: [mc[f, F]['fixed_allocation_quadrature_error_pct'] for f in FIELD] for F in (50, 100, 200)}
relc = {F: [mc[f, F]['continuous_relative_to_finest_pct'] for f in FIELD] for F in (50, 100, 200)}
ratio = [mc[f, 200]['own_allocation_quadrature_error_pct'] / float(width_pct(mesh[f, 200])) for f in FIELD]
check['mesh'] = dict(width_min_down=down(min(wm)), width_max_up=up(max(wm)), own={F: (round(min(v), 2), round(max(v), 2)) for F, v in own.items()}, fine={F: (round(min(v), 2), round(max(v), 2)) for F, v in fine.items()}, rel_to_finest={F: (round(min(v), 3), round(max(v), 3)) for F, v in relc.items()}, ratio_finest=(round(min(ratio), 1), round(max(ratio), 1)), monotone_in_h={f: [float(width_pct(mesh[f, F])) for F in (50, 100, 200)] for f in FIELD})
fe = [r for r in rows if r['L_float_above_feasible_exact']]
exc = {r['id']: float(Fraction(r['L_float']) - Fraction(r['Vfeas_exact'])) for r in fe}
rel = {r['id']: float((Fraction(r['L_float']) - Fraction(r['Vfeas_exact'])) / Fraction(r['U_exact'])) for r in fe}
check['numerical'] = dict(n_float_exceedance=len(fe), exceedance=exc, relative=rel, max_exceedance=max(exc.values()), max_relative=max(rel.values()), all_zero_field=all((r['field'] == 'zero' for r in fe)), max_enclosure=max((float(r['continuous_interval_width']) for r in rows)), max_relative_enclosure=max((float(r['bound_refine_relative_gap']) for r in rows)), extra_reference_total=sum((r['extra_reference'] for r in rows)), extra_by_group={g: sum((r['extra_reference'] for r in rows if r['group'] == g)) for g in ('target', 'distribution', 'mesh')}, width_change_from_v1_pp_max=max((abs(r['width_pct'] - r['v1_width_pct']) for r in rows)))
main = ['\\begin{table}[!tbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}', '\\caption{Verified relative certificate width $(\\overline V-\\underline V)/\\overline V$ in percent', '(three-candidate rule, exact-arithmetic verification; individual widths and maxima rounded up,', 'minima down, medians to nearest). A: benchmark kernel and fields, six targets. B: $30$ fields per', 'family at $T=200$; s.d.\\ and horizontal adjacent-cell correlation of the initial fields are family', 'means, $\\overline V$ the median exact variance of the selected allocation. C: cell-average Gaussian', 'model with cells of side $h$, each grid certified as its own discrete problem;', '$\\delta_{\\mathrm{own}}=1-V_h(\\mathbf{x}_h)/V_c(\\mathbf{x}_h)$ is the shortfall of the grid objective', '$V_h$ of the selected allocation below its analytic-integral variance $V_c$, a numerical', 'comparison that the certificate does not bound.}', '\\label{tab:sensitivity-ext}', '\\begin{tabular}{lrrrrrr}', '\\toprule', '\\multicolumn{7}{l}{A. Target thickness $T$ (widths in \\%)}\\\\', 'Field & 50 & 100 & 150 & 200 & 300 & 400\\\\', '\\midrule'] + main_A + [main_A_bn, '\\midrule', '\\multicolumn{7}{l}{B. Initial-field distribution, $T=200$, $30$ fields each}\\\\', 'Family & s.d. & corr. & $\\overline V$ med. & width med. & min--max & abs.\\ max\\\\', '\\midrule'] + main_B + ['\\midrule', '\\multicolumn{7}{l}{C. Grid resolution, cell-average model, $T=200$}\\\\', 'Field & $h{=}1$ & $0.5$ & $0.25$ & $\\delta_{\\mathrm{own}}$, $h{=}1$ & $0.5$ & $0.25$\\\\', '\\midrule'] + main_C + ['\\bottomrule', '\\end{tabular}', '\\end{table}']
(OUT / 'table_sensitivity_main.tex').write_text('\n'.join(main) + '\n')
(OUT / 'tableS13_target.tex').write_text('\n'.join(S13) + '\n')
(OUT / 'tableS14_distribution.tex').write_text('\n'.join(S14) + '\n')
(OUT / 'tableS15_mesh.tex').write_text('\n'.join(S15) + '\n')
(OUT / 'sensitivity_tables_check.json').write_text(json.dumps(check, indent=1, default=str))
print(json.dumps(check, indent=1, default=str))
