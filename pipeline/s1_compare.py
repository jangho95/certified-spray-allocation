from __future__ import annotations
from package_paths import load_json
import csv
import json
import re
import sys
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP / 'pipeline'), str(EXP / 'work' / 'stage1')]
import progress
from s1_lib import ARCH, SUPP_TEX, WORK1, compare_csv, compare_display, flag_for, read_rows, summarize, tex_table_rows
OUT = EXP / 'results' / 'stage1'
LOGS = EXP / 'logs' / 'stage1'
INP = EXP / 'inputs'
FIELD_KEYS = {'Zero': 'zero', 'Random': 'random', 'Center-30': 'center30', 'Center-100': 'center100'}

def num_or_none(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def same_order(rec, cell, value):
    from s1_lib import tex_number
    target = tex_number(cell)[0] if isinstance(cell, str) else cell
    ratio = value / target if target else float('inf')
    ok = target == value or (target and 1 / 3 <= ratio <= 3)
    rec['flag'] = 'ok' if ok else 'warn'
    rec['detail'] = f'기계 정밀도 근처 잔차: 크기 수준 비교 (비율 {ratio:.2f}, 기준 1/3–3)'

def disp(table, row, col, cell, value):
    return compare_display(table, row, col, cell, value)

def log_text(tid):
    return (LOGS / f"{tid.replace('.', '_')}.log").read_text(errors='replace')

def c_1_1():
    from types import SimpleNamespace
    from scip_compare import build_stamps, evaluate, load_warm_start, make_waypoints
    g = SimpleNamespace(field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
    st = build_stamps(g, make_waypoints(g))
    rows = []
    for name, inp in [('zero', 'zero_F50.csv'), ('random_1_30_seed7_sync', 'random_u1_30_seed7_F50.csv'), ('center30', 'center30_F50.csv'), ('center100', 'center100_F50.csv')]:
        s0 = np.array([float(t) for t in (INP / inp).read_text().replace('\n', ',').split(',') if t.strip()])
        xn = np.array(load_warm_start(WORK1 / f'{name}_allocation.csv', 625))
        xa = np.array(load_warm_start(ARCH / f'{name}_allocation.csv', 625))
        vn, va = (evaluate(list(xn), s0, st, 50)[1], evaluate(list(xa), s0, st, 50)[1])
        ndiff = int(np.sum(xn != xa))
        fl, d, rel = flag_for(va, vn)
        rows.append(dict(table='1.1 incumbent', row=name, column='할당·V_repair', archived=va, reproduced=vn, abs_diff=d, rel_diff=rel, flag='ok' if ndiff == 0 else fl, basis='아카이브', detail=f'서로 다른 waypoint {ndiff}개, B {int(xa.sum())}→{int(xn.sum())}'))
    return rows

def c_1_2():
    t = {r['case']: r for r in load_json((OUT / 'table5_s8.json').read_text())}
    rows = []
    for cells in tex_table_rows('tab:target-gaps'):
        key = cells[0].split(' ')[0].replace('$', '')
        name = {'Zero': 'Zero', 'Random': 'Random', 'Center-30': 'Center-30', 'Center-100': 'Center-100'}[key]
        r = t[name]
        for col, cell, val in [('B*', cells[1], r['B']), ('V_dual', cells[2], r['L']), ('V_round', cells[3], r['V_round']), ('V_repair', cells[4], r['V_repair']), ('V_UB', cells[5], r['U']), ('Gap at B* (%)', cells[6], r['gap_pct'])]:
            rows.append(disp('Table 5', name, col, cell, val))
    for cells in tex_table_rows('tab:residual-audit-si'):
        case = cells[0].replace('$', '').replace('C_{\\max}', 'Cmax').replace(' ', '')
        case = {'Zero': 'Zero', 'Random': 'Random', 'Center-30': 'Center-30', 'Center-100': 'Center-100'}.get(case, case)
        r = t[case]
        rows.append(dict(table='Table S8', row=case, column='Status', manuscript=cells[2], reproduced=r['status'], flag='ok' if cells[2] == r['status'] else 'fail', basis='원고'))
        for col, cell, val in [('V_QP', cells[3], r['V_QP']), ('V_dual', cells[4], r['L']), ('Prim. res.', cells[5], r['prim_res']), ('Dual res.', cells[6], r['dual_res']), ('Hz=v', cells[7], r['Hz_v'])]:
            rec = disp('Table S8', case, col, cell, val)
            if col in ('Prim. res.', 'Dual res.', 'Hz=v') and rec['flag'] != 'ok':
                same_order(rec, cell, val)
            rows.append(rec)
        fc = f"{('yes' if r['fallback'] else 'no')}, {('yes' if r['clip'] else 'no')}"
        rows.append(dict(table='Table S8', row=case, column='Fallback, clip', manuscript=cells[8], reproduced=fc, flag='ok' if fc == cells[8] else 'fail', basis='원고'))
        rows.append(dict(table='Table S8', row=case, column='Inc. viol.', manuscript=cells[9], reproduced=r['inc_viol'], flag='ok' if float(cells[9]) == r['inc_viol'] else 'fail', basis='원고'))
    return rows

def c_1_3():
    rows = compare_csv('Table 3 (CSV)', WORK1 / 'cmax_count_ablation.csv', ARCH / 'cmax_count_ablation.csv', ['cmax', 'budget'])
    new = {r['cmax']: r for r in read_rows(WORK1 / 'cmax_count_ablation.csv')}
    for cells in tex_table_rows('tab:cmax-ablation'):
        r = new[cells[0]]
        for col, cell, key in [('B/N', cells[2], 'avg_count'), ('V_dual', cells[3], 'var_lower'), ('V_round', cells[4], 'var_round'), ('V_repair', cells[5], 'var_repair'), ('V_UB', cells[6], 'var_upper'), ('Gap (%)', cells[7], 'gap_pct')]:
            rows.append(disp('Table 3', f'Cmax={cells[0]}', col, cell, float(r[key])))
    return rows

def c_1_4():
    txt = log_text('1.4')
    rows = []

    def grab(pattern, cast=float, group=1):
        m = re.search(pattern, txt)
        return cast(m.group(group)) if m else None
    refl = re.search('reflect\\s+min=([0-9.]+) max=([0-9.]+).*?distinct=(\\d+)', txt)
    trunc = re.search('truncate\\s+min=([0-9.]+) max=([0-9.]+).*?relative=([0-9.e+-]+).*?distinct=(\\d+)', txt)
    checks = []
    if trunc:
        checks += [('Distinct column masses (trunc.)', '10', float(trunc.group(4))), ('Relative column-mass spread (trunc.)', '6.48e-1', float(trunc.group(3))), ('Column-mass min (trunc.)', '6.759388', float(trunc.group(1))), ('Column-mass max (trunc.)', '17.638022', float(trunc.group(2)))]
    if refl:
        checks += [('Distinct column masses (refl.)', '1', float(refl.group(3)))]
    lam = re.findall('(reflect|truncate)\\s+lambda_min\\(2 A\\^T L A / M\\) = ([+-][0-9.e+-]+)', txt)
    for b, v in lam:
        checks.append((f'lambda_min(2A^TLA/M) ({b})', '9.7e-10' if b == 'reflect' else '1.5e-9', float(v)))
    dmu = grab('measured  delta-mu\\s+: ([+-][0-9.e+-]+)')
    if dmu is not None:
        checks.append(('Δμ interior→corner (trunc.)', '-4.351454e-1', dmu))
    pred = grab('predicted k\\(kappa_e-kappa_i\\)/M\\s+: ([+-][0-9.e+-]+)')
    if pred is not None:
        checks.append(('predicted k(κe−κi)/M', '-4.351454e-1', pred))
    env = re.search('truncate\\s+mean in \\[([0-9.]+), ([0-9.]+)\\]', txt)
    if env:
        checks += [('attainable-mean lower (trunc.)', '157.314239', float(env.group(1))), ('attainable-mean upper (trunc.)', '199.786971', float(env.group(2)))]
    for col, cell, val in checks:
        cell_tex = cell
        if 'e' in cell:
            mant, ex = cell.split('e')
            cell_tex = f'${mant}{{\\times}}10^{{{int(ex)}}}$'
        rows.append(disp('Table 4', 'κ 위반', col, cell_tex, val))
    if not rows:
        rows.append(dict(table='Table 4', row='κ 위반', column='(파싱)', flag='fail', basis='원고', detail='로그에서 값을 찾지 못함'))
    return rows

def c_1_5():
    txt = log_text('1.5')
    blocks = re.split('\\n\\$ ', txt)
    rows = []
    col_linf, rel_l2, ratio = ([], [], [])
    for b in blocks:
        head = b.splitlines()[0] if b.splitlines() else ''
        tag = 'truncate' if '--boundary truncate' in head else 'reflect'
        m1 = re.search('max \\|emp - A\\|\\s+: ([0-9.e+-]+)', b)
        m2 = re.search('relative l2 over A\\s+: ([0-9.e+-]+)', b)
        m3 = re.search('worst measured/tolerated ratio: ([0-9.]+)', b)
        mm = re.search('support mismatches\\s+: (\\d+)', b)
        if m1:
            col_linf.append((tag, float(m1.group(1))))
            rel_l2.append((tag, float(m2.group(1)) if m2 else None))
            ratio.append((tag, float(m3.group(1)) if m3 else None))
            rows.append(dict(table='§6.1 audit', row=b.splitlines()[0][:60], column='support mismatches', manuscript='0', reproduced=int(mm.group(1)) if mm else None, flag='ok' if mm and int(mm.group(1)) == 0 else 'fail', basis='원고'))
    ref = [v for t, v in col_linf if t == 'reflect']
    tr = [v for t, v in col_linf if t == 'truncate']
    if ref:
        rows.append(disp('§6.1 audit', 'reflect 3개 초기장', 'max |col discrepancy|', '$1.8{\\times}10^{-14}$', max(ref)))
        rows.append(disp('§6.1 audit', 'reflect 3개 초기장', 'max rel l2', '$3.2{\\times}10^{-15}$', max((v for t, v in rel_l2 if t == 'reflect' and v is not None))))
        rows.append(disp('§6.1 audit', 'reflect 3개 초기장', 'worst measured/allowed', '0.082', max((v for t, v in ratio if t == 'reflect' and v is not None))))
    if tr:
        rows.append(disp('§6.1 audit', 'truncate', 'max |col discrepancy|', '$1.3{\\times}10^{-15}$', max(tr)))
        rows.append(disp('§6.1 audit', 'truncate', 'worst measured/allowed', '0.051', max((v for t, v in ratio if t == 'truncate' and v is not None))))
    for r in rows:
        if r['column'].startswith(('max |col', 'max rel')) and r['flag'] != 'ok':
            same_order(r, r['manuscript'], r['reproduced'])
    return rows

def c_1_6():
    rows = compare_csv('Table 6 (CSV)', WORK1 / 'small_exact_miqp.csv', ARCH / 'small_exact_miqp.csv', ['instance'], skip_cols={'scip_gap'})
    new = {r['instance']: r for r in read_rows(WORK1 / 'small_exact_miqp.csv')}
    for cells in tex_table_rows('tab:exact-miqp'):
        r = new[cells[0].lower()]
        for col, cell, key in [('V_dual', cells[1], 'dual_lower'), ('V_QP*', cells[2], 'v_qp'), ('V_Z*', cells[3], 'v_z'), ('V_UB', cells[4], 'v_ub'), ('Relax. gap', cells[5], 'relax_gap'), ('Inc. subopt.', cells[6], 'incumbent_subopt')]:
            rows.append(disp('Table 6', cells[0], col, cell, float(r[key])))
    return rows

def verify_archived_allocations(prefix_fmt, cases, arch_rows, key_fn):
    from types import SimpleNamespace
    from scip_compare import build_stamps, evaluate, load_warm_start, make_waypoints
    out = {}
    for inst, s, inp in cases:
        cnt = (50 - 1) // s + 1
        g = SimpleNamespace(field_size=50, waypoint_rows=cnt, waypoint_cols=cnt, spray_interval=s, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
        st = build_stamps(g, make_waypoints(g))
        s0 = np.array([float(t) for t in (INP / inp).read_text().replace('\n', ',').split(',') if t.strip()])
        x = load_warm_start(ARCH / prefix_fmt.format(inst=inst, s=s), len(st))
        v = evaluate(x, s0, st, 50)[1]
        va = float(arch_rows[key_fn(inst, s)]['var_repair'])
        out[inst, str(s)] = (v, va, abs(v - va) <= 5e-07)
    return out

def c_1_7():
    rows = compare_csv('Table 7 (CSV)', WORK1 / 'interval_ablation.csv', ARCH / 'interval_ablation.csv', ['instance', 'interval'])
    arch = {(r['instance'], r['interval']): r for r in read_rows(ARCH / 'interval_ablation.csv')}
    inputs = {'zero': 'zero_F50.csv', 'random': 'random_u1_30_seed7_F50.csv', 'center30': 'center30_F50.csv', 'center100': 'center100_F50.csv'}
    ver = verify_archived_allocations('ablation_{inst}_s{s}_allocation.csv', [(i, s, inputs[i]) for i in inputs for s in (1, 2, 3)], arch, lambda i, s: (i, str(s)))
    for (inst, s), (v, va, ok) in ver.items():
        rows.append(dict(table='Table 7 (아카이브 할당 검증)', row=f'instance={inst}, interval={s}', column='V(아카이브 할당)', archived=va, reproduced=v, abs_diff=abs(v - va), flag='ok' if ok else 'fail', basis='아카이브', detail='아카이브 incumbent 할당을 현재 모델(정본 입력)로 재평가'))
    new = {(r['instance'], r['interval']): r for r in read_rows(WORK1 / 'interval_ablation.csv')}
    for cells in tex_table_rows('tab:interval-ablation'):
        r = new[FIELD_KEYS[cells[0]], cells[1]]
        for col, cell, key in [('V_UB', cells[3], 'var_upper'), ('V_dual', cells[4], 'var_lower'), ('Abs. gap', cells[5], 'abs_gap')]:
            rows.append(disp('Table 7', f'{cells[0]} s={cells[1]}', col, cell, float(r[key])))
    for r in rows:
        m = re.search('instance=(\\w+), interval=(\\d)', r['row']) or re.search('^(\\S+) s=(\\d)$', r['row'])
        if not m or r['flag'] == 'ok' or r['table'].startswith('Table 7 (아카이브'):
            continue
        inst = FIELD_KEYS.get(m.group(1), m.group(1))
        if r['column'] in ('var_round',):
            r['flag'] = 'warn'
            r['detail'] = '연속해 반올림 incumbent: s=1 조건수 약 4e11에서 연속해 마지막 자리가 달라 반올림 결과가 바뀜; V_UB는 repair'
        elif ver.get((inst, m.group(2)), (0, 0, False))[2] and r['column'] in ('var_repair', 'var_upper', 'abs_gap', 'gap_pct', 'V_UB', 'Abs. gap'):
            r['flag'] = 'warn'
            r['detail'] = 'repair 휴리스틱 재실행의 빌드 의존 경로 차이; 아카이브 할당은 현재 모델에서 원고 값으로 검증됨'
    return rows

def c_1_8():
    rows = compare_csv('Table S4 (CSV)', WORK1 / 'gaussian_distortion_ablation.csv', ARCH / 'gaussian_distortion_ablation.csv', ['instance', 'kernel'])
    new = {(r['instance'], r['kernel']): r for r in read_rows(WORK1 / 'gaussian_distortion_ablation.csv')}
    kmap = {'iso-7': 'isotropic-7', 'iso-11': 'isotropic-11', 'x7 raw': 'x-long-7-raw', 'y7 raw': 'y-long-7-raw', 'x7 norm': 'x-long-7-norm', 'y7 norm': 'y-long-7-norm', 'x11 raw': 'x-long-11-raw', 'y11 raw': 'y-long-11-raw'}
    for cells in tex_table_rows('tab:kernel-ablation-si'):
        k = kmap[cells[0].strip()]
        for fcell, fname in zip(cells[2:6], ['zero', 'random', 'center30', 'center100']):
            r = new[fname, k]
            u, l = fcell.split('/')
            rows.append(disp('Table S4', f'{cells[0]} {fname}', 'V_UB', u, float(r['var_upper'])))
            rows.append(disp('Table S4', f'{cells[0]} {fname}', 'V_dual', l, float(r['var_lower'])))
        gaps = [float(new[f, k]['gap_pct']) for f in ['zero', 'random', 'center30', 'center100']]
        lo, hi = cells[6].split('--')
        rows.append(disp('Table S4', cells[0], 'gap min (%)', lo, min(gaps)))
        rows.append(disp('Table S4', cells[0], 'gap max (%)', hi, max(gaps)))
    return rows

def c_1_9():
    rows = compare_csv('Table S5 (CSV)', WORK1 / 'random_seed_robustness.csv', ARCH / 'random_seed_robustness.csv', ['seed'])
    new = {r['seed']: r for r in read_rows(WORK1 / 'random_seed_robustness.csv')}
    for cells in tex_table_rows('tab:random-seeds-si'):
        if not cells[0].strip().isdigit():
            continue
        r = new[cells[0].strip()]
        for col, cell, key in [('B*', cells[1], 'budget'), ('V_UB', cells[2], 'variance_ub'), ('V_dual', cells[3], 'variance_lb'), ('Gap (%)', cells[4], 'gap_pct')]:
            rows.append(disp('Table S5', f'seed {cells[0].strip()}', col, cell, float(r[key])))
    g = np.array([float(r['gap_pct']) for r in new.values()])
    rows.append(disp('Table S5', '요약', 'gap mean', '0.174', g.mean()))
    rows.append(disp('Table S5', '요약', 'gap std', '0.016', g.std(ddof=1)))
    rows.append(disp('Table S5', '요약', 'gap max (본문)', '0.206', g.max()))
    return rows

def c_1_10():
    rows = compare_csv('Table S6 (CSV)', WORK1 / 'budget_direction_diagnostics.csv', ARCH / 'budget_direction_diagnostics.csv', ['case', 'cmax', 'B'], skip_first=True)
    first_new = (WORK1 / 'budget_direction_diagnostics.csv').read_text().splitlines()[0].split(',')
    first_arch = (ARCH / 'budget_direction_diagnostics.csv').read_text().splitlines()[0].split(',')
    for i, name in [(1, 'lambda_max_Q'), (3, 'gamma')]:
        fl, d, rel = flag_for(float(first_arch[i]), float(first_new[i]))
        rows.append(dict(table='Table S6 (CSV)', row='상수', column=name, archived=float(first_arch[i]), reproduced=float(first_new[i]), abs_diff=d, rel_diff=rel, flag=fl, basis='아카이브'))
    full = load_json((OUT / 'table_s6_full.json').read_text())
    rows.append(disp('본문 §6.2', '상수', 'lambda_max(Q)', '$3.5512{\\times}10^{-2}$', full['lambda_max_Q']))
    rows.append(disp('본문 §6.2', '상수', 'gamma', '$6.70655{\\times}10^{-8}$', full['gamma']))
    byc = {r['case']: r for r in full['rows']}
    for cells in tex_table_rows('tab:budget-direction-si'):
        case = cells[0].replace('$', '').replace('C_{\\max}', 'Cmax').replace(' ', '')
        case = {'Targetzero': 'Target zero', 'Targetcenter-30': 'Target center-30', 'Targetcenter-100': 'Target center-100'}.get(case, case)
        r = byc[case]
        cols = [('phi', cells[2], 'phi'), ('V_QP*', cells[3], 'V_QP')]
        if cells[4] != '--':
            cols += [('cert/phi', cells[4], 'cert_over_phi'), ('first/phi', cells[5], 'first_over_phi'), ('curv/phi', cells[6], 'curv_over_phi'), ('round/phi', cells[7], 'round_over_phi'), ('spect/phi', cells[8], 'spect_over_phi')]
        for col, cell, key in cols:
            rows.append(disp('Table S6', case, col, cell, r[key]))
    return rows

def c_1_11():
    rows = compare_csv('Table S7 runs (CSV)', WORK1 / 'pso_repeat_runs.csv', ARCH / 'pso_repeat_runs.csv', ['instance', 'run'])
    rows += compare_csv('Table S7 summary (CSV)', WORK1 / 'pso_repeat_summary.csv', ARCH / 'pso_repeat_summary.csv', ['instance'])
    rows += compare_csv('PSO points (CSV)', WORK1 / 'pso_points.csv', ARCH / 'pso_points.csv', ['instance'])
    summ = {r['instance']: r for r in read_rows(WORK1 / 'pso_repeat_summary.csv')}
    t5 = {r['case']: r for r in load_json((OUT / 'table5_s8.json').read_text())} if (OUT / 'table5_s8.json').exists() else {}
    names = {'Zero': 'zero', 'Random': 'random_1_30_seed7', 'Center-30': 'center30', 'Center-100': 'center100'}
    for cells in tex_table_rows('tab:pso-summary-si'):
        s = summ[names[cells[0]]]
        mu, musd = cells[1].replace('$', '').split('\\pm')
        va, vasd = cells[2].replace('$', '').split('\\pm')
        rows += [disp('Table S7', cells[0], 'PSO mean', mu, float(s['mean_mean'])), disp('Table S7', cells[0], 'PSO mean std', musd, float(s['mean_std'])), disp('Table S7', cells[0], 'PSO variance', va, float(s['variance_mean'])), disp('Table S7', cells[0], 'PSO variance std', vasd, float(s['variance_std']))]
        if cells[0] in t5:
            rows += [disp('Table S7', cells[0], 'V_UB', cells[3], t5[cells[0]]['U']), disp('Table S7', cells[0], 'PSO/V_UB', cells[4], float(s['variance_mean']) / t5[cells[0]]['U'])]
    return rows

def s9_rows():
    s = SUPP_TEX.read_text()
    i = s.find('\\label{tab:sweep-full-si}')
    body = s[s.find('\\endlastfoot', i) + len('\\endlastfoot'):s.find('\\end{longtable}', i)]
    return [[c.strip() for c in r.split('&')] for r in body.split('\\\\') if '&' in r]

def c_1_12():
    rows = []
    files = {'Zero': 'zero', 'Random': 'random_1_30_seed7', 'Center-30': 'center30', 'Center-100': 'center100'}
    for label, f in files.items():
        rows += compare_csv(f'Table S9 {label} (CSV)', WORK1 / f'pareto_{f}.csv', ARCH / f'pareto_{f}.csv', ['budget'])
    new = {label: {r['budget']: r for r in read_rows(WORK1 / f'pareto_{f}.csv')} for label, f in files.items()}
    for cells in s9_rows():
        label = cells[0]
        if label not in new or cells[1] not in new[label]:
            rows.append(dict(table='Table S9', row=f'{label} B={cells[1]}', column='(행)', flag='fail', basis='원고'))
            continue
        r = new[label][cells[1]]
        for col, cell, key in [('mu_B', cells[2], 'mean'), ('V_dual', cells[3], 'var_lower'), ('V_round', cells[4], 'var_round'), ('V_repair', cells[5], 'var_repair'), ('V_UB', cells[6], 'var_upper'), ('Gap (%)', cells[7], 'gap')]:
            rows.append(disp('Table S9', f'{label} B={cells[1]}', col, cell, float(r[key])))
    for cells in tex_table_rows('tab:target-gaps'):
        label = cells[0].split(' ')[0]
        g = max((float(r['gap']) for r in new[label].values()))
        rows.append(disp('Table 5', label, 'Max swept gap (%)', cells[7], g))
    return rows
TASKS = {'1.1': c_1_1, '1.2': c_1_2, '1.3': c_1_3, '1.4': c_1_4, '1.5': c_1_5, '1.6': c_1_6, '1.7': c_1_7, '1.8': c_1_8, '1.9': c_1_9, '1.10': c_1_10, '1.11': c_1_11, '1.12': c_1_12}
COLUMNS = [dict(key='table', label='표'), dict(key='row', label='행'), dict(key='column', label='항목'), dict(key='basis', label='기준'), dict(key='archived', label='기존값(아카이브)'), dict(key='manuscript', label='기존값(원고)'), dict(key='reproduced', label='재현값'), dict(key='reproduced_display', label='재현(표시)'), dict(key='abs_diff', label='절대차'), dict(key='rel_diff', label='상대차'), dict(key='flag', label='판정'), dict(key='detail', label='비고')]

def publish_task(tid, rows):
    (OUT / f"compare_{tid.replace('.', '_')}.json").write_text(json.dumps(rows, indent=1, default=float, ensure_ascii=False))
    progress.result(f"s1_{tid.replace('.', '_')}", f'S1 · {tid}', dict(columns=COLUMNS, rows=rows, note='기준=아카이브: JAMC/solver_cpp의 CSV와 전체 정밀도 비교(ok: 상대차 ≤1e-6 또는 절대차 ≤1e-9, 6자리로 저장된 열은 ±5e-7). 기준=원고: 재현값을 원고 표시 자리수로 반올림해 비교(같으면 ok, 마지막 자리 1 차이는 warn).'))

def aggregate():
    causes = load_json((EXP / 'pipeline' / 'stage1_causes.json').read_text()) if (EXP / 'pipeline' / 'stage1_causes.json').exists() else {}
    allrows, per_table = ([], {})
    for tid in TASKS:
        p = OUT / f"compare_{tid.replace('.', '_')}.json"
        if not p.exists():
            continue
        for r in load_json(p.read_text()):
            r['task'] = tid
            for key in (f"{r['table']}|{r['row']}|{r['column']}", f"{r['table']}|{r['column']}", r['table']):
                if key in causes and r.get('flag') != 'ok':
                    r['cause'], r['action'] = (causes[key]['cause'], causes[key]['action'])
                    break
            allrows.append(r)
            t = per_table.setdefault(r['table'], {'table': r['table'], 'task': tid, 'ok': 0, 'warn': 0, 'fail': 0})
            t[r.get('flag', 'ok')] += 1
    with open(OUT / 'comparison_full.csv', 'w', newline='') as fh:
        keys = ['task', 'table', 'row', 'column', 'basis', 'archived', 'manuscript', 'reproduced', 'reproduced_display', 'abs_diff', 'rel_diff', 'flag', 'detail', 'cause', 'action']
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(allrows)
    summary = []
    for t in per_table.values():
        t['flag'] = 'fail' if t['fail'] else 'warn' if t['warn'] else 'ok'
        c = causes.get(t['table'], {})
        t['cause'], t['action'] = (c.get('cause', ''), c.get('action', ''))
        summary.append(t)
    (OUT / 'comparison_summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    progress.result('s1_summary', 'S1 · 표별 요약', dict(columns=[dict(key='task', label='작업'), dict(key='table', label='표'), dict(key='ok', label='ok'), dict(key='warn', label='warn'), dict(key='fail', label='fail'), dict(key='flag', label='판정'), dict(key='cause', label='원인'), dict(key='action', label='수정 여부')], rows=summary, note='전체 행 비교는 results/stage1/comparison_full.csv (기존값/재현값/차이/원인/수정 여부).'))
    flagged = [r for r in allrows if r.get('flag') != 'ok']
    progress.result('s1_flagged', 'S1 · 불일치 행', dict(columns=COLUMNS + [dict(key='cause', label='원인'), dict(key='action', label='수정 여부')], rows=flagged, note='warn·fail 행만 모음.'))
    return allrows
if __name__ == '__main__':
    tid = sys.argv[sys.argv.index('--task') + 1]
    if tid == '1.13':
        rows = aggregate()
        s = summarize(rows)
        s['note'] = f'{len(rows)}개 값 비교'
    else:
        rows = TASKS[tid]()
        publish_task(tid, rows)
        s = summarize(rows)
    print(json.dumps(s, ensure_ascii=False))
