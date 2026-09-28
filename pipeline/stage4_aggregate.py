from package_paths import load_json
import csv
import json
import shutil
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import progress
EXP = Path(__file__).resolve().parents[1]
RAW = EXP / 'results' / 'stage4' / 'raw'
OUT = EXP / 'results' / 'stage4'
DASH = EXP / 'reports' / 'data'
PHASES = [('a_build', 'A 구성'), ('gram', 'Gram/H'), ('setup', 'OSQP setup'), ('solve', 'QP solve'), ('dual', 'dual 평가'), ('round', '반올림'), ('greedy_repair_cpp', 'greedy+repair (C++)'), ('round_repair_cpp', '반올림→repair (C++)'), ('eval', '후보 평가')]

def load(mode):
    return [load_json(p.read_text()) for p in sorted(RAW.glob(f'{mode}_F*_r[1-9]*.json'))]

def med(v):
    return float(np.median(v))

def iqr(v):
    return float(np.percentile(v, 75) - np.percentile(v, 25))

def phase_times(r):
    t = r['times']
    out = {'a_build': t.get('waypoints', 0) + t.get('stamps', 0) + t.get('dense_A', 0)}
    for k, _ in PHASES[1:]:
        out[k] = t.get(k, 0.0)
    out['e2e'] = sum((out[k] for k, _ in PHASES))
    return out

def single_table(runs):
    rows, long = ([], [])
    for F, field in sorted({(r['F'], r['field']) for r in runs}):
        rs = [r for r in runs if r['F'] == F and r['field'] == field]
        pt = [phase_times(r) for r in rs]
        row = dict(F=F, field=field, N=rs[0]['N'], M=rs[0]['M'], B=rs[0]['B_target'], reps=len(rs))
        for k, label in PHASES + [('e2e', '측정 단계 합계')]:
            vals = [p[k] for p in pt]
            row[k] = med(vals)
            long.append(dict(F=F, field=field, phase=label, median_s=med(vals), iqr_s=iqr(vals), min_s=min(vals), max_s=max(vals), reps=len(vals)))
        row['e2e_iqr'] = iqr([p['e2e'] for p in pt])
        row['worker'] = med([r['t_wall_main'] - r['t_import'] for r in rs])
        row['unmeasured'] = row['worker'] - row['e2e']
        row['t_import'] = med([r['t_import'] for r in rs])
        row['rss_py'] = med([r['rss_python_peak'] for r in rs])
        row['rss_cpp_greedy'] = med([r['budget']['cpp_greedy']['rss_mb'] for r in rs])
        row['rss_cpp_rr'] = med([r['budget']['cpp_round_repair']['rss_mb'] for r in rs])
        row['moves_greedy'] = rs[0]['budget']['cpp_greedy']['moves']
        row['moves_rr'] = rs[0]['budget']['cpp_round_repair']['moves']
        row['cpp_gram_ms'] = med([r['budget']['cpp_greedy']['gram_ms'] for r in rs])
        row['dense_arrays_mb'] = (rs[0]['A_bytes'] + 3 * rs[0]['gram_bytes']) / 2 ** 20
        b = rs[0]['budget']
        row['gap_greedy_pct'] = 100 * (min(b['V_greedy'], b['V_round']) - b['L']) / min(b['V_greedy'], b['V_round'])
        u = min(b['V_greedy'], b['V_round'], b['V_round_repair'])
        row['gap_best_pct'] = 100 * (u - b['L']) / u
        row['identical_reps'] = len({(r['budget']['L'], r['budget']['V_greedy'], r['budget']['V_round_repair']) for r in rs}) == 1
        row['flag'] = 'ok' if row['identical_reps'] else 'warn'
        rows.append(row)
    return (rows, long)

def sweep_table(sweeps, reuses):
    rows = []
    for F in sorted({r['F'] for r in sweeps}):
        s = [r for r in sweeps if r['F'] == F]
        u = [r for r in reuses if r['F'] == F]
        qp_impl = [sum((r['times'].get(k, 0) for k in ('gram', 'setup', 'solve', 'dual'))) for r in s]
        qp_reuse = [sum((r['times'].get(k, 0) for k in ('gram', 'setup', 'update', 'solve', 'dual'))) for r in u]
        tot = [sum((phase_times(r)[k] for k, _ in PHASES)) for r in s]
        rep = [r['times'].get('greedy_repair_cpp', 0) + r['times'].get('round_repair_cpp', 0) for r in s]
        exrep = [sum((b['cpp_greedy']['repair_ms'] + b['cpp_round_repair']['repair_ms'] for b in r['budgets'])) / 1000 for r in s]
        worker = [r['t_wall_main'] - r['t_import'] for r in s]
        worker_u = [r['t_wall_main'] - r['t_import'] for r in u]
        Li = np.array([b['L'] for b in s[0]['budgets']])
        Lr = np.array([b['L'] for b in u[0]['budgets']])
        it_i = [b['iters'] for b in s[0]['budgets']]
        it_r = [b['iters'] for b in u[0]['budgets']]
        rows.append(dict(F=F, N=s[0]['N'], budgets=len(s[0]['budgets']), reps=len(s), total_impl_s=med(tot), total_impl_iqr=iqr(tot), worker_impl_s=med(worker), repair_share=med(rep) / med(tot), exchange_share=med(exrep) / med(tot), worker_reuse_s=med(worker_u), qp_impl_s=med(qp_impl), gram_impl_s=med([r['times']['gram'] for r in s]), qp_reuse_s=med(qp_reuse), qp_speedup=med(qp_impl) / med(qp_reuse), L_max_rel_diff=float(np.max(np.abs(Li - Lr) / np.abs(Li))), L_max_abs_diff=float(np.max(np.abs(Li - Lr))), iters_equal=it_i == it_r, rss_py_impl=med([r['rss_python_peak'] for r in s]), rss_py_reuse=med([r['rss_python_peak'] for r in u]), rss_cpp_max=max((max(b['cpp_greedy']['rss_mb'], b['cpp_round_repair']['rss_mb']) for r in s for b in r['budgets'])), flag='ok' if float(np.max(np.abs(Li - Lr) / np.abs(Li))) < 1e-06 else 'warn'))
    return rows

def pso_table(runs):
    rows = []
    for F in sorted({r['F'] for r in runs}):
        rs = [r for r in runs if r['F'] == F]
        rows.append(dict(F=F, N=rs[0]['N'], reps=len(rs), t_model=med([r['t_model'] for r in rs]), t_opt=med([r['t_opt'] for r in rs]), t_total=med([r['t_total'] for r in rs]), t_opt_iqr=iqr([r['t_opt'] for r in rs]), rss_py=med([r['rss_python_peak'] for r in rs]), P_mb=rs[0]['P_bytes'] / 2 ** 20, variance_med=med([r['variance'] for r in rs]), flag='ok'))
    return rows
LIGHT = dict(surface='#fcfcfb', text='#0b0b0b', text2='#52514e', grid='#e6e5e0')
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']

def _style(ax, fig):
    fig.patch.set_facecolor(LIGHT['surface'])
    ax.set_facecolor(LIGHT['surface'])
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
    for sp_ in ('left', 'bottom'):
        ax.spines[sp_].set_color(LIGHT['grid'])
    ax.tick_params(colors=LIGHT['text2'], labelsize=9)
    ax.grid(True, which='major', color=LIGHT['grid'], linewidth=0.8)
    ax.xaxis.label.set_color(LIGHT['text2'])
    ax.yaxis.label.set_color(LIGHT['text2'])

def figures(single_rows):
    plt.rcParams['font.family'] = ['DejaVu Sans']
    z = [r for r in single_rows if r['field'] == 'zero']
    N = [r['N'] for r in z]
    series = [('a_build', 'A build'), ('gram', 'Gram / H'), ('solve', 'QP setup + solve'), ('dual', 'dual bound'), ('greedy_repair_cpp', 'greedy + repair (C++)'), ('round_repair_cpp', 'round → repair (C++)')]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    _style(ax, fig)
    for i, (k, lab) in enumerate(series):
        y = [r[k] + (r['setup'] if k == 'solve' else 0) for r in z]
        ax.plot(N, y, color=SERIES[i], linewidth=2, marker='o', markersize=6, markeredgecolor=LIGHT['surface'], markeredgewidth=1.5, label=lab)
    ax.plot(N, [r['e2e'] for r in z], color=LIGHT['text2'], linewidth=2, linestyle='--', marker='o', markersize=6, label='sum of measured phases (one budget)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(N)
    ax.set_xticklabels([f"{n}\n(F={r['F']})" for n, r in zip(N, z)])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel('waypoints N')
    ax.set_ylabel('median wall time (s)')
    ax.set_title('Stage 4 · time per phase, one target budget, zero field', color=LIGHT['text'], fontsize=11, loc='left')
    leg = ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    for t in leg.get_texts():
        t.set_color(LIGHT['text'])
    fig.tight_layout()
    fig.savefig(OUT / 'stage4_time_vs_N.png', facecolor=fig.get_facecolor())
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    _style(ax, fig)
    for i, (key, lab) in enumerate([('rss_py', 'Python peak RSS'), ('rss_cpp_greedy', 'C++ peak RSS (greedy + repair)'), ('dense_arrays_mb', 'dense A + Gram + H + G (arrays only)')]):
        ax.plot(N, [r[key] for r in z], color=SERIES[i], linewidth=2, marker='o', markersize=6, markeredgecolor=LIGHT['surface'], markeredgewidth=1.5, label=lab)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(N)
    ax.set_xticklabels([f"{n}\n(F={r['F']})" for n, r in zip(N, z)])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel('waypoints N')
    ax.set_ylabel('MiB')
    ax.set_title('Stage 4 · peak memory, one target budget, zero field', color=LIGHT['text'], fontsize=11, loc='left')
    leg = ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    for t in leg.get_texts():
        t.set_color(LIGHT['text'])
    fig.tight_layout()
    fig.savefig(OUT / 'stage4_memory_vs_N.png', facecolor=fig.get_facecolor())
    plt.close(fig)
    DASH.mkdir(parents=True, exist_ok=True)
    for n in ('stage4_time_vs_N.png', 'stage4_memory_vs_N.png'):
        shutil.copyfile(OUT / n, DASH / n)

def main():
    single, long = single_table(load('single'))
    sweeps = sweep_table(load('sweep'), load('sweep_reuse'))
    pso = pso_table(load('pso'))
    machine = load_json((OUT / 'machine.json').read_text())
    (OUT / 'single_summary.json').write_text(json.dumps(single, indent=1))
    (OUT / 'sweep_summary.json').write_text(json.dumps(sweeps, indent=1))
    (OUT / 'pso_summary.json').write_text(json.dumps(pso, indent=1))
    with open(OUT / 'single_phases_long.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(long[0]))
        w.writeheader()
        w.writerows(long)
    figures(single)
    mnote = f"장비: {machine['cpu']}, 논리 CPU {machine['logical_cpus']}, {machine['mem_total']}, {machine['note']}."
    cols = [('F', 'F'), ('field', '초기장'), ('N', 'N'), ('B', 'B'), ('reps', '반복')] + [(k, f'{lab} (s)') for k, lab in PHASES] + [('e2e', '측정 단계 합계 (s)'), ('e2e_iqr', '합계 IQR (s)'), ('worker', 'worker 경과, import 이후 (s)'), ('unmeasured', '합계 밖 시간 (s)'), ('t_import', '시작·import (s)'), ('moves_greedy', 'greedy 이동'), ('moves_rr', '반올림→repair 이동'), ('identical_reps', '반복 간 결과 동일'), ('flag', '판정')]
    progress.result('s4_single', 'S4 · 단계별 시간', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=single, images=[dict(src='data/stage4_time_vs_N.png', alt='단계별 시간과 N (log-log)')], note="값은 새 프로세스 반복의 중앙값(초). '측정 단계 합계'는 A 구성부터 후보 평가까지 표의 단계만 더한 값으로, 입력 로딩·Python CSV 입출력·후처리는 빠진다. 이를 포함한 시간은 'worker 경과, import 이후'. C++ 두 repair는 자식 프로세스 wall time(프로세스 생성·C++ 내부 stamp/Gram 재구성 포함). 반올림→repair가 greedy+repair보다 빠른지는 초기장에 따라 다름(F=100 center-100은 빠르고 zero는 느림). " + mnote))
    mcols = [('F', 'F'), ('field', '초기장'), ('N', 'N'), ('rss_py', 'Python peak RSS (MiB)'), ('rss_cpp_greedy', 'C++ peak RSS greedy+repair (MiB)'), ('rss_cpp_rr', 'C++ peak RSS 반올림→repair (MiB)'), ('dense_arrays_mb', 'dense A+Gram+H+G 이론 (MiB)'), ('cpp_gram_ms', 'C++ Gram 구성 (ms)'), ('gap_greedy_pct', 'gap 원고 규칙 (%)'), ('gap_best_pct', 'gap 후보 추가 (%)'), ('flag', '판정')]
    progress.result('s4_memory', 'S4 · 메모리', dict(columns=[dict(key=k, label=l) for k, l in mcols], rows=single, images=[dict(src='data/stage4_memory_vs_N.png', alt='peak 메모리와 N')], note='Python peak = 워커 프로세스 ru_maxrss. C++ peak = C++가 스스로 보고한 VmHWM(exec 이후). os.wait4의 ru_maxrss는 fork 시 부모 페이지를 포함하므로 C++ 메모리로 쓰지 않았다(raw JSON에 교차 확인값으로만 보존).'))
    scols = [('F', 'F'), ('N', 'N'), ('budgets', '예산 수'), ('reps', '반복'), ('total_impl_s', '측정 단계 합계, 현재 구현 (s)'), ('total_impl_iqr', 'IQR (s)'), ('worker_impl_s', 'worker 경과, 현재 구현 (s)'), ('repair_share', 'C++ 실행 전체 비중'), ('exchange_share', '순수 exchange repair 비중'), ('qp_impl_s', 'QP 단계 합, 현재 구현 (s)'), ('gram_impl_s', '그중 Gram 재계산 (s)'), ('qp_reuse_s', 'QP 단계 합, 재사용 (s)'), ('qp_speedup', 'QP 단계 합 비율'), ('L_max_abs_diff', 'L 최대 절대차'), ('L_max_rel_diff', 'L 최대 상대차'), ('iters_equal', 'OSQP 반복 수 동일'), ('rss_py_impl', 'Python RSS 현재 (MiB)'), ('rss_py_reuse', 'Python RSS 재사용 (MiB)'), ('rss_cpp_max', 'C++ RSS 최대 (MiB)'), ('flag', '판정')]
    progress.result('s4_sweep', 'S4 · sweep 재구성 vs 재사용', dict(columns=[dict(key=k, label=l) for k, l in scols], rows=sweeps, note="현재 구현: A만 한 번 만들고 예산마다 Gram·OSQP setup·dense dual 풀이와 C++ 두 repair(각각 stamp·Gram 재구성)를 다시 수행. 재사용(시험 구현, QP 측만): Gram·OSQP setup·H의 Cholesky를 한 번 하고 예산마다 q·l·u만 갱신(OSQP가 rho를 갱신하면 재인수분해되며 최대 2회 관측). 'QP 단계 합 비율'은 Gram·setup 재사용, dual 풀이 방식 변경(Cholesky), 반복 수 차이가 합쳐진 값이고 두 경로의 타이머 범위가 완전히 같지 않다(재사용 경로의 q·상수 계산은 미포함). C++ 실행 비중은 프로세스 생성·stamp·Gram·greedy 포함, 순수 exchange repair는 C++ 내부 repair_ms 합. 재사용 경로는 Cholesky 인자를 보관해 Python peak가 늘고(F=100: 473→528 MiB), 연속해가 달라져 반올림·repair 결과도 달라진다(S4 · 재사용 경로 정수해) — 하한이 거의 같아도 U를 옮겨 쓰면 안 된다."))
    pcols = [('F', 'F'), ('N', 'N'), ('reps', '반복'), ('t_model', '모델 구성 (s)'), ('t_opt', '최적화 (s)'), ('t_opt_iqr', '최적화 IQR (s)'), ('t_total', '전체 (s)'), ('rss_py', 'Python peak RSS (MiB)'), ('P_mb', 'P=AᵀA (MiB)'), ('variance_med', 'PSO 분산 중앙값'), ('flag', '판정')]
    progress.result('s4_pso', 'S4 · PSO', dict(columns=[dict(key=k, label=l) for k, l in pcols], rows=pso, note='100 particles × 500 iterations, zero field, seed=rep. 원 스크립트의 elapsed_s(최적화)는 Simulator 구성(stamp, A, P=AᵀA) 이후부터만 측정하므로 모델 구성 = 전체 − elapsed_s로 분리.'))
    z = {r['F']: r for r in single if r['field'] == 'zero'}
    c = {r['F']: r for r in single if r['field'] == 'center100'}
    progress.task('4.1', 'done', '측정 단계 합계, zero / center-100: ' + ' · '.join((f"F={F} (N={z[F]['N']}): {z[F]['e2e']:.3g}s / {c[F]['e2e']:.3g}s (greedy+repair {z[F]['greedy_repair_cpp']:.3g}/{c[F]['greedy_repair_cpp']:.3g}s, 반올림→repair {z[F]['round_repair_cpp']:.3g}/{c[F]['round_repair_cpp']:.3g}s)" for F in sorted(z))))
    progress.task('4.2', 'done', '21-budget sweep(현재 구현, zero), 측정 단계 합계 / worker 경과: ' + ' · '.join((f"F={r['F']}: {r['total_impl_s']:.3g}s / {r['worker_impl_s']:.3g}s (C++ 실행 {100 * r['repair_share']:.1f}%, 순수 exchange repair {100 * r['exchange_share']:.1f}%)" for r in sweeps)) + ' · 단일 예산(zero): ' + ' · '.join((f"F={F}: 합계 {z[F]['e2e']:.4g}s / worker {z[F]['worker']:.4g}s" for F in sorted(z))))
    progress.task('4.3', 'done', f"새 프로세스 반복(단일 5회, sweep·PSO 3회) 중앙값·IQR. 반복 간 결과 동일 {sum((r['identical_reps'] for r in single))}/{len(single)}. Python peak RSS " + ', '.join((f"F={F}: {z[F]['rss_py']:.0f} MiB" for F in sorted(z))) + ' · C++ peak RSS ' + ', '.join((f"F={F}: {z[F]['rss_cpp_greedy']:.1f} MiB" for F in sorted(z))))
    progress.task('4.4', 'done', 'PSO: ' + ' · '.join((f"F={r['F']}: 모델 {r['t_model']:.3g}s + 최적화 {r['t_opt']:.3g}s" for r in pso)))
    progress.task('4.5', 'done', 'QP 단계 합(시험 구현 재사용, setup 한 번 후 q·l·u 갱신): ' + ' · '.join((f"F={r['F']}: {r['qp_impl_s']:.3g}s → {r['qp_reuse_s']:.3g}s (비율 {r['qp_speedup']:.1f}, L 최대 절대차 {r['L_max_abs_diff']:.3g}·상대차 {r['L_max_rel_diff']:.3g}, Python peak {r['rss_py_impl']:.0f}→{r['rss_py_reuse']:.0f} MiB)" for r in sweeps)) + '. 비율은 여러 개선이 합쳐진 QP 단계 합의 값. 재사용 경로는 21예산 중 20개에서 반올림이 달라 정수 후보가 바뀜(별도 후보로 등록, 기존 최선 U 보존). 원고의 인수분해 재사용은 현재 코드에 없음')
    progress.log('Stage 4 집계 완료')
if __name__ == '__main__':
    main()
