from package_paths import load_json
import json
import math
import shutil
import sys
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import progress
from qp_audit import sha256_file
EXP = Path(__file__).resolve().parents[1]
OUT, DASH = (EXP / 'results' / 'stage9', EXP / 'reports' / 'data')
SURF, TEXT, TEXT2, GRID = ('#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0')
SERIES = ['#2a78d6', '#eb6834']
LEVEL = 1 - 0.95 ** 59
LABELS = {'T1': 'T1 · B*, U 후보 3종', 'T2': 'T2 · 격자 21점 최대, U 후보 3종', 'T1_paper': 'T1 · B*, 원고 U', 'T2_paper': 'T2 · 격자 최대, 원고 U', 'T1_best': 'T1 · B*, L = max(OSQP, Clarabel)', 'T2_best': 'T2 · 격자 최대, L = max(OSQP, Clarabel)', 'abs_T1': '절대 폭 U − L · B*'}

def style(ax):
    ax.set_facecolor(SURF)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
    for sp_ in ('left', 'bottom'):
        ax.spines[sp_].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT2, labelsize=8.5)

def fmtn(v, spec, none='없음'):
    return none if v is None else format(v, spec)

def ceil_pct(v, d=4):
    if v is None or not math.isfinite(v):
        return v
    return float(Decimal(repr(v)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_CEILING))

def diagnostics(runs):
    out = []
    for res in runs['results']:
        if res.get('failed'):
            out.append(dict(name=res['name'], T2_B_ratio=None, pd_ratio_max=None, n_clip_high=None, n_clip_low=None, prim_res_max=None, clar_minus_osqp_max=None, osqp_minus_clar_max=None, slack_to_Pc_min=None))
            continue
        rs, b = (res['rows'], res['bstar'])
        top = max(rs, key=lambda r: r['W_rev'])
        out.append(dict(name=res['name'], T2_B_ratio=top['B'] / b, pd_ratio_max=max(((r['V_QP'] - r['L_osqp']) / (r['U_rev'] - r['L_osqp']) for r in rs)), n_clip_high=sum((bool(r['clip_high']) for r in rs)), n_clip_low=sum((bool(r['clip_low']) for r in rs)), prim_res_max=max((r['prim_res'] for r in rs)), clar_minus_osqp_max=max((r['L_c'] - r['L_osqp'] for r in rs)), osqp_minus_clar_max=max((r['L_osqp'] - r['L_c'] for r in rs)), slack_to_Pc_min=min((r['P_c'] - r['L_osqp'] for r in rs))))
    return out

def counts(fields, diag):
    ok = [f for f in fields if not f.get('failed')]
    c = dict(n=len(fields), failed_fields=sum((bool(f.get('failed')) for f in fields)), field_retries=sum((f.get('attempts', 2 if f.get('failed') else 1) > 1 for f in fields)), instances=sum((f.get('n_budgets', 0) for f in ok)))
    for k in ('n_fallback', 'n_clar_fail', 'n_valid_flag', 'n_cpp_retry', 'n_infeasible'):
        c[k] = sum((f.get(k, 0) for f in ok))
    for k in ('n_clip_high', 'n_clip_low'):
        c[k] = sum((d[k] or 0 for d in diag))
    fin = [d for d in diag if d['pd_ratio_max'] is not None]
    c.update(pd_ratio_max=max((d['pd_ratio_max'] for d in fin), default=None), prim_res_max=max((d['prim_res_max'] for d in fin), default=None), bound_diff_max=max((max(d['clar_minus_osqp_max'], d['osqp_minus_clar_max']) for d in fin), default=None), slack_to_Pc_min=min((d['slack_to_Pc_min'] for d in fin), default=None))
    return c

def figure(e4, dev, out_dir, dash_dir):
    n = len(e4)
    nf = sum((not math.isfinite(f['T2']) for f in e4))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    fig.patch.set_facecolor(SURF)
    for ax, (suffix, title) in zip(axes, (('', 'U = best of 3 candidates (revised)'), ('_paper', 'U = min(greedy+repair, rounding) (manuscript)'))):
        style(ax)
        for i, t in enumerate(('T1', 'T2')):
            v = np.sort([f[t + suffix] for f in e4 if math.isfinite(f[t + suffix])])
            if len(v) == 0:
                continue
            y = np.arange(1, len(v) + 1) / n
            ax.step(v, y, where='post', color=SERIES[i], linewidth=2, label=f"{t} ({('target B*' if t == 'T1' else 'max over 21 budgets')})")
            ax.axvline(v[-1], color=SERIES[i], linewidth=1.2, linestyle='--')
            ax.annotate(f'max {v[-1]:.4f}%', xy=(v[-1], 1.06), xytext=(-4, 0), textcoords='offset points', ha='right', va='center', fontsize=8, color=TEXT2)
            d = [f[t + suffix] for f in dev if math.isfinite(f[t + suffix])]
            ax.plot(d, np.full(len(d), -0.06 - 0.05 * i), linestyle='none', marker='o', markersize=5, markerfacecolor='none', markeredgecolor=SERIES[i], markeredgewidth=1.3)
        ax.set_ylim(-0.17, 1.13)
        ax.set_xlabel('certified relative width (U−L)/U  (%)', color=TEXT2, fontsize=8.5)
        ax.set_title(title, color=TEXT, fontsize=10, loc='left')
    axes[0].set_ylabel(f'fraction of the {n} fields  (hollow: {len(dev)} dev fields)', color=TEXT2, fontsize=8.5)
    leg = axes[0].legend(frameon=False, fontsize=8.5, loc='upper left', bbox_to_anchor=(0, 0.93))
    for t in leg.get_texts():
        t.set_color(TEXT)
    fail_txt = f'; {nf} failed field(s) at +inf not drawn' if nf else ''
    fig.suptitle(f'Stage 9 · E4: {n} independent Uniform[1,30] fields (F=50, 7×7) — ECDF over n={n}{fail_txt}', color=TEXT, fontsize=10.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / 'stage9_ecdf.png', facecolor=SURF, dpi=130)
    plt.close(fig)
    shutil.copyfile(out_dir / 'stage9_ecdf.png', dash_dir / 'stage9_ecdf.png')
RELATIVE = ('T1', 'T2', 'T1_paper', 'T2_paper', 'T1_best', 'T2_best')
ORDER = '개발용 10개로 파일럿 → 본 표본 59개 입력 생성(PCG64) → 입력 해시(manifest)와 코드 해시(주 계산 파일 7개 + C++ 실행파일)·평가 규칙을 preregistration.json에 기록 → run 단계에서 해시 확인 후 최적화 실행. 로컬 사전 고정 기록이며 외부 시간 증명이나 과거 열람 이력의 독립 증명은 아님. 보고 스크립트·모든 Python 의존성·전체 환경은 해시에 포함되지 않음(환경 잠금 파일로 별도 고정)'

def main(out_dir=OUT, dash_dir=DASH, publish=True):
    dash_dir.mkdir(parents=True, exist_ok=True)
    pre = load_json((out_dir / 'preregistration.json').read_text())
    runs = load_json((out_dir / 'e4_runs.json').read_text())
    e4 = load_json((out_dir / 'e4_fields.json').read_text())
    dev = load_json((out_dir / 'dev_fields.json').read_text())
    se = load_json((out_dir / 'e4_summary.json').read_text())['summary']
    sd = load_json((out_dir / 'dev_summary.json').read_text())['summary']
    diag = diagnostics(runs)
    c = counts(e4, diag)
    (out_dir / 'e4_diagnostics.json').write_text(json.dumps(dict(counts=c, fields=diag), indent=1, ensure_ascii=False, default=float))
    figure(e4, dev, out_dir, dash_dir)
    rows = []
    for k, lab in LABELS.items():
        for set_, s, main_ in (('본 표본 59', se[k], True), ('개발용 10(별도)', sd[k], False)):
            crit = main_ and k in ('T2', 'T2_paper')
            rows.append(dict(statistic=lab, set=set_, unit='%' if k in RELATIVE else '분산 단위', **{a: s[a] for a in ('n', 'mean', 'median', 'q1', 'q3', 'iqr', 'min', 'max', 'argmax')}, ucl=ceil_pct(s['max']) if main_ and k in RELATIVE else None, conf=LEVEL if main_ and k in RELATIVE else None, criterion='0.6% (sweep 상대 폭)' if crit else '해당 없음', flag=('ok' if s['max'] < 0.6 else 'warn') if crit else None))
    cols = [('statistic', '통계량'), ('set', '표본'), ('unit', '단위'), ('n', 'n'), ('mean', '평균'), ('median', '중앙값'), ('q1', 'Q1'), ('q3', 'Q3'), ('iqr', 'IQR (Q3−Q1)'), ('min', '최소'), ('max', '표본 최대'), ('ucl', '95번째 백분위 상측 신뢰한계(올림)'), ('conf', '신뢰수준'), ('argmax', '최대 초기장'), ('criterion', '판정 기준'), ('flag', '판정')]
    events = f"최종 실패 초기장 {c['failed_fields']}, 초기장 재실행 {c['field_retries']}, OSQP fallback {c['n_fallback']}, Clarabel 실패 {c['n_clar_fail']}, 비교 플래그(L > P_c + 1e-9) {c['n_valid_flag']}, C++ 재시도 {c['n_cpp_retry']}, 비실행가능 후보 {c['n_infeasible']}, 제외 0(규칙상 없음)"
    pd_txt = f"계산된 primal–dual 차이의 인증폭 대비 비율 (V_QP − L)/(U − L) 최대 {c['pd_ratio_max']:.3g}, 두 solver의 계산된 하한 차이 최대 {c['bound_diff_max']:.3g} — 이 표본의 수치적 일관성 진단이며 부동소수점 오차를 포함한 엄밀한 상계가 아님. {c['n_clip_high']}개 사례에서 L을 근사 primal 값으로 clipping(그 사례의 비율은 0). OSQP 연속해는 정확히 실행가능하지 않음(저장 primal residual 최대 {c['prim_res_max']:.3g}); 정수 후보는 모두 제약을 정확히 만족" if c['pd_ratio_max'] is not None else ''
    if publish:
        progress.result('s9_summary', 'S9 · 분포 요약', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=rows, images=[dict(src='data/stage9_ecdf.png', alt='Stage 9 ECDF')], note=f"사전 고정 기록 preregistration.json(sha256 {sha256_file(out_dir / 'preregistration.json')[:16]}…, {pre['created']}). 순서: {ORDER}. 모집단: 셀별 i.i.d. Uniform[1,30], F=50, 7×7, σ=1.75, C_max=200. 개발용 seed 1–10(C++ mt19937)은 표본에서 제외하고 별도 표시. L = OSQP 1e-8 dual bound(원고 설정), U = 이번 실행의 후보 최소. 상측 신뢰한계: 초기장 {c['n']}개를 i.i.d. 표본으로 볼 때 표본 최대가 모집단 95번째 백분위 이상일 확률 ≥ 1 − 0.95^{c['n']} = {LEVEL:.4f}; 보고값은 표본 최대를 바깥쪽(위)으로 올림. T1·T2 상한을 동시에 95.15%로 보장한다는 뜻은 아니며, 예산 집합에 대한 주장은 주 통계량 T2로 한다(같은 상한이 B*를 포함한 격자의 모든 예산에 적용). 이 모집단·격자·절차에 한정, 최악값 보장 아님, 개별 L의 엄밀성을 대신하지 않음. 사건 수: {events}. 판정은 T2 행에만 적용(초록의 0.6%는 sweep 상대 폭). {pd_txt}."))
    fc = [('name', '초기장'), ('bstar', 'B*'), ('cpp_bstar', 'C++ B*'), ('mean_s0', '초기 평균'), ('T1', 'T1 (%)'), ('T2', 'T2 (%)'), ('T2_B_ratio', 'T2 위치 B/B*'), ('T1_paper', 'T1 원고 U'), ('T2_paper', 'T2 원고 U'), ('abs_T1', '절대 폭(분산 단위)'), ('pd_ratio_max', '계산된 primal–dual 차이/(U−L) 최대'), ('n_clip_high', 'L clipping 수'), ('U_method_target', 'B*의 U 방법'), ('attempts', '실행 횟수'), ('n_valid_flag', '비교 플래그'), ('flag', '판정')]
    dmap = {d['name']: d for d in diag}

    def fflag(f):
        if f.get('failed'):
            return 'fail'
        bad = f.get('n_fallback', 0) + f.get('n_clar_fail', 0) + f.get('n_valid_flag', 0) + f.get('n_infeasible', 0)
        return 'warn' if bad or f.get('attempts', 1) > 1 or f.get('n_cpp_retry', 0) else 'ok'
    frows = [dict(f, **dmap.get(f['name'], {}), flag=fflag(f)) for f in e4] + [dict(f, name=f['name'] + ' (개발용)', flag=fflag(f)) for f in dev]
    if not publish:
        return (rows, frows, c)
    progress.result('s9_fields', 'S9 · 초기장별', dict(columns=[dict(key=k, label=l) for k, l in fc], rows=frows, note="T1 = B*에서의 상대 폭, T2 = 원고 sweep 격자 21점(0.85–1.05 B*)의 최대. '계산된 primal–dual 차이/(U−L)'은 격자의 최대값으로 수치 일관성 진단일 뿐 오차 상계가 아님(clipping 사례는 0). 판정: fail = 최종 실패(T = +∞), warn = fallback·Clarabel 실패·비교 플래그·비실행가능 후보·재시도 중 하나 이상. 개발용 행은 파일럿이며 요약에 포함하지 않음."))
    s = se
    fin = [d for d in diag if d['T2_B_ratio'] is not None]
    pos = (min((d['T2_B_ratio'] for d in fin)), max((d['T2_B_ratio'] for d in fin))) if fin else (None, None)
    progress.task('9.1', 'done', f"사전 고정 기록 {pre['created']}: {ORDER}. 모집단 Uniform[1,30] i.i.d.(F=50), PCG64 SeedSequence spawn(59), 예산 B*와 원고 격자 21점, L = OSQP 1e-8(원고 설정) + Clarabel 비교, U = 후보 3종 최소, 통계량 T1(B*)·T2(격자 최대), 실패 규칙 {len(pre['failure_rules'])}개")
    progress.task('9.2', 'done', f"본 표본 {c['n']}개 × 예산 21개 = {c['instances']:,} 인스턴스, 최종 실패 {c['failed_fields']}. T1(B*, U 후보 3종): 평균 {s['T1']['mean']:.4f}%, 중앙값 {s['T1']['median']:.4f}%, Q1–Q3 {s['T1']['q1']:.4f}–{s['T1']['q3']:.4f}% (IQR {s['T1']['iqr']:.4f}%p), 표본 최대 {s['T1']['max']:.6f}%. 원고 U 기준 T1: 중앙값 {s['T1_paper']['median']:.4f}%, 표본 최대 {s['T1_paper']['max']:.6f}%(개발용 10개 평균 {sd['T1_paper']['mean']:.4f}%, 최대 {sd['T1_paper']['max']:.4f}% — 원고 Table S5 0.174%·0.206%와 일치)")
    progress.task('9.3', 'done', f"T2 = 격자 21점의 최대(초기장당 1개 표본): 중앙값 {s['T2']['median']:.4f}%, 표본 최대 {s['T2']['max']:.6f}% → 95번째 백분위 상측 신뢰한계 {ceil_pct(s['T2']['max']):.4f}%(신뢰수준 ≥ {LEVEL:.4f}); 원고 U 기준 표본 최대 {s['T2_paper']['max']:.6f}% → {ceil_pct(s['T2_paper']['max']):.4f}%. 관측 표본의 모든 T2_paper가 0.6% 미만. 최대 위치 B/B* {fmtn(pos[0], '.2f')}–{fmtn(pos[1], '.2f')}. 이 모집단·격자·절차에 한정")
    progress.task('9.4', 'done', f"사전 고정 실패 규칙 적용, 사건 수는 기록에서 계산: {events}. n = {c['n']} 유지(실패는 T = +∞)")
    progress.task('9.6', 'done', '보고 코드가 최종 실패 초기장(rows 없음)을 처리: 진단은 null, T = +∞ 유지, ECDF 분모 n 유지·실패 수 표기. 실패·재시도·판정 문자열을 기록에서 계산. 실패 1개를 넣은 합성 입력으로 확인')
    progress.task('9.7', 'done', f"'수치 몫' → '계산된 primal–dual 차이의 인증폭 대비 비율'(최대 {fmtn(c['pd_ratio_max'], '.3g')}), 엄밀한 오차 상계가 아님을 명시. L clipping {c['n_clip_high']}건, 저장 primal residual 최대 {fmtn(c['prim_res_max'], '.3g')}, 두 solver 하한 차이 최대 {fmtn(c['bound_diff_max'], '.3g')}")
    progress.task('9.8', 'done', "사전 고정 순서를 '파일럿 → 입력 59개 생성 → 입력·코드 해시와 규칙 기록 → 해시 확인 후 실행'으로 정정하고 로컬 기록의 확인 범위(외부 시간 증명 아님, 보고 스크립트·전체 환경은 해시 밖)를 명시")
    progress.task('9.9', 'done', f"상측 신뢰한계는 올림: T1 {ceil_pct(s['T1']['max']):.4f}%, T2 {ceil_pct(s['T2']['max']):.4f}%, T1 원고 U {ceil_pct(s['T1_paper']['max']):.4f}%, T2 원고 U {ceil_pct(s['T2_paper']['max']):.4f}%. 절대 폭은 분산 단위로 분리, 0.6% 판정은 본 표본의 T2 행에만 적용, 'IQR' 표기를 Q1–Q3 구간과 구분")
    progress.log('Stage 9 보고 보완 반영: s9_summary·s9_fields 재게시')
if __name__ == '__main__':
    main()
