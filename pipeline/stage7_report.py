from package_paths import load_json
import json
import shutil
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
sys.path.insert(0, str(Path(__file__).resolve().parent))
import progress
EXP = Path(__file__).resolve().parents[1]
OUT, DASH = (EXP / 'results' / 'stage7', EXP / 'reports' / 'data')
SURF, TEXT, TEXT2, GRID = ('#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0')
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
FIELDS = ['zero', 'random s7', 'center-30', 'center-100']
W = 'rel_width_final_pct'

def style(ax):
    ax.set_facecolor(SURF)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
    for sp_ in ('left', 'bottom'):
        ax.spines[sp_].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT2, labelsize=8.5)

def reference_lines(ax, xmax_label):
    for y, lab in ((1, '1%'), (5, '5%')):
        ax.axhline(y, color=TEXT2, linewidth=1.2, linestyle='--')
        ax.annotate(lab, xy=(xmax_label, y), xytext=(2, 2), textcoords='offset points', fontsize=8, color=TEXT2)

def attainment(rows, key_fields, xs_key):
    out = {}
    for r in rows:
        k = tuple((r[f] for f in key_fields))
        out.setdefault(k, []).append(r)
    res = []
    for k, rs in out.items():
        rs.sort(key=lambda r: r[xs_key])
        first = lambda thr: next((r[xs_key] for r in rs if r[W] <= thr), None)
        mono = all((a[W] >= b[W] - 1e-12 for a, b in zip(rs, rs[1:])))
        res.append(dict(zip(key_fields, k), rho_1pct=first(1.0), rho_5pct=first(5.0), width_at_min_rho=rs[0][W], width_at_max_rho=rs[-1][W], width_max=max((r[W] for r in rs)), monotone_in_rho=mono, flag='ok' if mono else 'warn'))
    return res

def main():
    rows = load_json((OUT / 'range_rows.json').read_text())['rows']
    A = [r for r in rows if r['series'] == 'A']
    B = [r for r in rows if r['series'] == 'B']
    attA = attainment(A, ['field'], 'rho')
    attB = attainment(B, ['field', 'sigma'], 'rho')
    (OUT / 'attainment.json').write_text(json.dumps(dict(A=attA, B=attB), indent=1, ensure_ascii=False))
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=150)
    fig.patch.set_facecolor(SURF)
    style(ax)
    for i, f in enumerate(FIELDS):
        rs = sorted([r for r in A if r['field'] == f], key=lambda r: r['BN'])
        ax.plot([r['BN'] for r in rs], [r[W] for r in rs], color=SERIES[i], linewidth=2, marker='o', markersize=6, markeredgecolor=SURF, markeredgewidth=1.5, label=f)
    ax.set_xscale('log')
    ax.set_yscale('log')
    xs = sorted({r['BN'] for r in A})
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{x:.3g}' for x in xs])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    reference_lines(ax, xs[0])
    ax.set_xlabel('average shots per waypoint  B/N', color=TEXT2)
    ax.set_ylabel('certified relative width (U−L)/U  (%)', color=TEXT2)
    ax.set_title('Stage 7 · series A (7×7, σ=1.75, F=50): width vs average shot count', color=TEXT, fontsize=10.5, loc='left')
    leg = ax.legend(frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(OUT / 'stage7_seriesA.png', facecolor=SURF)
    plt.close(fig)
    rhos = sorted({r['rho'] for r in B})
    fig, axes = plt.subplots(1, len(rhos), figsize=(11, 3.6), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, rho in zip(axes, rhos):
        style(ax)
        for i, f in enumerate(FIELDS):
            rs = sorted([r for r in B if r['field'] == f and r['rho'] == rho], key=lambda r: r['sigma'])
            ax.plot([r['sigma_over_s'] for r in rs], [r[W] for r in rs], color=SERIES[i], linewidth=2, marker='o', markersize=5, markeredgecolor=SURF, markeredgewidth=1.2, label=f)
        ax.set_yscale('log')
        reference_lines(ax, rs[0]['sigma_over_s'])
        ax.set_title(f'B/N = {rho:g}', color=TEXT, fontsize=10, loc='left')
        ax.set_xlabel('σ / s  (support ±⌈3σ⌉, raw mass)', color=TEXT2, fontsize=8.5)
    axes[0].set_ylabel('(U−L)/U  (%)', color=TEXT2)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.suptitle('Stage 7 · series B: width vs kernel width (separate from the 7×7 benchmark)', color=TEXT, fontsize=10.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / 'stage7_seriesB.png', facecolor=SURF)
    plt.close(fig)
    DASH.mkdir(parents=True, exist_ok=True)
    for n in ('stage7_seriesA.png', 'stage7_seriesB.png'):
        shutil.copyfile(OUT / n, DASH / n)
    cols = [('series', '계열'), ('field', '초기장'), ('kernel', 'kernel'), ('sigma', 'σ'), ('sigma_over_s', 'σ/s'), ('rho', 'ρ'), ('B', 'B'), ('BN', '실제 B/N'), ('mass', 'stamp 질량 κ'), ('gram_cond', 'cond(AᵀA)'), ('L_orig', 'L 보정 전 (OSQP 1e-8)'), ('rel_width_pct_orig', '상대 폭 % 보정 전'), ('L_final', 'L 최종'), ('L_source', 'L 출처'), ('x_ref_source', '기준 연속해'), ('x_ref_violation', '기준해 제약 위반'), ('V_QP_final', 'V(x_ref) 분산 크기'), ('U', 'U'), ('U_method', 'U 해'), ('abs_width_final', '절대 폭 U−L'), ('rel_width_final_pct', '상대 폭 %'), ('first_final', '1차항'), ('curv_final', '곡률 δᵀQδ'), ('numeric_final', '수치 V(x_ref)−L'), ('curv_share_final', '곡률 몫'), ('split_check_final', '분해 검산'), ('relax_lower_active_final', '기준해 하한 활성'), ('relax_upper_active_final', '기준해 상한 활성'), ('relax_x_max_final', '기준해 max x'), ('U_at_lower', 'U 해 x=1'), ('U_at_upper', 'U 해 x=C_max'), ('refined', '하한 보정'), ('flag', '판정')]
    note = '모든 폭·분해·활성 제약은 최종 하한 L 최종과 기준 연속해 x_ref로 일관되게 계산: U−L = 1차항 + 곡률 + 수치(V(x_ref)−L), δ = U 해 − x_ref, 활성 판정 허용오차 1e-5. 수치 몫이 폭의 10%를 넘은 11개(넓은 kernel)는 OSQP 1e-8·1e-10·Clarabel 승수 하한 중 최댓값을 쓰고, x_ref는 정밀 해(OSQP 1e-10, 또는 목적값·제약 위반이 모두 더 작은 Solved Clarabel 해). 보정 전 L과 상대 폭은 별도 열. 판정 ok ≤1%, warn ≤5%, fail >5%. 17×17·23×23의 cond(AᵀA) ≈ 2.1e12·4.1e13이므로 배정밀도 하한은 수치 결과이며 엄밀한 인증이 아니다. 계열 B는 원시 Gaussian(η=1), 지지 ±⌈3σ⌉ — σ=1.75이면 13×13으로 계열 A의 7×7과 다른 문제.'
    progress.result('s7_A', 'S7 · 계열 A (7×7 benchmark)', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=A, images=[dict(src='data/stage7_seriesA.png', alt='계열 A 상대 폭과 B/N')], note=note))
    progress.result('s7_B', 'S7 · 계열 B (kernel 폭)', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=B, images=[dict(src='data/stage7_seriesB.png', alt='계열 B 상대 폭과 σ/s')], note=note))
    acols = [('field', '초기장'), ('sigma', 'σ (계열 B)'), ('rho_1pct', '시험한 ρ 중 1% 이하가 처음 관측된 ρ'), ('rho_5pct', '시험한 ρ 중 5% 이하가 처음 관측된 ρ'), ('width_at_min_rho', '최소 ρ에서 폭 %'), ('width_at_max_rho', '최대 ρ에서 폭 %'), ('width_max', '시험한 모든 ρ의 최대 폭 %'), ('monotone_in_rho', 'ρ에 단조'), ('flag', '판정')]
    progress.result('s7_attain', 'S7 · 1%·5% 달성 범위', dict(columns=[dict(key=k, label=l) for k, l in acols], rows=[dict(x, sigma=x.get('sigma', '1.75 (A, 7×7)'), rho_1pct=x['rho_1pct'] if x['rho_1pct'] is not None else '미관측', rho_5pct=x['rho_5pct'] if x['rho_5pct'] is not None else '미관측') for x in attA + attB], note="시험한 ρ 격자(계열 A {1.5008, 2, 3, 8, 20, 45}, 계열 B {1.5, 3, 8, 45}) 안에서 상대 폭이 1%·5% 이하로 처음 관측된 ρ. 더 큰 ρ나 격자 사이 값에서의 보장이 아니며, 그림의 연결선은 시각적 안내일 뿐이다. 계열 B의 σ/s 결론도 시험한 σ ∈ {1.0, 1.25, 1.75, 2.5, 3.5}와 지지 ±⌈3σ⌉·원시 질량 규칙에 한정. '미관측' = 시험 범위에서 관측되지 않음. 판정 warn = 시험점에서 ρ에 단조가 아님."))

    def rng(rs, key):
        v = [r[key] for r in rs if r[key] is not None]
        return f'{min(v):.3g}–{max(v):.3g}'
    progress.task('7.1', 'done', f'계열 A {len(A)}개: F=50, s=2, 7×7, σ=1.75, C_max=200, 초기장 4개')
    progress.task('7.2', 'done', 'ρ = 1.5, 2, 3, 8, 20, 45')
    progress.task('7.3', 'done', 'B=round(ρN) 직접 입력, 실제 B/N ' + ', '.join(sorted({f"{r['BN']:.4g}" for r in A})))
    progress.task('7.4', 'done', f'계열 B {len(B)}개(별도): σ = 1.0, 1.25, 1.75, 2.5, 3.5 (σ/s 0.5–1.75), ρ = 1.5, 3, 8, 45')
    progress.task('7.5', 'done', '지지 ±⌈3σ⌉: σ=1.0→7×7, 1.25→9×9, 1.75→13×13(기존 7×7과 다른 문제), 2.5→17×17, 3.5→23×23')
    progress.task('7.6', 'done', '질량 정책: 정규화하지 않음(원고와 같은 원시 Gaussian, η=1). stamp 질량 κ ' + rng(B, 'mass') + '를 결과에 기록. 같은 ρ에서도 σ가 커지면 샷당 두께가 커짐')
    progress.task('7.7', 'done', f"최종 하한·기준 연속해 기준으로 절대 폭·상대 폭·분산 크기·활성 제약 보고. 상대 폭 계열 A {rng(A, 'rel_width_final_pct')}%, 계열 B {rng(B, 'rel_width_final_pct')}%. 기준해 상한 활성 최대 {max((r['relax_upper_active_final'] for r in rows))}(평균 B/N=45 < C_max여도 활성)")
    nref = sum((1 for r in rows if r.get('refined')))
    progress.task('7.8', 'done', f"보정 {nref}개(넓은 kernel): 하한·기준 연속해·목적값·분해·활성 제약을 함께 갱신, 감사 정보·x·승수 저장, records에 보정 run id·L 출처 연결(예 center-30 23×23 ρ=45: 절대 폭 6.12→0.0173, 곡률 몫 0.26%→{100 * next((r['curv_share_final'] for r in rows if r['field'] == 'center-30' and r['kernel'] == '23×23' and (r['rho'] == 45))):.1f}%). 분해 검산 최대 |오차| {max((abs(r['split_check_final']) for r in rows)):.1e}; 곡률 몫 {rng(rows, 'curv_share_final')}")
    progress.task('7.9', 'done', '시험한 B/N 중 처음 관측(계열 A): ' + '; '.join((f"{x['field']} 5% 이하 {x['rho_5pct']}, 1% 이하 {x['rho_1pct']}" for x in attA)) + '. 격자 사이·밖의 보장은 아님 (그림·S7 · 1%·5% 달성 범위 탭)')
    print(json.dumps(attA, indent=1, ensure_ascii=False))
if __name__ == '__main__':
    main()
