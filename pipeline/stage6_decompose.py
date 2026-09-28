from package_paths import load_json
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import progress
EXP = Path(__file__).resolve().parents[1]
OUT = EXP / 'results' / 'stage6'

def main():
    summ = load_json((OUT / 'miqp_summary.json').read_text())['rows']
    jobs = load_json((OUT / 'scip_jobs.json').read_text())
    by = {}
    for j in jobs:
        by.setdefault(j['instance'], {})[j['form']] = j
    rows = []
    for r in summ:
        e, s = (by[r['id']]['expanded'], by[r['id']]['soc'])
        U0, Uf, Q, L = (r['U_pipeline'], r['U_final'], r['V_QP'], r['L'])
        D = r['D']
        Dh = max(D, L) if D is not None else L
        W = U0 - L
        row = dict(id=r['id'], series=r['series'], N=r['N'], BN=r['BN'], L=L, Q=Q, U0=U0, Uf=Uf, D=D, exp_status=e['status'], exp_dual_end=e['dual_bound'], soc_status=s['status'], soc_dual_end=s['dual_bound'], exp_dual_zero=e['dual_bound'] <= 0.0, exp_dual_below_L=e['dual_bound'] < L - 1e-09, soc_dual_below_L=s['dual_bound'] < L - 1e-09, W0=W, rel_W0_pct=100 * W / U0, numeric_share=(Q - L) / W, share_after_scip=(D - L) / (Uf - L) if D is not None and Uf > L else None, share_original=(D - L) / W if D is not None else None)
        if r['V_Z'] is not None:
            VZ = r['V_Z']
            row.update(solved=True, V_Z=VZ, integ_lo=(VZ - Q) / W, integ_hi=(VZ - Q) / W, inc_lo=(U0 - VZ) / W, inc_hi=(U0 - VZ) / W, scip_tol_diff=max((abs(j['primal_bound'] - j['V_eval']) for j in (e, s) if j.get('status') == 'optimal' and j.get('V_eval') is not None)))
        else:
            row.update(solved=False, V_Z=None, integ_lo=max(0.0, Dh - Q) / W, integ_hi=(Uf - Q) / W, inc_lo=(U0 - Uf) / W, inc_hi=(U0 - Dh) / W, scip_tol_diff=None)
        row['unresolved_share'] = row['integ_hi'] - row['integ_lo']
        row['statement'] = '분해 확정' if row['solved'] else '원래 폭의 절반 이상이 incumbent 미최적성' if row['inc_lo'] > 0.5 else '원래 폭의 절반 이상이 정수화 차이' if row['integ_lo'] > 0.5 else '주원인 미분리'
        row['flag'] = 'ok' if row['solved'] else 'warn' if row['statement'] != '주원인 미분리' else 'fail'
        rows.append(row)
    new = [x for x in rows if x['series'] != 'N16 (Table 6)']
    stats = dict(new_instances=len(new), exp_optimal=sum((x['exp_status'] == 'optimal' for x in new)), soc_optimal=sum((x['soc_status'] == 'optimal' for x in new)), exp_dual_zero=sum((x['exp_dual_zero'] for x in new)), exp_dual_below_L=sum((x['exp_dual_below_L'] for x in new)), soc_dual_below_L=sum((x['soc_dual_below_L'] for x in new)), share_original_unsolved=[min((x['share_original'] for x in new if not x['solved'])), max((x['share_original'] for x in new if not x['solved']))], max_scip_tol_diff=max((x['scip_tol_diff'] for x in rows if x['scip_tol_diff'] is not None)))
    (OUT / 'formulation_decomposition.json').write_text(json.dumps(dict(stats=stats, rows=rows), indent=1, default=float, ensure_ascii=False))
    cols = [('id', '인스턴스'), ('N', 'N'), ('BN', 'B/N'), ('L', 'L'), ('Q', 'Q (연속 최적값)'), ('U0', 'U0 파이프라인'), ('Uf', 'Uf SCIP 후'), ('D', 'D 정수 하한'), ('V_Z', 'V_Z* (증명)'), ('rel_W0_pct', '원래 폭 (U0−L)/U0 %'), ('integ_lo', '정수화 몫 하한'), ('integ_hi', '정수화 몫 상한'), ('inc_lo', 'incumbent 몫 하한'), ('inc_hi', 'incumbent 몫 상한'), ('numeric_share', '수치 몫 (Q−L)/W0'), ('share_original', '(D−L)/(U0−L)'), ('share_after_scip', '(D−L)/(Uf−L) SCIP 후 남은 폭 기준'), ('statement', '현재 결론'), ('exp_status', '전개식 상태'), ('exp_dual_end', '전개식 종료 시 하한'), ('soc_status', 'SOC 상태'), ('soc_dual_end', 'SOC 종료 시 하한'), ('scip_tol_diff', '|SCIP primal − 직접 평가 V|'), ('flag', '판정')]
    progress.result('s6_forms', 'S6 · 정식화 비교·폭 분해', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=rows, note=f"몫은 모두 원래 파이프라인 폭 W0 = U0 − L로 정규화: W0 = (U0−VZ) + (VZ−Q) + (Q−L). 미해결 사례는 Dhat=max(D,L)로 구간 표시. (D−L)/(Uf−L)은 SCIP 실행 후 남은 폭 기준의 다른 지표. 하한은 종료 시점 값이며 시간별 이력은 저장하지 않았다. SCIP optimal은 SCIP 허용오차에서의 종료 상태(직접 평가와 최대 {stats['max_scip_tol_diff']:.2e} 차이). 재생성: .venv/bin/python pipeline/stage6_decompose.py"))
    progress.task('6.3', 'done', f"새 계열 {stats['new_instances']}개, 같은 SCIP·같은 초기해: 최적성 증명 SOC {stats['soc_optimal']} vs 전개식 {stats['exp_optimal']}(N=16은 둘 다 3/3). 종료 시점 전개식 정수 하한이 L보다 낮음 {stats['exp_dual_below_L']}/12(그중 0인 것 {stats['exp_dual_zero']}/12), SOC는 {stats['soc_dual_below_L']}/12. 미해결 사례의 (D−L)/(U0−L) {100 * stats['share_original_unsolved'][0]:.3g}–{100 * stats['share_original_unsolved'][1]:.3g}%. crash·외부 timeout 0건")
    print(json.dumps(stats, indent=1, default=float))
    for x in rows:
        print(f"{x['id']:22s} {x['statement']:28s} integ [{x['integ_lo']:.4f}, {x['integ_hi']:.4f}] inc [{x['inc_lo']:.4f}, {x['inc_hi']:.4f}]")
if __name__ == '__main__':
    main()
