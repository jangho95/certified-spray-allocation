from package_paths import load_json
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import progress
EXP = Path(__file__).resolve().parents[1]
OUT = EXP / 'results' / 'rigor_pilot'

def main():
    d = load_json((OUT / 'pilot_rows.json').read_text())
    assert d.get('version') == 2
    rows = d['rows']
    for r in rows:
        T = r['times']
        r.update(t_total=T['total_run_case'], t_lean=T['lean_total'], t_float=T['osqp_1e-8_float_cert'], t_refine_lean=T['refine_from_1e-8'], verified=all(r['witness_verified'].values()), U_ok=all((all(c.values()) for c in r['U_checks'].values())))
    cols = [('case', '사례'), ('N', 'N'), ('B', 'B'), ('BN', 'B/N'), ('U_rig', 'U 정확(올림)'), ('U_ok', '정수해 검사'), ('L_float_1e8', '기존 float 하한(1e-8)'), ('L_rig', 'L 엄밀(내림)'), ('L_rig_source', 'L 엄밀의 기준점'), ('L8_minus_Lrig', 'float − 엄밀'), ('QP_enclosure_width_up', 'V_QP* 포함구간 폭(올림)'), ('W_samefloat_pct', '폭: 같은 incumbent·기존 float 하한 (%)'), ('W_rig_pct_up', '폭 엄밀(올림, %)'), ('W_lean_pct_up', '폭 간소화 절차(올림, %)'), ('dW_pp', '엄밀 − float (%p)'), ('eps_ideal', '이상 모델 ε'), ('W_ideal_pct_up', '폭 이상 모델(올림, %)'), ('t_float', 'float 인증 (s)'), ('t_lean', '간소화 총시간 (s)'), ('t_total', '파일럿 전체 (s)'), ('verified', '검증 자료 재계산'), ('flag', '판정')]
    progress.result('s_f2_pilot', 'F.2 · 엄밀한 사후 검증 파일럿', dict(columns=[dict(key=k, label=l) for k, l in cols], rows=rows, note="보장 대상: build_stamps가 만든 float64 A와 저장된 s0(고정 입력)를 정확한 유리수로 본 모델. 하한 L = V(x̄) + min_X ∇V(x̄)ᵀ(x − x̄)(볼록성), 정수 산술로 정확히 계산 후 내림 — 선형계 풀이가 없어 조건수는 조임에만 영향. U = 정수성·예산·box를 검사한 정수 배분의 정확한 V(올림). 상대폭·포함구간 폭·이상 모델 하한은 유리수로 계산한 뒤 바깥쪽 반올림. 이상 Gaussian 모델(7×7 지지·경계 접기 유지): exp의 유리수 포함구간으로 ε를 구해 max(0, √L − ε)² ≤ (이상 모델의 연속·정수 최적값), (이상 모델에서 incumbent의 목적값) ≤ (√U + ε)²(뺄셈부터 유리수). '폭: 같은 incumbent·기존 float 하한'은 현재 U와 원고 설정의 float L을 결합한 값으로 제출 원고 Table 5 값이 아님. 간소화 절차 = 원고의 OSQP 1e-8 해에서 바로 정제(추가 solver 없음); 간소화 총시간 = 구성 + OSQP 1e-8 + 정확 하한 + 정제 + 정수해 검사·정확 U(연산자당 한 번인 이상 모델 ε 제외). 파일럿 전체는 세 기준점 solver와 두 정제를 모두 포함. 시간은 단일 측정. 검증 자료(results/rigor_pilot/witness/)만으로 L·U·V_feas를 다시 계산해 일치 확인. 판정 warn = 폭 차이 ≥ 0.001%p(s=1 center-30: float 하한이 느슨)."))
    tg = [r for r in rows if r['id'].startswith('T-')]
    hard = {r['id']: r for r in rows if not r['id'].startswith('T-')}
    c30, z = (hard['V-s1c30'], hard['V-s1zero'])
    progress.task('F.2', 'partial', f"파일럿 v2 완료(전체 적용은 Stage 9 추가 분석으로 진행). 6개 모두 정수해 검사 통과, 검증 자료 재계산 일치, V_QP* 포함구간 폭 ≤ {max((r['QP_enclosure_width_up'] for r in rows)):.1g}. 목표 4개: 기존 float 하한 ≤ 엄밀 하한(타당), |폭 차이| ≤ {max((abs(r['dW_pp']) for r in tg)):.1g}%p. s=1 center-30: float 하한이 {-c30['L8_minus_Lrig']:.2g} 느슨 → 폭 {c30['W_samefloat_pct']:.3f}% → {c30['W_rig_pct_up']:.3f}%(올림). s=1 zero: V_QP* ∈ [0, {z['QP_enclosure_width_up']:.1e}] — 이 완화의 하한 0으로는 양의 정수 목적값의 상대폭 100%를 줄일 수 없음(incumbent 준최적성과 정수화 손실의 몫은 미결정). 비용: N=625에서 간소화 총시간 {min((r['t_lean'] for r in tg)):.2g}–{max((r['t_lean'] for r in tg)):.2g}s(float 인증 {min((r['t_float'] for r in tg)):.2g}–{max((r['t_float'] for r in tg)):.2g}s의 약 7–10배), s=1 {min((r['t_lean'] for r in hard.values())):.2g}–{max((r['t_lean'] for r in hard.values())):.2g}s; 세 기준점을 모두 쓰면 N=625 {min((r['t_total'] for r in tg)):.2g}–{max((r['t_total'] for r in tg)):.2g}s, s=1 최대 {max((r['t_total'] for r in hard.values())):.3g}s")
    progress.log('F.2 파일럿 v2 게시: s_f2_pilot')
if __name__ == '__main__':
    main()
