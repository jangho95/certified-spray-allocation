from __future__ import annotations
from package_paths import load_json
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / 'pipeline'))
import progress
from qp_audit import THREAD_ENV
PY = sys.executable
WORK1 = EXP / 'work' / 'stage1'
LOGS = EXP / 'logs' / 'stage1'
PIPE = str(EXP / 'pipeline')
BIN = 'build/pufoam_solver'
MAXPAR = 6

def cpp(inp, prefix):
    return [BIN, '--initial-field', f'inputs/{inp}', '--quiet', '--out-prefix', prefix]
TASKS = {'1.1': dict(deps=[], cmds=[cpp('zero_F50.csv', 'zero'), cpp('random_u1_30_seed7_F50.csv', 'random_1_30_seed7_sync'), cpp('center30_F50.csv', 'center30'), cpp('center100_F50.csv', 'center100')]), '1.3': dict(deps=[], cmds=[[PY, 'cmax_count_ablation.py']]), '1.2': dict(deps=['1.1', '1.3'], cmds=[[PY, f'{PIPE}/s1_certificates.py', 'table5']]), '1.4': dict(deps=[], cmds=[[PY, 'kappa_violation.py']]), '1.5': dict(deps=[], cmds=[[PY, 'audit_equivalence.py', '--field', 'zero'], [PY, 'audit_equivalence.py', '--field', 'random', '--seed', '7'], [PY, 'audit_equivalence.py', '--field', 'center', '--center-height', '100'], [PY, 'audit_equivalence.py', '--field', 'zero', '--boundary', 'truncate']]), '1.6': dict(deps=[], cmds=[[PY, 'small_exact_miqp.py']]), '1.7': dict(deps=[], cmds=[[PY, 'interval_ablation.py']]), '1.8': dict(deps=[], cmds=[[PY, 'gaussian_distortion_ablation.py']]), '1.9': dict(deps=[], cmds=[[PY, 'random_seed_robustness.py']]), '1.10': dict(deps=['1.1', '1.3'], cmds=[[PY, 'budget_direction_diagnostics.py'], [PY, f'{PIPE}/s1_certificates.py', 's6']]), '1.11': dict(deps=[], cmds=[[PY, 'pso_repeated_runs.py']]), '1.12': dict(deps=[], cmds=[[PY, 'pareto_sweep.py']]), '1.13': dict(deps=[f'1.{i}' for i in range(1, 13)], cmds=[])}

def last_line(path: Path) -> str:
    try:
        with open(path, 'rb') as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 4096))
            lines = [l.strip() for l in fh.read().decode(errors='replace').splitlines() if l.strip()]
        return lines[-1][:180] if lines else ''
    except FileNotFoundError:
        return ''

def run_task(tid: str, spec: dict, results: dict) -> None:
    log = LOGS / f"{tid.replace('.', '_')}.log"
    log.write_text('')
    env = {**os.environ, **THREAD_ENV}
    t0 = time.time()
    progress.task(tid, 'running', f"{len(spec['cmds'])}개 명령 실행 시작")
    for k, cmd in enumerate(spec['cmds'], 1):
        with open(log, 'a') as fh:
            fh.write(f"$ {' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=WORK1, stdout=fh, stderr=subprocess.STDOUT, env=env)
            while proc.poll() is None:
                time.sleep(3)
                progress.detail(tid, f"[{k}/{len(spec['cmds'])}] {last_line(log)}")
        if proc.returncode != 0:
            progress.task(tid, 'failed', f'명령 {k} 실패 (exit {proc.returncode}): {last_line(log)}', {'run_s': round(time.time() - t0, 1)})
            results[tid] = 'failed'
            return
    run_s = time.time() - t0
    cmp = subprocess.run([PY, f'{PIPE}/s1_compare.py', '--task', tid], cwd=EXP, capture_output=True, text=True, env=env)
    with open(log, 'a') as fh:
        fh.write('\n# compare\n' + cmp.stdout + cmp.stderr)
    try:
        summ = load_json(cmp.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        progress.task(tid, 'warn', f'실행 완료, 비교 스크립트 오류: {last_line(log)}', {'run_s': round(run_s, 1)})
        results[tid] = 'warn'
        return
    status = 'done' if summ.get('fail', 0) == 0 and summ.get('warn', 0) == 0 else 'warn'
    det = f"실행 {run_s:.0f}s · 비교 ok {summ.get('ok', 0)} / warn {summ.get('warn', 0)} / fail {summ.get('fail', 0)}"
    if summ.get('note'):
        det += f" — {summ['note']}"
    progress.task(tid, status, det, {'run_s': round(run_s, 1)})
    results[tid] = status

def main(selected):
    LOGS.mkdir(parents=True, exist_ok=True)
    todo = [t for t in TASKS if not selected or t in selected]
    done = {t for t in TASKS if t not in todo}
    results: dict = {}
    running: dict[str, threading.Thread] = {}
    progress.log(f"Stage 1 시작: {', '.join(todo)}")
    while todo or running:
        for tid in list(running):
            if not running[tid].is_alive():
                running.pop(tid)
                done.add(tid)
        for tid in list(todo):
            if len(running) >= MAXPAR:
                break
            if all((d in done for d in TASKS[tid]['deps'])):
                failed_dep = [d for d in TASKS[tid]['deps'] if results.get(d) == 'failed']
                todo.remove(tid)
                if failed_dep:
                    progress.task(tid, 'skipped', f"선행 작업 실패: {', '.join(failed_dep)}")
                    done.add(tid)
                    continue
                th = threading.Thread(target=run_task, args=(tid, TASKS[tid], results), daemon=True)
                th.start()
                running[tid] = th
        time.sleep(1)
    progress.log('Stage 1 종료: ' + ', '.join((f'{k}={v}' for k, v in sorted(results.items()))))
if __name__ == '__main__':
    main(set(sys.argv[1:]))
