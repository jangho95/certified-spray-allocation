import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import progress
EXP = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGS = EXP / 'logs' / 'stage4'
OUT = EXP / 'results' / 'stage4'
SCALES = (25, 50, 100)
JOBS = [('single', F, f, r) for F in SCALES for f in ('zero', 'center100') for r in range(1, 6)] + [('pso', F, 'zero', r) for F in SCALES for r in range(1, 4)] + [('sweep_reuse', F, 'zero', r) for F in SCALES for r in range(1, 4)] + [('sweep', F, 'zero', r) for F in SCALES for r in range(1, 4)]
TASK = {'single': '4.1', 'pso': '4.4', 'sweep_reuse': '4.5', 'sweep': '4.2'}

def machine():
    cpu = next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')), '?')
    mem = next((l.split(':', 1)[1].strip() for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal')), '?')
    return dict(cpu=cpu, logical_cpus=os.cpu_count(), mem_total=mem, kernel=platform.release(), python=platform.python_version(), note='WSL2; measurements run one at a time, BLAS forced to 1 thread')

def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'machine.json').write_text(json.dumps(machine(), indent=1))
    progress.log(f'Stage 4 측정 시작: {len(JOBS)}개 프로세스, 순차 실행')
    started = set()
    t_all = time.time()
    for k, (mode, F, field, rep) in enumerate(JOBS, 1):
        tid = TASK[mode]
        if tid not in started:
            progress.task(tid, 'running', f'{mode} 측정 시작')
            started.add(tid)
        progress.detail(tid, f'[{k}/{len(JOBS)}] {mode} · F={F} · {field} · rep {rep}')
        log = LOGS / f'{mode}_F{F}_{field}_r{rep}.log'
        with open(log, 'w') as fh:
            p = subprocess.run([PY, str(EXP / 'pipeline' / 'stage4_worker.py'), '--mode', mode, '--F', str(F), '--field', field, '--rep', str(rep), '--job', f'[{k}/{len(JOBS)}]'], stdout=fh, stderr=subprocess.STDOUT, cwd=EXP)
        if p.returncode != 0:
            progress.task(tid, 'failed', f'{mode} F={F} {field} rep {rep} 실패 — {log.name}')
            raise SystemExit(1)
        nxt = JOBS[k][0] if k < len(JOBS) else None
        if nxt != mode:
            progress.detail(tid, f'{mode} 측정 {sum((1 for j in JOBS if j[0] == mode))}회 완료, 집계 대기')
    progress.log(f'Stage 4 측정 완료 ({time.time() - t_all:.0f}s) — 집계 시작')
    subprocess.run([PY, str(EXP / 'pipeline' / 'stage4_aggregate.py')], check=True, cwd=EXP)
if __name__ == '__main__':
    main()
