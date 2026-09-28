import determinism
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time

import sensitivity_extension as se
import sensitivity_extension_v2 as s2

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/sensitivity_extension_replay'
FINAL = ROOT / 'results/sensitivity_extension_v2_replay'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(phase):
    se.WORK = ROOT / 'work/sensitivity_extension_replay'
    se.OUT = BASE if phase == 1 else FINAL
    s2.V1 = BASE
    s2.OUT = FINAL


def main():
    parser = argparse.ArgumentParser(description='Repeat sensitivity cases on archived inputs in a new working copy.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--case', action='append', dest='cases')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    if not (ROOT / 'RUN_COPY.json').is_file():
        parser.error('Create a copy with tools/new_run.py DEST --build first.')
    for name in ['pufoam_solver', 'matrix_exchange']:
        if not (ROOT / 'build' / name).is_file():
            parser.error('Build the solvers with python tools/build.py.')
    if BASE.exists() or FINAL.exists():
        parser.error('Replay output already exists; use a new working copy.')
    original = ROOT / 'results/sensitivity_extension/protocol.json'
    jobs = json.loads(original.read_text())['jobs']
    if args.cases:
        unknown = set(args.cases) - {j['id'] for j in jobs}
        if unknown:
            parser.error('Unknown cases: ' + ', '.join(sorted(unknown)))
        jobs = [j for j in jobs if j['id'] in args.cases]
    for job in jobs:
        if digest(ROOT / job['input']) != job['input_sha256']:
            raise RuntimeError('Input hash mismatch: ' + job['id'])
    BASE.mkdir(parents=True)
    FINAL.mkdir(parents=True)
    setup(1)
    se.WORK.mkdir(parents=True, exist_ok=True)
    code = ['pipeline/replay_sensitivity.py', 'pipeline/sensitivity_extension.py',
            'pipeline/sensitivity_extension_v2.py', 'pipeline/rigor_pilot.py',
            'pipeline/qp_audit.py', 'pipeline/rounding.py',
            'src/matrix_exchange.cpp', 'build/matrix_exchange', 'build/pufoam_solver']
    protocol = dict(kind='retrospective replay with current public code and archived inputs',
                    historical_protocol_sha256=digest(original), jobs=jobs,
                    current_sha256={rel: digest(ROOT / rel) for rel in code},
                    cases=len(jobs), full_series=len(jobs) == 156,
                    threads=determinism.assert_single_thread())
    se.dump(BASE / 'replay_protocol.json', protocol)
    se.dump(FINAL / 'replay_protocol.json', protocol)
    for kind, F in sorted({(j['kind'], j['F']) for j in jobs}):
        path = se.save_operator(kind, F)
        shutil.copy2(path, FINAL / path.name)
    se.operator.cache_clear()
    start = time.perf_counter()
    rows = []
    for group in ['target', 'distribution', 'mesh']:
        batch = [j for j in jobs if j['group'] == group]
        if not batch:
            continue
        workers = min(args.workers, 2 if group == 'mesh' else args.workers)
        with ProcessPoolExecutor(max_workers=workers, initializer=setup, initargs=(1,)) as executor:
            for row in executor.map(se.solve_job, batch):
                rows.append(row)
                print(f"Allocation {len(rows)}/{len(jobs)}: {row['id']}", flush=True)
    se.dump(BASE / 'rows.json', rows)
    if any(r['failed'] for r in rows):
        raise RuntimeError('Some allocation cases failed; inspect replay rows.json.')
    setup(2)
    refined = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=setup, initargs=(2,)) as executor:
        for row in executor.map(s2.run_case, rows):
            refined.append(row)
            print(f"Certificate {len(refined)}/{len(rows)}: {row['id']}", flush=True)
    se.dump(FINAL / 'rows.json', refined)
    if any(r['failed'] for r in refined):
        raise RuntimeError('Some certificate cases failed; inspect v2 replay rows.json.')
    with ProcessPoolExecutor(max_workers=args.workers, initializer=setup, initargs=(2,)) as executor:
        verified = list(executor.map(se.verify_row, refined))
    se.dump(FINAL / 'verification.json', dict(n=len(refined), passed=sum(verified),
                                            failed=len(verified) - sum(verified)))
    if not all(verified):
        raise RuntimeError('Replay certificate verification failed.')
    s2.summary(refined)
    se.mesh_compare(refined)
    print(json.dumps(dict(cases=len(refined), verified=sum(verified),
                          full_series=len(refined) == 156, elapsed_s=time.perf_counter() - start)))


if __name__ == '__main__':
    main()
