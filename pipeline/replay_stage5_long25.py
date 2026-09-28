import json
from pathlib import Path
import subprocess
import sys

import progress
import stage5_long_runs as runs


def main():
    root = Path(__file__).resolve().parents[1]
    if not (root / 'RUN_COPY.json').is_file():
        raise SystemExit('Use tools/new_run.py DEST --stage pso-long.')
    runs.OUT = root / 'results/stage5_long25_replay'
    if runs.OUT.exists():
        raise SystemExit('Replay directory already exists; use a fresh working copy.')
    progress.task('5.8', 'running', 'Replicating the archived long-run comparison')
    runs.main()
    import stage5_long_analyze as analysis
    analysis.OUT = runs.OUT
    analysis.main()
    subprocess.run([sys.executable, str(root / 'pipeline/verify_stage5_long25.py'), str(runs.OUT), '--output', str(runs.OUT / 'verification.json')], check=True)
    import stage5_long_report as report
    report.OUT = runs.OUT
    report.main()
    print(json.dumps({'results': str(runs.OUT.relative_to(root)), 'design': 'Preserve the four archived seed-1 runs and repeat the 96 extension runs; archived outputs and protocol remain unchanged.'}))


if __name__ == '__main__':
    main()
