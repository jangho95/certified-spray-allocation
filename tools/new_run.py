import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    '1': ['stage1_run.py'],
    '2': ['stage2_e1.py'],
    '3': ['stage3_cmax.py', 'stage3_round_repair.py'],
    '4': ['stage4_run.py', 'stage4_reuse_integer_check.py'],
    '5': ['stage5_pso_same_budget.py', 'stage5_figure.py'],
    '6': ['stage6_miqp.py', 'stage6_decompose.py'],
    '7': ['stage7_range.py', 'stage7_refine.py', 'stage7_report.py'],
    '8': ['stage8_mass.py'],
    '9': ['replay_stage9.py'],
    'benchmark': ['bench_sweep.py'],
    'figures': ['figures_revised.py', 'initial_fields_3d.py', 'legacy_v5_figures.py', 'range_figures.py', 'stage5_long_report.py'],
    'pso-long': ['replay_stage5_long25.py'],
    'sensitivity': ['replay_sensitivity.py'],
    'sensitivity-tables': ['make_sensitivity_tables.py'],
    'tables': ['stage4_aggregate.py', 'stage5_figure.py', 'stage6_decompose.py', 'stage7_report.py', 'stage9_report.py'],
}

def main():
    p = argparse.ArgumentParser(description='Create an independent working copy before generating new outputs.')
    p.add_argument('destination', type=Path)
    p.add_argument('--stage', choices=STAGES)
    p.add_argument('--build', action='store_true')
    args = p.parse_args()
    dst = args.destination.expanduser().resolve()
    if dst.exists() or dst == ROOT or ROOT in dst.parents:
        p.error('Use a new destination outside this package.')
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns('.venv', '.git', '__pycache__', 'build', 'reports', '*.zip'), symlinks=True)
    (dst / 'RUN_COPY.json').write_text(json.dumps(dict(source='public reproducibility package', purpose='independent rerun',
                                                        historical_outputs='copied baseline; stages may replace these in this run copy'), indent=2))
    if args.build:
        subprocess.run([sys.executable, str(dst / 'tools/build.py')], check=True, cwd=dst)
    if args.stage:
        if args.stage not in ('tables', 'figures', 'pso-long', 'sensitivity-tables') and not (dst / 'build/pufoam_solver').exists():
            p.error('Computational stages require --build.')
        for script in STAGES[args.stage]:
            subprocess.run([sys.executable, str(dst / 'pipeline' / script)], check=True, cwd=dst)
    print(dst)

if __name__ == '__main__':
    main()
