import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / 'src'))
from scip_compare import make_initial
BIN = EXP / 'build' / 'pufoam_solver'
INP = EXP / 'inputs'
WORK = EXP / 'work' / 'stage0'
INP.mkdir(exist_ok=True)

def write17(path: Path, field: np.ndarray, f: int) -> None:
    rows = field.reshape(f, f)
    path.write_text('\n'.join((','.join((repr(float(v)) for v in r)) for r in rows)) + '\n')

def load(path: Path) -> np.ndarray:
    return np.array([float(t) for t in path.read_text().replace('\n', ',').split(',') if t.strip()])

def py_args(**kw):
    d = dict(field='zero', seed=7, random_low=1.0, random_high=30.0, center_height=100.0, center_size=16, field_size=50, initial_field=None)
    d.update(kw)
    return SimpleNamespace(**d)
manifest = {}

def register(name, desc, f):
    p = INP / name
    manifest[name] = dict(description=desc, field_size=f, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), mean=float(load(p).mean()))
for seed in range(1, 11):
    name = f'random_u1_30_seed{seed}_F50.csv'
    shutil.copyfile(WORK / f'regen_seed{seed}_initial.csv', INP / name)
    register(name, f'Uniform[1,30], std::mt19937 seed {seed} (C++ path); rounds to archived 10-digit CSV', 50)
checks = []
for name, cli, pyk in [('zero_F50.csv', ['--field', 'zero'], dict(field='zero')), ('center30_F50.csv', ['--field', 'center', '--center-height', '30'], dict(field='center', center_height=30.0)), ('center100_F50.csv', ['--field', 'center', '--center-height', '100'], dict(field='center', center_height=100.0))]:
    subprocess.run([str(BIN), *cli, '--budget', '625', '--max-exchanges', '0', '--quiet', '--write-initial', str(INP / name), '--out-prefix', str(WORK / 'tmp')], check=True, capture_output=True)
    py = make_initial(py_args(**pyk))
    checks.append(dict(input=name, max_abs_diff_cpp_vs_python=float(np.max(np.abs(load(INP / name) - py)))))
    register(name, f"deterministic {name.split('_')[0]} field (C++ writer, equals Python generator)", 50)
for name, pyk, desc in [('small_zero_F12.csv', dict(field='zero'), 'reduced zero field'), ('small_random_pcg64_seed7_F12.csv', dict(field='random', seed=7), 'reduced Uniform[1,30] field, NumPy PCG64 default_rng(7) — the only PCG64 path in the paper'), ('small_center30_c4_F12.csv', dict(field='center', center_height=30.0, center_size=4), 'reduced center field, height 30, 4x4')]:
    fld = make_initial(py_args(field_size=12, **pyk))
    write17(INP / name, fld, 12)
    assert np.array_equal(load(INP / name), fld)
    register(name, desc, 12)
outs = {}
for tag, cli in [('rng', ['--field', 'random', '--seed', '7', '--random-low', '1', '--random-high', '30']), ('csv', ['--initial-field', str(INP / 'random_u1_30_seed7_F50.csv')])]:
    p = subprocess.run([str(BIN), *cli, '--quiet', '--full-precision', '--out-prefix', str(WORK / f'initfield_{tag}')], check=True, capture_output=True, text=True)
    outs[tag] = dict(stdout=[l for l in p.stdout.splitlines() if l.startswith('final')][0], alloc=(WORK / f'initfield_{tag}_allocation.csv').read_bytes(), field=(WORK / f'initfield_{tag}_field.csv').read_bytes())
initfield = dict(final_line_rng=outs['rng']['stdout'], final_line_csv=outs['csv']['stdout'], identical_final_metrics=outs['rng']['stdout'] == outs['csv']['stdout'], identical_allocation=outs['rng']['alloc'] == outs['csv']['alloc'], identical_field=outs['rng']['field'] == outs['csv']['field'])
(INP / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
res = dict(deterministic_checks=checks, initial_field_option=initfield, n_inputs=len(manifest))
(EXP / 'results' / 'stage0' / 'inputs_check.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
