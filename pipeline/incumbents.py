from __future__ import annotations
from package_paths import load_json
import datetime as _dt
import hashlib
import json
import os
import shutil
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
ROOT = EXP / 'results' / 'incumbents'
ALLOC = ROOT / 'alloc'
REG = ROOT / 'registry.jsonl'
BEST = ROOT / 'best.json'

def new_run_id(script: str) -> str:
    return f'{Path(script).stem}-{_dt.datetime.now():%Y%m%dT%H%M%S}-{os.getpid()}'
INPUTS = EXP / 'inputs'

def input_sha(inp: str) -> str:
    return hashlib.sha256((INPUTS / inp).read_bytes()).hexdigest()

def key(inp: str, B: int, cap: int, *, F: int=50, kernel: int=7, sigma=(1.75, 1.75), stamp_mass=None, boundary: str='reflect', spacing: int=2) -> str:
    norm = 'raw' if stamp_mass is None else f'mass{stamp_mass:g}'
    return f'in:{input_sha(inp)[:16]}|F{F}|k{kernel}|sx{sigma[0]:g}|sy{sigma[1]:g}|{norm}|{boundary}|s{spacing}|B{B}|C{cap}'

def register(*, inp, B, cap, method, V, alloc_path, run_id, feasible, F=50, kernel=7, sigma=(1.75, 1.75), stamp_mass=None, boundary='reflect', spacing=2, **extra) -> dict:
    ALLOC.mkdir(parents=True, exist_ok=True)
    data = Path(alloc_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    stored = ALLOC / f'{sha}.csv'
    if not stored.exists():
        shutil.copyfile(alloc_path, stored)
    rec = dict(key=key(inp, B, cap, F=F, kernel=kernel, sigma=sigma, stamp_mass=stamp_mass, boundary=boundary, spacing=spacing), input=inp, input_sha256=input_sha(inp), B=B, cap=cap, F=F, kernel=kernel, sigma=list(sigma), stamp_mass=stamp_mass, boundary=boundary, spacing=spacing, method=method, V=float(V), feasible=bool(feasible), alloc_sha256=sha, alloc_file=str(stored.relative_to(EXP)), run_id=run_id, time=_dt.datetime.now().isoformat(timespec='seconds'), **extra)
    with open(REG, 'a') as fh:
        fh.write(json.dumps(rec, default=str, ensure_ascii=False) + '\n')
    return rec

def rebuild_best() -> dict:
    best = {}
    if REG.exists():
        for line in REG.read_text().splitlines():
            r = load_json(line)
            if r['feasible'] and (r['key'] not in best or r['V'] < best[r['key']]['V']):
                best[r['key']] = r
    BEST.write_text(json.dumps(best, indent=1, ensure_ascii=False))
    return best

def best_for(inp, B, cap, **model) -> dict | None:
    b = load_json(BEST.read_text()) if BEST.exists() else {}
    return b.get(key(inp, B, cap, **model))

def migrate_v1() -> int:
    if not REG.exists():
        return 0
    out, n = ([], 0)
    for line in REG.read_text().splitlines():
        r = load_json(line)
        if 'input_sha256' not in r:
            r['key_v1'] = r['key']
            r.update(input_sha256=input_sha(r['input']), F=50, sigma=[1.75, 1.75], stamp_mass=None, boundary='reflect')
            r['key'] = key(r['input'], r['B'], r['cap'], kernel=r.get('kernel', 7), spacing=r.get('spacing', 2))
            n += 1
        out.append(json.dumps(r, default=str, ensure_ascii=False))
    REG.write_text('\n'.join(out) + '\n')
    rebuild_best()
    return n
