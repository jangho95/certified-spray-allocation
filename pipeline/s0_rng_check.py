import csv
import json
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parents[1]
ARCH = EXP / 'reference' / 'archive'
WORK = EXP / 'work' / 'stage0'

def read_tokens(path):
    with open(path, newline='') as fh:
        return [t for row in csv.reader(fh) for t in row if t.strip()]
rows = []
for seed in range(1, 11):
    regen = np.array([float(t) for t in read_tokens(WORK / f'regen_seed{seed}_initial.csv')])
    arch_tok = read_tokens(ARCH / f'robust_random_seed{seed}_initial.csv')
    arch = np.array([float(t) for t in arch_tok])
    text_equal = sum((f'{v:.10g}' == t for v, t in zip(regen, arch_tok)))
    rows.append(dict(seed=seed, n=len(regen), text_equal_10sig=text_equal, max_abs_diff=float(np.max(np.abs(regen - arch))), max_rel_diff=float(np.max(np.abs(regen - arch) / np.abs(regen)))))
seed7 = (ARCH / 'robust_random_seed7_initial.csv').read_bytes()
same = {p.name: p.read_bytes() == seed7 for p in sorted(ARCH.glob('*random*_initial.csv')) if 'seed' not in p.name or 'seed7' in p.name}
out = dict(per_seed=rows, archived_seed7_copies_identical=same, all_text_equal=all((r['text_equal_10sig'] == r['n'] for r in rows)))
(EXP / 'results' / 'stage0' / 'rng_check.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
