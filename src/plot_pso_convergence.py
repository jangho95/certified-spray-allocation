from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FILES = [('zero', Path('pso_long_zero_seed1_history.csv'), 54.077695), ('random', Path('pso_long_random_seed1_history.csv'), 102.93372), ('center-30', Path('pso_long_center30_seed1_history.csv'), 55.86483), ('center-100', Path('pso_long_center100_seed1_history.csv'), 93.780756)]

def read_history(path: Path):
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r['iter']), float(r['gbest_variance'])))
    return rows

def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    print(f"{'instance':12s} {'V500':>10s} {'V5000':>10s} {'V_IQP':>10s} {'ratio500':>10s} {'ratio5000':>10s}")
    for ax, (name, path, viqp) in zip(axes.ravel(), FILES):
        rows = read_history(path)
        it = [r[0] for r in rows]
        var = [r[1] for r in rows]
        v500 = min(rows, key=lambda r: abs(r[0] - 500))[1]
        v5000 = rows[-1][1]
        print(f'{name:12s} {v500:10.3f} {v5000:10.3f} {viqp:10.3f} {v500 / viqp:10.3f} {v5000 / viqp:10.3f}')
        ax.plot(it, var, color='black', lw=1.6, label='PSO gbest')
        ax.axhline(viqp, color='tab:red', ls='--', lw=1.2, label='IQP-feas.')
        ax.axvline(500, color='gray', ls=':', lw=1)
        ax.set_title(name)
        ax.set_xlabel('PSO iteration')
        ax.set_ylabel('variance')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('pso_convergence.png', dpi=300)
    print('wrote pso_convergence.png')
if __name__ == '__main__':
    main()
