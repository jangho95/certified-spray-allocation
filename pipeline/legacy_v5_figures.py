from pathlib import Path
import csv
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reference/legacy_figure_inputs'
OUT = ROOT / 'results/revision_figures'

def read_csv(name):
    with (SOURCE / name).open(newline='') as handle:
        return list(csv.DictReader(handle))

def save(fig, stem):
    for extension in ('pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{extension}', dpi=300)
    plt.close(fig)

def stamp(sx, sy, size):
    radius = size // 2
    return np.array([[math.exp(-0.5 * ((x / sx) ** 2 + (y / sy) ** 2)) for x in range(-radius, radius + 1)] for y in range(-radius, radius + 1)])

def overview():
    initial = np.zeros((50, 50))
    initial[17:33, 17:33] = 100
    final = np.loadtxt(SOURCE / 'center100_field.csv', delimiter=',')
    allocation = read_csv('center100_allocation.csv')
    rows = np.array([int(r['row']) for r in allocation])
    cols = np.array([int(r['col']) for r in allocation])
    shots = np.array([int(r['shot_count']) for r in allocation])
    assert final.shape == initial.shape and len(shots) == 625
    assert shots.sum() == 26896 and np.all((shots >= 1) & (shots <= 200))
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 4.8), layout='constrained')
    ax = axes[0, 0]
    im = ax.imshow(initial, cmap='magma', vmin=0, vmax=100)
    ax.set_title('(a) Initial field', loc='left')
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.025, label='Thickness')
    ax = axes[0, 1]
    ax.imshow(initial, cmap='Greys', vmin=0, vmax=180, alpha=0.18)
    ax.scatter(cols, rows, s=2.5, c='0.5', linewidths=0)
    yy, xx = np.mgrid[-3:4, -3:4]
    kernel = stamp(1.75, 1.75, 7)
    for row, col, color in [(24, 24, '#0072B2'), (4, 4, '#009E73'), (44, 10, '#D55E00')]:
        ax.contour(col + xx, row + yy, kernel, levels=[0.2, 0.45, 0.7], colors=color, linewidths=0.8)
        ax.scatter([col], [row], marker='x', c=color, s=18)
    ax.set_title('(b) Waypoints, $s=2$', loc='left')
    ax = axes[1, 0]
    im = ax.scatter(cols, rows, c=shots, marker='s', s=10, cmap='viridis', linewidths=0)
    ax.set_title('(c) Integer allocation', loc='left')
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.025, label='Shots')
    ax = axes[1, 1]
    im = ax.imshow(final, cmap='magma', vmin=180, vmax=220)
    ax.set_title('(d) Final field', loc='left')
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.025, label='Thickness')
    for ax in axes.flat:
        ax.set(xlim=(-0.5, 49.5), ylim=(49.5, -0.5), aspect='equal', xlabel='Column (cell)', ylabel='Row (cell)', xticks=[0, 25, 49], yticks=[0, 25, 49])
    save(fig, 'problem_overview_rev')
    return {'budget': int(shots.sum()), 'variance_from_saved_field': float(final.var()), 'allocation_rows': len(shots), 'initial_patch': [17, 33, 100]}

def kernel_shapes():
    records = read_csv('gaussian_distortion_ablation.csv')
    base = [r for r in records if r['instance'] == 'zero'][:6]
    names = ['iso-7', 'iso-11', 'x7 raw', 'y7 raw', 'x7 norm', 'y7 norm']
    arrays, details = ([], [])
    for name, row in zip(names, base):
        size = int(row['kernel_size'])
        a = stamp(float(row['sigma_x']), float(row['sigma_y']), size)
        if row['mass_normalized'] == 'True':
            a *= float(base[0]['stamp_mass']) / a.sum()
        assert abs(float(a.sum()) - float(row['stamp_mass'])) < 1e-12
        arrays.append(a)
        details.append({'label': name, 'size': size, 'mass': float(a.sum()), 'sigma_x': float(row['sigma_x']), 'sigma_y': float(row['sigma_y'])})
    fig, axes = plt.subplots(3, 2, figsize=(5.8, 6.5), layout='constrained')
    vmax = max((float(a.max()) for a in arrays))
    for i, (ax, a, d) in enumerate(zip(axes.flat, arrays, details)):
        radius = d['size'] // 2
        im = ax.imshow(a, origin='lower', cmap='viridis', vmin=0, vmax=vmax, extent=(-radius - 0.5, radius + 0.5, -radius - 0.5, radius + 0.5))
        ax.set_title(f"({chr(97 + i)}) {d['label']}\n$k={d['size']}$, $\\kappa={d['mass']:.3f}$", loc='left', fontsize=9.5)
        ax.set(xlabel='Column offset (cell)', ylabel='Row offset (cell)', xticks=[-radius, 0, radius], yticks=[-radius, 0, radius])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.025, label='Kernel weight')
    save(fig, 'gaussian_kernel_shapes_rev')
    np.savez(OUT / 'kernel_shapes.npz', **{f'kernel_{i}': a for i, a in enumerate(arrays)})
    return details

def low_count():
    rows = [{k: float(v) for k, v in row.items()} for row in read_csv('cmax_count_ablation.csv')]
    assert len(rows) == 6
    x = [r['avg_count'] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(5.8, 6.4), sharex=True, layout='constrained')
    for key, label, color, marker in [('var_upper', 'Selected allocation (two candidates)', '#0072B2', 'o'), ('var_round', 'Rounding alone', '#D55E00', 's'), ('var_lower', 'Numerical lower bound', '#009E73', '^')]:
        axes[0].plot(x, [r[key] for r in rows], marker=marker, color=color, label=label, markersize=4, linewidth=1.2)
    axes[0].set(title='(a) Objective values', ylabel='Variance')
    axes[0].legend(frameon=False, fontsize=8.5, loc='upper left')
    axes[1].plot(x, [r['gap_pct'] for r in rows], 'o-', color='#0072B2', label='Selected allocation (two candidates)')
    axes[1].plot(x, [r['round_gap_pct'] for r in rows], 's--', color='#D55E00', label='Rounding alone')
    axes[1].set(title='(b) Numerical widths', ylabel='Relative width (%)')
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[2].plot(x, [r['time_s'] for r in rows], 'o-', color='#0072B2')
    axes[2].set(title='(c) Recorded time: single run per budget', ylabel='Recorded computation time (s)', xlabel='Average shots per waypoint, $B/N$')
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(x, [f'{v:.3g}' for v in x])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=0.22)
    save(fig, 'cmax_count_ablation_rev')
    return rows

def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    for name, entry in manifest.items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == entry['sha256']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42})
    evidence = {'overview': overview(), 'kernel_shapes': kernel_shapes(), 'low_count': low_count(), 'optimization_rerun': False, 'inputs': manifest}
    (OUT / 'legacy_v5_figure_data.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print('Redrew 3 figures from preserved inputs; wrote PDFs, PNGs and data record.')
if __name__ == '__main__':
    main()
