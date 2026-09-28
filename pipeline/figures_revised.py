import determinism
import json
import argparse
import hashlib
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from decimal import ROUND_CEILING, Decimal

def up4(v):
    return str(Decimal(repr(float(v))).quantize(Decimal('0.0001'), rounding=ROUND_CEILING))
EXP = Path(__file__).resolve().parents[1]
OUT = EXP / 'results' / 'revision_figures'
BS = EXP / 'results' / 'bench_sweep'
FIELDS = [('zero', 'Zero', 'zero_F50.csv'), ('random s7', 'Random', 'random_u1_30_seed7_F50.csv'), ('center-30', 'Center-30', 'center30_F50.csv'), ('center-100', 'Center-100', 'center100_F50.csv')]
T, M = (200.0, 2500)

def load_field(inp):
    return np.array([float(t) for t in (EXP / 'inputs' / inp).read_text().replace('\n', ',').split(',') if t.strip()])

def fronts():
    runs = {r['name']: r for r in json.loads((BS / 'runs.json').read_text())['results']}
    rig = {f['name']: f for f in json.loads((BS / 'rigor.json').read_text())['fields']}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.9, wspace=0.12, hspace=0.42)
    handles = {}
    for ax, (name, title, inp) in zip(axes.ravel(), FIELDS):
        r = runs[name]
        rows = sorted(r['rows'], key=lambda x: x['B'])
        rr = {x['B']: x for x in rig[name]['rows']}
        s0 = load_field(inp)
        with np.load(BS / 'operator.npz') as op:
            ptr = op['col_ptr']
            kappa = float(np.mean([sum(op['weights'][ptr[j]:ptr[j + 1]]) for j in range(len(ptr) - 1)]))
        mean = np.array([s0.mean() + kappa * x['B'] / M for x in rows])
        w2 = np.array([x['W_paper'] for x in rows])
        w3 = np.array([x['W_rev'] for x in rows])
        wv = np.array([rr[x['B']]['W_rig_pct_up'] for x in rows])
        ax.fill_between(mean, 0, w2, color='tab:orange', alpha=0.18, label='width, two-candidate rule')
        ax.plot(mean, w2, '-o', ms=3.5, lw=1.6, color='tab:red', label='rule (2), double precision')
        ax.fill_between(mean, 0, wv, color='tab:blue', alpha=0.22, label='width, rule (3)')
        ax.plot(mean, w3, '-s', ms=3.5, lw=1.6, color='tab:blue', label='rule (3), double precision')
        ax.plot(mean, wv, '--', lw=1.2, color='black', label='rule (3), verified (rounded up)')
        ax.axvline(T, color='gray', ls='--', lw=1, label='target T')
        k = int(np.argmin(np.abs(mean - T)))
        ax.scatter([mean[k]], [wv[k]], s=40, color='black', zorder=5, label='target-budget sample')
        ax.annotate(f'(3): {up4(wv[k])}%\n(2): {w2[k]:.3f}%', xy=(mean[k], wv[k]), xytext=(8, 14), textcoords='offset points', fontsize=8)
        ax.set_title(f"{title}\nB*={r['bstar']}, max (2) {w2.max():.3f}%, max (3) verified {up4(wv.max())}%", fontsize=9.5)
        ax.grid(alpha=0.28)
        ax.tick_params(labelsize=9)
        ax.set_ylim(0, 0.62)
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    for ax in axes[1]:
        ax.set_xlabel('mean thickness (model units)')
    for ax in axes[:, 0]:
        ax.set_ylabel('relative certificate width (%)')
    fig.legend(handles.values(), handles.keys(), loc='lower center', ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.savefig(OUT / 'pareto_fronts_rev.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def maps():
    with np.load(BS / 'operator.npz') as op:
        ptr, cells, weights = (op['col_ptr'], op['cells'], op['weights'])
        A = np.zeros((M, len(ptr) - 1))
        for j in range(len(ptr) - 1):
            np.add.at(A[:, j], cells[ptr[j]:ptr[j + 1]], weights[ptr[j]:ptr[j + 1]])
    runs = {r['name']: r for r in json.loads((BS / 'runs.json').read_text())['results']}
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.8), constrained_layout=True)
    info = []
    for (name, title, inp), (ax0, ax1) in zip(FIELDS, axes):
        B = runs[name]['bstar']
        record = next((r for r in runs[name]['rows'] if r['B'] == B))
        witness = BS / 'witness' / (name + '.npz')
        with np.load(witness) as w:
            k = int(np.flatnonzero(w['B'] == B)[0])
            x = w['alloc'][k].astype(float)
            assert np.array_equal(w['s0'], load_field(inp))
            assert np.all(x == np.rint(x)) and int(x.sum()) == B
            assert np.all((x >= 1) & (x <= int(w['Cmax'])))
        s0 = load_field(inp)
        f = A @ x + s0
        assert np.isclose(np.var(f), record['U_rev'], rtol=1e-12, atol=1e-12)
        info.append(dict(field=name, B=B, method=record['U_method'], V=record['U_rev'], witness='results/bench_sweep/witness/' + witness.name, witness_sha256=hashlib.sha256(witness.read_bytes()).hexdigest(), operator_sha256=hashlib.sha256((BS / 'operator.npz').read_bytes()).hexdigest(), mean=float(f.mean()), var=float(np.mean((f - f.mean()) ** 2))))
        im0 = ax0.imshow(x.reshape(25, 25), origin='lower', cmap='viridis', interpolation='nearest')
        im1 = ax1.imshow(f.reshape(50, 50), origin='lower', cmap='magma', interpolation='nearest')
        ax0.set_title(f'{title}: shot counts (B*={B})', fontsize=9)
        ax1.set_title(f"final field (mean {f.mean():.2f}, var {info[-1]['var']:.3f})", fontsize=9)
        for ax in (ax0, ax1):
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    fig.savefig(OUT / 'solution_maps_rev.png', dpi=300)
    plt.close(fig)
    (OUT / 'solution_maps_rev.json').write_text(json.dumps(info, indent=1, ensure_ascii=False))
    return info
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=OUT)
    OUT = parser.parse_args().output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    fronts()
    for i in maps():
        print(i)
