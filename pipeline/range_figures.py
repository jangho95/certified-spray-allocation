from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/stage7/range_rows.json'
OUT = ROOT / 'results/revision_figures'
FIELDS = [('zero', 'Zero'), ('random s7', 'Random'), ('center-30', 'Center-30'), ('center-100', 'Center-100')]
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#b87800']
WIDTH = 'rel_width_final_pct'

def axes_style(ax):
    ax.set_yscale('log')
    ax.grid(True, alpha=0.22)
    ax.tick_params(labelsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for y in (1, 5):
        ax.axhline(y, color='0.35', linewidth=0.8, linestyle='--')
        ax.text(0.02, y, f'{y}%', transform=ax.get_yaxis_transform(), fontsize=8, va='bottom', color='0.25')

def draw(ax, rows, xkey):
    for (field, label), color in zip(FIELDS, COLORS):
        values = sorted((r for r in rows if r['field'] == field), key=lambda r: r[xkey])
        ax.plot([r[xkey] for r in values], [r[WIDTH] for r in values], color=color, marker='o', markersize=3, linewidth=1.3, label=label)
    axes_style(ax)

def main():
    rows = json.loads(SOURCE.read_text())['rows']
    a = [r for r in rows if r['series'] == 'A']
    b = [r for r in rows if r['series'] == 'B']
    assert len(a) == 24 and len(b) == 80
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10})
    fig, ax = plt.subplots(figsize=(5.0, 2.7), layout='constrained')
    draw(ax, a, 'BN')
    ax.set_xscale('log')
    xs = sorted({r['BN'] for r in a})
    ax.set_xticks(xs, [f'{x:.3g}' for x in xs])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Average shots per waypoint, $B/N$')
    ax.set_ylabel('Relative width (%)')
    ax.set_title('Series A: $7\\times7$ kernel, $\\sigma=1.75$', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.savefig(OUT / 'stage7_seriesA.png', dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.1), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.88, hspace=0.52, wspace=0.12)
    for ax, rho in zip(axes.flat, sorted({r['rho'] for r in b})):
        draw(ax, [r for r in b if r['rho'] == rho], 'sigma_over_s')
        ax.set_title(f'$B/N={rho:g}$', fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel('Kernel width / spacing, $\\sigma/s$', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('Relative width (%)', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.55, 0.005))
    fig.suptitle('Series B: support $\\pm\\lceil3\\sigma\\rceil$, unnormalized mass', fontsize=10)
    fig.savefig(OUT / 'stage7_seriesB.png', dpi=300)
    plt.close(fig)
    (OUT / 'range_figure_data.json').write_text(json.dumps({'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'value_field': WIDTH, 'rows': [{k: r[k] for k in ('series', 'field', 'BN', 'rho', 'sigma_over_s', WIDTH)} for r in rows], 'outputs': ['figures/revision/stage7_seriesA.png', 'figures/revision/stage7_seriesB.png']}, ensure_ascii=False, indent=2) + '\n')
if __name__ == '__main__':
    main()
