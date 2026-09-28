from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent / 'paper'
TIMES_FONTS = [Path('/mnt/c/Windows/Fonts/times.ttf'), Path('/mnt/c/Windows/Fonts/timesbd.ttf'), Path('/mnt/c/Windows/Fonts/timesi.ttf'), Path('/mnt/c/Windows/Fonts/timesbi.ttf')]

def configure_fonts() -> str:
    family = 'DejaVu Serif'
    for font in TIMES_FONTS:
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 9.5, 'axes.titlesize': 11, 'axes.labelsize': 9.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'mathtext.fontset': 'stix', 'axes.linewidth': 0.7})
    return family

def read_field(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=',')

def read_allocation(path: Path):
    rows, cols, shots = ([], [], [])
    with path.open(newline='') as fh:
        for row in csv.DictReader(fh):
            rows.append(float(row['row']))
            cols.append(float(row['col']))
            shots.append(float(row['shot_count']))
    return (np.array(rows), np.array(cols), np.array(shots))

def center100_initial(size: int=50, center_size: int=16, height: float=100.0):
    field = np.zeros((size, size), dtype=float)
    start = (size - center_size) // 2
    field[start:start + center_size, start:start + center_size] = height
    return field

def gaussian_stamp(kernel_size: int=7, sigma: float=1.75):
    radius = kernel_size // 2
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    stamp = np.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
    stamp /= stamp.max()
    return (yy, xx, stamp)

def style_field_axis(ax):
    ax.set_xlim(-0.5, 49.5)
    ax.set_ylim(49.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('column')
    ax.set_ylabel('row')
    ax.set_xticks([0, 10, 20, 30, 40, 49])
    ax.set_yticks([0, 10, 20, 30, 40, 49])

def panel_label(ax, text):
    ax.text(0.015, 0.985, text, transform=ax.transAxes, ha='left', va='top', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.8))

def main():
    family = configure_fonts()
    initial = center100_initial()
    final = read_field(ROOT / 'center100_field.csv')
    row, col, shot = read_allocation(ROOT / 'center100_allocation.csv')
    way_rows, way_cols = np.meshgrid(np.arange(0, 50, 2), np.arange(0, 50, 2), indexing='ij')
    yy, xx, stamp = gaussian_stamp()
    fig = plt.figure(figsize=(13.2, 3.75), dpi=300)
    gs = fig.add_gridspec(1, 4, left=0.045, right=0.985, bottom=0.16, top=0.86, wspace=0.38)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    im0 = axes[0].imshow(initial, origin='upper', cmap='magma', vmin=0, vmax=220)
    axes[0].set_title('Initial thickness field')
    style_field_axis(axes[0])
    panel_label(axes[0], '(a)')
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.02)
    cb0.set_label('height')
    axes[1].imshow(initial, origin='upper', cmap='Greys', vmin=0, vmax=180, alpha=0.18)
    axes[1].scatter(way_cols.ravel(), way_rows.ravel(), s=7, color='#4b5563', alpha=0.55, linewidths=0)
    for cy, cx, color in [(24, 24, '#0072B2'), (4, 4, '#009E73'), (44, 10, '#D55E00')]:
        levels = [0.2, 0.45, 0.7]
        axes[1].contour(cx + xx, cy + yy, stamp, levels=levels, colors=color, linewidths=0.9, alpha=0.9)
        axes[1].scatter([cx], [cy], s=26, marker='x', color=color, linewidths=1.2)
    axes[1].set_title('Candidate waypoints, s = 2')
    style_field_axis(axes[1])
    panel_label(axes[1], '(b)')
    axes[1].text(0.5, -0.17, 'fixed boundary-folded Gaussian stamps', transform=axes[1].transAxes, ha='center', va='top', fontsize=8.5)
    sc = axes[2].scatter(col, row, c=shot, s=20, marker='s', cmap='viridis', edgecolor='none', vmin=shot.min(), vmax=shot.max())
    axes[2].set_title('Integer shot-count vector')
    style_field_axis(axes[2])
    panel_label(axes[2], '(c)')
    cb2 = fig.colorbar(sc, ax=axes[2], fraction=0.045, pad=0.02)
    cb2.set_label('shots')
    im3 = axes[3].imshow(final, origin='upper', cmap='magma', vmin=180, vmax=220)
    axes[3].set_title('Final thickness field')
    style_field_axis(axes[3])
    panel_label(axes[3], '(d)')
    cb3 = fig.colorbar(im3, ax=axes[3], fraction=0.045, pad=0.02)
    cb3.set_label('height')
    axes[3].text(0.5, -0.17, '$f(x)=s_0+Ax,\\quad x_j\\in\\mathbb{Z}$; minimize $|\\mu-T|$ and $V$', transform=axes[3].transAxes, ha='center', va='top', fontsize=8.5)
    fig.suptitle('Stop-and-spray allocation: choose integer shots at actual spray locations', y=0.965, fontsize=12.5, fontweight='bold')
    out_solver = ROOT / 'problem_overview.png'
    out_paper = PAPER / 'problem_overview.png'
    fig.savefig(out_solver, dpi=300)
    fig.savefig(out_paper, dpi=300)
    print(f'font={family}')
    print(f'wrote {out_solver}')
    print(f'wrote {out_paper}')
if __name__ == '__main__':
    main()
