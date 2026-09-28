from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'reference/initial_fields_3d_inputs'
OUT = ROOT / 'results/revision_figures'
FIELDS = [('zero_F50.csv', '(a) Zero'), ('random_u1_30_seed7_F50.csv', '(b) Random'), ('center30_F50.csv', '(c) Center-30'), ('center100_F50.csv', '(d) Center-100')]

def cell_surface(values, cmap, norm):
    faces, heights = ([], [])
    rows, cols = values.shape
    for r in range(rows):
        for c in range(cols):
            z = float(values[r, c])
            xl, xr, yl, yr = (c - 0.5, c + 0.5, r - 0.5, r + 0.5)
            faces.append([(xl, yl, z), (xr, yl, z), (xr, yr, z), (xl, yr, z)])
            heights.append(z)
            if c + 1 < cols and values[r, c + 1] != z:
                h = float(values[r, c + 1])
                faces.append([(xr, yl, z), (xr, yr, z), (xr, yr, h), (xr, yl, h)])
                heights.append((z + h) / 2)
            if r + 1 < rows and values[r + 1, c] != z:
                h = float(values[r + 1, c])
                faces.append([(xl, yr, z), (xr, yr, z), (xr, yr, h), (xl, yr, h)])
                heights.append((z + h) / 2)
    return Poly3DCollection(faces, facecolors=cmap(norm(heights)), edgecolors='none', antialiased=False, zsort='average', rasterized=True)

def main():
    manifest = json.loads((INPUT / 'manifest.json').read_text())
    fields, summaries = ([], {})
    for name, title in FIELDS:
        path = INPUT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[name]['sha256']
        values = np.loadtxt(path, delimiter=',')
        assert values.shape == (50, 50) and np.isfinite(values).all()
        fields.append(values)
        summaries[name] = {'sha256': manifest[name]['sha256'], 'minimum': float(values.min()), 'maximum': float(values.max()), 'mean': float(values.mean()), 'shape': list(values.shape)}
    assert np.all(fields[0] == 0)
    assert np.all((fields[1] >= 1) & (fields[1] <= 30))
    for values, height in zip(fields[2:], [30, 100]):
        expected = np.zeros((50, 50))
        expected[17:33, 17:33] = height
        assert np.array_equal(values, expected)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 12, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42})
    cmap, norm = (matplotlib.colormaps['viridis'], Normalize(0, 100))
    fig = plt.figure(figsize=(7.3, 6.6))
    grid = fig.add_gridspec(2, 2, left=0.035, right=0.94, bottom=0.13, top=0.96, wspace=0.14, hspace=0.18)
    for i, ((name, title), values) in enumerate(zip(FIELDS, fields)):
        ax = fig.add_subplot(grid[i // 2, i % 2], projection='3d')
        ax.add_collection3d(cell_surface(values, cmap, norm))
        ax.set(xlim=(-0.5, 49.5), ylim=(-0.5, 49.5), zlim=(0, 100), xticks=[0, 25, 49], yticks=[0, 25, 49], zticks=[0, 30, 60, 100])
        ax.set_xlabel('Column (cell)', labelpad=0)
        ax.set_ylabel('Row (cell)', labelpad=0)
        ax.set_zlabel('Initial thickness', labelpad=3)
        ax.set_title(title, loc='left', pad=2)
        ax.set_proj_type('ortho')
        ax.set_box_aspect((1, 1, 0.8))
        ax.view_init(elev=25, azim=-55)
        ax.tick_params(pad=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo['grid']['color'] = (0.7, 0.7, 0.7, 0.3)
    bar_ax = fig.add_axes([0.26, 0.07, 0.48, 0.017])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar_ax, orientation='horizontal', ticks=[0, 30, 60, 100])
    cb.set_label('Initial thickness (model units)', labelpad=3)
    cb.outline.set_linewidth(0.5)
    OUT.mkdir(exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(OUT / f'initial_fields_3d.{extension}', dpi=450)
    plt.close(fig)
    evidence = {'fields': summaries, 'z_limits': [0, 100], 'color_limits': [0, 100], 'representation': 'piecewise constant cell tops with vertical jumps', 'camera': {'projection': 'orthographic', 'elevation': 25, 'azimuth': -55}, 'optimization_rerun': False}
    (OUT / 'initial_fields_3d_data.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print('Generated initial_fields_3d.pdf and .png from four canonical CSV inputs.')
if __name__ == '__main__':
    main()
