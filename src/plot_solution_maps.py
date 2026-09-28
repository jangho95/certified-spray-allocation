from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
INSTANCES = [('zero', 'zero_allocation.csv', 'zero_field.csv', (25, 25)), ('random', 'random_1_30_seed7_sync_allocation.csv', 'random_1_30_seed7_sync_field.csv', (25, 25)), ('center-30', 'center30_allocation.csv', 'center30_field.csv', (25, 25)), ('center-100', 'center100_allocation.csv', 'center100_field.csv', (25, 25))]

def read_allocation(path: Path, shape: tuple[int, int]) -> np.ndarray:
    vals = np.zeros(shape[0] * shape[1], dtype=float)
    with path.open(newline='') as fh:
        for row in csv.DictReader(fh):
            vals[int(row['waypoint'])] = float(row['shot_count'])
    return vals.reshape(shape)

def read_field(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline='') as fh:
        for row in csv.reader(fh):
            rows.append([float(v) for v in row])
    return np.array(rows, dtype=float)

def main() -> None:
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7})
    data = []
    for title, alloc, field, shape in INSTANCES:
        data.append((title, read_allocation(Path(alloc), shape), read_field(Path(field))))
    shot_vmin = min((float(x.min()) for _, x, _ in data))
    shot_vmax = max((float(x.max()) for _, x, _ in data))
    field_vmin = min((float(f.min()) for _, _, f in data))
    field_vmax = max((float(f.max()) for _, _, f in data))
    fig, axes = plt.subplots(len(INSTANCES), 2, figsize=(7.2, 9.8), constrained_layout=True)
    shot_images = []
    field_images = []
    for r, (title, x, f) in enumerate(data):
        ax0, ax1 = axes[r]
        im0 = ax0.imshow(x, origin='lower', cmap='viridis', interpolation='nearest', vmin=shot_vmin, vmax=shot_vmax)
        im1 = ax1.imshow(f, origin='lower', cmap='magma', interpolation='nearest', vmin=field_vmin, vmax=field_vmax)
        shot_images.append(im0)
        field_images.append(im1)
        ax0.set_title(f'{title}: shots')
        ax1.set_title(f'{title}: thickness')
        ax0.set_xlabel('waypoint column')
        ax0.set_ylabel('waypoint row')
        ax1.set_xlabel('cell column')
        ax1.set_ylabel('cell row')
        ax0.set_aspect('equal')
        ax1.set_aspect('equal')
    fig.colorbar(shot_images[0], ax=axes[:, 0], shrink=0.78, label='shots')
    fig.colorbar(field_images[0], ax=axes[:, 1], shrink=0.78, label='thickness (mm)')
    fig.savefig('solution_maps.png', dpi=300)
    print('wrote solution_maps.png')
if __name__ == '__main__':
    main()
