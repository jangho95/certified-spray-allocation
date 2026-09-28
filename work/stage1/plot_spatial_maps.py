from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
FIELD_SIZE = 50
CENTER_SIZE = 16
INSTANCES = [{'name': 'zero', 'alloc': 'zero_allocation.csv', 'final': 'zero_field.csv', 'initial': None, 'kind': 'zero'}, {'name': 'random', 'alloc': 'random_1_30_seed7_sync_allocation.csv', 'final': 'random_1_30_seed7_sync_field.csv', 'initial': 'random_1_30_seed7_initial.csv', 'kind': 'csv'}, {'name': 'center-30', 'alloc': 'center30_allocation.csv', 'final': 'center30_field.csv', 'initial': None, 'kind': 'center30'}, {'name': 'center-100', 'alloc': 'center100_allocation.csv', 'final': 'center100_field.csv', 'initial': None, 'kind': 'center100'}]

def read_field(path: str | Path) -> np.ndarray:
    rows = []
    with Path(path).open(newline='') as fh:
        for row in csv.reader(fh):
            rows.append([float(v) for v in row])
    return np.array(rows, dtype=float)

def make_initial(spec: dict) -> np.ndarray:
    if spec['kind'] == 'csv':
        return read_field(spec['initial'])
    field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=float)
    if spec['kind'].startswith('center'):
        height = float(spec['kind'].replace('center', ''))
        r0 = (FIELD_SIZE - CENTER_SIZE) // 2
        c0 = (FIELD_SIZE - CENTER_SIZE) // 2
        field[r0:r0 + CENTER_SIZE, c0:c0 + CENTER_SIZE] = height
    return field

def read_allocation(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols, shots = ([], [], [])
    with Path(path).open(newline='') as fh:
        for row in csv.DictReader(fh):
            rows.append(float(row['row']))
            cols.append(float(row['col']))
            shots.append(float(row['shot_count']))
    return (np.array(rows), np.array(cols), np.array(shots))

def plot_field_grid(fields: list[np.ndarray], names: list[str], out: str, title_suffix: str, cmap: str, label: str) -> None:
    vmin = min((float(f.min()) for f in fields))
    vmax = max((float(f.max()) for f in fields))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2), constrained_layout=True)
    ims = []
    for ax, name, field in zip(axes.ravel(), names, fields):
        im = ax.imshow(field, origin='lower', cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
        ims.append(im)
        ax.set_title(f'{name}: {title_suffix}')
        ax.set_xlabel('cell column')
        ax.set_ylabel('cell row')
        ax.set_aspect('equal')
    fig.colorbar(ims[0], ax=axes.ravel().tolist(), shrink=0.82, label=label)
    fig.savefig(out, dpi=300)
    print(f'wrote {out}')

def plot_spray_locations(specs: list[dict]) -> None:
    allocs = [read_allocation(spec['alloc']) for spec in specs]
    shot_min = min((float(s.min()) for _, _, s in allocs))
    shot_max = max((float(s.max()) for _, _, s in allocs))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2), constrained_layout=True)
    scatters = []
    for ax, spec, (rows, cols, shots) in zip(axes.ravel(), specs, allocs):
        sc = ax.scatter(cols, rows, c=shots, s=22, marker='s', cmap='viridis', vmin=shot_min, vmax=shot_max, linewidths=0)
        scatters.append(sc)
        ax.set_title(f"{spec['name']}: spray locations")
        ax.set_xlim(-1, FIELD_SIZE)
        ax.set_ylim(-1, FIELD_SIZE)
        ax.set_xlabel('cell column')
        ax.set_ylabel('cell row')
        ax.set_aspect('equal')
        ax.grid(color='0.85', linewidth=0.4)
    fig.colorbar(scatters[0], ax=axes.ravel().tolist(), shrink=0.82, label='shot count')
    fig.savefig('spray_location_counts.png', dpi=300)
    print('wrote spray_location_counts.png')

def main() -> None:
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7})
    names = [spec['name'] for spec in INSTANCES]
    initials = [make_initial(spec) for spec in INSTANCES]
    finals = [read_field(spec['final']) for spec in INSTANCES]
    plot_field_grid(initials, names, 'initial_fields.png', 'initial field', 'magma', 'initial thickness (mm)')
    plot_spray_locations(INSTANCES)
    plot_field_grid(finals, names, 'final_fields.png', 'final field', 'magma', 'final thickness (mm)')
if __name__ == '__main__':
    main()
