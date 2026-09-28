from __future__ import annotations
import csv
import math
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from convex_qp_tools import dense_A, largest_remainder_round, solve_box_qp_relaxation
from scip_compare import build_stamps, evaluate, make_initial, make_waypoints
CPP = './build/pufoam_solver'
FINAL = re.compile('mean=([\\d.eE+-]+) variance=([\\d.eE+-]+).*?total_shots=(\\d+)')
TIME = re.compile('total_ms=(\\d+)')
TARGET = 200.0
DEFAULTS = dict(field='zero', seed=1, random_low=1.0, random_high=30.0, center_height=100.0, center_size=16, target=TARGET, field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, cmax=200, budget=None, initial_field=None)
INSTANCES = [('zero', ['--field', 'zero'], dict(field='zero')), ('random', ['--initial-field', 'inputs/random_u1_30_seed7_F50.csv'], dict(field='random', seed=7, random_low=1.0, random_high=30.0)), ('center30', ['--field', 'center', '--center-height', '30'], dict(field='center', center_height=30.0)), ('center100', ['--field', 'center', '--center-height', '100'], dict(field='center', center_height=100.0))]
ISO_MASS_7 = None
KERNELS = [('isotropic-7', 1.75, 1.75, 7, None), ('isotropic-11', 1.75, 1.75, 11, None), ('x-long-7-raw', 2.5, 1.225, 7, None), ('y-long-7-raw', 1.225, 2.5, 7, None), ('x-long-7-norm', 2.5, 1.225, 7, 'iso7'), ('y-long-7-norm', 1.225, 2.5, 7, 'iso7'), ('x-long-11-raw', 2.5, 1.225, 11, None), ('y-long-11-raw', 1.225, 2.5, 11, None)]

def run_cpp(cli: list[str], kernel: tuple[str, float, float], out_prefix: str, initial_path: Path) -> dict[str, float]:
    _, sx, sy, ksize, target_mass = kernel
    cmd = [CPP, *cli, '--sigma-x', str(sx), '--sigma-y', str(sy), '--kernel-size', str(ksize), '--quiet', '--out-prefix', out_prefix, '--write-initial', str(initial_path), '--full-precision']
    if target_mass is not None:
        cmd += ['--stamp-mass', str(iso7_mass())]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    mf = FINAL.search(out)
    mt = TIME.search(out)
    if mf is None or mt is None:
        raise RuntimeError(out)
    return {'mean': float(mf.group(1)), 'var_upper': float(mf.group(2)), 'budget': int(mf.group(3)), 'time_int_s': int(mt.group(1)) / 1000.0}

def iso7_mass() -> float:
    return float(kernel_stamp(1.75, 1.75, 7).sum())

def scale_stamps(stamps, target_mass: float | None):
    if target_mass is None:
        return stamps
    for st in stamps:
        scale = target_mass / st.mass
        st.weights = [w * scale for w in st.weights]
        st.mass = sum(st.weights)
    return stamps

def qp_lb(args: SimpleNamespace, budget: int) -> tuple[float, float, float, str]:
    pts = make_waypoints(args)
    stamps = build_stamps(args, pts)
    stamps = scale_stamps(stamps, getattr(args, 'stamp_mass', None))
    initial = make_initial(args)
    m = args.field_size ** 2
    n = len(stamps)
    mass = stamps[0].mass
    t0 = time.time()
    res = solve_box_qp_relaxation(dense_A(stamps, m, n), initial, budget, mass, args.cmax)
    qptime = time.time() - t0
    x_round = largest_remainder_round(res.x, budget, args.cmax)
    _, v_round, _, _ = evaluate(list(x_round), initial, stamps, args.field_size)
    return (res.dual_lower, v_round, qptime, res.status)

def kernel_stamp(sx: float, sy: float, size: int=7) -> np.ndarray:
    radius = size // 2
    stamp = np.zeros((size, size), dtype=float)
    for i, dr in enumerate(range(-radius, radius + 1)):
        for j, dc in enumerate(range(-radius, radius + 1)):
            z = dr * dr / (2.0 * sy * sy) + dc * dc / (2.0 * sx * sx)
            stamp[i, j] = math.exp(-z)
    return stamp

def main() -> None:
    rows = []
    for name, cli, ov in INSTANCES:
        for kernel in KERNELS:
            kname, sx, sy, ksize, target_mass = kernel
            out_prefix = f'kernel_{name}_{kname}'
            initial_path = Path(f'kernel_{name}_{kname}_initial.csv')
            cpp = run_cpp(cli, kernel, out_prefix, initial_path)
            args = SimpleNamespace(**{**DEFAULTS, **ov})
            args.sigma_x = sx
            args.sigma_y = sy
            args.kernel_size = ksize
            args.stamp_mass = iso7_mass() if target_mass is not None else None
            args.initial_field = initial_path
            lb, v_round, qptime, status = qp_lb(args, int(cpp['budget']))
            raw_mass = float(kernel_stamp(sx, sy, ksize).sum())
            mass = iso7_mass() if target_mass is not None else raw_mass
            v_ub = min(cpp['var_upper'], v_round)
            row = {'instance': name, 'kernel': kname, 'sigma_x': sx, 'sigma_y': sy, 'kernel_size': ksize, 'stamp_mass': mass, 'raw_stamp_mass': raw_mass, 'mass_normalized': target_mass is not None, 'budget': int(cpp['budget']), 'mean': cpp['mean'], 'var_repair': cpp['var_upper'], 'var_round': v_round, 'var_upper': v_ub, 'var_lower': lb, 'abs_gap': v_ub - lb, 'gap_pct': (v_ub - lb) / v_ub * 100.0 if v_ub > 1e-12 else float('nan'), 'int_s': cpp['time_int_s'], 'qp_s': qptime, 'status': status}
            rows.append(row)
            print(row)
    with open('gaussian_distortion_ablation.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for inst in [r['instance'] for r in rows if r['kernel'] == 'isotropic-7']:
        sub = [r for r in rows if r['instance'] == inst]
        labels = [r['kernel'] for r in sub]
        xs = np.arange(len(labels))
        axes[0].plot(xs, [r['var_upper'] for r in sub], '-o', label=inst)
        axes[1].plot(xs, [r['abs_gap'] for r in sub], '-o', label=inst)
        axes[2].plot(xs, [r['int_s'] + r['qp_s'] for r in sub], '-o', label=inst)
    for ax in axes:
        ax.set_xticks(range(len(KERNELS)))
        ax.set_xticklabels([k[0].replace('-', '\n') for k in KERNELS], fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('integer feasible variance')
    axes[1].set_ylabel('absolute gap (mm$^2$)')
    axes[2].set_ylabel('certification time (s)')
    axes[0].legend(fontsize=7)
    fig.savefig('gaussian_distortion_ablation.png', dpi=300)
    fig2, axes2 = plt.subplots(1, 6, figsize=(13.5, 2.8), constrained_layout=True)
    shape_kernels = KERNELS[:6]
    stamps = []
    for _, sx, sy, ksize, target_mass in shape_kernels:
        stamp = kernel_stamp(sx, sy, ksize)
        if target_mass is not None:
            stamp = stamp * (iso7_mass() / float(stamp.sum()))
        stamps.append(stamp)
    vmax = max((float(s.max()) for s in stamps))
    for ax, (name, sx, sy, ksize, target_mass), stamp in zip(axes2, shape_kernels, stamps):
        im = ax.imshow(stamp, origin='lower', cmap='viridis', vmin=0.0, vmax=vmax)
        suffix = '\nnormalized' if target_mass is not None else ''
        ax.set_title(f'{name}\n$k={ksize}$, mass={stamp.sum():.2f}{suffix}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.colorbar(im, ax=axes2, shrink=0.8, label='kernel weight')
    fig2.savefig('gaussian_kernel_shapes.png', dpi=300)
    print('wrote gaussian_distortion_ablation.csv, gaussian_distortion_ablation.png, gaussian_kernel_shapes.png')
if __name__ == '__main__':
    main()
