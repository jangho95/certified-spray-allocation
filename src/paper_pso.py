from __future__ import annotations
import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scip_compare import build_stamps, choose_budget, make_initial, make_waypoints
EPS = 1e-09

@dataclass
class ArchiveItem:
    x: np.ndarray
    mean: float
    variance: float
    delta_mu: float

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='paper pso')
    p.add_argument('--field', choices=['zero', 'random', 'center'], default='zero')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--pso-seed', type=int, default=123)
    p.add_argument('--random-low', type=float, default=1.0)
    p.add_argument('--random-high', type=float, default=30.0)
    p.add_argument('--center-height', type=float, default=100.0)
    p.add_argument('--center-size', type=int, default=16)
    p.add_argument('--target', type=float, default=200.0)
    p.add_argument('--field-size', type=int, default=50)
    p.add_argument('--waypoint-rows', type=int, default=25)
    p.add_argument('--waypoint-cols', type=int, default=25)
    p.add_argument('--spray-interval', type=int, default=2)
    p.add_argument('--kernel-size', type=int, default=7)
    p.add_argument('--sigma-x', type=float, default=1.75)
    p.add_argument('--sigma-y', type=float, default=1.75)
    p.add_argument('--cmax', type=int, default=200)
    p.add_argument('--budget', type=int, default=None)
    p.add_argument('--initial-field', type=Path, default=None)
    p.add_argument('--pop-size', type=int, default=100)
    p.add_argument('--iters', type=int, default=500)
    p.add_argument('--w', type=float, default=0.7)
    p.add_argument('--c1', type=float, default=1.5)
    p.add_argument('--c2', type=float, default=1.5)
    p.add_argument('--strategy', choices=['proposed', 'random', 'uniform', 'gradient', 'cluster'], default='proposed')
    p.add_argument('--gamma', type=float, default=None, help='shots/mm scale. Default M/(N*kernel_mass).')
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--beta', type=float, default=10.0)
    p.add_argument('--gradient-scale', type=float, default=10.0)
    p.add_argument('--adjust-iters', type=int, default=20)
    p.add_argument('--atf-min', type=float, default=0.05)
    p.add_argument('--atf-max', type=float, default=0.25)
    p.add_argument('--atf-initial', type=float, default=0.15)
    p.add_argument('--atf-tighten', type=float, default=0.005)
    p.add_argument('--atf-relax', type=float, default=0.01)
    p.add_argument('--relax-patience', type=int, default=3)
    p.add_argument('--diversify-patience', type=int, default=15)
    p.add_argument('--diversify-ratio', type=float, default=0.15)
    p.add_argument('--k-aw', type=float, default=5.0)
    p.add_argument('--w-min', type=float, default=0.1)
    p.add_argument('--w-max', type=float, default=0.9)
    p.add_argument('--archive-capacity', type=int, default=200)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--out-prefix', type=Path, default=Path('pso_solution'))
    return p.parse_args()

def dense_a(stamps, m: int, n: int) -> np.ndarray:
    A = np.zeros((m, n), dtype=np.float64)
    for j, stamp in enumerate(stamps):
        for cell, weight in zip(stamp.cells, stamp.weights):
            A[cell, j] += weight
    return A

def write_allocation(path: Path, pts: list[tuple[int, int]], x: np.ndarray) -> None:
    with path.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['waypoint', 'row', 'col', 'shot_count'])
        for i, ((r, c), val) in enumerate(zip(pts, x.astype(int))):
            writer.writerow([i, r, c, int(val)])

def write_field(path: Path, field: np.ndarray, side: int) -> None:
    np.savetxt(path, field.reshape(side, side), delimiter=',', fmt='%.10f')

def write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

def write_archive(path: Path, archive: list[ArchiveItem]) -> None:
    with path.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['archive_index', 'mean', 'variance', 'delta_mu', 'total_shots'])
        for i, item in enumerate(archive):
            writer.writerow([i, f'{item.mean:.9f}', f'{item.variance:.9f}', f'{item.delta_mu:.9f}', int(item.x.sum())])

class Simulator:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pts = make_waypoints(args)
        self.stamps = build_stamps(args, self.pts)
        self.initial = make_initial(args).astype(np.float64)
        self.m = args.field_size * args.field_size
        self.n = len(self.stamps)
        self.mass = self.stamps[0].mass
        self.budget = choose_budget(args, self.initial, self.mass, self.n)
        self.A = dense_a(self.stamps, self.m, self.n)
        self.P = self.A.T @ self.A
        self.lin = self.A.T @ self.initial
        self.initial_sum = float(self.initial.sum())
        self.initial_sq = float(self.initial @ self.initial)

    def batch_metrics(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xf = X.astype(np.float64, copy=False)
        total = Xf.sum(axis=1)
        mean = (self.initial_sum + self.mass * total) / self.m
        XP = Xf @ self.P
        sq = self.initial_sq + 2.0 * (Xf @ self.lin) + np.einsum('ij,ij->i', XP, Xf)
        var = np.maximum(sq / self.m - mean * mean, 0.0)
        return (mean, var)

    def field(self, x: np.ndarray) -> np.ndarray:
        return self.initial + self.A @ x.astype(np.float64)

    def metrics_one(self, x: np.ndarray) -> tuple[float, float, float, float]:
        f = self.field(x)
        mean = float(f.mean())
        var = float(((f - mean) ** 2).mean())
        return (mean, var, float(f.min()), float(f.max()))

def objective(mean: np.ndarray, var: np.ndarray, target: float, atf: float, k_aw: float, w_min: float, w_max: float) -> tuple[np.ndarray, np.ndarray]:
    delta = np.abs(mean - target)
    tau_aw = target * atf
    z = np.clip(k_aw * (delta - tau_aw), -60.0, 60.0)
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    w_mu = w_min + (w_max - w_min) * sigmoid
    fit = w_mu * delta / (target + EPS) + (1.0 - w_mu) * var / (target * target + EPS)
    return (fit, w_mu)

def local_patch(field2: np.ndarray, r: int, c: int, radius: int=1) -> np.ndarray:
    r0 = max(0, r - radius)
    r1 = min(field2.shape[0], r + radius + 1)
    c0 = max(0, c - radius)
    c1 = min(field2.shape[1], c + radius + 1)
    return field2[r0:r1, c0:c1]

def normalize01(v: np.ndarray) -> np.ndarray:
    lo = float(np.min(v))
    hi = float(np.max(v))
    if hi - lo < 1e-12:
        return np.zeros_like(v, dtype=np.float64)
    return (v - lo) / (hi - lo)

def simple_kmeans(features: np.ndarray, k: int, rng: np.random.Generator, iters: int=20) -> np.ndarray:
    n = features.shape[0]
    k = max(1, min(k, n))
    centers = features[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dist, axis=1)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = features[mask].mean(axis=0)
            else:
                centers[j] = features[rng.integers(0, n)]
    return labels

def target_adjust(X: np.ndarray, sim: Simulator, target: float, cmax: int, iters: int) -> np.ndarray:
    X = X.astype(np.float64, copy=True)
    prev_sign = np.zeros(X.shape[0])
    for t in range(max(1, iters)):
        mean, _ = sim.batch_metrics(np.rint(X).clip(1, cmax))
        scale = target / np.maximum(mean, 1e-12)
        alpha_t = 0.6 * (1.0 - t / max(1, iters))
        sign = np.sign(mean - target)
        alpha = np.full(X.shape[0], alpha_t)
        alpha[(prev_sign != 0) & (sign != 0) & (sign != prev_sign)] *= 0.5
        X = np.rint(X * (1.0 + alpha[:, None] * (scale[:, None] - 1.0)))
        X = np.clip(X, 1, cmax)
        prev_sign = sign
    return X.astype(np.int32)

def initialize_population(sim: Simulator, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    n = sim.n
    pop = args.pop_size
    cmax = args.cmax
    field2 = sim.initial.reshape(args.field_size, args.field_size)
    mu0 = float(sim.initial.mean())
    gamma = args.gamma if args.gamma is not None else sim.m / (sim.n * sim.mass)
    base = np.clip(gamma * (args.target - mu0), 1, int(np.floor(0.75 * cmax)))
    gy, gx = np.gradient(field2)
    grad = normalize01(np.sqrt(gx * gx + gy * gy))
    local_grad = np.array([local_patch(grad, r, c, radius=1).mean() for r, c in sim.pts])
    local_mean = np.array([local_patch(field2, r, c, radius=1).mean() for r, c in sim.pts])
    local_rough = np.array([local_patch(field2, r, c, radius=1).std() for r, c in sim.pts])
    coords = np.array(sim.pts, dtype=np.float64) / max(1, args.field_size - 1)
    feats = np.column_stack([coords, normalize01(local_mean), normalize01(local_rough)])
    labels = simple_kmeans(feats, int(np.floor(np.sqrt(n) / 2)), rng)
    cluster_mean = np.zeros(labels.max() + 1)
    cluster_rough = np.zeros(labels.max() + 1)
    for label in range(labels.max() + 1):
        mask = labels == label
        cluster_mean[label] = local_mean[mask].mean()
        cluster_rough[label] = local_rough[mask].mean()
    X = np.empty((pop, n), dtype=np.int32)
    for p in range(pop):
        if args.strategy == 'proposed':
            kind = ['uniform', 'gradient', 'cluster'][p % 3]
        else:
            kind = args.strategy
        if kind == 'random':
            x = rng.integers(1, cmax + 1, size=n)
        elif kind == 'uniform':
            x = np.rint(base * rng.uniform(0.8, 1.2, size=n))
        elif kind == 'gradient':
            x = np.rint((base + args.gradient_scale * local_grad) * rng.uniform(0.7, 1.3, size=n))
        elif kind == 'cluster':
            deviation = mu0 - cluster_mean[labels]
            value = base + args.alpha * deviation + args.beta * cluster_rough[labels]
            x = np.rint(value * rng.uniform(0.92, 1.08, size=n))
        else:
            raise ValueError(kind)
        X[p] = np.clip(x, 1, cmax)
    if args.strategy != 'random':
        X = target_adjust(X, sim, args.target, cmax, args.adjust_iters)
    return X

def dominates(a_delta: float, a_var: float, b_delta: float, b_var: float) -> bool:
    return (a_delta <= b_delta and a_var <= b_var) and (a_delta < b_delta or a_var < b_var)

def crowding_dist(items: list[ArchiveItem], target: float) -> np.ndarray:
    n = len(items)
    if n == 0:
        return np.array([])
    obj = np.array([[it.delta_mu / (target + EPS), it.variance / (target * target + EPS)] for it in items])
    cd = np.zeros(n)
    for k in range(2):
        order = np.argsort(obj[:, k])
        cd[order[0]] = cd[order[-1]] = np.inf
        span = obj[order[-1], k] - obj[order[0], k]
        if span < 1e-15:
            continue
        for i in range(1, n - 1):
            cd[order[i]] += (obj[order[i + 1], k] - obj[order[i - 1], k]) / span
    return cd

def update_archive(archive: list[ArchiveItem], x: np.ndarray, mean: float, var: float, target: float, capacity: int) -> list[ArchiveItem]:
    delta = abs(mean - target)
    for item in archive:
        if dominates(item.delta_mu, item.variance, delta, var):
            return archive
    kept = [item for item in archive if not dominates(delta, var, item.delta_mu, item.variance)]
    if not any((np.array_equal(item.x, x) for item in kept)):
        kept.append(ArchiveItem(x=x.astype(np.int32).copy(), mean=float(mean), variance=float(var), delta_mu=float(delta)))
    while len(kept) > capacity:
        cd = crowding_dist(kept, target)
        remove = int(np.argmin(cd))
        del kept[remove]
    return kept

def run_pso(args: argparse.Namespace):
    rng = np.random.default_rng(args.pso_seed)
    sim = Simulator(args)
    t0 = time.time()
    X = initialize_population(sim, args, rng).astype(np.float64)
    V = np.zeros_like(X)
    Xi = np.rint(X).clip(1, args.cmax).astype(np.int32)
    mean, var = sim.batch_metrics(Xi)
    fit, w_mu = objective(mean, var, args.target, args.atf_initial, args.k_aw, args.w_min, args.w_max)
    pbest = Xi.copy()
    pbest_mean = mean.copy()
    pbest_var = var.copy()
    gidx = int(np.argmin(fit))
    gbest = pbest[gidx].copy()
    gbest_fit = float(fit[gidx])
    gbest_mean = float(mean[gidx])
    gbest_var = float(var[gidx])
    archive: list[ArchiveItem] = []
    for i in range(args.pop_size):
        archive = update_archive(archive, Xi[i], float(mean[i]), float(var[i]), args.target, args.archive_capacity)
    atf = args.atf_initial
    stagnation = 0
    history: list[dict[str, float]] = []
    for it in range(1, args.iters + 1):
        r1 = rng.random(X.shape)
        r2 = rng.random(X.shape)
        V = args.w * V + args.c1 * r1 * (pbest.astype(np.float64) - X) + args.c2 * r2 * (gbest.astype(np.float64) - X)
        X = np.rint(X + V).clip(1, args.cmax)
        Xi = X.astype(np.int32)
        mean, var = sim.batch_metrics(Xi)
        fit, w_mu = objective(mean, var, args.target, atf, args.k_aw, args.w_min, args.w_max)
        pfit, _ = objective(pbest_mean, pbest_var, args.target, atf, args.k_aw, args.w_min, args.w_max)
        improved = fit < pfit
        if np.any(improved):
            pbest[improved] = Xi[improved]
            pbest_mean[improved] = mean[improved]
            pbest_var[improved] = var[improved]
        pfit, _ = objective(pbest_mean, pbest_var, args.target, atf, args.k_aw, args.w_min, args.w_max)
        best_idx = int(np.argmin(pfit))
        old_fit = gbest_fit
        if float(pfit[best_idx]) < gbest_fit - 1e-15:
            gbest = pbest[best_idx].copy()
            gbest_fit = float(pfit[best_idx])
            gbest_mean = float(pbest_mean[best_idx])
            gbest_var = float(pbest_var[best_idx])
            improved_global = True
        else:
            improved_global = False
        for i in range(args.pop_size):
            archive = update_archive(archive, Xi[i], float(mean[i]), float(var[i]), args.target, args.archive_capacity)
        if improved_global:
            stagnation = 0
            atf = max(args.atf_min, atf - args.atf_tighten)
        else:
            stagnation += 1
            if stagnation > args.relax_patience:
                atf = min(args.atf_max, atf + args.atf_relax)
            if stagnation > args.diversify_patience:
                nrep = max(1, int(round(args.diversify_ratio * args.pop_size)))
                worst = np.argsort(fit)[-nrep:]
                replacement = initialize_population(sim, args, rng)[:nrep]
                X[worst] = replacement
                V[worst] = 0.0
                stagnation = 0
        if it == 1 or it == args.iters or it % args.log_every == 0:
            history.append({'iter': it, 'gbest_fit': gbest_fit, 'gbest_mean': gbest_mean, 'gbest_variance': gbest_var, 'gbest_delta_mu': abs(gbest_mean - args.target), 'atf': atf, 'archive_size': len(archive), 'mean_w_mu': float(np.mean(w_mu)), 'best_iter_fit': float(np.min(fit)), 'old_gbest_fit': old_fit})
            if not args.quiet:
                print(f'iter={it:4d} fit={gbest_fit:.8g} mean={gbest_mean:.6f} var={gbest_var:.6f} dmu={abs(gbest_mean - args.target):.6f} atf={atf:.3f} archive={len(archive)}')
    final_mean, final_var, final_min, final_max = sim.metrics_one(gbest)
    elapsed = time.time() - t0
    return (sim, gbest, {'mean': final_mean, 'variance': final_var, 'min': final_min, 'max': final_max, 'total_shots': int(gbest.sum()), 'fit': gbest_fit, 'elapsed_s': elapsed, 'archive_size': len(archive)}, history, archive)

def main() -> None:
    args = parse_args()
    sim, x, metrics, history, archive = run_pso(args)
    alloc_path = Path(f'{args.out_prefix}_allocation.csv')
    field_path = Path(f'{args.out_prefix}_field.csv')
    history_path = Path(f'{args.out_prefix}_history.csv')
    archive_path = Path(f'{args.out_prefix}_archive.csv')
    write_allocation(alloc_path, sim.pts, x)
    write_field(field_path, sim.field(x), args.field_size)
    write_history(history_path, history)
    write_archive(archive_path, archive)
    print(f"final: mean={metrics['mean']:.6f} variance={metrics['variance']:.6f} min={metrics['min']:.6f} max={metrics['max']:.6f} total_shots={metrics['total_shots']}")
    print(f"pso_seconds={metrics['elapsed_s']:.3f} archive_size={metrics['archive_size']} budget_target={sim.budget}")
    print(f'wrote {alloc_path}, {field_path}, {history_path}, {archive_path}')
if __name__ == '__main__':
    main()
