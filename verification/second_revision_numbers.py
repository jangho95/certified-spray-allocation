import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def counts(path):
    with open(path, newline='') as handle:
        return [int(float(row['shot_count'])) for row in csv.DictReader(handle)]


def main():
    parser = argparse.ArgumentParser(description='Recompute numbers quoted in the text of the revised manuscript from the stored files. Standard library only.')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root
    out = {}

    long_dir = root / 'results/stage5_long25'
    runs = load(long_dir / 'runs.json')['rows']
    still_decreasing = 0
    for run in runs:
        with open(long_dir / run['history'], newline='') as handle:
            variance = {int(float(r['iter'])): float(r['gbest_variance']) for r in csv.DictReader(handle)}
        still_decreasing += variance[5000] < variance[4500]
    timed = [r['elapsed_s'] for r in runs if r['elapsed_s'] is not None]
    out['pso_long_runs'] = len(runs)
    out['pso_runs_with_lower_variance_at_5000_than_4500'] = still_decreasing
    out['pso_timed_runs'] = len(timed)
    out['pso_wall_time_s_min_median_max'] = [round(min(timed), 1), round(statistics.median(timed), 1), round(max(timed), 1)]
    assert (len(runs), still_decreasing, len(timed)) == (100, 97, 96)
    assert out['pso_wall_time_s_min_median_max'] == [16.6, 19.3, 22.8]

    target = 200.0
    eps = 1e-9
    weights = []
    rows_checked = 0
    for path in sorted((long_dir / 'history').glob('*.csv')):
        with open(path, newline='') as handle:
            for r in csv.DictReader(handle):
                d = abs(float(r['gbest_mean']) - target)
                v = float(r['gbest_variance']) / (target ** 2 + eps)
                denominator = d / (target + eps) - v
                rows_checked += 1
                if abs(denominator) > 1e-12:
                    weights.append((float(r['gbest_fit']) - v) / denominator)
    out['pso_history_rows'] = rows_checked
    out['pso_mean_weight_implied_by_stored_fitness_min_max'] = [min(weights), max(weights)]
    assert all(abs(w - 0.1) < 1e-9 for w in weights)

    short = [r['P'] for r in load(root / 'results/stage5/pso_runs.json')['rows'] if r['field'] == 'zero' and r['iters'] == 500]
    long_zero = [r['P'] for r in runs if r['field'] == 'zero']
    pilot = {r['id']: r for r in load(root / 'results/rigor_pilot/pilot_rows.json')['rows']}
    sd = lambda variance: round(math.sqrt(variance), 3)
    out['zero_field_sd_selected_pso500_pso5000'] = [sd(pilot['T-zero']['U_float']), sd(statistics.median(short)), sd(statistics.median(long_zero))]
    assert (len(short), len(long_zero)) == (25, 25)
    assert out['zero_field_sd_selected_pso500_pso5000'] == [7.344, 10.453, 8.657]

    fields = load(root / 'results/stage9/e4_runs.json')['results']
    cases = clipped = at_lower_end = 0
    clipped_fields = set()
    ratios = []
    for field in fields:
        rows = field['rows']
        cases += len(rows)
        for r in rows:
            if r['clip_high']:
                clipped += 1
                clipped_fields.add(field['name'])
        widest = max(rows, key=lambda r: r['W_rev'])
        ratios.append(widest['B'] / field['bstar'])
        at_lower_end += widest['B'] == min(r['B'] for r in rows)
    out['random_field_cases'] = cases
    out['random_field_cases_with_clipped_dual'] = [clipped, len(clipped_fields)]
    out['random_field_widest_budget_ratio_min_max'] = [round(min(ratios), 2), round(max(ratios), 2)]
    out['random_fields_widest_at_lowest_budget'] = at_lower_end
    assert (len(fields), cases, clipped, len(clipped_fields), at_lower_end) == (59, 1239, 132, 28, 15)
    assert out['random_field_widest_budget_ratio_min_max'] == [0.85, 0.93]

    moves = {}
    for name in ('zero', 'random', 'center30', 'center100'):
        text = (root / f'work/sensitivity_extension/target_{name}_T200_round_exchange.log').read_text()
        moves[name] = int(re.search(r'exchange_moves=(\d+)', text).group(1))
    out['exchange_moves_at_target_budgets'] = moves
    assert sorted(moves.values()) == [58, 62, 62, 68]

    a = counts(root / 'work/stage7/A_zero_F50_k7_s1.75_B12500_round_idx.csv')
    b = counts(root / 'work/stage3/t3_b12500_c50_round_idx_allocation.csv')
    out['b12500_rounded_starts_differ_at_waypoints'] = [i for i in range(len(a)) if a[i] != b[i]]
    assert (sum(a), sum(b)) == (12500, 12500)
    assert out['b12500_rounded_starts_differ_at_waypoints'] == [70, 113]

    phases = {r['field']: r['setup'] + r['solve'] + r['dual'] for r in load(root / 'results/stage4/single_summary.json') if r['F'] == 50}
    out['qp_setup_solve_dual_median_s'] = {k: round(v, 4) for k, v in sorted(phases.items())}
    calls = [pilot[k]['times']['osqp_1e-8_float_cert'] for k in pilot if k.startswith('T-')]
    out['solve_and_bound_call_s_min_max'] = [round(min(calls), 2), round(max(calls), 2)]
    assert [round(v, 2) for v in sorted(phases.values())] == [0.02, 0.06]
    assert out['solve_and_bound_call_s_min_max'] == [0.04, 0.13]

    out['passed'] = True
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
