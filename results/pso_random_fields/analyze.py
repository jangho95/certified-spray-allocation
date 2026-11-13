from common import *

def interval(lo, hi):
    return dict(low_exact=str(lo), high_exact='+Infinity' if hi == math.inf else str(hi), low_pct=pct(lo), high_pct=pct(hi, True))

def main():
    os.chdir(ROOT)
    p = check_protocol()
    check = json.loads((ROOT / 'independent_check.json').read_text())
    assert check['passed']
    data = json.loads((ROOT / 'verified_instances.json').read_text())
    instances = data['instances']
    assert not data['failures'] and len(instances) == data['planned_pairs']
    runs = [r for i in instances for r in i['runs']]
    assert len(runs) == 1475 and len({(r['field'], r['seed']) for r in runs}) == 1475
    fields = []
    for f in p['fields']:
        rs = sorted([r for r in runs if r['field'] == f['field']], key=lambda r: r['seed'])
        assert [r['seed'] for r in rs] == [100000 + 100 * f['index'] + r for r in range(1, 26)]
        if any((r['excess_low_exact'] is None for r in rs)):
            raise ValueError('Undefined relative excess retained; aggregate claim suppressed')
        low = [Fraction(r['excess_low_exact']) for r in rs]
        high = [math.inf if r['excess_high_exact'] == '+Infinity' else Fraction(r['excess_high_exact']) for r in rs]
        med = interval(quantile(low, Fraction(1, 2)), quantile(high, Fraction(1, 2)))
        item = dict(field=f['field'], index=f['index'], n_runs=25, median_excess=med)
        fields.append(item)
        atomic(Path('fields') / f['field'] / 'summary.json', item)
        with (Path('fields') / f['field'] / 'runs.csv').open('w', newline='') as h:
            keys = ['field', 'seed', 'B', 'P_exact', 'L_exact', 'U_exact', 'excess_low_exact', 'excess_high_exact', 'excess_low_pct', 'excess_high_pct', 'P_better_than_three', 'variance_lower_at_5000', 'allocation', 'history']
            w = csv.DictWriter(h, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(rs)
    lows = [Fraction(f['median_excess']['low_exact']) for f in fields]
    highs = [math.inf if f['median_excess']['high_exact'] == '+Infinity' else Fraction(f['median_excess']['high_exact']) for f in fields]
    summary = {name: interval(quantile(lows, prob), quantile(highs, prob)) for name, prob in [('minimum', 0), ('Q1', Fraction(1, 4)), ('median', Fraction(1, 2)), ('Q3', Fraction(3, 4)), ('maximum', 1)]}
    conf = 1 - Fraction(95, 100) ** 59
    bound = min(lows)
    width = [(Fraction(i['U_exact']) - Fraction(i['L_exact'])) / Fraction(i['U_exact']) for i in instances]
    better = [dict(field=i['field'], B=i['B'], seeds=i['pso_better_than_three']) for i in instances if i['pso_better_than_three']]
    reference_low, reference_high = (Fraction(266, 1000), Fraction(268, 1000))
    positional = dict(reference_interval_pct=['26.6', '26.8'], n_fields_definitely_above=sum((x > reference_high for x in lows)), n_fields_definitely_below=sum((x < reference_low for x in highs)), n_fields_overlapping=59 - sum((x > reference_high for x in lows)) - sum((x < reference_low for x in highs)))
    failures = json.loads((ROOT / 'runs.json').read_text())
    report = dict(status='complete', protocol_sha256=digest(ROOT / 'protocol.json'), n_fields=59, n_planned_runs=1475, n_completed_runs=len(runs), failed_attempts=len(failures['failures']), retry_attempts=sum((f['attempt'] == 1 for f in failures['failures'])), infeasible_runs=sum((not r['feasible'] for r in runs)), pso_wall_s=failures['wall_s'], verification_wall_s=data['wall_s'], independent_check_wall_s=check['wall_s'], field_median_excess_summary=summary, population_5th_percentile_lower_bound=dict(lower_bound_exact=str(bound), lower_bound_pct=pct(bound), confidence_exact=str(conf), confidence_decimal=float(conf), estimand='Population 5th percentile of the field-level 25-run sample-median relative excess, for the declared field and independent-seed procedure; not the population median over infinitely many PSO runs on each field.', statement='The sample minimum is a one-sided lower confidence bound for the population 5th percentile; conservative optimization-interval lower endpoints are used.'), n_verified_pairs=len(instances), relative_certificate_width_range_pct=dict(minimum=pct(min(width)), maximum=pct(max(width), True)), n_PSO_runs_better_than_three=sum((r['P_better_than_three'] for r in runs)), n_pairs_PSO_better_than_three=len(better), PSO_better_pairs=better, n_variance_lower_at_5000_than_4500=sum((r['variance_lower_at_5000'] for r in runs)), benchmark_random_position=positional, deviations=[])
    atomic(ROOT / 'summary.json', report)
    atomic(ROOT / 'field_summaries.json', fields)
    with (ROOT / 'field_summaries.csv').open('w', newline='') as h:
        w = csv.writer(h)
        w.writerow(['field', 'n_runs', 'median_low_exact', 'median_high_exact', 'median_low_pct', 'median_high_pct'])
        for f in fields:
            m = f['median_excess']
            w.writerow([f['field'], 25, m['low_exact'], m['high_exact'], m['low_pct'], m['high_pct']])
    print(json.dumps(report, indent=2), flush=True)
if __name__ == '__main__':
    main()
