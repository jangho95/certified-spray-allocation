from common import *
import argparse
import numpy as np
import pso_repeated_runs as repeated
from paper_pso import run_pso

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', default='e4_01')
    parser.add_argument('--seed', type=int, default=100001)
    args = parser.parse_args()
    os.chdir(ROOT)
    protocol = check_protocol()
    field = next((f for f in protocol['fields'] if f['field'] == args.field))
    assert args.seed in field['seeds'], 'Choose a seed from the recorded schedule'
    output = Path('replayed_runs') / f'{args.field}_seed_{args.seed}'
    assert not output.exists(), 'Output already exists; choose a fresh package copy to repeat'
    output.mkdir(parents=True)
    settings = repeated.make_args(dict(field='random', initial_field=Path(field['input']), iters=5000, log_every=10, out_prefix=Path('unused')), args.seed)
    threads()
    sim, x, met, hist, _ = run_pso(settings)
    recorded = json.loads((Path('fields') / args.field / 'pso' / f'seed_{args.seed}.json').read_text())
    expected = read_alloc(recorded['allocation'])
    identical = bool(np.array_equal(x, expected))
    variance_difference = abs(met['variance'] - recorded['P_reported']) / max(1, abs(recorded['P_reported']))
    write_alloc(output / 'allocation.csv', x)
    with (output / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(hist[0]))
        writer.writeheader()
        writer.writerows(hist)
    check = dict(field=args.field, seed=args.seed, allocation_identical=identical, budget_identical=int(x.sum()) == recorded['B'], relative_variance_difference=variance_difference, passed=identical and variance_difference <= 1e-12, execution_protocol_sha256=protocol['public_adaptation']['execution_protocol_sha256'])
    atomic(output / 'check.json', check)
    print(json.dumps(check, indent=2))
    assert check['passed'], 'Recorded PSO execution was not reproduced'
if __name__ == '__main__':
    main()
