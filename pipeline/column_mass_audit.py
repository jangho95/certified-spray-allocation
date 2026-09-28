import json
from pathlib import Path
import numpy as np
from stage8_mass import scenarios

ROOT = Path(__file__).resolve().parents[1]

def main():
    operators = scenarios()
    reference = float(operators['reflect'].sum(axis=0)[0])
    expected = {r['scenario']: r for r in json.loads((ROOT / 'results/stage8/mass_structure.json').read_text())['kappa']}
    rows = []
    for name, A in operators.items():
        masses = A.sum(axis=0)
        mean = float(masses.mean())
        assert abs(mean - expected[name]['kappa_mean']) < 1e-12
        assert abs(float(masses.min()) - expected[name]['kappa_min']) < 1e-12
        assert abs(float(masses.max()) - expected[name]['kappa_max']) < 1e-12
        deviations = np.abs(masses - mean)
        original = np.abs(masses - reference)
        rows.append(dict(scenario=name, kappa_reference=reference, kappa_mean=mean,
                         mean_abs_from_own_mean=float(deviations.mean()),
                         max_abs_from_own_mean=float(deviations.max()),
                         mean_abs_from_reflective_reference=float(original.mean()),
                         max_abs_from_reflective_reference=float(original.max())))
    output = ROOT / 'results/stage8/column_mass_audit.json'
    output.write_text(json.dumps(dict(aggregation='float64 column sums; own-mean and reflective-reference deviations are distinguished', rows=rows), indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(rows, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
