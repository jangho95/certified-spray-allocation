import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

def scaled(values):
    fractions = [Fraction(v) for v in values]
    exponents = [v.denominator.bit_length() - 1 for v in fractions]
    assert all((v.denominator == 1 << e for v, e in zip(fractions, exponents)))
    exponent = max(exponents)
    return ([v.numerator << exponent - e for v, e in zip(fractions, exponents)], exponent)

class Checker:

    def __init__(self, operator, s0):
        self.M, self.N = (operator['M'], operator['N'])
        a, self.ea = scaled([float.fromhex(w) for col in operator['columns'] for _, w in col])
        self.cols, offset = ([], 0)
        for col in operator['columns']:
            self.cols.append([(i, a[offset + j]) for j, (i, _) in enumerate(col)])
            offset += len(col)
        self.s0, self.es = scaled([float.fromhex(v) for v in s0])

    def field(self, x):
        xi, ex = scaled(x)
        ef = max(self.ea + ex, self.es)
        field = [v << ef - self.es for v in self.s0]
        for j, col in enumerate(self.cols):
            for i, a in col:
                field[i] += a * xi[j] << ef - self.ea - ex
        return (field, ef, xi, ex)

    def value(self, x):
        f, e, _, _ = self.field(x)
        return Fraction(self.M * sum((v * v for v in f)) - sum(f) ** 2, self.M ** 2 * (1 << 2 * e))

    def lower(self, x, B, cap):
        f, ef, xi, ex = self.field(x)
        center = [self.M * v - sum(f) for v in f]
        grad = [sum((a * center[i] for i, a in col)) for col in self.cols]
        remaining = B - self.N
        minimum = sum(grad)
        for g in sorted(grad):
            take = min(remaining, cap - 1)
            minimum += take * g
            remaining -= take
            if remaining == 0:
                break
        assert remaining == 0
        dot = Fraction(sum((g * v for g, v in zip(grad, xi))), 1 << ex)
        coefficient = Fraction(2, self.M ** 2 * (1 << self.ea + ef))
        return max(Fraction(0), self.value(x) + coefficient * (minimum - dot))

def read_allocation(path, n, B, cap):
    rows = list(csv.DictReader(path.open()))
    assert sorted((int(r['waypoint']) for r in rows)) == list(range(n))
    x = [0] * n
    for row in rows:
        v = Fraction(row['shot_count'])
        assert v.denominator == 1 and 1 <= v <= cap
        x[int(row['waypoint'])] = int(v)
    assert sum(x) == B
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parents[1] / 'results/stage5_long25')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root
    if not __debug__:
        raise SystemExit('Run without Python -O: assertion checks are required.')
    started = time.perf_counter()
    operator = json.loads((root / 'operator.json').read_text())
    result = json.loads((root / 'analysis.json').read_text())
    checked = {}
    for instance in result['instances']:
        w = json.loads((root / instance['witness']).read_text())
        csv_s0 = [float(v) for line in (root / 'inputs' / w['input']).read_text().splitlines() for v in line.split(',') if v.strip()]
        assert [float.fromhex(v) for v in w['s0_hex']] == csv_s0
        c = Checker(operator, w['s0_hex'])
        L = c.lower([float.fromhex(v) for v in w['xbar_hex']], w['B'], w['Cmax'])
        assert L == Fraction(w['L_exact']) == Fraction(instance['L_exact'])
        xu = read_allocation(root / w['U_alloc'], c.N, w['B'], w['Cmax'])
        assert xu == w['integer_alloc']
        U = c.value(xu)
        assert U == Fraction(w['U_exact']) == Fraction(instance['U_exact'])
        for p in w['pso']:
            xp = read_allocation(root / p['alloc'], c.N, w['B'], w['Cmax'])
            P = c.value(xp)
            assert P == Fraction(p['P_exact'])
            checked[p['alloc']] = (P, L, U)
    assert len(checked) == len(result['rows']) == 200
    for row in result['rows']:
        P, L, U = checked[row['alloc']]
        assert P == Fraction(row['P_exact'])
        assert P / U - 1 == Fraction(row['excess_low_exact'])
        assert P / L - 1 == Fraction(row['excess_high_exact'])
        assert Fraction(row['excess_low_pct']) <= 100 * (P / U - 1)
        assert Fraction(row['excess_high_pct']) >= 100 * (P / L - 1)
    out = dict(status='passed', implementation='standard-library exact-integer checker; no imports from analysis/solver modules', instances=len(result['instances']), pso_allocations=len(checked), checks='input fields; supporting-plane L; integer/box/budget constraints; exact U/P; outward-rounded per-run endpoints', elapsed_s=time.perf_counter() - started)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
