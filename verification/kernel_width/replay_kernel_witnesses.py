"""Solver-free verification of the eleven witnesses. Requires only NumPy.

Run: python replay_kernel_witnesses.py
Checks the exact dyadic operator, original integer allocation, affine-minorant
lower bound, feasible continuous upper bound, and reported rational width.
"""
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def integers(values):
    values = [v if isinstance(v, Fraction) else Fraction(float(v)) for v in values]
    assert all(v.denominator & (v.denominator - 1) == 0 for v in values)
    exponent = min(-(v.denominator.bit_length() - 1) for v in values)
    return [v.numerator << (-(v.denominator.bit_length() - 1) - exponent) for v in values], exponent


def verify(directory):
    record = json.loads((directory / "witness.json").read_text())
    operator = ROOT / record["operator"]
    assert sha256(operator.read_bytes()).hexdigest() == record["operator_sha256"]
    assert sha256((directory / "points.npz").read_bytes()).hexdigest() == record["points_sha256"]
    op, pt = np.load(operator), np.load(directory / "points.npz")
    ptr, cells = op["col_ptr"], op["cells"]
    weights, ea = integers(op["weights"])
    initial, es = integers(pt["s0"])
    m, n, budget, cap = record["M"], record["N"], record["B"], record["Cmax"]
    assert len(ptr) == n + 1 and ptr[-1] == len(weights) and len(initial) == m
    assert ptr[0] == 0 and np.all(np.diff(ptr) >= 0)
    assert len(cells) == len(weights) and np.all((0 <= cells) & (cells < m))
    source = record["source_manifest"]
    for key in ("input", "allocation", "reference"):
        path = ROOT / "provenance" / source[key]
        assert sha256(path.read_bytes()).hexdigest() == source[key + "_sha256"]
    source_field = np.loadtxt(ROOT / "provenance" / source["input"], delimiter=",").ravel()
    assert np.array_equal(source_field, pt["s0"])
    source_allocation = list(csv.DictReader((ROOT / "provenance" / source["allocation"]).open()))
    assert sorted(int(r["waypoint"]) for r in source_allocation) == list(range(n))
    for row in source_allocation:
        assert Fraction(row["shot_count"]) == int(pt["allocation"][int(row["waypoint"])])

    def field(x):
        counts, ex = integers(x)
        deposited = [0] * m
        for j in range(n):
            for k in range(int(ptr[j]), int(ptr[j + 1])):
                deposited[int(cells[k])] += weights[k] * counts[j]
        e = min(ea + ex, es)
        return [(v << (ea + ex - e)) + (s << (es - e)) for v, s in zip(deposited, initial)], e

    def objective(x):
        f, e = field(x)
        return Fraction(m * sum(v * v for v in f) - sum(f)**2, m * m) * Fraction(2)**(2 * e)

    xbar = [Fraction(float(v)) for v in pt["xbar"]]
    f, e = field(xbar)
    total = sum(f)
    residual = [m * v - total for v in f]
    gradient_integer = [sum(weights[k] * residual[int(cells[k])] for k in range(int(ptr[j]), int(ptr[j + 1]))) for j in range(n)]
    multiplier = Fraction(2)**(ea + e + 1) / (m * m)
    minimizer = [1] * n
    remaining = budget - n
    for j in sorted(range(n), key=gradient_integer.__getitem__):
        take = min(cap - 1, remaining)
        minimizer[j] += take
        remaining -= take
        if remaining == 0:
            break
    assert remaining == 0 and sum(minimizer) == budget
    lower = max(Fraction(0), objective(xbar) + multiplier * sum(gradient_integer[j] * (minimizer[j] - xbar[j]) for j in range(n)))
    allocation = [Fraction(int(v)) for v in pt["allocation"]]
    assert np.array_equal(pt["allocation"], np.asarray([int(v) for v in allocation]))
    assert sum(allocation) == budget and all(1 <= v <= cap for v in allocation)
    feasible = [Fraction(v) for v in record["feasible_continuous_point"]]
    assert sum(feasible) == budget and all(1 <= v <= cap for v in feasible)
    upper, continuous = objective(allocation), objective(feasible)
    assert lower == Fraction(record["L_exact"])
    assert upper == Fraction(record["U_exact"])
    assert continuous == Fraction(record["V_feas_exact"])
    assert lower <= continuous <= upper
    assert (upper - lower) / upper == Fraction(record["relative_width_exact"])
    assert (continuous - lower) / max(upper, 1) <= Fraction(1, 10**9)
    return dict(id=record["id"], exact_replay=True)


if __name__ == "__main__":
    results = [verify(p.parent) for p in sorted((ROOT / "witness").glob("*/witness.json"))]
    assert len(results) == 11
    print(json.dumps(dict(cases=len(results), all_passed=all(r["exact_replay"] for r in results)), indent=2))
