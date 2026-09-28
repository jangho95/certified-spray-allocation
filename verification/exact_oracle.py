import math
from fractions import Fraction as Q
import numpy as np

def scaled(values):
    qs = [v if isinstance(v, Q) else Q(int(v)) if isinstance(v, np.integer) else Q(float(v)) for v in values]
    den = max((v.denominator for v in qs))
    assert all((den % v.denominator == 0 for v in qs))
    return ([v.numerator * (den // v.denominator) for v in qs], den)

class Oracle:

    def __init__(self, op, s0):
        self.m = len(s0)
        an, self.ad = scaled(op['weights'])
        ptr = op['col_ptr']
        self.cols = [list(zip(map(int, op['cells'][ptr[j]:ptr[j + 1]]), an[ptr[j]:ptr[j + 1]])) for j in range(len(ptr) - 1)]
        self.sn, self.sd = scaled(s0)

    def evaluate(self, x, B=None, C=200):
        xn, xd = scaled(x)
        assert len(xn) == len(self.cols)
        den = self.ad * xd * self.sd
        f = [v * self.ad * xd for v in self.sn]
        for col, xj in zip(self.cols, xn):
            for i, a in col:
                f[i] += a * xj * self.sd
        total = sum(f)
        z = [self.m * v - total for v in f]
        value = Q(sum((v * v for v in z)), self.m ** 3 * den ** 2)
        if B is None:
            return value
        g = [sum((a * z[i] for i, a in col)) for col in self.cols]
        n = len(g)
        assert n <= B <= n * C
        full, rem = divmod(B - n, C - 1)
        gs = sorted(g)
        lp = sum(g) + (C - 1) * sum(gs[:full]) + (rem * gs[full] if rem else 0)
        bound = value + Q(2, self.ad * self.m ** 2 * den) * (lp - Q(sum((a * b for a, b in zip(g, xn))), xd))
        return (max(Q(0), bound), value)

def up(q):
    f = float(q)
    return f if Q(f) >= q else math.nextafter(f, math.inf)

def down(q):
    f = float(q)
    return f if Q(f) <= q else math.nextafter(f, -math.inf)

def checked_alloc(x, B, C=200):
    q = [Q(int(v)) if isinstance(v, np.integer) else Q(float(v)) for v in x]
    assert all((v.denominator == 1 and 1 <= v <= C for v in q))
    assert sum(q) == B
