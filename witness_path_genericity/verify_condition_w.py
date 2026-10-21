"""Exact verification of Condition W and of gamma(w_0) > 0.29 at w_0 (Appendix H-E, Table II).

With x = (u, v(u), -1), v(u) = (a3 - a1 u)/a2, the constraint X_1 = 0 holds identically and X_3 = F(u),
F(u) = a1 u^3 + (a3 - a1 u)^3 / a2^2 - a3. All arithmetic is in exact rationals; exp is bounded by
partial sums of its series (lower bounds) and partial sums plus a geometric tail (upper bounds).
"""
import itertools
import json
import os
from fractions import Fraction as Fr
from math import factorial

OUT = os.path.join('witness_path_genericity', 'output')
WIT = json.load(open(os.path.join('moment_preservation', 'output', 'moment_path.json')))['K3']['witness']
W = [Fr(x) for x in WIT['w']]
A1, A2, A3 = (1 / x for x in W)


def exp_lower(x, n=30):
    return sum(x ** k / factorial(k) for k in range(n + 1))


def exp_upper(x, n=30):
    tail = x ** (n + 1) / factorial(n + 1) / (1 - x / (n + 2))
    return exp_lower(x, n) + tail


def v(u):
    return (A3 - A1 * u) / A2


def F(u):
    return A1 * u ** 3 + (A3 - A1 * u) ** 3 / A2 ** 2 - A3


def main():
    lo, hi = Fr(829, 1000), Fr(831, 1000)
    checks = {
        'F(0.829) < 0': F(lo) < 0,
        'F(0.831) > 0': F(hi) > 0,
        'v(0.829) < -0.145': v(lo) < Fr(-145, 1000),
        'v(0.831) > -0.147': v(hi) > Fr(-147, 1000),
        'a2/a1 < exp(0.6) (series lower bound)': A2 / A1 < exp_lower(Fr(6, 10)),
        'a2/a1 > 1 (log positive)': A2 / A1 > 1,
        'a2/a3 > exp(1.16) (series upper bound)': A2 / A3 > exp_upper(Fr(116, 100)),
        'a1*0.831*0.6 - a3*1.16 < -0.19': A1 * hi * Fr(6, 10) - A3 * Fr(116, 100) < Fr(-19, 100),
    }
    gaps = [abs(s1 * A1 + s2 * A2 + s3 * A3) for s1, s2, s3 in itertools.product((-1, 0, 1), repeat=3)
            if (s1, s2, s3) != (0, 0, 0)]
    checks['gamma(w) > 0.29'] = min(gaps) > Fr(29, 100)
    lines = [f'{k}: {v_}' for k, v_ in checks.items()]
    lines.append(f'F(0.829) = {float(F(lo)):.6e}, F(0.831) = {float(F(hi)):.6e}, gamma = {float(min(gaps)):.6f}, '
                 f'bound on L(x) = {float(A1 * hi * Fr(6, 10) - A3 * Fr(116, 100)):.6f}')
    lines.append('ALL EXACT CHECKS PASS' if all(checks.values()) else 'SOME CHECK FAILS')
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'condition_w.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
