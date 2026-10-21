"""Exact Delta-expansion of the chi^2 divergence between the endpoints of the searched pair scaled toward theta*
(Appendix H-C, Table II).

chi^2(P_0 || P_1) = sum_A P_q(A) (L_0 - L_1)^2 / L_1 with the pattern representation of fisher_expansion.py,
evaluated with mpmath at 60 digits for n = 2..7 and expanded to order D in Delta. For each amplitude s it
reports the largest coefficient below Delta^6 relative to the Delta^6 coefficient (n >= 3), the relative error
of the Delta^6 coefficient against C(n, 3) mu_2^3 (Delta C_3)^2, the largest |c_{k,v}| with v > k/2 + 1
relative to max_v |c_{k,v}| (6 <= k <= 12), and the relative error of c_{8,5} against 125 mu_2^4 (Delta S_5)^2.
"""
import json
import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import mpmath as mp
import numpy as np

mp.mp.dps = 60
OUT = os.path.join('H-C_exact_expansions', 'output')
D = 12
NMAX = 7
Q = mp.mpf(1) / 3
S_VALUES = ['1', '1/2', '1/4', '1/8']


def zero():
    return [mp.mpf(0)] * (D + 1)


def smul(a, b):
    out = zero()
    for i, ai in enumerate(a):
        if ai:
            for j in range(D + 1 - i):
                out[i + j] += ai * b[j]
    return out


def sinv(a):
    out = zero()
    out[0] = 1 / a[0]
    for k in range(1, D + 1):
        out[k] = -mp.fsum(a[j] * out[k - j] for j in range(1, k + 1)) / a[0]
    return out


def binom_series(j, x):
    return [mp.mpf(math.comb(j, i)) * x ** i if i <= j else mp.mpf(0) for i in range(D + 1)]


def witness():
    wit = json.load(open(os.path.join('H-B_moment_preservation', 'output', 'moment_path.json')))['K3']['witness']
    w = [mp.mpf(x) for x in wit['w']]
    th = [mp.mpf(x) for x in wit['theta']]
    thp = [mp.mpf(x) for x in wit['theta_prime']]
    return w, th, thp


def scaled(th, w, s):
    a = [1 / x for x in w]
    mu = 1 / mp.fsum(a)
    return [a[c] * (mu + s * (w[c] * th[c] - mu)) for c in range(3)]


def L_series_factory(th, w, q):
    alpha = [x / q for x in w]
    beta = [-x / (1 - q) for x in w]
    pcache, cache = {}, {}

    def pw(c, j, plus):
        key = (c, j, plus)
        if key not in pcache:
            pcache[key] = binom_series(j, alpha[c] if plus else beta[c])
        return pcache[key]

    def pattern(p):
        if p not in cache:
            k, nin = p[:3], p[3:]
            ser = [mp.mpf(1)] + [mp.mpf(0)] * D
            for c in range(3):
                n_out = k[c] * (k[c] - 1) // 2 - nin[c]
                ser = smul(ser, smul(pw(c, nin[c], True), pw(c, n_out, False)))
            thk = th[0] ** k[0] * th[1] ** k[1] * th[2] ** k[2]
            cache[p] = [thk * x for x in ser]
        return cache[p]
    return pattern


def chi2_series(s_str):
    s = mp.mpf(eval(s_str))
    w, th, thp = witness()
    t0, t1 = scaled(th, w, s), scaled(thp, w, s)
    graphs = pickle.load(open(os.path.join(OUT, 'graph_classes.pkl'), 'rb'))
    f0, f1 = L_series_factory(t0, w, Q), L_series_factory(t1, w, Q)
    out = {}
    for n in range(2, NMAX + 1):
        N = n * (n - 1) // 2
        tot = zero()
        for g in graphs[n]:
            L0, L1 = zero(), zero()
            for p, cnt in zip(g['patterns'], g['counts']):
                a, b = f0(p), f1(p)
                for i in range(D + 1):
                    L0[i] += cnt * a[i]
                    L1[i] += cnt * b[i]
            diff = [x - y for x, y in zip(L0, L1)]
            term = smul(smul(diff, diff), sinv(L1))
            weight = g['size'] * Q ** g['nE'] * (1 - Q) ** (N - g['nE'])
            for i in range(D + 1):
                tot[i] += weight * term[i]
        out[n] = tot
    m = lambda th_, v, e: mp.fsum(th_[c] ** v * w[c] ** e for c in range(3))
    dC3 = m(t0, 3, 3) - m(t1, 3, 3)
    dS5 = m(t0, 5, 4) - m(t1, 5, 4)
    return s_str, {n: [str(x) for x in out[n]] for n in out}, str(dC3), str(dS5)


def main():
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(chi2_series, S_VALUES))
    mu2 = 1 / (Q * (1 - Q))
    report, lines = [], []
    for s_str, ser, dC3, dS5 in results:
        I = {int(n): [mp.mpf(x) for x in v] for n, v in ser.items()}
        dC3, dS5 = mp.mpf(dC3), mp.mpf(dS5)
        c = {(k, v): mp.fsum((-1) ** (v - j) * math.comb(v, j) * I[j][k] for j in range(2, v + 1))
             for k in range(D + 1) for v in range(2, NMAX + 1)}
        low = max(abs(I[n][k]) / abs(I[n][6]) for n in I if n >= 3 for k in range(6))
        tri = max(abs(I[n][6] / (math.comb(n, 3) * mu2 ** 3 * dC3 ** 2) - 1) for n in I if n >= 3)
        blow = 0
        for k in range(6, D + 1):
            scale_k = max(abs(c[(k, v)]) for v in range(2, NMAX + 1))
            for v in range(k // 2 + 2, NMAX + 1):
                blow = max(blow, abs(c[(k, v)]) / scale_k)
        c85 = abs(c[(8, 5)] / (125 * mu2 ** 4 * dS5 ** 2) - 1)
        row = dict(s=s_str, low_rel=float(low), delta6_rel_err=float(tri), blowup_rel=float(blow),
                   c85_rel_err=float(c85), dC3=float(dC3), dS5=float(dS5),
                   c_float={f'{k},{v}': float(c[(k, v)]) for k in range(D + 1) for v in range(2, NMAX + 1)},
                   chi2_coeffs={n: [float(x) for x in I[n]] for n in I})
        report.append(row)
        lines.append(f's = {s_str}: max |coef_k<6|/|coef_6| = {float(low):.2e}; Delta^6 vs triangle rel err = {float(tri):.2e}; '
                     f'blow-up rel = {float(blow):.2e}; c85 vs 125 mu2^4 dS5^2 rel err = {float(c85):.2e}; '
                     f'dC3 = {float(dC3):.6e}, dS5 = {float(dS5):.6e}')
    w, th, thp = witness()
    ss = [mp.mpf(10) ** (-4 + 0.5 * i) for i in range(5)]
    m3 = lambda th_: mp.fsum(th_[c] ** 3 * w[c] ** 3 for c in range(3))
    m5 = lambda th_: mp.fsum(th_[c] ** 5 * w[c] ** 4 for c in range(3))
    y = [float(mp.log((m3(scaled(th, w, s)) - m3(scaled(thp, w, s))) ** 2)) for s in ss]
    slope = float(np.polyfit([float(mp.log(s)) for s in ss], y, 1)[0])
    r10 = [(m5(scaled(th, w, mp.mpf(eval(s)))) - m5(scaled(thp, w, mp.mpf(eval(s))))) ** 2 / mp.mpf(eval(s)) ** 10
           for s in S_VALUES]
    spread = float(max(abs(r / r10[0] - 1) for r in r10))
    lines.append(f'(Delta C_3(s))^2 slope on s in [1e-4, 1e-2]: {slope:.5f}; '
                 f'(Delta S_5(s))^2 / s^10 relative spread over s in {S_VALUES}: {spread:.2e}')
    json.dump(dict(rows=report, dC3_sq_slope=slope, dS5_s10_spread=spread),
              open(os.path.join(OUT, 'endpoint_chi2.json'), 'w'), indent=1)
    open(os.path.join(OUT, 'endpoint_chi2.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
