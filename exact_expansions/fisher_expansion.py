"""Exact Delta-expansion of the Fisher information along S_1..S_4-preserving directions (Appendix H-C, Table II).

With Xt_e = (A_e - q)/(q(1-q)), the likelihood ratio against Erdos-Renyi(q) is
L(A) = sum_z theta^z prod_{edges inside class c} (1 + Delta w_c Xt_e), a function of the labelling
pattern only. The Fisher information in a direction (theta_dot, w_dot) is
I(n) = sum_A P_q(A) (dL)^2 / L, expanded here in Delta to order D with exact rationals.

At each rational point (q; theta; w) the directions are the two vectors of the reduced row echelon basis of
the kernel of the Jacobian of (S_1, ..., S_4) with respect to (theta, w), and the first plus twice the second.
Writing the Delta^k coefficient as sum_v c_{k,v} C(n, v), each direction is checked for (a) no term below
Delta^6, (b) a Delta^6 coefficient equal to C(n, 3) mu_2^3 (dC_3)^2, (c) c_{k,v} = 0 for v > k/2 + 1 and
(d) c_{8,5} = 125 mu_2^4 (dS_5)^2. The sum of the basis vectors of the kernel for (S_1, S_2) is a control
direction, checked for no term below Delta^4 and c_{4,3} = 3 mu_2^2 (dS_3)^2.
"""
import json
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as Fr

import sympy as sp

OUT = os.path.join('exact_expansions', 'output')
D = 12
NMAX = 7

POINTS = {
    'P1': (Fr(1, 3), [Fr(1, 2), Fr(3, 10), Fr(1, 5)], [Fr(4, 5), Fr(1, 2), Fr(7, 5)]),
    'P2': (Fr(2, 5), [Fr(2, 5), Fr(7, 20), Fr(1, 4)], [Fr(3, 2), Fr(2, 3), Fr(1)]),
    'P3': (Fr(1, 4), [Fr(1, 3), Fr(1, 2), Fr(1, 6)], [Fr(1, 2), Fr(6, 5), Fr(9, 10)]),
}


def zero():
    return [Fr(0)] * (D + 1)


def smul(a, b):
    out = zero()
    for i, ai in enumerate(a):
        if ai:
            for j in range(D + 1 - i):
                if b[j]:
                    out[i + j] += ai * b[j]
    return out


def sinv(a):
    out = zero()
    out[0] = 1 / a[0]
    for k in range(1, D + 1):
        out[k] = -sum(a[j] * out[k - j] for j in range(1, k + 1)) / a[0]
    return out


def shift(a):
    return [Fr(0)] + a[:D]


def scale(a, x):
    return [x * ai for ai in a]


def sadd(*terms):
    out = zero()
    for t in terms:
        for i in range(D + 1):
            out[i] += t[i]
    return out


def binom_series(j, x):
    """(1 + x Delta)^j truncated."""
    return [Fr(math.comb(j, i)) * x ** i if i <= j else Fr(0) for i in range(D + 1)]


def S(v, th, w):
    return sum(th[c] ** v * w[c] ** (v - 1) for c in range(3))


def dmom(v, e, th, w, dth, dw):
    """Directional derivative of m_{v,e} = sum_c theta_c^v w_c^e."""
    return sum(v * th[c] ** (v - 1) * dth[c] * w[c] ** e
               + (e * th[c] ** v * w[c] ** (e - 1) * dw[c] if e else 0) for c in range(3))


def directions(th, w):
    t1, t2, t3, w1, w2, w3 = sp.symbols('t1 t2 t3 w1 w2 w3')
    tv, wv = [t1, t2, t3], [w1, w2, w3]
    Svs = [sum(tv[c] ** v * wv[c] ** (v - 1) for c in range(3)) for v in range(1, 5)]
    subs = {t1: sp.Rational(th[0].numerator, th[0].denominator),
            t2: sp.Rational(th[1].numerator, th[1].denominator),
            t3: sp.Rational(th[2].numerator, th[2].denominator),
            w1: sp.Rational(w[0].numerator, w[0].denominator),
            w2: sp.Rational(w[1].numerator, w[1].denominator),
            w3: sp.Rational(w[2].numerator, w[2].denominator)}
    J = sp.Matrix([[sp.diff(Sv, x).subs(subs) for x in tv + wv] for Sv in Svs])
    ker4 = J.nullspace()
    ker2 = J[:2, :].nullspace()
    to_fr = lambda vec: [Fr(int(sp.fraction(x)[0]), int(sp.fraction(x)[1])) for x in vec]
    assert len(ker4) == 2, len(ker4)
    k1, k2 = to_fr(ker4[0]), to_fr(ker4[1])
    dirs = {'ker1': k1, 'ker2': k2, 'ker1+2ker2': [a + 2 * b for a, b in zip(k1, k2)],
            'control_S1S2': [sum(col) for col in zip(*[to_fr(v) for v in ker2])]}
    return dirs


def fisher_series(q, th, w, dth, dw, graphs):
    alpha = [wc / q for wc in w]
    beta = [-wc / (1 - q) for wc in w]
    cache_p, cache_m = {}, {}

    def pw(c, j, plus):
        cache = cache_p if plus else cache_m
        key = (c, j)
        if key not in cache:
            cache[key] = binom_series(j, alpha[c] if plus else beta[c])
        return cache[key]

    cls_cache = {}

    def class_series(c, n_in, n_out):
        key = (c, n_in, n_out)
        if key not in cls_cache:
            P = smul(pw(c, n_in, True), pw(c, n_out, False))
            terms = []
            if n_in:
                terms.append(scale(shift(smul(pw(c, n_in - 1, True), pw(c, n_out, False))), n_in / q))
            if n_out:
                terms.append(scale(shift(smul(pw(c, n_in, True), pw(c, n_out - 1, False))), -n_out / (1 - q)))
            dP = scale(sadd(*terms), dw[c]) if terms else zero()
            cls_cache[key] = (P, dP)
        return cls_cache[key]

    pat_cache = {}

    def pattern_series(p):
        if p not in pat_cache:
            k, nin = p[:3], p[3:]
            Ps, dPs = [], []
            for c in range(3):
                n_out = k[c] * (k[c] - 1) // 2 - nin[c]
                P, dP = class_series(c, nin[c], n_out)
                Ps.append(P)
                dPs.append(dP)
            thk = th[0] ** k[0] * th[1] ** k[1] * th[2] ** k[2]
            P123 = smul(smul(Ps[0], Ps[1]), Ps[2])
            lab = sum(k[c] * dth[c] / th[c] for c in range(3))
            dF = sadd(scale(P123, lab),
                      smul(smul(dPs[0], Ps[1]), Ps[2]),
                      smul(smul(Ps[0], dPs[1]), Ps[2]),
                      smul(smul(Ps[0], Ps[1]), dPs[2]))
            pat_cache[p] = (scale(P123, thk), scale(dF, thk))
        return pat_cache[p]

    out = {}
    for n in range(2, NMAX + 1):
        N = n * (n - 1) // 2
        tot = zero()
        for g in graphs[n]:
            L, dL = zero(), zero()
            for p, cnt in zip(g['patterns'], g['counts']):
                F, dF = pattern_series(p)
                for i in range(D + 1):
                    if F[i]:
                        L[i] += cnt * F[i]
                    if dF[i]:
                        dL[i] += cnt * dF[i]
            assert L[0] == 1
            weight = g['size'] * q ** g['nE'] * (1 - q) ** (N - g['nE'])
            term = smul(smul(dL, dL), sinv(L))
            for i in range(D + 1):
                tot[i] += weight * term[i]
        out[n] = tot
    return out


def decompose(I):
    """c_{k,v} from I_k(n) = sum_v c_{k,v} C(n, v), n = 2..NMAX."""
    c = {}
    for k in range(D + 1):
        for v in range(2, NMAX + 1):
            c[(k, v)] = sum((-1) ** (v - j) * math.comb(v, j) * I[j][k] for j in range(2, v + 1))
    return c


def run(job):
    pname, dname = job
    q, th, w = POINTS[pname]
    graphs = pickle.load(open(os.path.join(OUT, 'graph_classes.pkl'), 'rb'))
    d = directions(th, w)[dname]
    dth, dw = d[:3], d[3:]
    I = fisher_series(q, th, w, dth, dw, graphs)
    c = decompose(I)
    mu2 = 1 / (q * (1 - q))
    dC3 = dmom(3, 3, th, w, dth, dw)
    dS = {v: dmom(v, v - 1, th, w, dth, dw) for v in range(1, 6)}
    res = dict(point=pname, direction=dname, dtheta=[str(x) for x in dth], dw=[str(x) for x in dw],
               dS={v: str(dS[v]) for v in dS}, dC3=str(dC3))
    if dname.startswith('ker'):
        res['a_low_orders_zero'] = all(I[n][k] == 0 for n in I for k in range(6))
        res['b_delta6_triangle'] = all(I[n][6] == math.comb(n, 3) * mu2 ** 3 * dC3 ** 2 for n in I)
        blow = {f'{k},{v}': str(c[(k, v)]) for k in range(6, D + 1) for v in range(2, NMAX + 1)
                if v > k // 2 + 1 and c[(k, v)] != 0}
        res['c_no_blowup'] = not blow
        res['c_nonzero_blowup_entries'] = blow
        res['d_c85_tree'] = c[(8, 5)] == 125 * mu2 ** 4 * dS[5] ** 2
        res['c85'] = float(c[(8, 5)])
        res['c85_pred'] = float(125 * mu2 ** 4 * dS[5] ** 2)
    else:
        res['control_low_orders_zero'] = all(I[n][k] == 0 for n in I for k in range(4))
        res['control_c43'] = float(c[(4, 3)])
        res['control_c43_pred'] = float(3 * mu2 ** 2 * dS[3] ** 2)
        res['control_c43_exact'] = c[(4, 3)] == 3 * mu2 ** 2 * dS[3] ** 2
        res['control_c4_other'] = {v: float(c[(4, v)]) for v in range(2, NMAX + 1) if v != 3}
    res['c_float'] = {f'{k},{v}': float(c[(k, v)]) for k in range(D + 1) for v in range(2, NMAX + 1)}
    res['I_float'] = {n: [float(x) for x in I[n]] for n in I}
    return res


def main():
    jobs = [(p, d) for p in POINTS for d in ('ker1', 'ker2', 'ker1+2ker2', 'control_S1S2')]
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run, jobs))
    json.dump(results, open(os.path.join(OUT, 'fisher_expansion.json'), 'w'), indent=1)
    lines = []
    for r in results:
        if r['direction'].startswith('ker'):
            lines.append(f"{r['point']} {r['direction']:11s} dw={r['dw']} | (a) {r['a_low_orders_zero']} "
                         f"(b) {r['b_delta6_triangle']} (c) {r['c_no_blowup']} (d) {r['d_c85_tree']} "
                         f"c85={r['c85']:.6e} pred={r['c85_pred']:.6e}")
        else:
            lines.append(f"{r['point']} {r['direction']:11s} low orders zero {r['control_low_orders_zero']}, "
                         f"c43={r['control_c43']:.6e} pred={r['control_c43_pred']:.6e} exact {r['control_c43_exact']}")
    open(os.path.join(OUT, 'fisher_expansion.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        NMAX = 5
        print(run(('P1', 'ker1'))['a_low_orders_zero'])
    else:
        main()
