"""Numerical cross-check of the weighted entropy modulus of Theorem 2 (K = 3, w = w_0; Appendix H-D, Table II).

Checks, on random and adversarial pairs (theta, theta'):
  L1  Lemma 6: the sign-change polynomial P satisfies P*G >= 0 and int P G = -sum_v q_v dS_v,
      |int P G| <= (eps/2)(1+U)^r;
  L2  Lemma 7: W1 <= C1 eps^(1/5) for eps <= 1;
  L3  Lemma 8: max_c |u_c - u'_c| <= W1 / gamma;
  L4  Lemma 9: in the boundary regime the single-linkage structure of the proof holds, and W1 <= C2 eps^(1/3)
      for eps <= eps1;
  T   Theorem 2(b): |H - H'| <= C eps^(1/5);
and the sharpness of the exponent on the reflected pair and on the searched pair of Appendix H-B.
A = ||a||_1, and rho and beta are rho_b and b_0 of Lemma 9. The case constants are the terms of C_K(w) in (4):
case 0 (eps_gt_1), case 1 (I_interior) and the boundary cases 2 to 4 (II_*); C_3(w) is their maximum.
"""
import itertools
import json
import math
import os

import mpmath as mp
import numpy as np
from scipy.optimize import brentq, minimize

OUT = os.path.join('modulus_and_variance', 'output')
K = 3
M = 2 * K - 1
WIT = json.load(open(os.path.join('moment_preservation', 'output', 'moment_path.json')))['K3']['witness']
W = np.array([float(x) for x in WIT['w']])
A_ = 1 / W
mp.mp.dps = 50
WM = [mp.mpf(x) for x in WIT['w']]
AM = [1 / x for x in WM]
RNG = np.random.default_rng(20261013)


def subset_gap(a):
    return min(abs(sum(s * x for s, x in zip(signs, a)))
               for signs in itertools.product((-1, 0, 1), repeat=len(a)) if any(signs))


def constants():
    A, U, wmin, amax = A_.sum(), W.max(), W.min(), A_.max()
    gamma = subset_gap(A_)
    C1 = (1 + U) ** (2 * K - 2) / 2 + (4 * K - 4) * A
    rho, d0, tau = wmin / (4 * K * U), wmin / (8 * K ** 2), wmin / (4 * K ** 2)
    beta = min(1.0, tau / 2)
    eps1 = beta ** ((2 * K - 2) * (2 * K - 3) / (2 * K - 4))
    C2 = (1 + U) ** (2 * K - 2) / 2 + (K // 2) * (4 * K - 8) * A * beta ** (-(2 * K - 2) / (2 * K - 4))
    al = (2 * K - 3) / M
    cases = {
        'I_interior': (1 + math.log(1 / rho)) * A * C1 / gamma,
        'II_D_ge_d0': math.log(K) * C1 / (gamma * d0),
        'II_eps_gt_eps1': math.log(K) * eps1 ** (-1 / M),
        'II_small_Theta': K * M / (2 * math.e) * (amax * C2 / gamma) ** al,
        'II_large_Theta': math.log(K) * (math.e * amax * C2 / gamma) ** al,
        'eps_gt_1': math.log(K),
    }
    return dict(w=W.tolist(), a=A_.tolist(), A=A, U=U, wmin=wmin, amax=amax, gamma=gamma, C1=C1, rho=rho,
                d0=d0, tau=tau, beta=beta, eps1=eps1, C2=C2, cases=cases, C=max(cases.values()))


CST = constants()


def mom(th):
    return np.array([np.sum(th ** v * W ** (v - 1)) for v in range(1, M + 1)])


def ent(th):
    t = th[th > 0]
    return float(-np.sum(t * np.log(t)))


def cdf_diff(th, thp):
    """Sorted atom locations z and the value G_i of F_nu - F_nu' on [z_i, z_{i+1})."""
    pts = np.concatenate([W * th, W * thp])
    mass = np.concatenate([A_, -A_])
    o = np.argsort(pts, kind='stable')
    return pts[o], np.cumsum(mass[o])


def sign_poly(z, G, tol):
    """Lemma 6: one root at z_{i+1} for each sign change between consecutive nonzero intervals."""
    nz = [i for i in range(len(z) - 1) if abs(G[i]) > tol and z[i + 1] > z[i]]
    roots = [z[i + 1] for i, j in zip(nz, nz[1:]) if G[i] * G[j] < 0]
    p = np.poly1d([1.0])
    for t in roots:
        p = p * np.poly1d([1.0, -t])
    if nz and p((z[nz[0]] + z[nz[0] + 1]) / 2) * G[nz[0]] < 0:
        p = -p
    return p, roots, nz


def pair_report(th, thp):
    z, G = cdf_diff(th, thp)
    tol = 1e-12 * CST['A']
    W1 = float(np.sum(np.abs(G[:-1]) * np.diff(z)))
    dS = mom(th) - mom(thp)
    eps = float(np.max(np.abs(dS[1:])))
    D = float(np.max(np.abs(W * th - W * thp)))
    p, roots, nz = sign_poly(z, G, tol)
    sign_ok = all(p((z[i] + z[i + 1]) / 2) * G[i] >= -1e-14 for i in nz)
    Q = p.integ()
    lhs = float(sum(G[i] * (Q(z[i + 1]) - Q(z[i])) for i in range(len(z) - 1)))
    q = Q.coeffs[::-1]
    rhs = -float(sum(q[v] * dS[v - 1] for v in range(1, len(q))))
    return dict(W1=W1, eps=eps, D=D, dH=abs(ent(th) - ent(thp)), r=len(roots), sign_ok=sign_ok,
                ibp_err=abs(lhs - rhs), ibp_scale=abs(lhs) + 1e-300,
                ibp_bound_ok=abs(lhs) <= eps / 2 * (1 + CST['U']) ** len(roots) * (1 + 1e-9) + 1e-15,
                z=z, G=G, th=th, thp=thp)


def tangent(k=K):
    x = RNG.standard_normal(k)
    return x - x.mean()


def proj_simplex(x):
    s = np.sort(x)[::-1]
    c = np.cumsum(s) - 1
    r = np.nonzero(s - c / np.arange(1, len(x) + 1) > 0)[0][-1]
    return np.maximum(x - c[r] / (r + 1), 0)


def sample_pairs(n):
    mu = 1 / CST['A']
    out = []
    for _ in range(n):
        kind = RNG.integers(5)
        if kind == 0:
            al = RNG.choice([0.3, 1.0, 3.0])
            th, thp = RNG.dirichlet([al] * K), RNG.dirichlet([al] * K)
        elif kind == 1:
            th = RNG.dirichlet([RNG.choice([0.3, 1.0, 3.0])] * K)
            thp = proj_simplex(th + 10 ** RNG.uniform(-5, -1) * tangent())
        elif kind == 2:
            x, xp = tangent(), tangent()
            x, xp = x - (A_ @ x) / A_.sum(), xp - (A_ @ xp) / A_.sum()
            th = A_ * (mu + 10 ** RNG.uniform(-2.5, -0.5) * x / np.abs(x).max() * mu)
            thp = A_ * (mu + 10 ** RNG.uniform(-2.5, -0.5) * xp / np.abs(xp).max() * mu)
        elif kind == 3:
            th = RNG.dirichlet([1.0] * K)
            small = RNG.choice(K, size=RNG.integers(1, K), replace=False)
            th[small] = 10 ** RNG.uniform(-7, -1, size=len(small))
            th /= th.sum()
            thp = proj_simplex(th + 10 ** RNG.uniform(-6, -2) * tangent())
        else:
            th = RNG.dirichlet([1.0] * K)
            th[RNG.integers(K)] = 0.0
            th /= th.sum()
            thp = proj_simplex(th + 10 ** RNG.uniform(-5, -1) * tangent())
        out.append((th, thp))
    return out


def clusters(th, thp, tau):
    """Single-linkage clusters of the K pairs {u_c, u'_c} at threshold tau (graph: set distance < tau)."""
    seg = [(min(W[c] * th[c], W[c] * thp[c]), max(W[c] * th[c], W[c] * thp[c])) for c in range(K)]
    dist = lambda i, j: max(0.0, seg[j][0] - seg[i][1], seg[i][0] - seg[j][1])
    parent = list(range(K))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i, j in itertools.combinations(range(K), 2):
        if dist(i, j) < tau:
            parent[find(i)] = find(j)
    groups = {}
    for c in range(K):
        groups.setdefault(find(c), []).append(c)
    hulls = sorted((min(seg[c][0] for c in g), max(seg[c][1] for c in g), g) for g in groups.values())
    gaps = [hulls[i + 1][0] - hulls[i][1] for i in range(len(hulls) - 1)]
    return hulls, gaps


def lemma4_structure(rep):
    th, thp = rep['th'], rep['thp']
    k = int(np.argmin(np.minimum(th, thp)))
    big = th if th[k] <= thp[k] else thp
    L = int(np.argmax(big))
    hulls, gaps = clusters(th, thp, CST['tau'])
    where = {c: i for i, h in enumerate(hulls) for c in h[2]}
    z, G = rep['z'], rep['G']
    gap_zero = True
    for i in range(len(hulls) - 1):
        lo, hi = hulls[i][1], hulls[i + 1][0]
        for j in range(len(z) - 1):
            if z[j] >= lo and z[j + 1] <= hi and z[j + 1] > z[j] and abs(G[j]) > 1e-12:
                gap_zero = False
    return dict(separate=where[k] != where[L], max_pairs=max(len(h[2]) for h in hulls),
                min_gap=min(gaps) if gaps else math.inf, gap_zero=gap_zero)


def mp_pair(th, thp):
    mom_ = lambda t: [mp.fsum(t[c] ** v * WM[c] ** (v - 1) for c in range(K)) for v in range(2, M + 1)]
    ent_ = lambda t: -mp.fsum(x * mp.log(x) for x in t if x > 0)
    eps = max(abs(x - y) for x, y in zip(mom_(th), mom_(thp)))
    pts = sorted([(WM[c] * th[c], AM[c]) for c in range(K)] + [(WM[c] * thp[c], -AM[c]) for c in range(K)])
    G, W1 = mp.mpf(0), mp.mpf(0)
    for (x0, m0), (x1, _) in zip(pts, pts[1:]):
        G += m0
        W1 += abs(G) * (x1 - x0)
    D = max(abs(WM[c] * (th[c] - thp[c])) for c in range(K))
    return eps, W1, D, abs(ent_(th) - ent_(thp)), abs(mp.fsum(th) - mp.fsum(thp))


def reflected_pairs():
    """Directions x in the plane sum a_c x_c = 0 with X_3(x) = 0; ratio limit 2^(4/5)|L(x)|/|X_5(x)|^(1/5)."""
    b1 = [AM[1], -AM[0], mp.mpf(0)]
    b2 = [AM[1] * b1[2] - AM[2] * b1[1], AM[2] * b1[0] - AM[0] * b1[2], AM[0] * b1[1] - AM[1] * b1[0]]
    b1 = [x / mp.norm(b1) for x in b1]
    b2 = [x / mp.norm(b2) for x in b2]
    vec = lambda t: [mp.cos(t) * b1[c] + mp.sin(t) * b2[c] for c in range(K)]
    Xj = lambda x, j: mp.fsum(AM[c] * x[c] ** j for c in range(K))
    f = lambda t: float(Xj(vec(mp.mpf(t)), 3))
    grid = np.linspace(0, np.pi, 2001)
    mu = 1 / mp.fsum(AM)
    moms = lambda t: [mp.fsum(t[c] ** v * WM[c] ** (v - 1) for c in range(K)) for v in range(2, M + 1)]
    H = lambda t: -mp.fsum(x * mp.log(x) for x in t)
    res = []
    for g0, g1 in zip(grid, grid[1:]):
        if f(g0) * f(g1) >= 0:
            continue
        phi = mp.findroot(lambda t: Xj(vec(t), 3), mp.mpf(brentq(f, g0, g1, xtol=1e-15)))
        x = vec(phi)
        x = [v / max(abs(t) for t in x) for v in x]
        L = -mp.fsum(AM[c] * x[c] * mp.log(AM[c]) for c in range(K))
        X5 = Xj(x, 5)
        series = []
        for s in [mp.mpf(10) ** (-j) for j in range(1, 6)]:
            th = [AM[c] * (mu + s * mu * x[c]) for c in range(K)]
            thp = [AM[c] * (mu - s * mu * x[c]) for c in range(K)]
            d = [abs(p - q) for p, q in zip(moms(th), moms(thp))]
            series.append(dict(s=float(s), dS2_4=float(max(d[:3])), dS5=float(d[3]),
                               dS5_pred=float(2 * (s * mu) ** 5 * abs(X5)),
                               ratio=float(abs(H(th) - H(thp)) / d[3] ** mp.mpf('0.2'))))
        res.append(dict(phi=float(phi), x=[float(v) for v in x], L=float(L), X5=float(X5), X1=float(Xj(x, 1)),
                        X3=float(Xj(x, 3)), limit=float(2 ** mp.mpf('0.8') * abs(L) / abs(X5) ** mp.mpf('0.2')),
                        series=series))
    return res


def adversarial(n_starts=400):
    def unpack(y):
        e = np.exp(y - y.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def obj(y, which):
        th, thp = unpack(y.reshape(2, K))
        dS = mom(th) - mom(thp)
        eps = float(np.max(np.abs(dS[1:])))
        if eps < 1e-9:
            return 50.0 + math.log(1e-9 / max(eps, 1e-300))
        if which == 'T':
            return -math.log(abs(ent(th) - ent(thp)) + 1e-300) + 0.2 * math.log(eps)
        z, G = cdf_diff(th, thp)
        W1 = float(np.sum(np.abs(G[:-1]) * np.diff(z)))
        if which == 'L2':
            return -math.log(W1 + 1e-300) + 0.2 * math.log(eps)
        D = float(np.max(np.abs(W * th - W * thp)))
        return -math.log(D + 1e-300) + math.log(W1 + 1e-300)
    best = {}
    for which in ('T', 'L2', 'L3'):
        top = (math.inf, None)
        for _ in range(n_starts):
            y0 = RNG.normal(0, 2, 2 * K)
            r = minimize(obj, y0, args=(which,), method='Nelder-Mead',
                         options=dict(maxiter=4000, xatol=1e-10, fatol=1e-12))
            if r.fun < top[0]:
                top = (r.fun, r.x)
        th, thp = unpack(top[1].reshape(2, K))
        best[which] = dict(value=math.exp(-top[0]), theta=th.tolist(), theta_prime=thp.tolist())
    best['L3']['value_gamma_D_over_W1'] = CST['gamma'] * best['L3'].pop('value')
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = [f"w = {W.tolist()}, a = {A_.tolist()}",
             f"A = {CST['A']:.6f}, U = {CST['U']:.6f}, gamma = {CST['gamma']:.6f}, C1 = {CST['C1']:.4f}, "
             f"rho = {CST['rho']:.5f}, d0 = {CST['d0']:.6f}, tau = {CST['tau']:.6f}, beta = {CST['beta']:.6f}, "
             f"eps1 = {CST['eps1']:.3e}, C2 = {CST['C2']:.4e}",
             'case constants: ' + ', '.join(f'{k} = {v:.4g}' for k, v in CST['cases'].items()),
             f"C_3(w) = {CST['C']:.4g}"]
    reps = [pair_report(th, thp) for th, thp in sample_pairs(200000)]
    ok = [r for r in reps if r['eps'] >= 1e-9]
    stats = dict(
        n=len(reps), n_eps_ge_1e9=len(ok),
        L1_sign_fail=sum(not r['sign_ok'] for r in reps),
        L1_ibp_max_rel_err=max(r['ibp_err'] / max(r['ibp_scale'], 1e-12) for r in ok),
        L1_bound_fail=sum(not r['ibp_bound_ok'] for r in ok),
        L1_max_r=max(r['r'] for r in reps),
        L2_max_ratio=max(r['W1'] / r['eps'] ** 0.2 for r in ok if r['eps'] <= 1),
        L3_max_gammaD_over_W1=max(CST['gamma'] * r['D'] / r['W1'] for r in reps if r['W1'] > 1e-12),
        T_max_ratio=max(r['dH'] / r['eps'] ** 0.2 for r in ok),
    )
    b = [r for r in reps if min(r['th'].min(), r['thp'].min()) < CST['rho'] and r['D'] < CST['d0']]
    st4 = [lemma4_structure(r) for r in b]
    stats.update(L4_n=len(b), L4_separate_fail=sum(not s['separate'] for s in st4),
                 L4_max_pairs=max(s['max_pairs'] for s in st4), L4_min_gap=min(s['min_gap'] for s in st4),
                 L4_gap_zero_fail=sum(not s['gap_zero'] for s in st4))
    lines.append('random pairs: ' + ', '.join(f'{k} = {v:.4g}' if isinstance(v, float) else f'{k} = {v}'
                                             for k, v in stats.items()))
    l4 = {'generic': [], 'vertex_split': [], 'boundary_pair_flip': []}
    mpm = lambda t: [mp.fsum(t[c] ** v * WM[c] ** (v - 1) for c in range(K)) for v in range(2, M + 1)]
    for _ in range(300):
        th = RNG.dirichlet([1.0] * K)
        small = RNG.choice(K, size=RNG.integers(1, K), replace=False)
        th[small] = 10 ** RNG.uniform(-6, -2, size=len(small))
        thm = [mp.mpf(x) for x in th]
        total = mp.fsum(thm)
        thm = [x / total for x in thm]
        step = mp.mpf(10) ** RNG.uniform(-16, -12)
        direction = [mp.mpf(y) for y in tangent()]
        direction[-1] = -mp.fsum(direction[:-1])
        thp = [x + step * y for x, y in zip(thm, direction)]
        if min(thp) > 0:
            l4['generic'].append((thm, thp))
    for _ in range(300):
        L, i, j = RNG.permutation(K)
        e, ph, php = mp.mpf(10) ** RNG.uniform(-9, -7), mp.mpf(RNG.uniform()), mp.mpf(RNG.uniform())

        def vert(ee, pp):
            t = [mp.mpf(0)] * K
            t[L], t[i], t[j] = 1 - ee, ee * pp, ee * (1 - pp)
            return t
        ep = mp.findroot(lambda x: mpm(vert(x, php))[0] - mpm(vert(e, ph))[0], e)
        l4['vertex_split'].append((vert(e, ph), vert(ep, php)))
    for _ in range(300):
        k, i, j = RNG.permutation(K)
        t3, t = mp.mpf(10) ** RNG.uniform(-4, -2), mp.mpf(10) ** RNG.uniform(-6.5, -5)
        m = (1 - t3) / (AM[i] + AM[j])
        th, thp = [mp.mpf(0)] * K, [mp.mpf(0)] * K
        th[k] = thp[k] = t3
        th[i], th[j] = AM[i] * (m + AM[j] * t), AM[j] * (m - AM[i] * t)
        thp[i], thp[j] = AM[i] * (m - AM[j] * t), AM[j] * (m + AM[i] * t)
        l4['boundary_pair_flip'].append((th, thp))
    l4_res = {}
    for fam, pairs in l4.items():
        rows = []
        for th, thp in pairs:
            eps, W1, D, dH, dS1 = mp_pair(th, thp)
            assert dS1 < mp.mpf(10) ** -45, (fam, float(dS1), float(eps))
            if eps <= CST['eps1'] and D < CST['d0'] and min(min(th), min(thp)) < CST['rho']:
                rows.append(dict(eps=float(eps), W1_over_eps13=float(W1 / eps ** (mp.mpf(1) / 3)),
                                 W1_over_eps12=float(W1 / eps ** (mp.mpf(1) / 2)),
                                 T_ratio=float(dH / eps ** mp.mpf('0.2'))))
        l4_res[fam] = rows
        lines.append(f"L4 regime, {fam} (mpmath, eps <= eps1): n = {len(rows)}, max W1/eps^(1/3) = "
                     f"{max(x['W1_over_eps13'] for x in rows):.4g} (C2 = {CST['C2']:.4g}), max W1/eps^(1/2) = "
                     f"{max(x['W1_over_eps12'] for x in rows):.4g}, max |dH|/eps^(1/5) = "
                     f"{max(x['T_ratio'] for x in rows):.4g}")
    refl = reflected_pairs()
    for r in refl:
        lines.append(f"reflected pair phi = {r['phi']:.6f} (X1 = {r['X1']:.1e}, X3 = {r['X3']:.1e}): L = {r['L']:.6f}, X5 = {r['X5']:.6e}, limit ratio = "
                     f"{r['limit']:.6f}; series " + '; '.join(
                         f"s={x['s']:.0e}: max|dS2..4| = {x['dS2_4']:.1e}, ratio = {x['ratio']:.6f}" for x in r['series']))
    wit = abs(float(WIT['dH_over_s'])) / abs(float(WIT['dS5'])) ** 0.2
    lines.append(f'endpoint witness (searched pair): limit ratio |dH/s| / |dS5(1)|^(1/5) = {wit:.6f}')
    adv = adversarial()
    lines.append('adversarial search: ' + '; '.join(
        f"{k}: {v.get('value', v.get('value_gamma_D_over_W1')):.6f} at theta = {np.round(v['theta'], 5).tolist()}, "
        f"theta' = {np.round(v['theta_prime'], 5).tolist()}" for k, v in adv.items()))
    open(os.path.join(OUT, 'modulus.txt'), 'w').write('\n'.join(lines) + '\n')
    json.dump(dict(constants=CST, random=stats, lemma4_mp=l4_res, reflected=refl, witness_ratio=wit, adversarial=adv),
              open(os.path.join(OUT, 'modulus.json'), 'w'), indent=1, default=float)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
