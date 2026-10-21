"""Checks of the genericity of Condition W for every K >= 3 (Appendix G; Appendix H-E, Table II).

(1) Phi_K(y) = sum_c pi_c(y) log|pi_c(y)| at the inductive points y = (0, 1, 2, T_4, ..., T_K), and its
    asymptotics Phi_K(y', T) ~ -Phi_{K-1}(y')/T (Lemma 22).
(2) Near-collision points x(h) in Omega_id with x_c^2 = 1 + h y_c: h^(K-1) Lambda(x(h)) -> Phi_K(y) (Lemma 23).
(3) Random weight vectors a = 1/w for K = 3, 4, 5: all real zeros of the odd-moment system found by
    multi-start root finding; Proposition 3(a) (distinct nonzero squares, a proportional to alpha(x)) and the
    largest |L(x)|/A over the zeros of each a, where L(x) = ell(x) and A = ||a||_1.
pi_c(z) = prod_{d != c} 1/(z_c - z_d), alpha_c(x) = pi_c(x^2)/x_c, Lambda(x) = sum_c pi_c(x^2) log alpha_c(x).
"""
import itertools
import json
import os

import mpmath as mp
import numpy as np
from scipy.optimize import root

mp.mp.dps = 40
OUT = os.path.join('witness_path_genericity', 'output')


def pi_vec(z):
    return [1 / mp.fprod(z[c] - z[d] for d in range(len(z)) if d != c) for c in range(len(z))]


def phi(y):
    return mp.fsum(p * mp.log(abs(p)) for p in pi_vec(y))


def lam(x):
    z = [v * v for v in x]
    p = pi_vec(z)
    al = [p[c] / x[c] for c in range(len(x))]
    assert all(v > 0 for v in al), 'x not in Omega'
    return mp.fsum(p[c] * mp.log(al[c]) for c in range(len(x)))


def omega_point(z_sorted):
    K = len(z_sorted)
    return [(-1) ** (K - 1 - j) * mp.sqrt(z_sorted[j]) for j in range(K)]


def gamma(a):
    best = np.inf
    for s in itertools.product((-1, 0, 1), repeat=len(a)):
        if any(s):
            best = min(best, abs(np.dot(s, a)))
    return best


def real_zeros(a, starts, rng):
    K = len(a)

    def fun(x):
        return np.array([np.dot(a, x ** (2 * i - 1)) for i in range(1, K)] + [np.dot(x, x) - 1])

    def jac(x):
        rows = [a * (2 * i - 1) * x ** (2 * i - 2) for i in range(1, K)] + [2 * x]
        return np.array(rows)

    found = []
    for _ in range(starts):
        x0 = rng.standard_normal(K)
        x0 /= np.linalg.norm(x0)
        sol = root(fun, x0, jac=jac, method='hybr', options={'xtol': 1e-14})
        x = sol.x
        if np.max(np.abs(fun(x))) > 1e-11:
            continue
        z = x ** 2
        p = np.array([1 / np.prod([z[c] - z[d] for d in range(K) if d != c]) for c in range(K)])
        al = p / x
        if al[0] < 0:
            x, al = -x, -al
        if not any(np.max(np.abs(x - y)) < 1e-7 for y in found):
            found.append(x)
    return found


def main():
    os.makedirs(OUT, exist_ok=True)
    lines, res = [], {}

    phi3 = phi([mp.mpf(0), mp.mpf(1), mp.mpf(2)])
    lines.append(f'Phi_3(0,1,2) = {mp.nstr(phi3, 20)}, -log 2 = {mp.nstr(-mp.log(2), 20)}')
    res['phi3'] = float(phi3)

    y = [mp.mpf(0), mp.mpf(1), mp.mpf(2)]
    chain = []
    for K in range(4, 9):
        prev = phi(y)
        rows = []
        for T in (mp.mpf(10) ** K, mp.mpf(10) ** (K + 1), mp.mpf(10) ** (K + 2)):
            val = phi(y + [T])
            rows.append((float(T), float(val), float(val * T / (-prev))))
        T_pick = mp.mpf(10) ** K
        y = y + [T_pick]
        cur = phi(y)
        chain.append(dict(K=K, phi=float(cur), asym=rows))
        lines.append(f'K = {K}: Phi_K(y, T)·T/(-Phi_(K-1)(y)) at T = 10^{K}, 10^{K + 1}, 10^{K + 2}: '
                     + ', '.join(f'{r[2]:.6f}' for r in rows) + f'; Phi_K at T = 10^{K}: {mp.nstr(cur, 8)}')
    res['induction'] = chain

    coll = []
    for K in range(3, 7):
        yK = [mp.mpf(j * (j + 1) // 2) for j in range(K)]
        target = phi(yK)
        vals = []
        for h in (mp.mpf('1e-2'), mp.mpf('1e-4'), mp.mpf('1e-6')):
            x = omega_point([1 + h * v for v in yK])
            vals.append(float(h ** (K - 1) * lam(x) / target))
        coll.append(dict(K=K, phi=float(target), ratio=vals))
        lines.append(f'K = {K}: Phi_K(y), y_j = j(j+1)/2: {mp.nstr(target, 8)}; h^(K-1) Lambda(x(h))/Phi_K at h = 1e-2, 1e-4, 1e-6: '
                     + ', '.join(f'{v:.8f}' for v in vals))
    res['collision'] = coll

    rng = np.random.default_rng(20261013)
    rand = {}
    for K, n_a, starts in ((3, 2000, 60), (4, 400, 300), (5, 150, 800)):
        stats = dict(n=n_a, min_maxL=np.inf, min_sq_gap=np.inf, min_abs_x=np.inf, max_prop_err=0.0,
                     zeros_hist={}, bad=0, min_gamma=np.inf, worst=None)
        for _ in range(n_a):
            a = np.exp(rng.uniform(np.log(0.2), np.log(5.0), K))
            g = gamma(a)
            stats['min_gamma'] = min(stats['min_gamma'], g)
            zs = real_zeros(a, starts, rng)
            stats['zeros_hist'][len(zs)] = stats['zeros_hist'].get(len(zs), 0) + 1
            if not zs:
                stats['bad'] += 1
                continue
            Ls = []
            for x in zs:
                z = x ** 2
                gaps = [abs(z[c] - z[d]) for c in range(K) for d in range(c + 1, K)]
                stats['min_sq_gap'] = min(stats['min_sq_gap'], min(gaps))
                stats['min_abs_x'] = min(stats['min_abs_x'], np.min(np.abs(x)))
                p = np.array([1 / np.prod([z[c] - z[d] for d in range(K) if d != c]) for c in range(K)])
                al = p / x
                err = np.max(np.abs(a / np.linalg.norm(a) - al / np.linalg.norm(al)))
                stats['max_prop_err'] = max(stats['max_prop_err'], err)
                Ls.append(abs(-np.sum(a * x * np.log(a))) / np.sum(a))
            mL = max(Ls)
            if mL < stats['min_maxL']:
                w = 1 / a
                gw = (w[0] ** 2 * np.log(w[1] / w[2]) ** 3 + w[1] ** 2 * np.log(w[2] / w[0]) ** 3
                      + w[2] ** 2 * np.log(w[0] / w[1]) ** 3) if K == 3 else None
                stats['worst'] = dict(a=a.tolist(), gamma=g, maxL=mL, zeros=len(zs),
                                      g_w=(float(gw) if gw is not None else None))
            stats['min_maxL'] = min(stats['min_maxL'], mL)
            if mL < 1e-6:
                stats['bad'] += 1
        stats['zeros_hist'] = {str(k): v for k, v in sorted(stats['zeros_hist'].items())}
        rand[K] = stats
        lines.append(f"random a, K = {K}: {n_a} draws, real zero directions found by multistart per draw {stats['zeros_hist']}; "
                     f"min squared-coordinate gap {stats['min_sq_gap']:.3e}, min |x_c| {stats['min_abs_x']:.3e}, "
                     f"max |a/|a| - alpha/|alpha|| {stats['max_prop_err']:.2e}; min over draws of max_x |L(x)|/A "
                     f"{stats['min_maxL']:.3e}; draws with no zero or all |L| < 1e-6: {stats['bad']}; min gamma {stats['min_gamma']:.3e}")
        wst = stats['worst']
        lines.append(f"   worst draw (smallest max |L|): a = {np.round(wst['a'], 6).tolist()}, gamma = {wst['gamma']:.3e}, "
                     f"real zero directions found {wst['zeros']}" + (f", g(w) = {wst['g_w']:.3e}" if wst['g_w'] is not None else ''))
    res['random'] = {str(k): {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv) for kk, vv in v.items()}
                     for k, v in rand.items()}

    a = np.array([1.0, 2.0, 3.0])
    x = np.array([1.0, 1.0, -1.0]) / np.sqrt(3)
    lines.append(f'gamma = 0 example a = (1, 2, 3): x = (1, 1, -1)/sqrt(3) gives X_1 = {np.dot(a, x):.1e}, '
                 f'X_3 = {np.dot(a, x ** 3):.1e} with a repeated coordinate')

    open(os.path.join(OUT, 'genericity.txt'), 'w').write('\n'.join(lines) + '\n')
    json.dump(res, open(os.path.join(OUT, 'genericity.json'), 'w'), indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
