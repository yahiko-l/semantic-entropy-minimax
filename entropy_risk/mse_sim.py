"""End-to-end entropy risk of the projected star-moment estimator at matched budgets (K = 3; Appendix H-F,
Fig. 5, Table III).

Two judges share q, Delta and n: a homogeneous one, w = (1, 1, 1), and a meaning-dependent one,
w = (1, 2, 4), at least as sensitive on every meaning. Each runs Algorithm 1 of the paper; the
projection uses S_2..S_{2K-1} for the meaning-dependent judge and S_2..S_K for the homogeneous one,
the orders that identify the entropy for each channel; for the meaning-dependent judge the projection on
S_2..S_K alone is also recorded. The verdict graph is simulated exactly; the
star statistics depend on the graph only through the degree sequence, because X_ij takes the value
(1-q)/Delta on an edge and -q/Delta otherwise, so only the degrees are kept.
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np

OUT = os.path.join('entropy_risk', 'output')
K = 3
Q = 0.5
JUDGES = {'homogeneous': (np.array([1.0, 1.0, 1.0]), K),
          'meaning-dependent': (np.array([1.0, 2.0, 4.0]), 2 * K - 1)}
_a = 1 / JUDGES['meaning-dependent'][0]
_mu = 1 / _a.sum()


def _witness(a):
    """x with X_1(x) = X_3(x) = 0 and x_3 = -1, X_j(x) = sum_c a_c x_c^j (Newton's method from the Figure 2 value)."""
    x = np.array([-0.149, 0.797])
    for _ in range(50):
        f = np.array([a[0] * x[0] + a[1] * x[1] - a[2], a[0] * x[0] ** 3 + a[1] * x[1] ** 3 - a[2]])
        J = np.array([[a[0], a[1]], [3 * a[0] * x[0] ** 2, 3 * a[1] * x[1] ** 2]])
        x = x - np.linalg.solve(J, f)
    return np.array([x[0], x[1], -1.0])


_x = _witness(_a)
S_REFLECT = 0.4
THETAS = {'collision': _a / _a.sum(),
          'reflected +': _a * (_mu + S_REFLECT * _x),
          'reflected -': _a * (_mu - S_REFLECT * _x),
          'uniform': np.full(K, 1 / K),
          'interior': np.array([0.5, 0.3, 0.2]),
          'rare 0.1': np.array([0.8, 0.1, 0.1]),
          'rare 0.03': np.array([0.94, 0.03, 0.03]),
          'rare 0.01': np.array([0.98, 0.01, 0.01])}
GRID_M = 300
REFINE = 12


def entropy(t):
    t = np.asarray(t, float)
    t = t[t > 0]
    return float(-(t * np.log(t)).sum())


def tree_moments(theta, w, vmax):
    return np.array([np.sum(theta ** v * w ** (v - 1), axis=-1) for v in range(2, vmax + 1)]).T


def simplex_grid(m):
    i, j = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing='ij')
    keep = i + j <= m
    t1, t2 = i[keep] / m, j[keep] / m
    return np.stack([t1, t2, 1 - t1 - t2], axis=1)


def degrees(counts, pw, q, rng, chunk=2048):
    """Exact degree sequence of the verdict graph; nodes are sorted by meaning."""
    n = int(counts.sum())
    starts = np.concatenate([[0], np.cumsum(counts)])
    deg = np.zeros(n, dtype=np.int64)
    for c in range(K):
        a, b = starts[c], starts[c + 1]
        nc = b - a
        for r0 in range(0, nc, chunk):
            r1 = min(nc, r0 + chunk)
            M = rng.random((r1 - r0, nc), dtype=np.float32) < pw[c]
            M &= np.arange(nc)[None, :] > np.arange(r0, r1)[:, None]
            deg[a + r0:a + r1] += M.sum(1)
            deg[a:b] += M.sum(0)
        for c2 in range(c + 1, K):
            a2, b2 = starts[c2], starts[c2 + 1]
            for r0 in range(0, nc, chunk):
                r1 = min(nc, r0 + chunk)
                M = rng.random((r1 - r0, b2 - a2), dtype=np.float32) < q
                deg[a + r0:a + r1] += M.sum(1)
                deg[a2:b2] += M.sum(0)
    return deg


def binom_cols(x, kmax):
    """C(x, j) for j = 0..kmax, as floats, for an integer array x."""
    out = np.ones((len(x), kmax + 1))
    for j in range(1, kmax + 1):
        out[:, j] = out[:, j - 1] * (x - j + 1) / j
    return out


def star_stats_from_degrees(deg, n, q, Delta, vmax):
    """S_hat_v, v = 2..vmax: (v-1)! sum_i e_{v-1}(X_i.) / (n)_v, with e_k from the degree d_i."""
    al, be = (1 - q) / Delta, -q / Delta
    kmax = vmax - 1
    cd = binom_cols(deg.astype(float), kmax) * al ** np.arange(kmax + 1)
    cn = binom_cols((n - 1 - deg).astype(float), kmax) * be ** np.arange(kmax + 1)
    out = []
    for v in range(2, vmax + 1):
        k = v - 1
        ek = sum(cd[:, j] * cn[:, k - j] for j in range(k + 1))
        out.append(math.factorial(k) * ek.sum() / math.prod(range(n - v + 1, n + 1)))
    return np.array(out)


def project(S_hat, w, vmax, grid, grid_S):
    """A minimizer of max_v |S_v(theta) - S_hat_v| over the simplex: grid search, then local refinement."""
    obj = np.abs(grid_S - S_hat).max(axis=1)
    best_t, best_f = None, np.inf
    for idx in np.argsort(obj)[:3]:
        c = grid[idx]
        h = 1.5 / GRID_M
        u = np.linspace(-h, h, 2 * REFINE + 1)
        d1, d2 = np.meshgrid(u, u, indexing='ij')
        t = np.stack([c[0] + d1.ravel(), c[1] + d2.ravel()], axis=1)
        t = np.column_stack([t, 1 - t.sum(1)])
        t = t[(t >= 0).all(1)]
        f = np.abs(tree_moments(t, w, vmax) - S_hat).max(axis=1)
        i = int(np.argmin(f))
        if f[i] < best_f:
            best_f, best_t = f[i], t[i]
    return best_t, best_f


def run_cell(args):
    judge, tname, n, delta, reps, seed = args
    w, vmax = JUDGES[judge]
    theta = THETAS[tname]
    Delta = math.sqrt(delta / n)
    pw = Q + Delta * w
    assert pw.max() <= 1
    rng = np.random.default_rng(seed)
    grid = simplex_grid(GRID_M)
    grid_S = tree_moments(grid, w, vmax)
    H0 = entropy(theta)
    S0 = tree_moments(theta, w, 2 * K - 1)
    low = vmax > K
    grid_low = grid_S[:, :K - 1]
    errs, errs_low, S_hats = [], [], []
    t0 = time.time()
    for _ in range(reps):
        counts = rng.multinomial(n, theta)
        deg = degrees(counts, pw, Q, rng)
        S_hat = star_stats_from_degrees(deg, n, Q, Delta, 2 * K - 1)
        t_hat, _ = project(S_hat[:vmax - 1], w, vmax, grid, grid_S)
        errs.append(entropy(t_hat) - H0)
        if low:
            t_low, _ = project(S_hat[:K - 1], w, K, grid, grid_low)
            errs_low.append(entropy(t_low) - H0)
        S_hats.append(S_hat)
    errs, S_hats = np.array(errs), np.array(S_hats)

    def summary(e):
        sq = e ** 2
        return dict(mse=float(sq.mean()), mse_se=float(sq.std(ddof=1) / math.sqrt(len(e))), bias=float(e.mean()),
                    rmse=float(math.sqrt(sq.mean())), median_abs=float(np.median(np.abs(e))),
                    frac_abs_gt_half=float(np.mean(np.abs(e) > 0.5)))

    out = dict(judge=judge, theta_name=tname, theta=theta.tolist(), n=n, delta=delta, Delta=Delta,
               reps=reps, seed=seed, H=H0, **summary(errs),
               S_true=S0.tolist(), S_hat_mean=S_hats.mean(0).tolist(), S_hat_sd=S_hats.std(0, ddof=1).tolist(),
               S_hat_z=((S_hats.mean(0) - S0) / (S_hats.std(0, ddof=1) / math.sqrt(reps))).tolist(),
               seconds=time.time() - t0)
    if low:
        out['order_K'] = summary(np.array(errs_low))
    return out


def self_test():
    """Degree formula against the recursion of Algorithm 1 on explicit small graphs."""
    rng = np.random.default_rng(1)
    for n, Delta in ((12, 0.3), (25, 0.15)):
        A = np.triu(rng.random((n, n)) < 0.5, 1)
        A = (A | A.T).astype(float)
        X = (A - Q) / Delta
        np.fill_diagonal(X, 0.0)
        e = np.zeros((n, 2 * K - 1))
        e[:, 0] = 1.0
        for j in range(n):
            for k in range(2 * K - 2, 0, -1):
                e[:, k] += np.where(np.arange(n) != j, X[:, j], 0.0) * e[:, k - 1]
        rec = np.array([math.factorial(v - 1) * e[:, v - 1].sum() / math.prod(range(n - v + 1, n + 1))
                        for v in range(2, 2 * K)])
        fast = star_stats_from_degrees(A.sum(1).astype(np.int64), n, Q, Delta, 2 * K - 1)
        assert np.allclose(rec, fast, rtol=1e-9, atol=1e-9), (rec, fast)
    print('self-test: degree formula matches the recursion of Algorithm 1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns', default='100,200,500,1000,2000,4000,8000')
    ap.add_argument('--kappa', type=float, default=0.1, help='window n Delta^2 = n^-kappa')
    ap.add_argument('--reps', type=int, default=400)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--tag', default='mse')
    args = ap.parse_args()
    self_test()
    os.makedirs(OUT, exist_ok=True)
    ns = [int(x) for x in args.ns.split(',')]
    cells, seed = [], 20261013
    for judge in JUDGES:
        for tname in THETAS:
            for n in ns:
                seed += 1
                cells.append((judge, tname, n, n ** -args.kappa, args.reps, seed))
    cells.sort(key=lambda c: -c[2])
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = pool.map(run_cell, cells, chunksize=1)
    res.sort(key=lambda r: (r['judge'], r['theta_name'], r['n']))
    lines = []
    for r in res:
        lines.append(f"{r['judge']:<18} {r['theta_name']:<10} n={r['n']:<5} delta={r['delta']:.3f} "
                     f"mse={r['mse']:.3e}+-{r['mse_se']:.1e} bias={r['bias']:+.3e} med|e|={r['median_abs']:.3f} "
                     f"P(|e|>.5)={r['frac_abs_gt_half']:.3f} "
                     + (f"orderK_mse={r['order_K']['mse']:.3e} " if 'order_K' in r else '')
                     + f"Sz={','.join(f'{z:+.1f}' for z in r['S_hat_z'])} t={r['seconds']:.0f}s")
    worst = {}
    for judge in JUDGES:
        for n in ns:
            rs = [r for r in res if r['judge'] == judge and r['n'] == n]
            top = max(rs, key=lambda r: r['mse'])
            worst.setdefault(judge, []).append(dict(n=n, mse=top['mse'], mse_se=top['mse_se'], theta_name=top['theta_name']))
    for judge, rows in worst.items():
        lines.append(f'worst over thetas, {judge}: ' + '; '.join(f"n={x['n']} {x['mse']:.3e} ({x['theta_name']})" for x in rows))
    lines.append(f'total seconds {time.time() - t0:.0f}')
    print('\n'.join(lines))
    with open(os.path.join(OUT, f'{args.tag}.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(OUT, f'{args.tag}.json'), 'w') as f:
        json.dump(dict(K=K, q=Q, kappa=args.kappa, reps=args.reps, grid_M=GRID_M, refine=REFINE,
                       witness_x=_x.tolist(), reflect_s=S_REFLECT,
                       judges={k: dict(w=v[0].tolist(), vmax=v[1]) for k, v in JUDGES.items()},
                       thetas={k: v.tolist() for k, v in THETAS.items()}, cells=res, worst=worst), f, indent=1)


if __name__ == '__main__':
    main()
