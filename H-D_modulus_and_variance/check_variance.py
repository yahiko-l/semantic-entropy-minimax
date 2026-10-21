"""Monte Carlo check of the star statistics of Lemma 1 (K = 3, w = w_0, q = 1/3; Appendix H-D, Table II).

For each configuration (theta, n, delta = n Delta^2) it simulates the verdict graph, computes the star
U-statistics S_hat_v (v = 2..5) and compares their mean with S_v (unbiasedness), and their variance with the
bound of Lemma 1 and with the leading noise term d_v rho_q^(v-1) / (n)_v, where d_v is the number of
automorphisms of the star on v vertices: d_2 = 2 and d_v = (v-1)! for v >= 3.
"""
import json
import math
import os

import numpy as np

OUT = os.path.join('H-D_modulus_and_variance', 'output')
WIT = json.load(open(os.path.join('H-B_moment_preservation', 'output', 'moment_path.json')))['K3']['witness']
W = np.array([float(x) for x in WIT['w']])
K, M = 3, 5
U = W.max()
U1 = max(1.0, U)
Q = 1 / 3
REPS, BATCH = 20000, 2000
RNG = np.random.default_rng(20261014)


def falling(n, v):
    return math.prod(range(n - v + 1, n + 1))


def star_stats(theta, n, delta_):
    """S_hat_v, v = 2..M, for BATCH independent graphs; star U-statistics via row elementary symmetric sums."""
    Z = RNG.choice(K, size=(BATCH, n), p=theta)
    same = Z[:, :, None] == Z[:, None, :]
    p = Q + delta_ * W[Z][:, :, None] * same
    up = np.triu(RNG.random((BATCH, n, n)) < p, 1)
    A = (up | np.transpose(up, (0, 2, 1))).astype(float)
    X = (A - Q) / delta_
    X[:, np.arange(n), np.arange(n)] = 0.0
    e = np.zeros((BATCH, n, M))
    e[:, :, 0] = 1.0
    for j in range(n):
        x = X[:, :, j]
        for k in range(M - 1, 0, -1):
            e[:, :, k] += x * e[:, :, k - 1]
    return np.stack([math.factorial(v - 1) * e[:, :, v - 1].sum(axis=1) / falling(n, v)
                     for v in range(2, M + 1)], axis=1)


def run(theta, n, dl):
    Delta = math.sqrt(dl / n)
    assert Q + Delta * U <= 1
    rho = max([Q * (1 - Q)] + [(Q + Delta * w) * (1 - Q - Delta * w) for w in W]) / Delta ** 2
    rho_q = Q * (1 - Q) / Delta ** 2
    vals = np.concatenate([star_stats(theta, n, Delta) for _ in range(REPS // BATCH)])
    rows = []
    for i, v in enumerate(range(2, M + 1)):
        Sv = float(np.sum(theta ** v * W ** (v - 1)))
        mean, var = float(vals[:, i].mean()), float(vals[:, i].var(ddof=1))
        Nv = n ** v / (U ** 2 + rho) ** (v - 1)
        bound = 2 ** (2 * v + 1) * math.factorial(v) * U1 ** (2 * v - 2) * max(1 / n, 1 / Nv)
        aut = 2 if v == 2 else math.factorial(v - 1)
        lead = aut * rho_q ** (v - 1) / falling(n, v)
        rows.append(dict(v=v, S=Sv, mean=mean, z=(mean - Sv) / math.sqrt(var / len(vals)), var=var,
                         bound=bound, var_over_bound=var / bound, lead=lead, var_over_lead=var / lead,
                         weak=bool(U ** 2 + rho >= n)))
    return dict(theta=theta.tolist(), n=n, delta=dl, Delta=Delta, rho=rho, rows=rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    thetas = {'centre': (1 / W) / np.sum(1 / W), 'generic': np.array([0.5, 0.3, 0.2])}
    res, lines = [], []
    for name, th in thetas.items():
        for n in (16, 32):
            for dl in (0.05, 0.2, 1.0, 2.0):
                r = run(th, n, dl)
                r['name'] = name
                res.append(r)
                lines.append(f"{name} n={n} delta={dl} Delta={r['Delta']:.4f} rho={r['rho']:.2f} "
                             f"weak={r['rows'][0]['weak']}: " + '; '.join(
                                 f"v={x['v']} z={x['z']:+.2f} var/bound={x['var_over_bound']:.2e} "
                                 f"var/lead={x['var_over_lead']:.3f}" for x in r['rows']))
                print(lines[-1], flush=True)
    zs = [x['z'] for r in res for x in r['rows']]
    summary = dict(n_tests=len(zs), max_abs_z=max(abs(z) for z in zs),
                   max_var_over_bound=max(x['var_over_bound'] for r in res for x in r['rows']))
    lines.append('summary: ' + json.dumps(summary))
    print(lines[-1])
    open(os.path.join(OUT, 'variance.txt'), 'w').write('\n'.join(lines) + '\n')
    json.dump(dict(reps=REPS, q=Q, w=W.tolist(), results=res, summary=summary),
              open(os.path.join(OUT, 'variance.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
