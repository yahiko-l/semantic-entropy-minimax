"""Data behind Figures 2 to 4: the reflected pair and its moment-preserving path for w = (1, 2, 4).

The witness x solves X_1(x) = X_3(x) = 0 with max |x_c| = 1, X_j(x) = sum_c a_c x_c^j, a = 1/w.
Path (Lemma 2): the K-atomic positive measure with moments X_0..X_4 of nu_x and fifth moment
(1 - 2t) X_5, atoms xi_k(t), masses alpha_k(t); theta_k = alpha_k (mu + s xi_k), w_k = 1/alpha_k.
Outputs reflected_pair_w124/output/path.json, reflected.json and random_pairs.json.
"""
import json
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 40
DATA = Path("reflected_pair_w124") / "output"
DATA.mkdir(parents=True, exist_ok=True)
W = [mp.mpf(1), mp.mpf(2), mp.mpf(4)]
K = len(W)
M = 2 * K - 1
A = [1 / w for w in W]
MU = 1 / mp.fsum(A)


def dump(name, obj):
    path = DATA / name
    path.write_text(json.dumps(obj, indent=1) + "\n")
    print(f"Saved: {path}")


def witness():
    b1 = [A[1], -A[0], mp.mpf(0)]
    b2 = [A[1] * b1[2] - A[2] * b1[1], A[2] * b1[0] - A[0] * b1[2], A[0] * b1[1] - A[1] * b1[0]]

    def vec(t):
        return [mp.cos(t) * b1[c] + mp.sin(t) * b2[c] for c in range(K)]

    roots = []
    for k in range(400):
        try:
            r = mp.findroot(lambda t: mp.fsum(A[c] * vec(t)[c] ** 3 for c in range(K)), mp.pi * k / 400) % mp.pi
        except (ValueError, ZeroDivisionError):
            continue
        if all(abs(r - q) > mp.mpf('1e-20') for q in roots):
            roots.append(r)
    assert len(roots) == 1, roots
    x = vec(roots[0])
    return [v / max(abs(u) for u in x) for v in x]


def measure_from_moments(X):
    H = mp.matrix(K, K)
    rhs = mp.matrix(K, 1)
    for i in range(K):
        for j in range(K):
            H[i, j] = X[i + j]
        rhs[i] = -X[i + K]
    b = mp.lu_solve(H, rhs)
    roots = mp.polyroots([mp.mpf(1)] + [b[K - 1 - j] for j in range(K)], maxsteps=200, extraprec=200)
    assert max(abs(mp.im(r)) for r in roots) < mp.mpf('1e-30')
    xs = sorted(mp.re(r) for r in roots)
    V = mp.matrix(K, K)
    for i in range(K):
        for j in range(K):
            V[i, j] = xs[j] ** i
    al = mp.lu_solve(V, mp.matrix([X[i] for i in range(K)]))
    return [al[j] for j in range(K)], xs


def S(theta, w, v):
    return mp.fsum(th ** v * wc ** (v - 1) for th, wc in zip(theta, w))


def ent(theta):
    return -mp.fsum(t * mp.log(t) for t in theta if t > 0)


def main():
    x = witness()
    X = [mp.fsum(A[c] * x[c] ** j for c in range(K)) for j in range(M + 1)]
    ell = -mp.fsum(A[c] * x[c] * mp.log(A[c]) for c in range(K))
    s_fig = mp.mpf('0.05')
    ts = [mp.mpf(i) / 200 for i in range(201)]
    rows = []
    for t in ts:
        al, xi = measure_from_moments(X[:M] + [(1 - 2 * t) * X[M]])
        th = [al[k] * (MU + s_fig * xi[k]) for k in range(K)]
        wk = [1 / a for a in al]
        rows.append(dict(t=float(t), xi=[float(v) for v in xi], alpha=[float(v) for v in al],
                         w=[float(v) for v in wk], theta=[float(v) for v in th],
                         S={str(v): float(S(th, wk, v)) for v in range(1, 8)}))
    S0 = rows[0]['S']
    for r in rows:
        r['dS'] = {v: r['S'][v] - S0[v] for v in r['S']}
    dump('path.json', dict(w=[float(v) for v in W], x=[float(v) for v in x], X=[float(v) for v in X],
                           ell=float(ell), mu=float(MU), s=float(s_fig), rows=rows))

    refl = []
    for e in np.linspace(-3.5, -0.75, 23):
        s = mp.mpf(10) ** mp.mpf(e)
        tp = [A[c] * (MU + s * x[c]) for c in range(K)]
        tm = [A[c] * (MU - s * x[c]) for c in range(K)]
        eps = max(abs(S(tp, W, v) - S(tm, W, v)) for v in range(2, M + 1))
        refl.append(dict(s=float(s), eps=float(eps), dH=float(abs(ent(tp) - ent(tm))),
                         dS_low=float(max(abs(S(tp, W, v) - S(tm, W, v)) for v in range(2, M)))))
    dump('reflected.json', dict(w=[float(v) for v in W], x=[float(v) for v in x], ell=float(ell),
                                X5=float(X[M]), rows=refl,
                                limit_ratio=float(2 ** (1 - mp.mpf(1) / M) * abs(ell) / abs(X[M]) ** (mp.mpf(1) / M))))

    rng = np.random.default_rng(20261013)
    w = np.array([float(v) for v in W])
    pairs = []
    for i in range(20000):
        th = rng.dirichlet(np.ones(K) * rng.choice([0.3, 1.0, 5.0]))
        scale = 10 ** rng.uniform(-4, 0)
        th2 = np.clip(th + scale * rng.standard_normal(K), 0, None)
        if th2.sum() == 0:
            continue
        th2 /= th2.sum()
        eps = max(abs(np.sum(th ** v * w ** (v - 1)) - np.sum(th2 ** v * w ** (v - 1))) for v in range(2, M + 1))
        h = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))
        if eps > 1e-12:
            pairs.append([float(eps), float(abs(h(th) - h(th2)))])
    dump('random_pairs.json', dict(w=w.tolist(), n=len(pairs), pairs=pairs))


if __name__ == '__main__':
    main()
