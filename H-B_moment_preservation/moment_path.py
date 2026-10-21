"""Moment-preserving path between the searched pair at w_0 scaled toward theta* (Appendix H-B, Table II).

The pair from search_pair.py is refined in 80-digit arithmetic, scaled by the amplitude s and joined by the
path of Lemma 11 built on its own moment sequence: K-atomic measures with the moments X_0..X_4 of the pair
and fifth moment interpolating linearly between its two endpoints. Along the path the script records the
drift of S_1..S_4 and the size of the derivatives of every motif moment m_{v,e} = sum_c theta_c^v w_c^e,
v <= 7, and fits their order in s.
"""
import json
import os

import mpmath as mp
import numpy as np

mp.mp.dps = 80
OUT = os.path.join('H-B_moment_preservation', 'output')


def S(v, th, w):
    return mp.fsum(th[c] ** v * w[c] ** (v - 1) for c in range(len(th)))


def ent(th):
    return -mp.fsum(t * mp.log(t) for t in th)


def refine(th0, thp0, w, vmax):
    """Solve S_v(th) = S_v(th') for v = 2..vmax with th_1 fixed (one point of the scaling orbit)."""
    K = len(th0)
    w = [mp.mpf(x) for x in w]
    t1 = mp.mpf(th0[0])

    def unpack(z):
        th = [t1] + list(z[:K - 2])
        th.append(1 - mp.fsum(th))
        thp = list(z[K - 2:2 * K - 3])
        thp.append(1 - mp.fsum(thp))
        return th, thp

    def F(*z):
        th, thp = unpack(z)
        return [S(v, th, w) - S(v, thp, w) for v in range(2, vmax + 1)]

    z0 = [mp.mpf(x) for x in list(th0[1:K - 1]) + list(thp0[:K - 1])]
    z = mp.findroot(F, z0, tol=mp.mpf(10) ** (-75), maxsteps=200)
    z = [z[i] for i in range(z.rows)] if isinstance(z, mp.matrix) else [z]
    th, thp = unpack(z)
    res = max(abs(f) for f in F(*z))
    return th, thp, w, res


def measure_from_moments(X, K):
    """Unique K-atom measure with moments X[0..2K-1] (positive definite Hankel)."""
    H = mp.matrix(K, K)
    rhs = mp.matrix(K, 1)
    for i in range(K):
        for j in range(K):
            H[i, j] = X[i + j]
        rhs[i] = -X[i + K]
    b = mp.lu_solve(H, rhs)
    coeffs = [mp.mpf(1)] + [b[K - 1 - j] for j in range(K)]
    roots = mp.polyroots(coeffs, maxsteps=200, extraprec=200)
    imag = max(abs(mp.im(r)) for r in roots)
    xs = sorted(mp.re(r) for r in roots)
    V = mp.matrix(K, K)
    for i in range(K):
        for j in range(K):
            V[i, j] = xs[j] ** i
    a = mp.lu_solve(V, mp.matrix([X[i] for i in range(K)]))
    return [a[j] for j in range(K)], xs, imag


def derivatives(a, x, dX):
    """d(a_r, x_r)/dt from sum_r a_r x_r^j = X_j(t), j = 0..2K-1, with dX_j/dt given."""
    K = len(a)
    M = mp.matrix(2 * K, 2 * K)
    for j in range(2 * K):
        for r in range(K):
            M[j, r] = x[r] ** j
            M[j, K + r] = j * a[r] * x[r] ** (j - 1) if j > 0 else 0
    d = mp.lu_solve(M, mp.matrix(dX))
    return [d[r] for r in range(K)], [d[K + r] for r in range(K)]


def path_study(th, thp, w, K, log, tag, ngrid=2001, s_list=None, motif_v=7):
    a = [1 / wc for wc in w]
    u = [w[c] * th[c] for c in range(K)]
    up = [w[c] * thp[c] for c in range(K)]
    mu = 1 / mp.fsum(a)
    x = [uc - mu for uc in u]
    xp = [uc - mu for uc in up]
    X = [mp.fsum(a[c] * x[c] ** j for c in range(K)) for j in range(2 * K)]
    Xp = [mp.fsum(a[c] * xp[c] ** j for c in range(K)) for j in range(2 * K)]
    match = max(abs(X[j] - Xp[j]) for j in range(2 * K - 1))
    dXm = Xp[2 * K - 1] - X[2 * K - 1]
    log(f'[{tag}] mu = {mp.nstr(mu, 12)}; max |X_j - X\'_j| (j <= {2*K-2}) = {mp.nstr(match, 5)}; '
        f'Delta X_{2*K-1} = {mp.nstr(dXm, 12)}')
    ts = [mp.mpf(i) / (ngrid - 1) for i in range(ngrid)]
    rows = []
    for t in ts:
        Xt = X[:2 * K - 1] + [X[2 * K - 1] + t * dXm]
        at, xt, im = measure_from_moments(Xt, K)
        da, dx = derivatives(at, xt, [0] * (2 * K - 1) + [dXm])
        rows.append((t, at, xt, da, dx, im))
    a_min = min(min(r[1]) for r in rows)
    a_max = max(max(r[1]) for r in rows)
    x_max = max(max(abs(v) for v in r[2]) for r in rows)
    im_max = max(r[5] for r in rows)
    log(f'[{tag}] path: a_min = {mp.nstr(a_min, 8)}, a_max = {mp.nstr(a_max, 8)}, '
        f'max|x| = {mp.nstr(x_max, 8)}, max imag part of roots = {mp.nstr(im_max, 3)}')

    def as_set(av, xv):
        return sorted(zip(xv, av))
    e0 = max(max(abs(p[0] - q[0]), abs(p[1] - q[1]))
             for p, q in zip(as_set(rows[0][1], rows[0][2]), as_set(a, x)))
    e1 = max(max(abs(p[0] - q[0]), abs(p[1] - q[1]))
             for p, q in zip(as_set(rows[-1][1], rows[-1][2]), as_set(a, xp)))
    rank0 = [sorted(range(K), key=lambda c: x[c]).index(c) for c in range(K)]
    rank1 = [sorted(range(K), key=lambda c: xp[c]).index(c) for c in range(K)]
    log(f'[{tag}] endpoint error t=0: {mp.nstr(e0, 3)}, t=1: {mp.nstr(e1, 3)}; '
        f'rank of class c at t=0: {rank0}, at t=1: {rank1}')
    result = dict(mu=str(mu), a_min=float(a_min), a_max=float(a_max), x_max=float(x_max),
                  endpoint_err=[float(e0), float(e1)], dX_last=float(dXm),
                  rank0=rank0, rank1=rank1, by_s={})
    for s in s_list:
        s = mp.mpf(s)
        Svals = {v: [] for v in range(1, 2 * K - 1)}
        dm = {}
        th_min, w_max = mp.inf, 0
        for (t, at, xt, da, dx, im) in rows:
            thr = [at[r] * (mu + s * xt[r]) for r in range(K)]
            wr = [1 / at[r] for r in range(K)]
            dth = [da[r] * (mu + s * xt[r]) + at[r] * s * dx[r] for r in range(K)]
            dw = [-da[r] / at[r] ** 2 for r in range(K)]
            th_min = min(th_min, min(thr))
            w_max = max(w_max, max(wr))
            for v in range(1, 2 * K - 1):
                Svals[v].append(mp.fsum(thr[r] ** v * wr[r] ** (v - 1) for r in range(K)))
            for v in range(2, motif_v + 1):
                for e in range(v - 1, v * (v - 1) // 2 + 1):
                    d = abs(mp.fsum(v * thr[r] ** (v - 1) * dth[r] * wr[r] ** e
                                    + e * thr[r] ** v * wr[r] ** (e - 1) * dw[r] for r in range(K)))
                    dm[(v, e)] = max(dm.get((v, e), 0), d)
        drift = {v: float(max(abs(val - Svals[v][0]) for val in Svals[v]) / abs(Svals[v][0]))
                 for v in Svals}
        result['by_s'][mp.nstr(s, 6)] = dict(
            theta_min=float(th_min), w_max=float(w_max), S_rel_drift=drift,
            sup_dm={f'{v},{e}': float(val) for (v, e), val in dm.items()})
        log(f'[{tag}] s = {mp.nstr(s, 4)}: min theta = {mp.nstr(th_min, 6)}, max w = {mp.nstr(w_max, 6)}, '
            f'max rel drift S_1..S_{2*K-2} = {max(drift.values()):.2e}')
    return result


def orders(result, log, tag, K):
    keys = sorted({k for d in result['by_s'].values() for k in d['sup_dm']},
                  key=lambda k: tuple(map(int, k.split(','))))
    ss = [k for k in result['by_s'] if float(k) < 0.5]
    xs = np.log([float(k) for k in ss])
    table = {}
    for key in keys:
        ys = np.array([result['by_s'][k]['sup_dm'][key] for k in ss])
        v, e = map(int, key.split(','))
        kind = 'tree' if e == v - 1 else ('unicyclic' if e == v else 'multicyclic')
        if kind == 'tree' and v <= 2 * K - 2:
            pred = None
            slope = None
        else:
            pred = (2 * K - 1) if kind == 'tree' else (1 if kind == 'unicyclic' else 0)
            slope = float(np.polyfit(xs, np.log(ys), 1)[0])
        table[key] = dict(kind=kind, pred=pred, slope=slope, sup_max=float(ys.max()), sup_min=float(ys.min()))
        log(f'[{tag}] (v,e)=({v},{e}) {kind:11s} sup|dm/dt| in [{ys.min():.3e}, {ys.max():.3e}]'
            + (f'  order {slope:.4f} (pred {pred})' if slope is not None else '  (constant tree moment)'))
    return table


def write_digits(w, th, thp, res):
    lines = ["Searched pair at w0 sharing S_1, ..., S_4 (Appendix H-B), refined in 80-digit arithmetic",
             "with theta_1 held at its double-precision value.", "",
             "w0, exact values of its double-precision coordinates:"]
    lines += [f"  w0_{c + 1} = {x}" for c, x in enumerate(w)]
    lines += ["", "theta:"] + [f"  theta_{c + 1} = {x}" for c, x in enumerate(th)]
    lines += ["", "theta':"] + [f"  theta'_{c + 1} = {x}" for c, x in enumerate(thp)]
    lines += ["", f"max over v = 2, 3, 4 of |S_v(theta) - S_v(theta')| = {mp.nstr(res, 3)}"]
    open(os.path.join(OUT, 'searched_pair_80_digits.txt'), 'w').write('\n'.join(lines) + '\n')


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    found = json.load(open(os.path.join(OUT, 'searched_pair.json')))
    th0, thp0 = np.array(found['pair']['theta']), np.array(found['pair']['theta_prime'])
    iw, w = found['weight_row'], np.array(found['w0'])
    log(f'[K3] searched pair theta = {th0}, theta\' = {thp0}; w0 = {w!r}')
    th, thp, wm, res = refine(th0, thp0, w, 4)
    log(f'[K3] refined: residual = {mp.nstr(res, 3)}; theta = {[mp.nstr(t, 15) for t in th]}; '
        f'theta\' = {[mp.nstr(t, 15) for t in thp]}')
    dS5 = S(5, th, wm) - S(5, thp, wm)
    a = [1 / x for x in wm]
    mu = 1 / mp.fsum(a)
    u = [wm[c] * th[c] for c in range(3)]
    up = [wm[c] * thp[c] for c in range(3)]
    s = mp.mpf('1e-8')
    ths = [a[c] * (mu + s * (u[c] - mu)) for c in range(3)]
    thps = [a[c] * (mu + s * (up[c] - mu)) for c in range(3)]
    dHs = (ent(ths) - ent(thps)) / s
    log(f'[K3] Delta S_5(1) = {mp.nstr(dS5, 12)}; '
        f'Delta H(s)/s at s=1e-8 = {mp.nstr(dHs, 12)}; '
        f'H(theta) = {mp.nstr(ent(th), 8)}, H(theta\') = {mp.nstr(ent(thp), 8)}')
    s_list = ['1', '1e-1', mp.mpf(10) ** mp.mpf('-1.5'), '1e-2', mp.mpf(10) ** mp.mpf('-2.5'), '1e-3']
    r3 = path_study(th, thp, wm, 3, log, 'K3', s_list=s_list)
    r3['orders'] = orders(r3, log, 'K3', 3)
    r3['witness'] = dict(weight_row=iw, w=[str(x) for x in wm], theta=[str(x) for x in th],
                         theta_prime=[str(x) for x in thp], residual=float(res),
                         dS5=str(dS5), dH_over_s=str(dHs))
    write_digits([str(x) for x in wm], [str(x) for x in th], [str(x) for x in thp], res)

    json.dump(dict(K3=r3), open(os.path.join(OUT, 'moment_path.json'), 'w'), indent=1)
    open(os.path.join(OUT, 'moment_path.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
