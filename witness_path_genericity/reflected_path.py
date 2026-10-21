"""The moment-preserving path of Lemma 2 between the reflected pair at w_0 (K = 3; Appendix H-E, Table II).

The scaled measure nu_t = sum_k alpha_k(t) delta_{xi_k(t)} has moments M_0..M_4 of nu_x and
M_5(t) = (1 - 2t) X_5(x); the path is theta_k = alpha_k (mu + s xi_k), w_k = 1/alpha_k. The script
records the s-free constants of the path and checks the orders of the motif-moment derivatives.
"""
import json
import math
import os

import mpmath as mp

mp.mp.dps = 50
OUT = os.path.join('witness_path_genericity', 'output')
MOD = json.load(open(os.path.join('modulus_and_variance', 'output', 'modulus.json')))
WIT = json.load(open(os.path.join('moment_preservation', 'output', 'moment_path.json')))['K3']['witness']
K = 3
M = 2 * K - 1
W = [mp.mpf(x) for x in WIT['w']]
A = [1 / x for x in W]
MU = 1 / mp.fsum(A)


def reflected_direction():
    """x with sum a_c x_c = sum a_c x_c^3 = 0 and max |x_c| = 1 (refined from the modulus check)."""
    b1 = [A[1], -A[0], mp.mpf(0)]
    b2 = [A[1] * b1[2] - A[2] * b1[1], A[2] * b1[0] - A[0] * b1[2], A[0] * b1[1] - A[1] * b1[0]]
    vec = lambda t: [mp.cos(t) * b1[c] + mp.sin(t) * b2[c] for c in range(K)]
    phi = mp.findroot(lambda t: mp.fsum(A[c] * vec(t)[c] ** 3 for c in range(K)), mp.mpf(MOD['reflected'][0]['phi']))
    x = vec(phi)
    return [v / max(abs(t) for t in x) for v in x]


def measure_from_moments(X):
    H = mp.matrix(K, K)
    rhs = mp.matrix(K, 1)
    for i in range(K):
        for j in range(K):
            H[i, j] = X[i + j]
        rhs[i] = -X[i + K]
    b = mp.lu_solve(H, rhs)
    roots = mp.polyroots([mp.mpf(1)] + [b[K - 1 - j] for j in range(K)], maxsteps=200, extraprec=200)
    imag = max(abs(mp.im(r)) for r in roots)
    xs = sorted(mp.re(r) for r in roots)
    V = mp.matrix(K, K)
    for i in range(K):
        for j in range(K):
            V[i, j] = xs[j] ** i
    al = mp.lu_solve(V, mp.matrix([X[i] for i in range(K)]))
    return [al[j] for j in range(K)], xs, imag


def derivatives(al, xi, dX):
    Mx = mp.matrix(2 * K, 2 * K)
    for j in range(2 * K):
        for r in range(K):
            Mx[j, r] = xi[r] ** j
            Mx[j, K + r] = j * al[r] * xi[r] ** (j - 1) if j > 0 else 0
    d = mp.lu_solve(Mx, mp.matrix(dX))
    return [d[r] for r in range(K)], [d[K + r] for r in range(K)]


def ent(th):
    return -mp.fsum(t * mp.log(t) for t in th)


def main():
    os.makedirs(OUT, exist_ok=True)
    x = reflected_direction()
    X = [mp.fsum(A[c] * x[c] ** j for c in range(K)) for j in range(M + 3)]
    L = -mp.fsum(A[c] * x[c] * mp.log(A[c]) for c in range(K))
    dM = [mp.mpf(0)] * M + [-2 * X[M]]
    grid = [mp.mpf(i) / 2000 for i in range(2001)]
    rec = dict(min_gap=mp.inf, min_alpha=mp.inf, max_alpha=0, max_dalpha=0, max_dxi=0, max_imag=0,
               max_abs_xi=0, max_dM=[0, 0, 0])
    path = []
    for t in grid:
        Mt = X[:M] + [(1 - 2 * t) * X[M]]
        al, xi, imag = measure_from_moments(Mt)
        dal, dxi = derivatives(al, xi, dM)
        dMj = [mp.fsum(dal[k] * xi[k] ** j + j * al[k] * xi[k] ** (j - 1) * dxi[k] for k in range(K))
               for j in (5, 6, 7)]
        rec['min_gap'] = min(rec['min_gap'], min(xi[k + 1] - xi[k] for k in range(K - 1)))
        rec['min_alpha'] = min(rec['min_alpha'], min(al))
        rec['max_alpha'] = max(rec['max_alpha'], max(al))
        rec['max_dalpha'] = max(rec['max_dalpha'], max(abs(v) for v in dal))
        rec['max_dxi'] = max(rec['max_dxi'], max(abs(v) for v in dxi))
        rec['max_imag'] = max(rec['max_imag'], imag)
        rec['max_abs_xi'] = max(rec['max_abs_xi'], max(abs(v) for v in xi))
        rec['max_dM'] = [max(a_, abs(b_)) for a_, b_ in zip(rec['max_dM'], dMj)]
        path.append((t, al, xi, dal, dxi))
    al0, xi0 = path[0][1], path[0][2]
    al1, xi1 = path[-1][1], path[-1][2]
    start = sorted(zip(x, A))
    end = sorted(zip([-v for v in x], A))
    end_err = max(max(abs(p - q[0]), abs(r - q[1])) for p, r, q in zip(xi0, al0, start))
    end_err = max(end_err, max(max(abs(p - q[0]), abs(r - q[1])) for p, r, q in zip(xi1, al1, end)))
    lines = [f"x = {[mp.nstr(v, 12) for v in x]}, X_1 = {mp.nstr(X[1], 3)}, X_3 = {mp.nstr(X[3], 3)}, "
             f"X_5 = {mp.nstr(X[5], 12)}, L(x) = {mp.nstr(L, 12)}, mu = {mp.nstr(MU, 12)}",
             f"path (s-free): min atom gap = {mp.nstr(rec['min_gap'], 6)}, alpha in [{mp.nstr(rec['min_alpha'], 6)}, "
             f"{mp.nstr(rec['max_alpha'], 6)}] (w in [{mp.nstr(1 / rec['max_alpha'], 6)}, {mp.nstr(1 / rec['min_alpha'], 6)}]), "
             f"max|alpha'| = {mp.nstr(rec['max_dalpha'], 6)}, max|xi'| = {mp.nstr(rec['max_dxi'], 6)}, "
             f"max|xi| = {mp.nstr(rec['max_abs_xi'], 6)}, max|M_5'|,|M_6'|,|M_7'| = {[mp.nstr(v, 6) for v in rec['max_dM']]}, "
             f"max imag part of roots = {mp.nstr(rec['max_imag'], 3)}, endpoint error = {mp.nstr(end_err, 3)}"]
    rows = []
    for s in [mp.mpf('0.1'), mp.mpf('0.03'), mp.mpf('0.01'), mp.mpf('0.003')]:
        st = dict(s=float(s), theta_min=mp.inf, dS_le4=0, dS5_s5=0, dS6_s5=0, dS7_s5=0, uni_s=0, multi=0,
                  eta=0, ratio=0)
        for t, al, xi, dal, dxi in path:
            th = [al[k] * (MU + s * xi[k]) for k in range(K)]
            dth = [dal[k] * (MU + s * xi[k]) + al[k] * s * dxi[k] for k in range(K)]
            dw = [-dal[k] / al[k] ** 2 for k in range(K)]
            u = [MU + s * xi[k] for k in range(K)]
            du = [s * dxi[k] for k in range(K)]
            dm = lambda v, e: mp.fsum((v - e) * al[k] ** (v - e - 1) * dal[k] * u[k] ** v
                                      + al[k] ** (v - e) * v * u[k] ** (v - 1) * du[k] for k in range(K))
            st['theta_min'] = min(st['theta_min'], min(th))
            st['dS_le4'] = max(st['dS_le4'], max(abs(dm(v, v - 1)) for v in range(2, 5)))
            for v in (5, 6, 7):
                st[f'dS{v}_s5'] = max(st[f'dS{v}_s5'], abs(dm(v, v - 1)) / s ** 5)
            st['uni_s'] = max(st['uni_s'], max(abs(dm(v, v)) / s for v in range(3, 8)))
            st['multi'] = max(st['multi'], max(abs(dm(v, e)) for v in range(4, 8) for e in range(v + 1, v * (v - 1) // 2 + 1)))
            st['eta'] = max(st['eta'], mp.fsum(abs(v) for v in dth) + max(abs(v) for v in dw))
            st['ratio'] = max(st['ratio'], max(abs(dth[k]) / th[k] for k in range(K)))
        th0 = [A[c] * (MU + s * x[c]) for c in range(K)]
        th1 = [A[c] * (MU - s * x[c]) for c in range(K)]
        gap = ent(th0) - ent(th1)
        st['gap_over_2sL'] = gap / (2 * s * L)
        rows.append({k: (float(v) if not isinstance(v, float) else v) for k, v in st.items()})
        lines.append(f"s = {mp.nstr(s, 3)}: theta_min = {mp.nstr(st['theta_min'], 6)}, max|dS_2..4| = {mp.nstr(st['dS_le4'], 3)}, "
                     f"max|dS_5|/s^5 = {mp.nstr(st['dS5_s5'], 6)}, max|dS_6|/s^5 = {mp.nstr(st['dS6_s5'], 6)}, "
                     f"max|dS_7|/s^5 = {mp.nstr(st['dS7_s5'], 6)}, max unicyclic |dm|/s = {mp.nstr(st['uni_s'], 6)}, "
                     f"max multicyclic |dm| = {mp.nstr(st['multi'], 6)}, eta = {mp.nstr(st['eta'], 6)}, "
                     f"max|theta'|/theta = {mp.nstr(st['ratio'], 6)}, (H0 - H1)/(2 s L) = {mp.nstr(st['gap_over_2sL'], 8)}")
    open(os.path.join(OUT, 'reflected_path.txt'), 'w').write('\n'.join(lines) + '\n')
    json.dump(dict(x=[float(v) for v in x], L=float(L), X5=float(X[5]), mu=float(MU),
                   path={k: (float(v) if not isinstance(v, list) else [float(t) for t in v]) for k, v in rec.items()},
                   endpoint_err=float(end_err), by_s=rows), open(os.path.join(OUT, 'reflected_path.json'), 'w'), indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
