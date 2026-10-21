"""Shared pieces for the pair search: weighted motif moments, entropy, sensitivity draws, Newton solvers.

Channel: A_ij ~ Bern(q + Delta * w_c * 1{Z_i = Z_j = c}). A connected motif on v vertices with e edges has
mean M_{v,e}(theta) = sum_c theta_c^v w_c^e; trees give S_v = M_{v,v-1}, cycles and denser motifs e >= v.
"""

import json
import os

import mpmath as mp
import numpy as np

OUT = os.path.join("moment_preservation", "output")
os.makedirs(OUT, exist_ok=True)

N_DRAWS = 200
W_LO, W_HI = 0.3, 1.7


def draw_weights(K, n=N_DRAWS):
    return np.random.default_rng(20261012 + K).uniform(W_LO, W_HI, size=(n, K))


def cycle_family(K):
    return [(v, e) for v in range(3, K + 2) for e in range(v, v * (v - 1) // 2 + 1)]


def high_trees(K):
    return [(v, v - 1) for v in range(K + 1, 2 * K)]


def moment_np(theta, w, v, e):
    return np.sum(theta ** v * w ** e, axis=-1)


def entropy_np(theta):
    t = np.clip(theta, 1e-300, None)
    return -np.sum(np.where(theta > 0, theta * np.log(t), 0.0), axis=-1)


def moment_mp(theta, w, v, e):
    return mp.fsum(t ** v * wc ** e for t, wc in zip(theta, w))


def entropy_mp(theta):
    return -mp.fsum(t * mp.log(t) for t in theta if t > 0)


def to_mp(xs):
    return [mp.mpf(float(x)) for x in xs]


# ---------------------------------------------------------------- star-moment system on the simplex
# Unknowns phi = theta_1..theta_{K-1}, theta_K = 1 - sum(phi); equations S_v(theta) = target_v for v in vs.

def full_theta_np(phi):
    return np.concatenate([phi, 1.0 - phi.sum(axis=-1, keepdims=True)], axis=-1)


def newton_batch(phi0, w, target, vs, iters=100, tol=1e-13):
    """Damped Newton for many starts at once. phi0: (M, K-1). Returns phi, converged mask, residual."""
    phi = phi0.astype(float).copy()
    K = phi.shape[1] + 1
    vs = list(vs)

    def resid(p):
        th = full_theta_np(p)
        return np.stack([np.sum(th ** v * w ** (v - 1), axis=1) for v in vs], axis=1) - target

    def jac(p):
        th = full_theta_np(p)
        u = th * w
        return np.stack([v * (u[:, :K - 1] ** (v - 1) - u[:, K - 1:K] ** (v - 1)) for v in vs], axis=1)

    F = resid(phi)
    nF = np.max(np.abs(F), axis=1)
    active = np.isfinite(nF)
    for _ in range(iters):
        act = active & (nF >= tol)
        if not act.any():
            break
        idx = np.where(act)[0]
        J = jac(phi[idx])
        det = np.linalg.det(J)
        bad = ~np.isfinite(det) | (np.abs(det) < 1e-300)
        J[bad] += 1e-12 * np.eye(K - 1)
        try:
            d = np.linalg.solve(J, -F[idx][..., None])[..., 0]
        except np.linalg.LinAlgError:
            d = np.stack([np.linalg.lstsq(Ji, -Fi, rcond=None)[0] for Ji, Fi in zip(J, F[idx])])
        alpha = np.ones(len(idx))
        newphi = phi[idx] + d
        for _ls in range(30):
            newF = resid(newphi)
            newn = np.max(np.abs(newF), axis=1)
            ok = np.isfinite(newn) & (newn < nF[idx])
            if ok.all():
                break
            alpha = np.where(ok, alpha, alpha / 2)
            newphi = np.where(ok[:, None], newphi, phi[idx] + alpha[:, None] * d)
        newphi = np.clip(newphi, -3.0, 4.0)
        newF = resid(newphi)
        newn = np.max(np.abs(newF), axis=1)
        improved = np.isfinite(newn) & (newn < nF[idx])
        stalled = idx[~improved]
        phi[idx[improved]] = newphi[improved]
        F[idx[improved]] = newF[improved]
        nF[idx[improved]] = newn[improved]
        active[stalled] = False
    return phi, nF < tol, nF


def polish_mp(theta, w_mp, target_mp, vs, iters=8):
    """Newton in mpmath at the current precision on S_v(theta) = target_v (v in vs), theta on the simplex."""
    K = len(w_mp)
    phi = [mp.mpf(x) if not isinstance(x, mp.mpf) else x for x in list(theta)[:K - 1]]
    for _ in range(iters):
        th = phi + [1 - mp.fsum(phi)]
        F = [moment_mp(th, w_mp, v, v - 1) - t for v, t in zip(vs, target_mp)]
        J = mp.matrix(len(vs), K - 1)
        for i, v in enumerate(vs):
            for j in range(K - 1):
                J[i, j] = v * ((w_mp[j] * th[j]) ** (v - 1) - (w_mp[K - 1] * th[K - 1]) ** (v - 1))
        d = mp.lu_solve(J, mp.matrix([-f for f in F]))
        phi = [phi[j] + d[j] for j in range(K - 1)]
    th = phi + [1 - mp.fsum(phi)]
    res = max(abs(moment_mp(th, w_mp, v, v - 1) - t) for v, t in zip(vs, target_mp))
    return th, res


def dump(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)
