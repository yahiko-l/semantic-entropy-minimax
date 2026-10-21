"""Numerical search for a pair theta != theta' at w_0 that shares the tree moments S_1, ..., S_4 (K = 3).

w_0 is the profile with index 14 among the 200 seeded profiles of common.draw_weights(3). The search has
three stages:
  1. for random target points, and target points near the walls where two locations w_c theta_c coincide,
     the preimages of their (S_2, S_3) found by multistart damped Newton and polished in 50-digit
     arithmetic; pairs of distinct preimages with entropy gap above 1e-3 are kept (run_draw);
  2. from each of the first N_PAIRS such pairs, a Nelder-Mead minimisation of |S_4(theta) - S_4(theta')|
     with theta' kept on the fibre of (S_2, S_3) through theta, polished in 50-digit arithmetic (search);
  3. Newton refinement in 60-digit arithmetic of S_v(theta) = S_v(theta'), v = 2, 3, 4, with the leading
     coordinate of theta held fixed (refine_exact).
The pair kept is the unique exact zero with distinct points, entropy gap above 1e-3 and positive coordinates.
"""

import itertools
import math
import sys

import mpmath as mp
import numpy as np
from scipy.optimize import minimize

from common import (cycle_family, draw_weights, dump, entropy_mp, entropy_np, high_trees, moment_mp, moment_np,
                    newton_batch, polish_mp, to_mp)

DPS = 50
DPS_EXACT = 60
N_RAND, N_WALL, N_STARTS = 48, 24, 32
PEN = 10.0
N_PAIRS = 5
W0_INDEX = 14


def make_targets(K, w, rng):
    T = list(rng.dirichlet(np.ones(K), size=N_RAND))
    pairs = list(itertools.combinations(range(K), 2))
    walls, tries = [], 0
    while len(walls) < N_WALL and tries < 100000:
        tries += 1
        c, cp = pairs[len(walls) % len(pairs)]
        th = rng.dirichlet(np.ones(K))
        t = (w[cp] * th[cp] - w[c] * th[c]) / (w[c] + w[cp])
        th[c] += t
        th[cp] -= t
        r = rng.uniform(0.01, 0.1) * rng.choice([-1.0, 1.0])
        th[c] += r
        th[cp] -= r
        if th.min() > 1e-6:
            walls.append(th)
    return T + walls, len(walls)


def separations(a, b, w_mp, K):
    sep = {}
    for (v, e) in high_trees(K) + cycle_family(K):
        sep[f"{v},{e}"] = float(abs(moment_mp(a, w_mp, v, e) - moment_mp(b, w_mp, v, e)))
    return sep


def run_draw(args):
    K, idx, w = args
    mp.mp.dps = DPS
    rng = np.random.default_rng(1000003 * K + idx)
    w_mp = to_mp(w)
    vs = list(range(2, K + 1))
    targets, n_wall = make_targets(K, w, rng)
    n_pre_hist, qual, n_multi, n_multi_rand = {}, [], 0, 0
    perm_complete_rand = 0
    max_res = 0.0
    for ti, tf in enumerate(targets):
        tphi = to_mp(tf[:K - 1])
        tth = tphi + [1 - mp.fsum(tphi)]
        target_mp = [moment_mp(tth, w_mp, v, v - 1) for v in vs]
        target_f = np.array([float(x) for x in target_mp])
        starts = np.vstack([rng.dirichlet(np.ones(K), size=N_STARTS), tf[None, :]])[:, :K - 1]
        phi, conv, _ = newton_batch(starts, w, target_f, vs)
        sols = []
        for p in phi[conv]:
            th = np.append(p, 1 - p.sum())
            if th.min() <= 1e-10:
                continue
            if all(np.max(np.abs(th - s)) > 1e-8 for s in sols):
                sols.append(th)
        polished = []
        for s in sols:
            th_mp, res = polish_mp(to_mp(s), w_mp, target_mp, vs)
            if min(th_mp) <= 0 or res > 1e-30:
                continue
            if all(max(abs(x - y) for x, y in zip(th_mp, q)) > mp.mpf(10) ** -20 for q in polished):
                polished.append(th_mp)
                max_res = max(max_res, float(res))
        m = len(polished)
        n_pre_hist[m] = n_pre_hist.get(m, 0) + 1
        if m >= 2:
            n_multi += 1
            if ti < N_RAND:
                n_multi_rand += 1
        if ti < N_RAND and m == math.factorial(K):
            perm_complete_rand += 1
        H = [entropy_mp(t) for t in polished]
        for i, j in itertools.combinations(range(m), 2):
            a, b = polished[i], polished[j]
            dH = float(abs(H[i] - H[j]))
            dS = float(max(abs(moment_mp(a, w_mp, v, v - 1) - moment_mp(b, w_mp, v, v - 1)) for v in vs))
            if dH > 1e-3 and dS < 1e-10:
                ua = [wc * t for wc, t in zip(w_mp, a)]
                ub = [wc * t for wc, t in zip(w_mp, b)]
                walls = [(c, cp) for c, cp in itertools.combinations(range(K), 2)
                         if (ua[c] - ua[cp]) * (ub[c] - ub[cp]) < 0]
                gap = min(min(abs(ua[c] - ua[cp]), abs(ub[c] - ub[cp])) for c, cp in itertools.combinations(range(K), 2))
                qual.append({"target": ti, "theta_a": [float(x) for x in a], "theta_b": [float(x) for x in b],
                             "theta_a_str": [mp.nstr(x, 40) for x in a], "theta_b_str": [mp.nstr(x, 40) for x in b],
                             "dH": dH, "dS_2toK": dS, "walls_between": walls, "min_u_gap": float(gap),
                             "sep": separations(a, b, w_mp, K)})
    return {"K": K, "idx": idx, "w": list(map(float, w)), "n_targets": len(targets), "n_wall_targets": n_wall,
            "n_preimages_hist": {str(k): v for k, v in sorted(n_pre_hist.items())}, "n_targets_multi": n_multi,
            "n_rand_targets_multi": n_multi_rand, "perm_complete_rand": perm_complete_rand,
            "max_polish_residual": max_res, "qualifying_pairs": qual}


def search(args):
    K, idx, w, a, b = args
    fam = [(K + 1, K)]
    w = np.asarray(w)
    vs = list(range(2, K + 1))
    state = {"tp": np.asarray(b, float), "best": (np.inf, None, None)}

    def phi_fun(p):
        th = np.append(p, 1 - p.sum())
        if th.min() < 1e-6:
            return PEN
        target = np.array([np.sum(th ** v * w ** (v - 1)) for v in vs])
        q, conv, _ = newton_batch(state["tp"][None, :K - 1], w, target, vs, iters=60, tol=1e-14)
        if not conv[0]:
            return PEN
        tp = np.append(q[0], 1 - q[0].sum())
        if tp.min() < 1e-6 or np.max(np.abs(th - tp)) < 1e-4 or abs(entropy_np(th) - entropy_np(tp)) < 1e-3:
            return PEN
        state["tp"] = tp
        val = float(np.sqrt(sum((moment_np(th, w, v, e) - moment_np(tp, w, v, e)) ** 2 for v, e in fam)))
        if val < state["best"][0]:
            state["best"] = (val, th.copy(), tp.copy())
        return val

    p0 = np.asarray(a, float)[:K - 1]
    start_val = phi_fun(p0)
    minimize(phi_fun, p0, method="Nelder-Mead", options={"maxfev": 400, "xatol": 1e-12, "fatol": 1e-16})
    val, th, tp = state["best"]
    rec = {"K": K, "idx": idx, "start_val": start_val, "min_val_float": val}
    if th is not None:
        mp.mp.dps = 50
        w_mp = to_mp(w)
        tm = to_mp(th[:K - 1])
        tm.append(1 - mp.fsum(tm))
        target = [moment_mp(tm, w_mp, v, v - 1) for v in vs]
        tpm, res = polish_mp(to_mp(tp), w_mp, target, vs, iters=20)
        rec["min_val_mp"] = float(mp.sqrt(mp.fsum((moment_mp(tm, w_mp, v, e) - moment_mp(tpm, w_mp, v, e)) ** 2
                                                  for v, e in fam)))
        rec["dH_at_min"] = float(abs(entropy_mp(tm) - entropy_mp(tpm)))
        rec["polish_residual"] = float(res)
        rec["theta"] = [float(x) for x in tm]
        rec["theta_prime"] = [float(x) for x in tpm]
    return rec


def refine_exact(args):
    K, w, th, tp = args
    mp.mp.dps = DPS_EXACT
    fam = [(v, v - 1) for v in range(2, K + 1)] + [(K + 1, K)]
    E = len(fam)
    m = 2 * K - 2 - E
    w_mp = to_mp(w)
    fixed = to_mp(th[:m])
    z = to_mp(th[m:K - 1]) + to_mp(tp[:K - 1])
    nfree = K - 1 - m

    def unpack(z):
        a = fixed + z[:nfree]
        a = a + [1 - mp.fsum(a)]
        b = list(z[nfree:])
        b = b + [1 - mp.fsum(b)]
        return a, b

    def dM(t, v, e, c):
        return v * (t[c] ** (v - 1) * w_mp[c] ** e - t[K - 1] ** (v - 1) * w_mp[K - 1] ** e)

    conv = False
    for _ in range(80):
        a, b = unpack(z)
        F = [moment_mp(b, w_mp, v, e) - moment_mp(a, w_mp, v, e) for v, e in fam]
        J = mp.matrix(E, E)
        for i, (v, e) in enumerate(fam):
            for j in range(nfree):
                J[i, j] = -dM(a, v, e, m + j)
            for j in range(K - 1):
                J[i, nfree + j] = dM(b, v, e, j)
        try:
            d = mp.lu_solve(J, mp.matrix([-f for f in F]))
        except ZeroDivisionError:
            break
        z = [zi + di for zi, di in zip(z, d)]
        if max(abs(di) for di in d) < mp.mpf(10) ** -(DPS_EXACT - 10):
            conv = True
            break
    a, b = unpack(z)
    res = max(abs(moment_mp(b, w_mp, v, e) - moment_mp(a, w_mp, v, e)) for v, e in fam)
    return {"K": K, "converged": conv, "residual": float(res),
            "dist": float(max(abs(x - y) for x, y in zip(a, b))), "dH": float(abs(entropy_mp(a) - entropy_mp(b))),
            "min_theta": float(min(min(a), min(b))), "theta": [float(x) for x in a], "theta_prime": [float(x) for x in b]}


def main():
    K = 3
    w = draw_weights(K)[W0_INDEX]
    draw = run_draw((K, W0_INDEX, w))
    starts = draw["qualifying_pairs"][:N_PAIRS]
    fibre = [search((K, W0_INDEX, draw["w"], p["theta_a"], p["theta_b"])) for p in starts]
    exact = [refine_exact((K, draw["w"], r["theta"], r["theta_prime"])) for r in fibre if "theta" in r]
    zeros = [r for r in exact if r["converged"] and r["residual"] < 1e-45 and r["dist"] > 1e-4 and r["dH"] > 1e-3
             and r["min_theta"] > 0]
    assert len(zeros) == 1, len(zeros)
    pair = dict(theta=zeros[0]["theta"], theta_prime=zeros[0]["theta_prime"])
    dump(dict(K=K, weight_row=W0_INDEX, w0=draw["w"], n_pairs_sharing_S2_S3=len(draw["qualifying_pairs"]),
              starts=[dict(theta_a=p["theta_a"], theta_b=p["theta_b"]) for p in starts],
              fibre_minima=fibre, exact_zeros=exact, pair=pair), "searched_pair.json")
    print(f"w0 = {draw['w']}")
    print(f"pairs of distinct preimages of (S_2, S_3) with entropy gap > 1e-3: {len(draw['qualifying_pairs'])}")
    print(f"exact common zeros of S_2, S_3, S_4 from the first {N_PAIRS}: {len(zeros)}; "
          f"theta = {pair['theta']}, theta' = {pair['theta_prime']}, |H - H'| = {zeros[0]['dH']:.6f}")


if __name__ == "__main__":
    sys.exit(main())
