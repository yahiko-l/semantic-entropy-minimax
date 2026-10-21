"""Tree moments and entropies quoted with the matched-budget simulation (K = 3, w = (1, 2, 4); Appendix H-F):
the reflected pair at s = 0.4, and a near-vertex distribution next to an interior one with nearly the same S_2."""
import json
import os

import numpy as np

from mse_sim import JUDGES, OUT, S_REFLECT, THETAS, entropy, tree_moments

W = JUDGES['meaning-dependent'][0]


def describe(theta):
    return dict(theta=np.asarray(theta).tolist(), H=entropy(theta), S=tree_moments(np.asarray(theta), W, 5).tolist())


def main():
    plus, minus = describe(THETAS['reflected +']), describe(THETAS['reflected -'])
    vertex, interior = describe([0.98, 0.01, 0.01]), describe([0.17, 0.55, 0.28])
    out = dict(
        w=W.tolist(), s=S_REFLECT,
        reflected=dict(plus=plus, minus=minus,
                       S_diff=(np.array(plus['S']) - np.array(minus['S'])).tolist(),
                       H_diff=minus['H'] - plus['H']),
        confusable=dict(vertex=vertex, interior=interior,
                        S_diff=(np.array(interior['S']) - np.array(vertex['S'])).tolist(),
                        H_diff=interior['H'] - vertex['H']))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'moment_comparisons.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
