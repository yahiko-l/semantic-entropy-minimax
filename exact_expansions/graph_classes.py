"""Isomorphism classes of graphs on n <= 7 vertices, class sizes, and K = 3 labelling-pattern histograms.

A pattern of a labelling z is (k_1, k_2, k_3, n_1, n_2, n_3): k_c vertices carry label c and n_c edges of
the graph join two of them. The likelihood ratio of the graph against Erdos-Renyi(q) depends on z only
through its pattern, so each class is stored as its edge count and pattern histogram.
"""
import itertools
import math
import os
import pickle
import sys

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher

OUT = os.path.join('exact_expansions', 'output')


def classes(n):
    out = []
    for G in nx.graph_atlas_g():
        if G.number_of_nodes() != n:
            continue
        aut = sum(1 for _ in GraphMatcher(G, G).isomorphisms_iter())
        out.append((sorted(G.edges()), math.factorial(n) // aut))
    assert sum(size for _, size in out) == 2 ** (n * (n - 1) // 2)
    return out


def pattern_histogram(n, edges, Z):
    k = np.stack([(Z == c).sum(1) for c in range(3)], 1)
    if edges:
        E = np.array(edges)
        same = Z[:, E[:, 0]] == Z[:, E[:, 1]]
        lab = Z[:, E[:, 0]]
        inside = np.stack([(same & (lab == c)).sum(1) for c in range(3)], 1)
    else:
        inside = np.zeros((len(Z), 3), dtype=int)
    keys, counts = np.unique(np.concatenate([k, inside], 1), axis=0, return_counts=True)
    return [tuple(int(x) for x in key) for key in keys], [int(c) for c in counts]


def main(nmax=7):
    data = {}
    for n in range(2, nmax + 1):
        Z = np.array(list(itertools.product(range(3), repeat=n)), dtype=np.int8)
        rows = []
        for edges, size in classes(n):
            pats, cnts = pattern_histogram(n, edges, Z)
            rows.append(dict(edges=edges, size=size, nE=len(edges), patterns=pats, counts=cnts))
        data[n] = rows
        npat = len({p for r in rows for p in r['patterns']})
        print(f'n = {n}: {len(rows)} classes, {npat} distinct patterns, '
              f'max patterns per graph {max(len(r["patterns"]) for r in rows)}', flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'graph_classes.pkl'), 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
