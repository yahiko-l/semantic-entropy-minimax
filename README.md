# Meaning-Dependent Judge Sensitivity Changes the Minimax Rate of Semantic Entropy

Xiaoli Yu and Jiamiao Liu

The paper studies the estimation of semantic entropy from pairwise judgments by a judge whose sensitivity
depends on the meaning: two answers with meaning c are declared equivalent with probability q + Delta w_c, and
two answers with different meanings with probability q, where q is bounded away from 0 and 1 and the
sensitivity profile w is known. For K >= 3 meanings and almost every w, the minimax squared risk on every fixed
polynomial window n Delta^2 <= n^-kappa is of order min{1, N^(-1/(2K-1))} with N = n^(2K-1) Delta^(4K-4), and the
entropy is uniformly consistently estimable exactly when N tends to infinity. For three meanings and
Delta = n^-beta, the threshold falls from beta < 3/4 for a homogeneous judge to beta < 5/8, even for a judge that
recognizes every meaning at least as well as a homogeneous one.

These results are theorems, proved in the paper. This repository contains the programs behind every number in
the paper that comes from a computation, the outputs that hold these numbers, and the instructions that
reproduce them. The programs compute

- the risk of the projected star-moment estimator of the paper (Algorithm 1) in simulation, for a homogeneous
  judge and for w = (1, 2, 4) with the same q, Delta and n (Fig. 5, Table III);
- the example at w = (1, 2, 4) behind the lower bound: the witness, the reflected pairs at the interior point
  theta proportional to 1/w, and the moment-preserving path between them (Figs. 2 to 4);
- checks of the mechanisms and constants the proofs rely on (Table II).

The estimator is implemented by `star_stats_from_degrees` and `project` in `entropy_risk/mse_sim.py`. The first
computes the star statistics from the degrees of the verdict graph, which determine them because a centered
verdict takes one value on an edge and another off it; the second solves the projection on a grid of the
simplex, refined locally.

The computations are synthetic: no script reads external data, and every file a script reads is written by an
earlier step. Everything runs on a CPU. The high-precision computations use mpmath, the exact ones use Python
fractions and SymPy, and the rest uses NumPy and SciPy.

## Contents

| Folder | What it computes | Paper |
| :-- | :-- | :-- |
| `entropy_risk/` | matched-budget simulation of the estimator, tree moments and entropies quoted with it | Fig. 5, Table III |
| `reflected_pair_w124/` | witness, reflected pairs and their moment-preserving path at w = (1, 2, 4), random pairs of Fig. 4 | Figs. 2 to 4, Table II |
| `moment_preservation/` | search for a pair sharing S_1, ..., S_4 at the profile w0 of the checks, its 80-digit values, the moment-preserving path between its endpoints scaled toward theta* | Table II |
| `exact_expansions/` | graph classes on at most 7 vertices, exact Fisher expansion, endpoint chi-square expansion | Table II |
| `modulus_and_variance/` | entropy modulus, variance of the star statistics | Table II |
| `witness_path_genericity/` | Condition W at w0, reflected path at w0, genericity of Condition W | Table II |

Appendix H of the paper describes the setting of every check and of the simulation. Every folder keeps its
scripts at the top and its outputs in `output/`. The files in the `output/` folders are the outputs used in the
paper, and running a script rewrites its own outputs in place.

## Paper items, scripts and outputs

Paths are relative to the repository root. A file name without a folder lives in the `output/` folder of the
script in the same row.

### Figures and tables

| Paper item | Script | Output (where the paper's numbers are) |
| :-- | :-- | :-- |
| Fig. 1 | none | the exponents are closed-form expressions; it involves no computation |
| Fig. 2 | `reflected_pair_w124/compute_figure_data.py` | `reflected.json` (`w`, witness `x`); the entropies 0.838 and 0.896 of the pair in (b) are `H` of the cells `reflected +` and `reflected -` in `entropy_risk/output/mse.json` |
| Fig. 3 | `reflected_pair_w124/compute_figure_data.py` | `path.json` |
| Fig. 4 | `reflected_pair_w124/compute_figure_data.py` | `reflected.json` (`ell`, `limit_ratio` = c), `random_pairs.json` (`n` = 19,950) |
| Fig. 5 | `entropy_risk/mse_sim.py` | `mse.json` (`cells[*]`: `mse` and `mse_se`, with `order_K` for the projection on S_2, S_3 only) |
| Table I | none | summary of the theorems and of the cited homogeneous results; it involves no computation |
| Table II | the scripts in the rows below | the outputs in the rows below; a bound over sampled points or runs is their maximum or minimum, rounded outward as the table note states |
| Table II, row "Moment preservation and fifth-order scaling" | `moment_preservation/moment_path.py` | `moment_path.txt` (lines `max rel drift S_1..S_4` and `(v,e)=(5,4) tree ... order`), `moment_path.json` |
| Table II, row "Exact Fisher information" | `exact_expansions/graph_classes.py`, `exact_expansions/fisher_expansion.py` | `fisher_expansion.txt` (checks (a) to (d) per direction), `fisher_expansion.json` (`c_float`: coefficients c_{k,v}) |
| Table II, row "Endpoint chi^2" | `exact_expansions/endpoint_chi2.py` | `endpoint_chi2.txt`, `endpoint_chi2.json` (`low_rel`, `blowup_rel`, `c85_rel_err`, `c_float`) |
| Table II, row "Modulus" | `modulus_and_variance/check_modulus.py` | `modulus.txt`, `modulus.json` (`random`, `adversarial.T.value`, `constants.C`) |
| Table II, row "Variance" | `modulus_and_variance/check_variance.py` | `variance.txt`, `variance.json` (`var_over_bound`, `var_over_lead`) |
| Table II, row "Condition W at w0" | `witness_path_genericity/verify_condition_w.py` | `condition_w.txt` |
| Table II, row "Reflected path" | `witness_path_genericity/reflected_path.py` | `reflected_path.txt`, `reflected_path.json` (`path.min_gap`, `path.min_alpha`, `path.max_alpha`, `by_s`), where w = 1/alpha |
| Table II, row "Genericity"; numerical checks of Lemmas 22 and 23 and of the gamma(w) = 0 example of Appendix G | `witness_path_genericity/check_genericity.py` | `genericity.txt` (lines `random a, K = ...` and `worst draw`), `genericity.json` |
| Table II, row "Fig. 3 path" | `reflected_pair_w124/compute_figure_data.py` | `path.json`: the range of `rows[*].w` and the smallest gap between consecutive entries of `rows[*].xi` |
| Table III | `entropy_risk/mse_sim.py` | `mse.json` (`cells[*]`: `H` and `mse`, with `order_K.mse` in the last column); the table gives the errors in units of 10^-3 |

### Numbers quoted in the text

| Number | Script | Output |
| :-- | :-- | :-- |
| the profile w0 of the checks | `moment_preservation/search_pair.py` | `searched_pair.json` (`w0`) |
| gamma(w0) | `witness_path_genericity/verify_condition_w.py` | `condition_w.txt` (`gamma = 0.294418`) |
| the three rational points of the exact Fisher expansion | `exact_expansions/fisher_expansion.py` | defined in the script (`POINTS`) |
| the searched pair theta, theta' | `moment_preservation/search_pair.py`, `moment_preservation/moment_path.py` | `searched_pair.json` (`pair`), `searched_pair_80_digits.txt` (80-digit values) |
| errors, ratios and fractions of large errors quoted with Fig. 5 and Table III | `entropy_risk/mse_sim.py` | `mse.json` (`cells[*]`: `mse`, `mse_se`, `frac_abs_gt_half`, `order_K`), `mse.txt` |
| tree moments and entropies quoted with Fig. 5 and Table III | `entropy_risk/moment_comparisons.py` | `moment_comparisons.json` |

Closed-form values in the main text and in Appendices B to G, such as gamma(w) = 1/4 and g(w) = 15 log^3 2
for w = (1, 2, 4), are derived in the paper and need no code.

## Environment

- Python 3.12.7
- NumPy 1.26.4
- SciPy 1.13.1
- mpmath 1.3.0
- SymPy 1.13.1
- NetworkX 3.3

The scripts import nothing else beyond the Python standard library. `requirements.txt` pins these
versions:

    pip install -r requirements.txt

No GPU is used. The stored outputs were produced with these versions on an x86-64 Linux CPU, with NumPy
linked against Intel MKL 2023.1 and mpmath on its pure-Python backend.

## Running

Run every command from the repository root. Steps 1 to 3 do not depend on the outputs of any other step.
Steps 4 to 13 run in the order below, because later steps read the outputs of earlier ones. Runtimes are
wall-clock times on a 24-core AMD Ryzen Threadripper PRO 5965WX shared with other jobs; the parallel steps
start the stated number of processes. The whole sequence takes about 40 minutes, half of it in step 1, which
needs 2.5 to 3 CPU hours in total.

| Step | Command | Paper items | Runtime |
| :-- | :-- | :-- | :-- |
| 1 | `python entropy_risk/mse_sim.py` | Fig. 5, Table III | 20 min (8 processes) |
| 2 | `python entropy_risk/moment_comparisons.py` | tree moments and entropies quoted with Fig. 5 | under 1 s |
| 3 | `python reflected_pair_w124/compute_figure_data.py` | Figs. 2 to 4, Table II | 10 s |
| 4 | `python moment_preservation/search_pair.py` | w0, searched pair | 10 s |
| 5 | `python moment_preservation/moment_path.py` | searched pair to 80 digits, Table II | 1.5 min |
| 6 | `python exact_expansions/graph_classes.py` | input of steps 7 and 8 | 10 s |
| 7 | `python exact_expansions/fisher_expansion.py` | Table II | 1.5 min (8 processes) |
| 8 | `python exact_expansions/endpoint_chi2.py` | Table II | 1 min (4 processes) |
| 9 | `python modulus_and_variance/check_modulus.py` | Table II | 8 min |
| 10 | `python modulus_and_variance/check_variance.py` | Table II | 1 min |
| 11 | `python witness_path_genericity/verify_condition_w.py` | gamma(w0), Table II | under 1 s |
| 12 | `python witness_path_genericity/reflected_path.py` | Table II | 30 s |
| 13 | `python witness_path_genericity/check_genericity.py` | Table II | 8.5 min |

Step 2 imports constants from the script of step 1. Step 5 reads step 4; steps 8 to 12 read the searched pair
from step 5; steps 7 and 8 read the graph classes of step 6; step 12 also reads step 9. Step 6 writes
`exact_expansions/output/graph_classes.pkl` (5.5 MB), which is not stored in the repository. `mse_sim.py`
accepts `--workers` to change the number of processes, which does not change the results, and `--tag` to
write `<tag>.json` and `<tag>.txt` instead of the stored `mse.json` and `mse.txt`.

## Randomness

Every random number comes from a NumPy `default_rng` generator with a fixed seed:

| Script | Seeds |
| :-- | :-- |
| `mse_sim.py` | one generator per judge, distribution and n, seeded 20261014 to 20261125 in the order judge, distribution, n; the self-test uses seed 1 |
| `compute_figure_data.py` | `default_rng(20261013)` for the random pairs of Fig. 4 |
| `search_pair.py` | the 200 candidate profiles are `default_rng(20261015).uniform(0.3, 1.7, size=(200, 3))` and w0 is the one with index 14; the targets and starts at w0 use `default_rng(3000023)` |
| `check_modulus.py` | `default_rng(20261013)`, used in turn by the 200,000 random pairs, the boundary families and the 400 starts per ratio |
| `check_variance.py` | `default_rng(20261014)` |
| `check_genericity.py` | `default_rng(20261013)` |

At K = 3, the same search, started from the first five pairs sharing S_2 and S_3 at each of the first 20
profiles (indices 0 to 19), finds 43 exact pairs sharing S_2, S_3 and S_4. The pair with the largest entropy
gap among them, 0.3255, is the searched pair, and its profile is w0. The next largest gap is 0.3234, at the
profile with index 2. `search_pair.py` reruns the search at w0 only.

`moment_comparisons.py`, `moment_path.py`, `graph_classes.py`, `fisher_expansion.py`, `endpoint_chi2.py`,
`verify_condition_w.py` and `reflected_path.py` use no randomness. Results do not depend on the number of
processes, because each parallel task carries its own seed or none.

## Reproducibility

With the versions above, rerunning the scripts reproduces the stored outputs exactly, apart from the timing
fields of `mse.json` (`seconds`) and `mse.txt` (`t=...` and `total seconds`). Other versions or platforms can
change the last digits of double-precision results and the paths of the multistart and Nelder-Mead searches.
The 80-digit values of the searched pair depend on mpmath only: from the stored `searched_pair.json`, step 5
reproduces them on any platform.

## License

Released under the MIT License; see `LICENSE`.
