# Meaning-Dependent Judge Sensitivity Changes the Minimax Rate of Semantic Entropy

Xiaoli Yu and Jiamiao Liu

This repository contains the code of every numerical check of the paper, the outputs behind the numbers in
its text, figures and tables, and the instructions that reproduce them. The checks are synthetic: no script
reads external data, and every file a script reads is written by an earlier step. Everything runs on a CPU.
The high-precision computations use mpmath, the exact ones use Python fractions and SymPy, and the rest uses
NumPy and SciPy.

## Layout

| Folder | Paper | Contents |
| :-- | :-- | :-- |
| `H-B_moment_preservation/` | Appendices H-A and H-B, Table II | search for the pair at w0, its 80-digit values, the moment-preserving path |
| `H-C_exact_expansions/` | Appendix H-C, Table II | graph classes on at most 7 vertices, exact Fisher expansion, endpoint chi-square expansion |
| `H-D_modulus_and_variance/` | Appendix H-D, Table II | checks of the entropy modulus and of the variance of the star statistics |
| `H-E_witness_path_genericity/` | Appendix H-E, Table II | Condition W at w0, reflected path at w0, genericity |
| `H-F_entropy_risk/` | Appendix H-F, Fig. 5, Table III | matched-budget simulation of the estimator, moments quoted in the text |
| `reflected_pair_w124/` | Figs. 2 to 4, Table II | witness, reflected pairs and their moment-preserving path at w = (1, 2, 4), random pairs of Fig. 4 |

Every folder keeps its scripts at the top and its outputs in `output/`. The files in the `output/` folders
are the outputs used in the paper, and running a script rewrites its own outputs in place.

## Paper items, scripts and outputs

Paths are relative to the repository root. A file name without a folder lives in the `output/` folder of the
script in the same row.

| Paper item | Script | Output (where the paper's numbers are) |
| :-- | :-- | :-- |
| Fig. 1 | none | the exponents are closed-form expressions; it involves no computation |
| Fig. 2 | `reflected_pair_w124/compute_figure_data.py` | `reflected.json` (`w`, witness `x`); the entropies 0.838 and 0.896 of the pair in (b) are `H` of the cells `reflected +` and `reflected -` in `H-F_entropy_risk/output/mse.json` |
| Fig. 3 | `reflected_pair_w124/compute_figure_data.py` | `path.json` |
| Fig. 4 | `reflected_pair_w124/compute_figure_data.py` | `reflected.json` (`ell`, `limit_ratio` = c), `random_pairs.json` (`n` = 19,950) |
| Fig. 5 | `H-F_entropy_risk/mse_sim.py` | `mse.json` (`cells[*]`: `mse` and `mse_se`, with `order_K` for the projection on S_2, S_3 only) |
| Table I | none | summary of the theorems and of the cited homogeneous results; it involves no computation |
| Table II | the scripts in the rows below | the outputs in the rows below; a bound over sampled points or runs is their maximum or minimum, rounded outward as the table note states |
| Table III | `H-F_entropy_risk/mse_sim.py` | `mse.json` (`cells[*]`: `H` and `mse`, with `order_K.mse` in the last column); the table gives the errors in units of 10^-3 |
| Appendix H-A: w0 | `H-B_moment_preservation/search_pair.py` | `searched_pair.json` (`w0`) |
| Appendix H-A: gamma(w0) | `H-E_witness_path_genericity/verify_condition_w.py` | `condition_w.txt` (`gamma = 0.294418`) |
| Appendix H-A: three rational points | `H-C_exact_expansions/fisher_expansion.py` | defined in the script (`POINTS`) |
| Appendix H-B: searched pair theta, theta' | `H-B_moment_preservation/search_pair.py`, `H-B_moment_preservation/moment_path.py` | `searched_pair.json` (`pair`), `searched_pair_80_digits.txt` (80-digit values) |
| Appendix H-B, Table II row "Moment preservation and fifth-order scaling" | `H-B_moment_preservation/moment_path.py` | `moment_path.txt` (lines `max rel drift S_1..S_4` and `(v,e)=(5,4) tree ... order`), `moment_path.json` |
| Appendix H-C, Table II row "Exact Fisher information" | `H-C_exact_expansions/graph_classes.py`, `H-C_exact_expansions/fisher_expansion.py` | `fisher_expansion.txt` (checks (a) to (d) per direction), `fisher_expansion.json` (`c_float`: coefficients c_{k,v}) |
| Appendix H-C, Table II row "Endpoint chi^2" | `H-C_exact_expansions/endpoint_chi2.py` | `endpoint_chi2.txt`, `endpoint_chi2.json` (`low_rel`, `blowup_rel`, `c85_rel_err`, `c_float`) |
| Appendix H-D, Table II row "Modulus" | `H-D_modulus_and_variance/check_modulus.py` | `modulus.txt`, `modulus.json` (`random`, `adversarial.T.value`, `constants.C`) |
| Appendix H-D, Table II row "Variance" | `H-D_modulus_and_variance/check_variance.py` | `variance.txt`, `variance.json` (`var_over_bound`, `var_over_lead`) |
| Appendix H-E, Table II row "Condition W at w0" | `H-E_witness_path_genericity/verify_condition_w.py` | `condition_w.txt` |
| Appendix H-E, Table II row "Reflected path" | `H-E_witness_path_genericity/reflected_path.py` | `reflected_path.txt`, `reflected_path.json` (`path.min_gap`, `path.min_alpha`, `path.max_alpha`, `by_s`), where w = 1/alpha |
| Appendix H-E, Table II row "Genericity"; numerical checks of Lemmas 22 and 23 and of the gamma(w) = 0 example of Appendix G | `H-E_witness_path_genericity/check_genericity.py` | `genericity.txt` (lines `random a, K = ...` and `worst draw`), `genericity.json` |
| Table II row "Fig. 3 path" | `reflected_pair_w124/compute_figure_data.py` | `path.json`: the range of `rows[*].w` and the smallest gap between consecutive entries of `rows[*].xi` |
| Appendix H-F: errors, ratios and fractions of large errors quoted in the text | `H-F_entropy_risk/mse_sim.py` | `mse.json` (`cells[*]`: `mse`, `mse_se`, `frac_abs_gt_half`, `order_K`), `mse.txt` |
| Appendix H-F: moments and entropies quoted in the text | `H-F_entropy_risk/moment_comparisons.py` | `moment_comparisons.json` |

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

Run every command from the repository root, in the order below; later steps read the outputs of earlier
ones. Runtimes are wall-clock times on a 24-core AMD Ryzen Threadripper PRO 5965WX shared with other jobs;
the parallel steps start the stated number of processes. The whole sequence takes about 40 minutes, half of
it in step 11, which needs 2.5 to 3 CPU hours in total.

| Step | Command | Paper items | Runtime |
| :-- | :-- | :-- | :-- |
| 1 | `python H-B_moment_preservation/search_pair.py` | Appendices H-A, H-B | 10 s |
| 2 | `python H-B_moment_preservation/moment_path.py` | Appendix H-B, Table II | 1.5 min |
| 3 | `python H-C_exact_expansions/graph_classes.py` | input of steps 4 and 5 | 10 s |
| 4 | `python H-C_exact_expansions/fisher_expansion.py` | Appendix H-C, Table II | 1.5 min (8 processes) |
| 5 | `python H-C_exact_expansions/endpoint_chi2.py` | Appendix H-C, Table II | 1 min (4 processes) |
| 6 | `python H-D_modulus_and_variance/check_modulus.py` | Appendix H-D, Table II | 8 min |
| 7 | `python H-D_modulus_and_variance/check_variance.py` | Appendix H-D, Table II | 1 min |
| 8 | `python H-E_witness_path_genericity/verify_condition_w.py` | Appendices H-A, H-E, Table II | under 1 s |
| 9 | `python H-E_witness_path_genericity/reflected_path.py` | Appendix H-E, Table II | 30 s |
| 10 | `python H-E_witness_path_genericity/check_genericity.py` | Appendix H-E, Table II | 8.5 min |
| 11 | `python H-F_entropy_risk/mse_sim.py` | Appendix H-F, Fig. 5, Table III | 20 min (8 processes) |
| 12 | `python H-F_entropy_risk/moment_comparisons.py` | Appendix H-F | under 1 s |
| 13 | `python reflected_pair_w124/compute_figure_data.py` | Figs. 2 to 4, Table II | 10 s |

Dependencies between the steps: step 2 reads step 1; steps 5 to 9 read the searched pair from step 2;
steps 4 and 5 read the graph classes of step 3; step 9 also reads step 6; step 12 imports constants from the
script of step 11. Step 3 writes `H-C_exact_expansions/output/graph_classes.pkl` (5.5 MB), which is not
stored in the repository. `mse_sim.py` accepts `--workers` to change the number of processes, which does not
change the results, and `--tag` to write `<tag>.json` and `<tag>.txt` instead of the stored `mse.json` and
`mse.txt`.

## Randomness

Every random number comes from a NumPy `default_rng` generator with a fixed seed:

| Script | Seeds |
| :-- | :-- |
| `search_pair.py` | the 200 candidate profiles are `default_rng(20261015).uniform(0.3, 1.7, size=(200, 3))` and w0 is the one with index 14; the targets and starts at w0 use `default_rng(3000023)` |
| `check_modulus.py` | `default_rng(20261013)`, used in turn by the 200,000 random pairs, the boundary families and the 400 starts per ratio |
| `check_variance.py` | `default_rng(20261014)` |
| `check_genericity.py` | `default_rng(20261013)` |
| `mse_sim.py` | one generator per judge, distribution and n, seeded 20261014 to 20261125 in the order judge, distribution, n; the self-test uses seed 1 |
| `compute_figure_data.py` | `default_rng(20261013)` for the random pairs of Fig. 4 |

At K = 3, the same search, started from the first five pairs sharing S_2 and S_3 at each of the first 20
profiles (indices 0 to 19), finds 43 exact pairs sharing S_2, S_3 and S_4. The pair with the largest entropy
gap among them, 0.3255, is the searched pair of Appendix H-B, and its profile is w0. The next largest gap is
0.3234, at the profile with index 2. `search_pair.py` reruns the search at w0 only.

`moment_path.py`, `graph_classes.py`, `fisher_expansion.py`, `endpoint_chi2.py`, `verify_condition_w.py`,
`reflected_path.py` and `moment_comparisons.py` use no randomness. Results do not depend on the number of
processes, because each parallel task carries its own seed or none.

## Reproducibility

With the versions above, rerunning the scripts reproduces the stored outputs exactly, apart from the timing
fields of `mse.json` (`seconds`) and `mse.txt` (`t=...` and `total seconds`). Other versions or platforms can
change the last digits of double-precision results and the paths of the multistart and Nelder-Mead searches.
The 80-digit values of the searched pair depend on mpmath only: from the stored `searched_pair.json`, step 2
reproduces them on any platform.

## License

Released under the MIT License; see `LICENSE`.
