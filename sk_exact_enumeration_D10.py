"""
sk_exact_enumeration_D10.py
=============================
Brute-force exact enumeration of a D=10 SK spin glass (2^10 = 1024
states -- small enough to fully enumerate). Computes the EXACT
susceptibility, Binder cumulant, and overlap distribution from the true
Boltzmann distribution, for direct comparison against KC/TB/MCMC's
SAMPLED estimates of the same quantities at D=10 (from the size sweep).

This is the ground-truth validation point for the size-scaling figure:
at D=10, we don't have to trust any sampler's approximation -- we can
compute the true answer directly and see how close each method's
samples come to it.

Uses the SAME coupling construction as spin_glass_lib.SpinEnv("sk", ...)
and the SAME J_seed convention as run_tc_sweep.py (J_seed = seed*1000 + d
for disorder realization d within outer --seed), so this can be run with
matching (seed, d) pairs to the actual sk_size_sweep/D10 runs for an
apples-to-apples comparison.

Usage:
  python sk_exact_enumeration_D10.py --p_af 0.5 --T_min 0.2 --T_max 0.4 \
      --n_T 2 --seed 0 --n_disorder 3 --outdir ./exact_D10
"""
import argparse
import itertools
import json
import os
import numpy as np


def build_sk_couplings(D, p_af, J_seed):
    """Identical construction to spin_glass_lib.SpinEnv('sk', ...)."""
    rng = np.random.default_rng(J_seed)
    pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]
    signs = np.where(rng.random(len(pairs)) < p_af, -1.0, 1.0)
    vals = signs / np.sqrt(D)
    J = {pair: float(v) for pair, v in zip(pairs, vals)}
    return J, pairs


def enumerate_all_states(D):
    """All 2^D spin configurations as a (2^D, D) array of +-1 values."""
    n_states = 2 ** D
    states = np.zeros((n_states, D), dtype=np.int8)
    for idx, bits in enumerate(itertools.product([0, 1], repeat=D)):
        states[idx] = [1 if b else -1 for b in bits]
    return states


def compute_energies(states, J, pairs):
    """Energy of every enumerated state: E = -sum_{i<j} J_ij * s_i * s_j."""
    E = np.zeros(states.shape[0], dtype=np.float64)
    for (i, j) in pairs:
        E -= J[(i, j)] * states[:, i] * states[:, j]
    return E


def exact_observables(D, p_af, J_seed, T):
    """
    Exact partition function and observables at temperature T:
      Z            = sum_s exp(-E(s)/T)
      p(s)         = exp(-E(s)/T) / Z
      chi_SG       = D * <q^2>  where <q^2> = sum_{s,s'} p(s) p(s') q(s,s')^2
      U4           = 1 - <q^4> / (3 <q^2>^2)
    <q^2> and <q^4> are computed by brute-force double summation over all
    pairs of the 2^D states (2^D x 2^D = ~1.05M pairs at D=10 -- feasible).
    """
    J, pairs = build_sk_couplings(D, p_af, J_seed)
    states = enumerate_all_states(D)
    E = compute_energies(states, J, pairs)

    logw = -E / T
    logw -= logw.max()  # numerical stability
    w = np.exp(logw)
    Z = w.sum()
    p = w / Z  # Boltzmann probability of each state

    # overlap matrix q(s_a, s_b) = (1/D) * s_a . s_b for ALL pairs at once
    # (states @ states.T gives D * q for every pair, in one matrix op)
    q_matrix = (states.astype(np.float64) @ states.astype(np.float64).T) / D

    # <q^2> and <q^4> under the product measure p(s_a) * p(s_b)
    q2_matrix = q_matrix ** 2
    q4_matrix = q_matrix ** 4
    p_outer = np.outer(p, p)

    mean_q2 = float(np.sum(p_outer * q2_matrix))
    mean_q4 = float(np.sum(p_outer * q4_matrix))

    chi_SG = D * mean_q2
    U4 = 1.0 - mean_q4 / (3.0 * mean_q2 ** 2 + 1e-12)

    return {"T": T, "chi_SG": chi_SG, "U4": U4, "Z": float(Z),
            "mean_q2": mean_q2, "mean_q4": mean_q4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=10,
                     help="Must stay small -- 2^D states enumerated exactly. "
                          "D=10 (1024 states) is fast; D=14+ gets slow/memory-heavy "
                          "fast due to the 2^D x 2^D overlap matrix.")
    ap.add_argument("--p_af", type=float, default=0.5)
    ap.add_argument("--T_min", type=float, default=0.2)
    ap.add_argument("--T_max", type=float, default=0.4)
    ap.add_argument("--n_T", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0,
                     help="outer seed, matched to run_tc_sweep.py's --seed")
    ap.add_argument("--n_disorder", type=int, default=3,
                     help="number of J-realizations to average over, matched to "
                          "run_tc_sweep.py's --n_disorder (J_seed = seed*1000 + d)")
    ap.add_argument("--outdir", type=str, default="./exact_D10")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    T_grid = np.linspace(args.T_min, args.T_max, args.n_T)

    all_chi = np.zeros((args.n_disorder, args.n_T))
    all_u4 = np.zeros((args.n_disorder, args.n_T))

    for d in range(args.n_disorder):
        J_seed = args.seed * 1000 + d
        for i, T in enumerate(T_grid):
            result = exact_observables(args.D, args.p_af, J_seed, T)
            all_chi[d, i] = result["chi_SG"]
            all_u4[d, i] = result["U4"]
            print(f"seed={args.seed} d={d} (J_seed={J_seed})  T={T:.4f}  "
                  f"chi_SG_exact={result['chi_SG']:.5f}  U4_exact={result['U4']:.5f}")

    chi_mean = all_chi.mean(axis=0)
    chi_sem = (all_chi.std(axis=0, ddof=1) / np.sqrt(args.n_disorder)
               if args.n_disorder > 1 else np.zeros_like(chi_mean))
    u4_mean = all_u4.mean(axis=0)
    u4_sem = (all_u4.std(axis=0, ddof=1) / np.sqrt(args.n_disorder)
              if args.n_disorder > 1 else np.zeros_like(u4_mean))

    payload = {
        "model": "sk_exact", "D": args.D, "p_af": args.p_af,
        "T_grid": T_grid.tolist(), "seed": args.seed, "n_disorder": args.n_disorder,
        "chi_exact_mean": chi_mean.tolist(), "chi_exact_sem": chi_sem.tolist(),
        "u4_exact_mean": u4_mean.tolist(), "u4_exact_sem": u4_sem.tolist(),
    }
    out_path = os.path.join(args.outdir, f"sk_D{args.D}_exact_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
