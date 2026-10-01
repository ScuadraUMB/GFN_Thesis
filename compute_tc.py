"""
compute_tc.py
==============
Aggregates the per-seed Tc_{method}_chi_peak and Tcross_{method}_binder
values already present in {tag}_chi_sweep.json files into a mean +/- SEM
across seeds, for each method (TB / KC / MCMC).

Reuses load_seed_jsons() from make_plots.py, so it expects to sit in the
same directory as that script (or adjust the import path below).

Usage
-----
  python compute_tc.py --base_dir ./run_logz_fix/outputs/sk_d100 --tag sk_D100
  python compute_tc.py --base_dir ./run_logz_fix/outputs/ising_n10 --tag ising

Output: a small table per estimator (chi_peak, binder_cross), plus
optional --latex to also print a ready-to-paste table row.
"""
import argparse
import numpy as np

from make_plots import load_seed_jsons  # same directory as make_plots.py


def mean_sem(values):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = arr.mean()
    sem = arr.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return mean, sem, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_dir", required=True, help="dir containing seed_*/ subdirs")
    ap.add_argument("--tag", required=True, help="file prefix, e.g. 'sk_D100' or 'ising'")
    ap.add_argument("--latex", action="store_true", help="also print a LaTeX table row per method")
    args = ap.parse_args()

    payloads, paths = load_seed_jsons(args.base_dir, args.tag)
    seed_labels = [p.split("/")[-2] for p in paths]
    print(f"Loaded {len(payloads)} seed(s): {seed_labels}\n")

    methods = [m.lower() for m in payloads[0]["methods"]]

    for estimator_key, estimator_label in [
        ("chi_peak", "T_c via chi peak"),
        ("binder", "T_c via Binder cumulant crossing"),
    ]:
        print(f"--- {estimator_label} ---")
        for m in methods:
            field = f"Tc_{m}_{estimator_key}" if estimator_key == "chi_peak" else f"Tcross_{m}_{estimator_key}"
            if field not in payloads[0]:
                print(f"  {m.upper():>6}:  [no '{field}' key in this JSON -- skipping]")
                continue
            values = [p[field] for p in payloads]
            mean, sem, n = mean_sem(values)
            spread = "  ".join(f"{v:.4f}" for v in values)
            print(f"  {m.upper():>6}:  T = {mean:.4f} +/- {sem:.4f}   (n={n} seeds; per-seed: {spread})")
            if args.latex:
                print(f"    LaTeX:  {m.upper()} & ${mean:.3f} \\pm {sem:.3f}$ \\\\")
        print()


if __name__ == "__main__":
    main()
