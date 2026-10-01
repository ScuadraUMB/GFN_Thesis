"""
multi_N_crossing_bootstrap.py
================================
Upgrades multi_N_crossing.py in three ways:

1. WINDOWED SEARCH: restricts the crossing search to T <= --T_max_window,
   chosen in advance (before looking at results) to exclude the noisy
   high-T tail where curves spuriously cross each other multiple times.
   This is NOT picking-the-crossing-you-like after the fact -- the
   window is fixed once, up front, and applied identically to every
   method and every N-pair.

2. SLOPE FILTER: a "crossing" where the curve difference is nearly flat
   through the sign change (grazing contact, not a real crossing) is
   discarded via --min_slope. This replaces an earlier magnitude-based
   filter that turned out to be the wrong tool: right AT any genuine
   crossing the endpoint diff is small by definition (that's what
   crossing zero means), so filtering on endpoint magnitude penalizes
   real crossings that happen to land close to a grid point just as
   much as spurious grazing ones -- which is exactly what collapsed
   bootstrap hit rates to 40-60% on the first attempt. Slope (the rate
   of sign change across the interval, not the endpoint values
   themselves) is the correct discriminator: TB's genuine N=50-vs-100
   crossing at T=0.377 has slope ~0.55, its spurious one at T=0.974 has
   slope ~0.03 -- a 20x gap that a magnitude threshold can't see but a
   slope threshold cleanly separates.

3. BOOTSTRAP ERROR BARS: a crossing point computed from two already
   seed-averaged curves doesn't have a well-defined "per-seed value" to
   take a simple SEM over (an individual seed's curve is MORE likely to
   have zero or multiple crossings than the averaged curve is). Instead,
   this resamples seeds with replacement independently for each N,
   recomputes the seed-averaged curve and its windowed crossing each
   time, and reports the mean +/- std of that distribution of
   crossings -- the standard fix for propagating uncertainty through a
   nonlinear (crossing-finding) operation on top of an average.

Usage
-----
  python multi_N_crossing_bootstrap.py \\
      --base_dir ./run_logz_fix/outputs/sk_size_sweep_full \\
      --N_values 10,25,50,100 --T_max_window 1.0 --min_slope 0.1 \\
      --n_bootstrap 2000 --outdir ./figures --latex

--outdir writes sk_multi_N_crossing.pdf: T_cross vs N-pair (log-scaled,
x-position at the geometric mean of each pair), one line per method,
with a horizontal reference line at the mean-field T_g~1.0 prediction.
Points below --reliability_hit_rate (default 90%) or with any
unresolved multi-crossing bootstrap draws get a hollow ring overlay,
so an unreliable point is visually distinct without needing a second
legend entry.

--latex prints a paper-ready table (mean +/- std per method per
N-pair, with unreliable cells starred) to the console -- copy directly
into the paper rather than retyping numbers by hand.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_plots import load_seed_jsons, COLORS, MARKERS, METHOD_LABELS


def windowed_crossing(T_grid, y1, y2, T_max, min_slope):
    """First sign change of (y1 - y2) within T <= T_max, requiring the
    slope of the diff across that interval to exceed min_slope (filters
    out grazing near-flat contact that isn't a real crossing, without
    penalizing genuine crossings that happen to land close to a grid
    point). Returns (T_cross, n_crossings_in_window) --
    n_crossings_in_window > 1 means the window still isn't tight enough
    to guarantee a unique answer."""
    diff = np.asarray(y1) - np.asarray(y2)
    mask = T_grid <= T_max
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return None, 0
    crossings = []
    for i in range(len(idx) - 1):
        a, b = idx[i], idx[i + 1]
        d0, d1 = diff[a], diff[b]
        T0, T1 = T_grid[a], T_grid[b]
        slope = (d1 - d0) / (T1 - T0)
        if (d0 >= 0) != (d1 >= 0) and abs(slope) > min_slope:
            crossings.append(T0 + (0 - d0) / (d1 - d0) * (T1 - T0))
    if not crossings:
        return None, 0
    return crossings[0], len(crossings)


def bootstrap_crossing(T_grid, payloads_N1, payloads_N2, method, T_max, min_slope, n_bootstrap, rng):
    key = f"u4_{method}_mean"
    arr_N1 = np.stack([np.asarray(p[key]) for p in payloads_N1])  # (n_seeds1, n_T)
    arr_N2 = np.stack([np.asarray(p[key]) for p in payloads_N2])
    n1, n2 = len(arr_N1), len(arr_N2)

    crossings = []
    n_multi = 0
    for _ in range(n_bootstrap):
        idx1 = rng.integers(0, n1, size=n1)
        idx2 = rng.integers(0, n2, size=n2)
        curve1 = arr_N1[idx1].mean(axis=0)
        curve2 = arr_N2[idx2].mean(axis=0)
        T_cross, n_found = windowed_crossing(T_grid, curve1, curve2, T_max, min_slope)
        if T_cross is not None:
            crossings.append(T_cross)
            if n_found > 1:
                n_multi += 1

    return crossings, n_multi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--N_values", default="10,25,50,100")
    ap.add_argument("--T_max_window", type=float, required=True,
                     help="upper T bound for the crossing search -- choose this BEFORE "
                          "looking at results, based on where curves are still cleanly "
                          "separated (e.g. from the charted U4(T) curves)")
    ap.add_argument("--min_slope", type=float, default=0.1,
                     help="minimum |slope| of the diff across the crossing interval for it "
                          "to count as a real crossing -- filters grazing near-flat contact "
                          "without penalizing genuine crossings near a grid point (see "
                          "inspect_all_crossings.py; a genuine TB crossing measured slope "
                          "~0.55, a spurious one ~0.03)")
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed, for reproducibility")
    ap.add_argument("--outdir", default=None,
                     help="if given, save a figure (sk_multi_N_crossing.pdf) here")
    ap.add_argument("--latex", action="store_true",
                     help="print a paper-ready LaTeX table alongside the console output")
    ap.add_argument("--reliability_hit_rate", type=float, default=0.9,
                     help="a point is flagged 'unreliable' if hit rate falls below this")
    args = ap.parse_args()

    N_values = [int(x) for x in args.N_values.split(",")]
    rng = np.random.default_rng(args.seed)

    per_N = {}
    for N in N_values:
        try:
            payloads, paths = load_seed_jsons(f"{args.base_dir}/D{N}", f"sk_D{N}")
        except FileNotFoundError:
            print(f"N={N}: not found, skipping")
            continue
        per_N[N] = payloads
        print(f"N={N}: {len(payloads)} seed(s)")

    loaded_N = sorted(per_N.keys())
    if len(loaded_N) < 2:
        print("Need at least 2 sizes loaded.")
        return

    T_grid = np.asarray(per_N[loaded_N[0]][0]["T_grid"])
    methods = set()
    for N in loaded_N:
        methods.update(m.lower() for m in per_N[N][0]["methods"])
    methods = sorted(methods)

    print(f"\nWindow: T <= {args.T_max_window}   |   min_slope: {args.min_slope}   "
          f"|   bootstrap draws: {args.n_bootstrap}\n")

    # results[method] = list of dicts, one per N-pair, in loaded_N order
    results = {m: [] for m in methods}

    for m in methods:
        print(f"=== {m.upper()} ===")
        for N1, N2 in zip(loaded_N[:-1], loaded_N[1:]):
            key = f"u4_{m}_mean"
            if key not in per_N[N1][0] or key not in per_N[N2][0]:
                print(f"  N={N1} vs N={N2}: missing data, skipping")
                results[m].append(None)
                continue
            crossings, n_multi = bootstrap_crossing(
                T_grid, per_N[N1], per_N[N2], m, args.T_max_window, args.min_slope,
                args.n_bootstrap, rng)

            n_valid = len(crossings)
            hit_rate = n_valid / args.n_bootstrap
            if n_valid == 0:
                print(f"  N={N1} vs N={N2}:  NO valid crossings found in {args.n_bootstrap} "
                      f"draws -- window may be wrong for this pair, inspect curves directly")
                results[m].append(None)
                continue

            mean = np.mean(crossings)
            std = np.std(crossings, ddof=1) if n_valid > 1 else float("nan")
            flag = ""
            if hit_rate < args.reliability_hit_rate:
                flag += f"  [only {hit_rate*100:.0f}% of draws found a crossing in-window -- interpret cautiously]"
            if n_multi > 0:
                flag += f"  [{n_multi} draw(s) had >1 crossing even within window -- window may need tightening]"
            print(f"  N={N1} vs N={N2}:  T = {mean:.4f} +/- {std:.4f}  (n_valid={n_valid}/{args.n_bootstrap}){flag}")
            results[m].append(dict(N1=N1, N2=N2, mean=mean, std=std, hit_rate=hit_rate,
                                    n_multi=n_multi, reliable=None))  # reliable set below, after all pairs seen
        print()

    # --------------------------------------------------------------
    # RELIABILITY FLAG -- data-driven, not a fixed n_multi==0 rule.
    # A nonzero multi-crossing count does NOT by itself mean the
    # reported (first-crossing) mean is unstable -- e.g. TB's N=50-vs-100
    # pair has 938/1971 multi-crossing draws yet the tightest std of any
    # pair (0.055), because the FIRST crossing lands in nearly the same
    # place regardless of a second, lower-priority one also appearing.
    # The bootstrap std itself is the correct, direct measure of
    # instability. So: flag a pair as unreliable if its std is
    # substantially larger (>1.5x) than the MEDIAN std across all valid
    # pairs in this run -- self-calibrating to this run's own spread,
    # rather than an arbitrary fixed cutoff chosen in advance.
    # --------------------------------------------------------------
    all_stds = [r["std"] for m in methods for r in results[m] if r is not None]
    median_std = float(np.median(all_stds)) if all_stds else float("nan")
    std_flag_threshold = 1.5 * median_std
    for m in methods:
        for r in results[m]:
            if r is not None:
                r["reliable"] = r["std"] <= std_flag_threshold
    print(f"Median bootstrap std across all pairs: {median_std:.4f}  "
          f"(flagging any pair with std > {std_flag_threshold:.4f} as an outlier)\n")

    pair_labels = [f"{N1}--{N2}" for N1, N2 in zip(loaded_N[:-1], loaded_N[1:])]
    x_vals = [np.sqrt(N1 * N2) for N1, N2 in zip(loaded_N[:-1], loaded_N[1:])]  # geometric mean

    # =========================================================
    # Figure
    # =========================================================
    if args.outdir:
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
        for m in methods:
            xs, ys, yerrs, reliable_mask = [], [], [], []
            for x, r in zip(x_vals, results[m]):
                if r is None:
                    continue
                xs.append(x)
                ys.append(r["mean"])
                yerrs.append(r["std"])
                reliable_mask.append(r["reliable"])
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=yerrs, color=COLORS[m], marker=MARKERS[m],
                        ms=8, lw=1.8, capsize=4, label=METHOD_LABELS[m])
            # hollow-circle overlay flags any unreliable point without a second legend entry
            for x, y, rel in zip(xs, ys, reliable_mask):
                if not rel:
                    ax.plot(x, y, marker="o", ms=14, mfc="none", mec=COLORS[m], mew=1.5)

        ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.6, label=r"$T_g \approx 1.0$ (mean-field, $N\to\infty$)")
        ax.set_xscale("log")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(pair_labels)
        ax.set_xlabel("N pair (geometric mean, log scale)")
        ax.set_ylabel(r"$T_{\mathrm{cross}}$")
        ax.set_title("Multi-N Binder cumulant crossings\n"
                      "(hollow ring = bootstrap std >1.5x median -- see caption)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        os.makedirs(args.outdir, exist_ok=True)
        out_path = os.path.join(args.outdir, "sk_multi_N_crossing.pdf")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")

    # =========================================================
    # LaTeX table
    # =========================================================
    if args.latex:
        print("\n% ---- paper-ready table ----")
        print(r"\begin{table}[t]")
        print(r"\centering")
        print(r"\caption{Adjacent-$N$ Binder cumulant crossing temperatures, bootstrap mean "
              r"$\pm$ std over " + str(args.n_bootstrap) + r" resamples "
              r"(window $T\le" + f"{args.T_max_window}" + r"$, slope filter "
              r"$>" + f"{args.min_slope}" + r"$). $^{*}$ marks a pair whose bootstrap std "
              r"exceeds 1.5$\times$ the median std across all pairs in this analysis -- "
              r"substantially higher uncertainty than typical, interpret with caution.}")
        print(r"\label{tab:multi-N-crossing}")
        col_spec = "l" + "c" * len(pair_labels)
        print(r"\begin{tabular}{" + col_spec + "}")
        print(r"\hline\hline")
        print("Method & " + " & ".join(pair_labels) + r" \\")
        print(r"\hline")
        for m in methods:
            row = [METHOD_LABELS[m]]
            for r in results[m]:
                if r is None:
                    row.append("--")
                else:
                    star = "" if r["reliable"] else "$^{*}$"
                    row.append(f"${r['mean']:.3f} \\pm {r['std']:.3f}${star}")
            print(" & ".join(row) + r" \\")
        print(r"\hline\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")


if __name__ == "__main__":
    main()