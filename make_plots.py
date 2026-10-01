"""
make_plots.py
==============
Generates every plot from data already saved by run_tc_sweep.py /
sk_exact_enumeration_D10.py. Never reruns a sweep -- only reads
{tag}_chi_sweep.json and {tag}_raw_data.npz files.

# ------------------------------------------------------------------
# CHANGE LOG (this revision):
#   - All plots now save as .pdf instead of .png (vector output,
#     sharper in a twocolumn print layout; dpi bumped 160 -> 300,
#     which still matters for any rasterized sub-elements like
#     marker glyphs or hatching even inside a vector file).
#   - "D" (system size) renamed to "N" throughout CLI args, variable
#     names, axis labels, titles, and docstrings, to match the
#     paper's notation.
#   - TWO THINGS THAT NEED YOUR ATTENTION, marked "ACTION NEEDED"
#     below -- I didn't guess on these since they depend on how your
#     data is actually laid out on disk:
#       1. size_sweep now globs for directories named N{value}
#          (e.g. N10/, N25/) instead of D{value}. If your existing
#          sk_size_sweep data (already run, per your notes) is still
#          sitting in D10/, D25/, ... directories, you'll need to
#          either rename those directories or tell me and I'll add
#          a --dir_prefix override instead.
#       2. paf_sweep and logz_ablation used to hardcode the tag
#          "sk_D100" for loading their (fixed-N=100) sweep data.
#          That's now a required --tag argument instead of a guess,
#          since I don't know whether your already-generated paf/
#          logz sweep data was tagged "sk_D100" or something else.
#          Pass whatever tag your run_tc_sweep.py actually used.
# ------------------------------------------------------------------

Subcommands
-----------
  chi              chi_SG (or U4) vs T, averaged across seeds, one line
                    per method, error bars from seed-to-seed spread.
                    Use for the Ising baseline and SK N=100 showcase.

  mcmc_diag         MCMC acceptance_rate and tau_int vs T, averaged
                    across seeds (from the same chi_sweep.json files).

  distributions     Per-seed order-parameter histogram comparison
                    (low T and high T), one method per color, from a
                    single seed's raw_data.npz.

  loss              TB vs KC loss-per-iteration curves, one subplot per
                    logged T-index, from a single seed's raw_data.npz
                    (only seeds run with --log_loss have this data).

  size_sweep        chi_SG (or U4) vs N, one line per method, optional
                    exact-enumeration ground truth overlay at N=10.

  paf_sweep         chi_SG (or U4) vs p_af, one line per method.

  logz_ablation     TB chi_SG and U4 vs log_Z init value, one line per
                    init value, shared/log-scaled axes.

Examples
--------
  python make_plots.py chi --base_dir ./run_logz_fix/outputs/ising_n10 \\
      --tag ising --model_kind ising --outdir ./figures

  # --tag here matches whatever tag your sweep data was actually saved
  # under -- "sk_D100" for existing/older runs, "sk_N100" if you've
  # since renamed your data-generation pipeline. Either works, it's
  # just a filename prefix.
  python make_plots.py chi --base_dir ./run_logz_fix/outputs/sk_d100 \\
      --tag sk_D100 --model_kind sk --outdir ./figures

  python make_plots.py distributions \\
      --seed_dir ./run_logz_fix/outputs/sk_d100/seed_0 --tag sk_D100 \\
      --outdir ./figures

  python make_plots.py loss \\
      --seed_dir ./run_logz_fix/outputs/sk_d100/seed_0 --tag sk_D100 \\
      --outdir ./figures

  # ACTION NEEDED (see note above): assumes N10/, N25/, N50/, N100/
  # subdirectories under --base_dir. Rename on disk if needed.
  python make_plots.py size_sweep --base_dir ./run_logz_fix/outputs/sk_size_sweep \\
      --N_values 10,25,50,100 --T_index 0 --exact_json ./exact_D10/sk_D10_exact_seed0.json \\
      --outdir ./figures

  # ACTION NEEDED (see note above): --tag is now required, pass
  # whatever tag your paf-sweep data was actually saved under.
  python make_plots.py paf_sweep --base_dir ./run_logz_fix/outputs/sk_paf_sweep \\
      --paf_values 0.0,0.1,0.2,0.3,0.4,0.5 --T_index 0 --tag sk_D100 \\
      --outdir ./figures

  # ACTION NEEDED (see note above): --tag is now required, pass
  # whatever tag your log_Z-ablation data was actually saved under.
  python make_plots.py logz_ablation --base_dir ./run_logz_fix/outputs/tb_logz_ablation \\
      --init_values 0,35,150,300 --tag sk_D100 --outdir ./figures
"""
import argparse
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_LABELS = {"tb": "TB-GFlowNet", "kc": "KC-GFlowNet", "mcmc": "MCMC"}
COLORS = {"tb": "#4A90D9", "kc": "#27AE60", "mcmc": "#E74C3C"}
MARKERS = {"tb": "o", "kc": "^", "mcmc": "s"}


# =====================================================
# Shared loading helpers
# =====================================================
def load_seed_jsons(base_dir, tag, seed_glob="seed_*"):
    pattern = os.path.join(base_dir, seed_glob, f"{tag}_chi_sweep.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No JSON files found matching {pattern}")
    return [json.load(open(p)) for p in paths], paths


def stack_and_average(payloads, key):
    """Stack a per-seed array-valued key across payloads and average
    (each payload's own value is already averaged over its internal
    --n_disorder; this averages across the OUTER --seed array index)."""
    stack = np.stack([np.asarray(p[key]) for p in payloads])
    mean = stack.mean(axis=0)
    n = len(payloads)
    sem = stack.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, sem


def methods_present(payload):
    return [m.lower() for m in payload["methods"]]


# =====================================================
# Subcommand: chi (or U4) vs T, averaged across seeds
# =====================================================
def cmd_chi(args):
    payloads, paths = load_seed_jsons(args.base_dir, args.tag)
    print(f"Loaded {len(payloads)} seed(s): {[os.path.basename(os.path.dirname(p)) for p in paths]}")
    T_grid = np.asarray(payloads[0]["T_grid"])
    methods = methods_present(payloads[0])
    is_spin_glass = args.model_kind != "ising"

    fig, axes = plt.subplots(1, 2 if is_spin_glass else 1,
                              figsize=(12, 5) if is_spin_glass else (7, 5), facecolor="white")
    if not is_spin_glass:
        axes = [axes]

    for m in methods:
        chi_mean, chi_sem = stack_and_average(payloads, f"chi_{m}_mean")
        axes[0].errorbar(T_grid, chi_mean, yerr=chi_sem, color=COLORS[m], marker=MARKERS[m],
                          ms=5, lw=1.8, capsize=3, label=METHOD_LABELS[m])
    axes[0].set_xlabel("Temperature T")
    axes[0].set_ylabel("chi (ferromagnetic)" if not is_spin_glass else "chi_SG (spin glass)")
    axes[0].set_title(f"{args.tag} susceptibility  (avg over {len(payloads)} seeds)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    if is_spin_glass:
        for m in methods:
            u4_mean, u4_sem = stack_and_average(payloads, f"u4_{m}_mean")
            axes[1].errorbar(T_grid, u4_mean, yerr=u4_sem, color=COLORS[m], marker=MARKERS[m],
                              ms=5, lw=1.8, capsize=3, label=METHOD_LABELS[m])
        axes[1].axhline(2/3, color="gray", ls="--", lw=1, alpha=0.5)
        axes[1].set_xlabel("Temperature T")
        axes[1].set_ylabel("Binder cumulant U4")
        axes[1].set_title(f"{args.tag} Binder cumulant  (avg over {len(payloads)} seeds)")
        axes[1].legend()
        axes[1].grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{args.tag}_chi_combined.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# =====================================================
# Subcommand: MCMC diagnostics vs T, averaged across seeds
# =====================================================
def cmd_mcmc_diag(args):
    payloads, paths = load_seed_jsons(args.base_dir, args.tag)
    print(f"Loaded {len(payloads)} seed(s)")
    T_grid = np.asarray(payloads[0]["T_grid"])

    if "mcmc_diag_acceptance_rate_mean" not in payloads[0]:
        raise KeyError("No MCMC diagnostics found in this JSON -- was MCMC run, "
                       "and was this sweep generated by the restructured run_tc_sweep.py?")

    accept_mean, accept_sem = stack_and_average(payloads, "mcmc_diag_acceptance_rate_mean")
    tau_mean, tau_sem = stack_and_average(payloads, "mcmc_diag_tau_int_mean")
    cluster_mean, cluster_sem = stack_and_average(payloads, "mcmc_diag_mean_cluster_fraction_mean")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    has_metropolis = not np.all(np.isnan(accept_mean))
    has_wolff = not np.all(np.isnan(cluster_mean))
    if has_metropolis:
        ax1.errorbar(T_grid, accept_mean, yerr=accept_sem, color="#E74C3C", marker="s",
                     ms=5, lw=1.8, capsize=3, label="Metropolis acceptance rate")
    if has_wolff:
        ax1.errorbar(T_grid, cluster_mean, yerr=cluster_sem, color="#9B59B6", marker="d",
                     ms=5, lw=1.8, capsize=3, label="Wolff mean cluster fraction")
    ax1.set_xlabel("Temperature T")
    ax1.set_ylabel("Rate / fraction")
    ax1.set_title(f"{args.tag} MCMC move statistics  (avg over {len(payloads)} seeds)")
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2.errorbar(T_grid, tau_mean, yerr=tau_sem, color="#E74C3C", marker="s",
                 ms=5, lw=1.8, capsize=3, label="tau_int (energy autocorrelation)")
    ax2.set_yscale("log")
    ax2.set_xlabel("Temperature T")
    ax2.set_ylabel("Integrated autocorrelation time (log scale)")
    ax2.set_title(f"{args.tag} MCMC mixing  (avg over {len(payloads)} seeds)\n"
                  f"large tau_int = poor mixing / possible basin-trapping")
    ax2.legend()
    ax2.grid(alpha=0.25, which="both")

    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{args.tag}_mcmc_diagnostics.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# =====================================================
# Subcommand: per-seed distribution comparison
# =====================================================
def cmd_distributions(args):
    npz_path = os.path.join(args.seed_dir, f"{args.tag}_raw_data.npz")
    data = np.load(npz_path)
    seed_label = os.path.basename(args.seed_dir.rstrip("/"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")
    bins = np.linspace(-1.05, 1.05, 40)
    xlabel = "Magnetization m" if args.model_kind == "ising" else "Pairwise overlap q"

    for ax, label in zip(axes, ["lowT", "highT"]):
        any_data = False
        for m in ["tb", "kc", "mcmc"]:
            key = f"samples_{label}_{m}"
            if key not in data:
                continue
            any_data = True
            samples = data[key]  # (n_eval, N)
            if args.model_kind == "ising":
                order_param = samples.mean(axis=1)  # magnetization
            else:
                # pairwise overlap via disjoint-pair trick, matching spin_glass_lib
                n = len(samples)
                idx = np.random.permutation(n)
                n_pairs = n // 2
                order_param = np.array([
                    float(np.mean(samples[idx[2 * k]] * samples[idx[2 * k + 1]]))
                    for k in range(n_pairs)
                ])
            T_val = float(data[f"samples_{label}_{m}_T"])
            ax.hist(order_param, bins=bins, density=True, alpha=0.5,
                    color=COLORS[m], label=METHOD_LABELS[m])
        if not any_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{label}  T={T_val:.3f}  ({seed_label})")
        ax.legend()
        ax.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{args.tag}_{seed_label}_distributions.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# =====================================================
# Subcommand: TB vs KC loss-per-iteration
# =====================================================
def cmd_loss(args):
    npz_path = os.path.join(args.seed_dir, f"{args.tag}_raw_data.npz")
    data = np.load(npz_path)
    seed_label = os.path.basename(args.seed_dir.rstrip("/"))

    loss_keys = [k for k in data.files if k.startswith("loss_") and not k.endswith("_T")]
    t_indices = sorted(set(int(k.split("Tidx")[1]) for k in loss_keys))
    if not t_indices:
        raise KeyError(f"No loss data found in {npz_path} -- was this sweep run with --log_loss?")

    fig, axes = plt.subplots(1, len(t_indices), figsize=(6 * len(t_indices), 5),
                              facecolor="white", squeeze=False)
    axes = axes[0]

    for ax, t_idx in zip(axes, t_indices):
        for m in ["tb", "kc"]:
            key = f"loss_{m}_Tidx{t_idx}"
            if key not in data:
                continue
            arr = data[key]  # (n_points, 2): [iteration, loss]
            T_val = float(data[f"{key}_T"])
            ax.plot(arr[:, 0], arr[:, 1], color=COLORS[m], marker="o", ms=4, lw=1.8,
                    label=METHOD_LABELS[m])
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"T={T_val:.3f}  ({seed_label})")
        ax.legend()
        ax.grid(alpha=0.25)

    fig.suptitle(f"{args.tag} TB vs KC loss per iteration", fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{args.tag}_{seed_label}_loss.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# =====================================================
# Subcommand: size sweep (chi/U4 vs N)
# =====================================================
def cmd_size_sweep(args):
    N_values = [int(x) for x in args.N_values.split(",")]
    methods = ["tb", "kc", "mcmc"]
    results = {m: {"N": [], "chi": [], "chi_sem": [], "u4": [], "u4_sem": []} for m in methods}

    for N in N_values:
        # CONFIRMED from run_tc_sweep.py's own source: it always tags SK
        # runs as f"{model}_D{D}" (main()'s `tag` line) -- there is no
        # N-based naming anywhere in the actual pipeline. The CLI flag
        # here is still called --N_values (matches the paper's N
        # notation), but it maps to D-named directories/tags on disk,
        # since that's what really gets written. An earlier version of
        # this script guessed N-based naming before we'd read the
        # source -- that guess was wrong and is corrected here.
        payloads, _ = load_seed_jsons(os.path.join(args.base_dir, f"D{N}"), f"sk_D{N}", seed_glob="seed_*")
        for m in methods:
            chi_arr = np.stack([np.asarray(p[f"chi_{m}_mean"])[args.T_index] for p in payloads])
            u4_arr = np.stack([np.asarray(p[f"u4_{m}_mean"])[args.T_index] for p in payloads])
            n = len(payloads)
            results[m]["N"].append(N)
            results[m]["chi"].append(chi_arr.mean())
            results[m]["chi_sem"].append(chi_arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
            results[m]["u4"].append(u4_arr.mean())
            results[m]["u4_sem"].append(u4_arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)

    fig, (ax_chi, ax_u4) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    for m in methods:
        r = results[m]
        ax_chi.errorbar(r["N"], r["chi"], yerr=r["chi_sem"], color=COLORS[m], marker=MARKERS[m],
                         ms=7, lw=2, capsize=4, label=METHOD_LABELS[m])
        ax_u4.errorbar(r["N"], r["u4"], yerr=r["u4_sem"], color=COLORS[m], marker=MARKERS[m],
                       ms=7, lw=2, capsize=4, label=METHOD_LABELS[m])

    if args.exact_json:
        with open(args.exact_json) as f:
            exact = json.load(f)
        exact_chi = np.asarray(exact["chi_exact_mean"])[args.T_index]
        exact_chi_sem = np.asarray(exact["chi_exact_sem"])[args.T_index]
        exact_u4 = np.asarray(exact["u4_exact_mean"])[args.T_index]
        exact_u4_sem = np.asarray(exact["u4_exact_sem"])[args.T_index]
        # NOTE: the exact-enumeration JSON (from sk_exact_enumeration_D10.py,
        # itself not renamed) still stores this under the key "D" -- read as-is,
        # just renaming the local variable for consistency with everything else.
        exact_N = exact["D"]
        ax_chi.errorbar([exact_N], [exact_chi], yerr=[exact_chi_sem], color="black", marker="*",
                        ms=14, capsize=4, label="Exact enumeration (ground truth)")
        ax_u4.errorbar([exact_N], [exact_u4], yerr=[exact_u4_sem], color="black", marker="*",
                      ms=14, capsize=4, label="Exact enumeration (ground truth)")

    for ax, ylabel, title in [(ax_chi, "chi_SG", "Susceptibility"), (ax_u4, "Binder cumulant U4", "Binder cumulant")]:
        ax.set_xscale("log")
        ax.set_xticks(N_values)
        ax.set_xticklabels([str(n) for n in N_values])
        ax.set_xlabel("N (system size)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} vs system size")
        ax.legend()
        ax.grid(alpha=0.25, which="both")

    fig.suptitle("SK size-scaling comparison: does KC's advantage over TB grow with N?",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "sk_size_scaling_combined.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# =====================================================
# Subcommand: p_af sweep (chi/U4 vs p_af, fixed N)
# =====================================================
def cmd_paf_sweep(args):
    paf_values = [float(x) for x in args.paf_values.split(",")]
    methods = ["tb", "kc", "mcmc"]
    results = {m: {"paf": [], "chi": [], "chi_sem": [], "u4": [], "u4_sem": []} for m in methods}

    for paf in paf_values:
        # args.tag now required (was hardcoded "sk_D100") -- pass whatever
        # tag your fixed-N=100 paf-sweep data was actually saved under.
        payloads, _ = load_seed_jsons(os.path.join(args.base_dir, f"paf{paf}"), args.tag, seed_glob="seed_*")
        for m in methods:
            chi_arr = np.stack([np.asarray(p[f"chi_{m}_mean"])[args.T_index] for p in payloads])
            u4_arr = np.stack([np.asarray(p[f"u4_{m}_mean"])[args.T_index] for p in payloads])
            n = len(payloads)
            results[m]["paf"].append(paf)
            results[m]["chi"].append(chi_arr.mean())
            results[m]["chi_sem"].append(chi_arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
            results[m]["u4"].append(u4_arr.mean())
            results[m]["u4_sem"].append(u4_arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)

    fig, (ax_chi, ax_u4) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    for m in methods:
        r = results[m]
        ax_chi.errorbar(r["paf"], r["chi"], yerr=r["chi_sem"], color=COLORS[m], marker=MARKERS[m],
                        ms=7, lw=2, capsize=4, label=METHOD_LABELS[m])
        ax_u4.errorbar(r["paf"], r["u4"], yerr=r["u4_sem"], color=COLORS[m], marker=MARKERS[m],
                      ms=7, lw=2, capsize=4, label=METHOD_LABELS[m])

    for ax, ylabel, title in [(ax_chi, "chi_SG", "Susceptibility"), (ax_u4, "Binder cumulant U4", "Binder cumulant")]:
        ax.set_xlabel("p_af (frustration probability)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} vs frustration")
        ax.legend()
        ax.grid(alpha=0.25)

    fig.suptitle("SK frustration sweep: where does KC's advantage over TB appear?",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "sk_paf_sweep_combined.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# =====================================================
# Subcommand: TB log_Z ablation (fixed N)
# =====================================================
def cmd_logz_ablation(args):
    init_values = [int(x) for x in args.init_values.split(",")]
    results = {}
    for init_val in init_values:
        # args.tag now required (was hardcoded "sk_D100") -- pass whatever
        # tag your fixed-N=100 log_Z-ablation data was actually saved under.
        payloads, _ = load_seed_jsons(os.path.join(args.base_dir, f"init{init_val}"), args.tag)
        chi_mean, chi_sem = stack_and_average(payloads, "chi_tb_mean")
        u4_mean, u4_sem = stack_and_average(payloads, "u4_tb_mean")
        T_grid = np.asarray(payloads[0]["T_grid"])
        results[init_val] = dict(T_grid=T_grid, chi_mean=chi_mean, chi_sem=chi_sem,
                                  u4_mean=u4_mean, u4_sem=u4_sem)

    fig, (ax_chi, ax_u4) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(init_values)))
    for color, init_val in zip(colors, init_values):
        r = results[init_val]
        label = f"log_Z init = {init_val}"
        ax_chi.errorbar(r["T_grid"], r["chi_mean"], yerr=r["chi_sem"], marker="o", ms=5,
                        lw=1.8, capsize=3, color=color, label=label)
        ax_u4.errorbar(r["T_grid"], r["u4_mean"], yerr=r["u4_sem"], marker="o", ms=5,
                      lw=1.8, capsize=3, color=color, label=label)

    ax_chi.set_yscale("log")
    ax_chi.axhline(100.0, color="gray", ls="--", lw=1, alpha=0.6)
    ax_chi.set_xlabel("Temperature T")
    ax_chi.set_ylabel("chi_SG (log scale)")
    ax_chi.set_title("TB susceptibility vs log_Z init")
    ax_chi.legend(fontsize=8)
    ax_chi.grid(alpha=0.25, which="both")

    ax_u4.axhline(2/3, color="gray", ls="--", lw=1, alpha=0.6)
    ax_u4.set_ylim(-0.1, 0.75)
    ax_u4.set_xlabel("Temperature T")
    ax_u4.set_ylabel("Binder cumulant U4")
    ax_u4.set_title("TB Binder cumulant vs log_Z init")
    ax_u4.legend(fontsize=8)
    ax_u4.grid(alpha=0.25)

    fig.suptitle("TB log_Z-initialization sensitivity", fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "tb_logz_ablation_combined.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# =====================================================
# CLI
# =====================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_chi = sub.add_parser("chi", help="chi/U4 vs T, averaged across seeds")
    p_chi.add_argument("--base_dir", required=True, help="dir containing seed_*/ subdirs")
    p_chi.add_argument("--tag", required=True, help="file prefix, e.g. 'ising' or 'sk_D100'/'sk_N100'")
    p_chi.add_argument("--model_kind", required=True, choices=["ising", "sk", "ea"])
    p_chi.add_argument("--outdir", required=True)
    p_chi.set_defaults(func=cmd_chi)

    p_mcmc = sub.add_parser("mcmc_diag", help="MCMC acceptance rate / tau_int vs T")
    p_mcmc.add_argument("--base_dir", required=True)
    p_mcmc.add_argument("--tag", required=True)
    p_mcmc.add_argument("--outdir", required=True)
    p_mcmc.set_defaults(func=cmd_mcmc_diag)

    p_dist = sub.add_parser("distributions", help="per-seed order-parameter histograms")
    p_dist.add_argument("--seed_dir", required=True, help="a single seed_N/ directory")
    p_dist.add_argument("--tag", required=True)
    p_dist.add_argument("--model_kind", required=True, choices=["ising", "sk", "ea"])
    p_dist.add_argument("--outdir", required=True)
    p_dist.set_defaults(func=cmd_distributions)

    p_loss = sub.add_parser("loss", help="TB vs KC loss-per-iteration curves")
    p_loss.add_argument("--seed_dir", required=True)
    p_loss.add_argument("--tag", required=True)
    p_loss.add_argument("--outdir", required=True)
    p_loss.set_defaults(func=cmd_loss)

    p_size = sub.add_parser("size_sweep", help="chi/U4 vs N")
    p_size.add_argument("--base_dir", required=True, help="dir containing N{n}/ subdirs")
    p_size.add_argument("--N_values", default="10,25,50,100")
    p_size.add_argument("--T_index", type=int, default=0)
    p_size.add_argument("--exact_json", default=None, help="optional path to exact-enumeration JSON for N=10 overlay")
    p_size.add_argument("--outdir", required=True)
    p_size.set_defaults(func=cmd_size_sweep)

    p_paf = sub.add_parser("paf_sweep", help="chi/U4 vs p_af")
    p_paf.add_argument("--base_dir", required=True, help="dir containing paf{p}/ subdirs")
    p_paf.add_argument("--paf_values", default="0.0,0.1,0.2,0.3,0.4,0.5")
    p_paf.add_argument("--T_index", type=int, default=0)
    p_paf.add_argument("--tag", required=True, help="tag matching the fixed-N=100 sweep data, e.g. 'sk_D100' or 'sk_N100'")
    p_paf.add_argument("--outdir", required=True)
    p_paf.set_defaults(func=cmd_paf_sweep)

    p_logz = sub.add_parser("logz_ablation", help="TB chi/U4 vs log_Z init")
    p_logz.add_argument("--base_dir", required=True, help="dir containing init{v}/ subdirs")
    p_logz.add_argument("--init_values", default="0,35,150,300")
    p_logz.add_argument("--tag", required=True, help="tag matching the fixed-N=100 sweep data, e.g. 'sk_D100' or 'sk_N100'")
    p_logz.add_argument("--outdir", required=True)
    p_logz.set_defaults(func=cmd_logz_ablation)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
