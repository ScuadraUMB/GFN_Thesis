"""
run_tc_sweep.py
================
Cluster-ready driver for a single model's critical-temperature sweep,
comparing TB-GFlowNet, KC-GFlowNet, and MCMC.

*** THIS SCRIPT PRODUCES NO PLOTS. *** It only runs the sweep and saves
every number needed to make plots later. Use make_plots.py (separate
script) to generate figures from the saved data -- this split means
you never need to rerun a sweep just to change a title, axis, or color.

Usage (examples)
-----------------
  python run_tc_sweep.py --model ising  --N 5 --T_min 1.5 --T_max 4.0 --n_T 14
  python run_tc_sweep.py --model sk     --D 100 --T_min 0.3 --T_max 2.0 --n_T 20 --n_disorder 3

  # restrict to a subset of methods, e.g. skip MCMC for a quick GFlowNet-only check:
  python run_tc_sweep.py --model sk --D 100 --T_min 0.3 --T_max 2.0 --n_T 14 --methods TB,KC

  # log TB/KC loss-per-iteration at the lowest and highest T (default):
  python run_tc_sweep.py --model sk --D 100 --T_min 0.3 --T_max 2.0 --n_T 20 --log_loss

Each model is a separate process/job -- run all three (e.g. via a SLURM
array script) rather than looping over models in one process.

Outputs (written to --outdir, default ./outputs)
--------------------------------------------------
  {tag}_chi_sweep.json     chi/U4 means+sems per method per T, Tc estimates,
                           MCMC diagnostics (acceptance rate / tau_int) means+sems
                           per T -- SAME schema as before, so existing scripts
                           that read this file keep working unmodified.
  {tag}_raw_data.npz       NEW: everything chi_sweep.json can't hold --
                           raw sample arrays (for distribution-comparison
                           plots) at the lowest and highest T, from the
                           FIRST disorder realization (d=0) only, per
                           method; loss-per-iteration curves for TB/KC
                           (only if --log_loss is set, only at the
                           T-indices in --log_loss_T_indices, only for
                           d=0) as (iteration, loss) arrays.
  {tag}_tc_estimates.txt   lightweight human-readable summary (Tc
                           estimates, MCMC diagnostics, warnings) --
                           NOT a substitute for make_plots.py, just a
                           quick sanity-check while a job is running.
  {tag}_checkpoint.pt      progress checkpoint (deleted on successful
                           completion -- see below)

Methods
-------
--methods TB,KC,MCMC (default: all three). TB and KC are both trained
GFlowNets sharing the same batched policy architecture and training
loop (see spin_glass_lib.py), differing only in their loss: TB fits an
explicit log_Z against the reward; KC eliminates log_Z via a
telescoping identity between trajectories and instead requires
trajectory residuals to agree with each other.

TB's log_Z gets its own Adam param group at --lr_logz_mult x the
policy's learning rate (default 50x; see TBGFlowNet in spin_glass_lib.py).

Warm start
----------
--warm_start sweeps from T_max down to T_min and initializes each
GFlowNet method's policy (and, for TB, log_Z) from its own previous
(higher-T) trained state, rather than retraining from scratch at every T.

Loss logging
------------
--log_loss turns on loss-per-iteration recording for TB and KC (MCMC has
no "loss"). Off by default -- adds a small amount of memory/disk per
logged T-point, and is only useful for a subset of runs (e.g. one
representative seed), not every seed in a sweep.
  --log_loss_every N     record every Nth iteration (default 25; iteration
                          1 and the final iteration are always recorded too)
  --log_loss_T_indices   comma-separated indices into the T_grid array to
                          log (supports negative indexing, e.g. "0,-1" for
                          first and last T-point -- the default)
Loss is only recorded for the FIRST disorder realization (d=0), matching
the existing convention for saved raw samples.

MCMC diagnostics
----------------
Always computed (cheap: ~3s overhead per T-point at D=100). Reports:
  - acceptance_rate: fraction of proposed single-spin flips accepted
    (Metropolis; NaN for Wolff, which always accepts by construction)
  - mean_cluster_fraction: mean cluster size / D per flip (Wolff only)
  - tau_int: integrated autocorrelation time of the energy trace during
    the sampling phase -- large tau_int relative to --mcmc_thin signals
    the chain hasn't decorrelated between recorded samples (a sign of
    basin-trapping or --mcmc_thin being too small)
Disable with --no_mcmc_diagnostics if you want to skip the overhead.

Checkpointing / preemption
---------------------------
On a preemptible QOS (e.g. a "scavenger" queue), the scheduler can kill
this job at any time to reclaim borrowed compute, with a short grace
period (a SIGTERM) before a hard kill. This script:
  - writes a checkpoint to {outdir}/{tag}_checkpoint.pt after every
    T-point completes (for every disorder seed's sweep independently)
  - catches SIGTERM (and SIGUSR1) and checkpoints + exits promptly
  - a T-point interrupted mid-training is discarded entirely, never
    saved as a reduced-iteration result
  - on restart with the SAME arguments, automatically resumes (verified
    via an argument signature; mismatched arguments are refused)
  - --no_resume ignores any existing checkpoint and starts fresh
  - the checkpoint file is deleted automatically on successful completion
"""
import argparse
import os
import json
import signal
import sys
import numpy as np
import torch

from spin_glass_lib import (
    SpinEnv, TBGFlowNet, KCGFlowNet, run_mcmc, susceptibility, binder_cumulant_sg,
    make_batch_size_schedule, MCMC_WARMUP, MCMC_THIN, LR_LOGZ_MULT,
)

GFN_CLASSES = {"TB": TBGFlowNet, "KC": KCGFlowNet}
ALL_METHODS = ["TB", "KC", "MCMC"]
METHOD_LABELS = {"TB": "TB-GFlowNet", "KC": "KC-GFlowNet", "MCMC": "MCMC"}

LITERATURE_NOTES = {
    "ising": (
        "Ising 2D (periodic): exact infinite-lattice T_c = 2/ln(1+sqrt(2)) "
        "= 2.2692 J (Onsager). Finite-size rounding shifts the "
        "susceptibility peak ABOVE this value -- that is expected, not an error."
    ),
    "ea": (
        "EA spin glass, 2D nearest-neighbor (as coded here): T_c = 0 "
        "rigorously (lower critical dimension > 2). Any peak found is a "
        "FINITE-SIZE CROSSOVER, not a real T_c."
    ),
    "sk": (
        "SK model (fully connected, bimodal J=+-1/sqrt(D)): exact mean-"
        "field result is T_c = 1 (in these units) only as D -> infinity. "
        "chi_SG does NOT have a clean peak at finite D (saturates as T->0). "
        "The primary T_c estimate is instead the steepest-slope crossover "
        "of the Binder cumulant U4. Disorder-average over multiple J-seeds "
        "(--n_disorder) before trusting a single number."
    ),
}


# =====================================================
#  Signal handling for graceful preemption
# =====================================================
_STOP_REQUESTED = [False]


def _signal_handler(signum, frame):
    if not _STOP_REQUESTED[0]:
        print(f"\n[checkpoint] Caught signal {signum} -- will checkpoint and "
              f"exit as soon as it's safe to do so.", flush=True)
    _STOP_REQUESTED[0] = True


def _install_signal_handlers():
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        signal.signal(signal.SIGUSR1, _signal_handler)
    except (AttributeError, ValueError):
        pass


def save_checkpoint(path, payload):
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"WARNING: failed to load checkpoint at {path} ({e}); "
              f"starting fresh instead.", flush=True)
        return None


_SIGNATURE_KEYS = [
    "model", "N", "D", "p_af", "T_min", "T_max", "n_T", "methods",
    "n_disorder", "seed", "warm_start", "n_iter", "n_eval",
    "batch_start", "batch_end", "entropy_coef0", "entropy_anneal_frac",
    "mcmc_warmup", "mcmc_thin", "mcmc_method", "lr_logz_mult",
    "init_log_z_override", "log_loss", "log_loss_every", "log_loss_T_indices",
    "mcmc_diagnostics",
]


def make_signature(args):
    return {k: getattr(args, k) for k in _SIGNATURE_KEYS}


def parse_methods(s):
    methods = [m.strip().upper() for m in s.split(",") if m.strip()]
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}; choose from {ALL_METHODS}")
    if not methods:
        raise ValueError("--methods must not be empty")
    return methods


def resolve_T_indices(spec, n_T):
    """Parse '0,-1' style comma-separated (possibly negative) indices
    into actual non-negative indices into a length-n_T array."""
    out = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        i = int(tok)
        if i < 0:
            i = n_T + i
        if not (0 <= i < n_T):
            raise ValueError(f"--log_loss_T_indices index {tok} out of range for n_T={n_T}")
        out.add(i)
    return out


def build_env(args, J_seed):
    if args.model == "ising":
        return SpinEnv("ising", N=args.N)
    elif args.model == "ea":
        return SpinEnv("ea", N=args.N, p_af=args.p_af, J_seed=J_seed)
    elif args.model == "sk":
        return SpinEnv("sk", D=args.D, p_af=args.p_af, J_seed=J_seed)
    else:
        raise ValueError(args.model)


def peak_estimate(T_grid, chi):
    T_grid = np.asarray(T_grid, dtype=float)
    chi = np.asarray(chi, dtype=float)
    i = int(np.argmax(chi))
    if i == 0 or i == len(chi) - 1:
        return float(T_grid[i]), float(chi[i]), False
    x0, x1, x2 = T_grid[i - 1], T_grid[i], T_grid[i + 1]
    y0, y1, y2 = chi[i - 1], chi[i], chi[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-12:
        return float(x1), float(y1), True
    T_peak = x1 + 0.5 * (y0 - y2) / denom * (x1 - x0)
    return float(T_peak), float(y1), True


def crossover_from_binder(T_grid, U4):
    T_grid = np.asarray(T_grid, dtype=float)
    U4 = np.asarray(U4, dtype=float)
    dU4 = np.diff(U4)
    dT = np.diff(T_grid)
    slope = dU4 / dT
    T_mid = 0.5 * (T_grid[:-1] + T_grid[1:])
    i = int(np.argmax(np.abs(slope)))
    interior = 0 < i < len(slope) - 1
    return float(T_mid[i]), float(slope[i]), interior


def run_sweep_for_disorder(env_kind, args, J_seed, resume=None, on_T_done=None,
                            collect_extras=False, log_T_indices=None):
    """
    Runs (or resumes) the T-sweep for ONE disorder realization.

    collect_extras: if True, this disorder realization's run also
      collects raw samples (all T, for later distribution/loss plots --
      only ever true for d=0, matching the existing convention) and,
      if args.log_loss is set, loss-per-iteration curves at the
      T-indices in log_T_indices.
    """
    T_grid = np.linspace(args.T_min, args.T_max, args.n_T)
    order = np.argsort(-T_grid) if args.warm_start else np.arange(len(T_grid))

    env = build_env(args, J_seed)
    gfn_methods = [m for m in args.methods if m in GFN_CLASSES]
    run_mcmc_flag = "MCMC" in args.methods
    log_T_indices = log_T_indices or set()

    if resume is not None:
        completed_T_idx = set(int(i) for i in resume["completed_T_idx"])
        chi = resume["chi"]
        u4 = resume["u4"]
        samples_by_T = resume["samples_by_T"]
        prev_state_dict = resume["prev_state_dict"]
        prev_log_z = resume.get("prev_log_z", {m: None for m in gfn_methods})
        mcmc_diag = resume.get("mcmc_diag", {
            "acceptance_rate": np.full(args.n_T, np.nan),
            "mean_cluster_fraction": np.full(args.n_T, np.nan),
            "tau_int": np.full(args.n_T, np.nan),
        })
        loss_logs = resume.get("loss_logs", {m: {} for m in gfn_methods})
        print(f"[{env_kind} seed={J_seed}] resuming: "
              f"{len(completed_T_idx)}/{args.n_T} T-points already done", flush=True)
    else:
        completed_T_idx = set()
        chi = {m: np.zeros(args.n_T) for m in args.methods}
        u4 = {m: np.zeros(args.n_T) for m in args.methods}
        samples_by_T = {m: {} for m in args.methods}
        prev_state_dict = {m: None for m in gfn_methods}
        prev_log_z = {m: None for m in gfn_methods}
        mcmc_diag = {
            "acceptance_rate": np.full(args.n_T, np.nan),
            "mean_cluster_fraction": np.full(args.n_T, np.nan),
            "tau_int": np.full(args.n_T, np.nan),
        }
        loss_logs = {m: {} for m in gfn_methods}  # {method: {T_idx: [(it, loss), ...]}}

    bsf = make_batch_size_schedule(args.batch_start, args.batch_end)
    stopped_early = False

    for idx in order:
        idx = int(idx)
        if idx in completed_T_idx:
            continue
        T = T_grid[idx]
        T_interrupted = False
        log_parts = []

        for m in gfn_methods:
            extra_kwargs = {}
            if m == "TB":
                extra_kwargs["lr_logz_mult"] = args.lr_logz_mult
                if args.init_log_z_override is not None:
                    extra_kwargs["init_log_z"] = args.init_log_z_override
                else:
                    extra_kwargs["init_log_z"] = prev_log_z[m] if args.warm_start else None
            model = GFN_CLASSES[m](
                env, T, device=args.device,
                init_state_dict=prev_state_dict[m] if args.warm_start else None,
                entropy_coef0=args.entropy_coef0,
                entropy_anneal_frac=args.entropy_anneal_frac,
                **extra_kwargs)

            this_loss_log = None
            if collect_extras and args.log_loss and idx in log_T_indices:
                this_loss_log = []

            completed_fully = model.train(
                args.n_iter, batch_size_fn=bsf, stop_check=lambda: _STOP_REQUESTED[0],
                loss_log=this_loss_log, log_every=args.log_loss_every)
            if not completed_fully:
                T_interrupted = True
                break
            if args.warm_start:
                prev_state_dict[m] = model.state_dict()
                if m == "TB":
                    prev_log_z[m] = float(model.log_Z.item())
            samp = model.sample(args.n_eval)
            chi[m][idx], _ = susceptibility(env, samp, T)
            if collect_extras:
                samples_by_T[m][float(T)] = samp
                if this_loss_log is not None:
                    loss_logs[m][idx] = this_loss_log
            if env_kind != "ising":
                u4[m][idx] = binder_cumulant_sg(env, samp)
            log_parts.append(
                f"chi_{m}={chi[m][idx]:.4f}"
                + (f" U4_{m}={u4[m][idx]:.4f}" if env_kind != "ising" else "")
                + (f" logZ={prev_log_z[m]:.3f}" if m == "TB" and prev_log_z[m] is not None else ""))

        if T_interrupted:
            print(f"[{env_kind} seed={J_seed}] T={T:.3f} interrupted mid-training "
                  f"-- discarding this T-point's partial work.", flush=True)
            stopped_early = True
            break

        if run_mcmc_flag:
            if args.mcmc_diagnostics:
                samp_mcmc, diag = run_mcmc(env, T, args.n_eval, warmup=args.mcmc_warmup,
                                            thin=args.mcmc_thin, method=args.mcmc_method,
                                            return_diagnostics=True)
                mcmc_diag["acceptance_rate"][idx] = diag["acceptance_rate"]
                mcmc_diag["mean_cluster_fraction"][idx] = diag["mean_cluster_fraction"]
                mcmc_diag["tau_int"][idx] = diag["tau_int"]
                log_parts.append(f"mcmc_accept={diag['acceptance_rate']:.3f} "
                                  f"mcmc_tau_int={diag['tau_int']:.1f}")
            else:
                samp_mcmc = run_mcmc(env, T, args.n_eval, warmup=args.mcmc_warmup,
                                      thin=args.mcmc_thin, method=args.mcmc_method)
            chi["MCMC"][idx], _ = susceptibility(env, samp_mcmc, T)
            if collect_extras:
                samples_by_T["MCMC"][float(T)] = samp_mcmc
            if env_kind != "ising":
                u4["MCMC"][idx] = binder_cumulant_sg(env, samp_mcmc)
            log_parts.append(
                f"chi_MCMC={chi['MCMC'][idx]:.4f}"
                + (f" U4_MCMC={u4['MCMC'][idx]:.4f}" if env_kind != "ising" else ""))

        print(f"[{env_kind} seed={J_seed}] T={T:.3f}  " + "  ".join(log_parts), flush=True)
        completed_T_idx.add(idx)
        if on_T_done is not None:
            on_T_done(completed_T_idx, chi, u4, samples_by_T, prev_state_dict, prev_log_z,
                      mcmc_diag, loss_logs)

        if _STOP_REQUESTED[0]:
            stopped_early = True
            break

    return T_grid, chi, u4, samples_by_T, mcmc_diag, loss_logs, stopped_early


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ising", "ea", "sk"], required=True)
    p.add_argument("--methods", type=str, default="TB,KC,MCMC")
    p.add_argument("--N", type=int, default=5)
    p.add_argument("--D", type=int, default=5)
    p.add_argument("--p_af", type=float, default=0.5)
    p.add_argument("--T_min", type=float, required=True)
    p.add_argument("--T_max", type=float, required=True)
    p.add_argument("--n_T", type=int, default=14)
    p.add_argument("--n_iter", type=int, default=1500)
    p.add_argument("--n_eval", type=int, default=800)
    p.add_argument("--n_disorder", type=int, default=1)
    p.add_argument("--mcmc_warmup", type=int, default=MCMC_WARMUP)
    p.add_argument("--mcmc_thin", type=int, default=MCMC_THIN)
    p.add_argument("--mcmc_method", choices=["auto", "metropolis", "wolff"], default="auto")
    p.add_argument("--mcmc_diagnostics", action="store_true", default=True,
                    help="compute acceptance rate / tau_int (default on; cheap, ~3s/T-point at D=100)")
    p.add_argument("--no_mcmc_diagnostics", dest="mcmc_diagnostics", action="store_false")
    p.add_argument("--batch_start", type=int, default=16)
    p.add_argument("--batch_end", type=int, default=16)
    p.add_argument("--entropy_coef0", type=float, default=0.05)
    p.add_argument("--entropy_anneal_frac", type=float, default=0.8)
    p.add_argument("--init_log_z_override", type=float, default=None)
    p.add_argument("--lr_logz_mult", type=float, default=LR_LOGZ_MULT)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--warm_start", action="store_true")
    p.add_argument("--disagree_tol", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="./outputs")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--log_loss", action="store_true",
                    help="record TB/KC loss every --log_loss_every iterations, at the "
                         "T-indices in --log_loss_T_indices, for the FIRST disorder "
                         "realization (d=0) only. Off by default.")
    p.add_argument("--log_loss_every", type=int, default=25)
    p.add_argument("--log_loss_T_indices", type=str, default="0,-1",
                    help="comma-separated (possibly negative) indices into the T_grid "
                         "array to log loss at. Default '0,-1' = first and last T-point.")
    args = p.parse_args()
    args.methods = parse_methods(args.methods)
    log_T_indices = resolve_T_indices(args.log_loss_T_indices, args.n_T)

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.model}_D{args.D}" if args.model == "sk" else args.model
    ckpt_path = os.path.join(args.outdir, f"{tag}_checkpoint.pt")

    _install_signal_handlers()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gfn_methods = [m for m in args.methods if m in GFN_CLASSES]
    is_spin_glass = args.model != "ising"
    n_disorder = 1 if args.model == "ising" else args.n_disorder

    signature = make_signature(args)
    ckpt = None if args.no_resume else load_checkpoint(ckpt_path)
    if ckpt is not None and ckpt.get("signature") != signature:
        diffs = {k: (ckpt["signature"].get(k), signature.get(k))
                 for k in _SIGNATURE_KEYS if ckpt["signature"].get(k) != signature.get(k)}
        print(f"WARNING: checkpoint at {ckpt_path} doesn't match current arguments -- "
              f"starting fresh. Differing keys: {diffs}", flush=True)
        ckpt = None

    per_d_state = ckpt["per_d"] if ckpt is not None else [None] * n_disorder
    if ckpt is not None:
        n_done = sum(1 for s in per_d_state if s is not None and s.get("done"))
        print(f"[checkpoint] Resuming from {ckpt_path}: {n_done}/{n_disorder} "
              f"disorder seed(s) already complete.", flush=True)

    T_grid = np.linspace(args.T_min, args.T_max, args.n_T)
    global_stop = False

    for d in range(n_disorder):
        if global_stop:
            break
        state = per_d_state[d]
        if state is not None and state.get("done"):
            continue

        J_seed = args.seed * 1000 + d
        collect_extras = (d == 0)  # raw samples + loss only saved for the first disorder realization

        def on_T_done(completed_T_idx, chi, u4, samples_by_T, prev_state_dict, prev_log_z,
                      mcmc_diag, loss_logs, d=d):
            per_d_state[d] = {
                "done": False, "chi": chi, "u4": u4,
                "samples_by_T": samples_by_T if d == 0 else {},
                "completed_T_idx": set(completed_T_idx),
                "prev_state_dict": prev_state_dict, "prev_log_z": prev_log_z,
                "mcmc_diag": mcmc_diag,
                "loss_logs": loss_logs if d == 0 else {},
            }
            save_checkpoint(ckpt_path, {"signature": signature, "per_d": per_d_state})

        _, chi, u4, samples_by_T, mcmc_diag, loss_logs, stopped_early = run_sweep_for_disorder(
            args.model, args, J_seed, resume=state, on_T_done=on_T_done,
            collect_extras=collect_extras, log_T_indices=log_T_indices)

        if stopped_early:
            any_progress = any(s is not None for s in per_d_state)
            print(f"\n[checkpoint] Exiting early due to preemption signal. "
                  f"{'Progress saved to ' + ckpt_path if any_progress else 'Nothing to checkpoint yet.'} "
                  f"Resubmit the SAME command to resume.", flush=True)
            global_stop = True
        else:
            per_d_state[d] = {
                "done": True, "chi": chi, "u4": u4,
                "samples_by_T": samples_by_T if d == 0 else {},
                "completed_T_idx": set(range(args.n_T)), "prev_state_dict": {}, "prev_log_z": {},
                "mcmc_diag": mcmc_diag,
                "loss_logs": loss_logs if d == 0 else {},
            }
            save_checkpoint(ckpt_path, {"signature": signature, "per_d": per_d_state})

    if global_stop:
        sys.exit(0)

    # =========================================================
    # All disorder seeds complete -- aggregate and SAVE (no plotting)
    # =========================================================
    all_chi = {m: [per_d_state[d]["chi"][m] for d in range(n_disorder)] for m in args.methods}
    all_u4 = {m: [per_d_state[d]["u4"][m] for d in range(n_disorder)] for m in args.methods}
    first_samples = per_d_state[0]["samples_by_T"]
    first_loss_logs = per_d_state[0]["loss_logs"]

    chi_mean, chi_sem = {}, {}
    for m in args.methods:
        chi_mean[m] = np.mean(all_chi[m], axis=0)
        chi_sem[m] = (np.std(all_chi[m], axis=0, ddof=1) / np.sqrt(n_disorder)
                      if n_disorder > 1 else np.zeros_like(chi_mean[m]))

    Tc, Tc_ok = {}, {}
    for m in args.methods:
        Tc[m], _, Tc_ok[m] = peak_estimate(T_grid, chi_mean[m])

    u4_mean, u4_sem, Tcross, Tcross_ok = {}, {}, {}, {}
    if is_spin_glass:
        for m in args.methods:
            u4_mean[m] = np.mean(all_u4[m], axis=0)
            u4_sem[m] = (np.std(all_u4[m], axis=0, ddof=1) / np.sqrt(n_disorder)
                         if n_disorder > 1 else np.zeros_like(u4_mean[m]))
            Tcross[m], _, Tcross_ok[m] = crossover_from_binder(T_grid, u4_mean[m])

    flagged = {}
    if "MCMC" in args.methods:
        for m in gfn_methods:
            rel_diff = np.abs(chi_mean[m] - chi_mean["MCMC"]) / np.maximum(chi_mean["MCMC"], 1e-8)
            flagged[m] = T_grid[rel_diff > args.disagree_tol]

    mcmc_diag_stack = {
        k: np.stack([per_d_state[d]["mcmc_diag"][k] for d in range(n_disorder)])
        for k in ["acceptance_rate", "mean_cluster_fraction", "tau_int"]
    } if "MCMC" in args.methods else {}
    # mean_cluster_fraction is legitimately all-NaN for non-Wolff runs (Metropolis has
    # no notion of cluster size) -- nanmean/nanstd over an all-NaN slice correctly
    # returns NaN but numpy warns about it; suppress that expected, harmless warning.
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mcmc_diag_mean = {k: np.nanmean(v, axis=0) for k, v in mcmc_diag_stack.items()}
            mcmc_diag_sem = {
                k: (np.nanstd(v, axis=0, ddof=1) / np.sqrt(n_disorder) if n_disorder > 1
                    else np.zeros(args.n_T))
                for k, v in mcmc_diag_stack.items()
            }

    # =========================================================
    # Save {tag}_chi_sweep.json -- SAME schema as before (chi/u4/Tc),
    # PLUS new mcmc_diag_* keys.
    # =========================================================
    json_payload = {
        "model": args.model, "D": args.D, "N": args.N, "T_grid": T_grid.tolist(),
        "methods": args.methods, "n_disorder": n_disorder,
    }
    for m in args.methods:
        ml = m.lower()
        json_payload[f"chi_{ml}_mean"] = chi_mean[m].tolist()
        json_payload[f"chi_{ml}_sem"] = chi_sem[m].tolist()
        json_payload[f"Tc_{ml}_chi_peak"] = Tc[m]
        if is_spin_glass:
            json_payload[f"u4_{ml}_mean"] = u4_mean[m].tolist()
            json_payload[f"u4_{ml}_sem"] = u4_sem[m].tolist()
            json_payload[f"Tcross_{ml}_binder"] = Tcross[m]
    if "MCMC" in args.methods:
        for k, v in mcmc_diag_mean.items():
            json_payload[f"mcmc_diag_{k}_mean"] = v.tolist()
        for k, v in mcmc_diag_sem.items():
            json_payload[f"mcmc_diag_{k}_sem"] = v.tolist()
    with open(os.path.join(args.outdir, f"{tag}_chi_sweep.json"), "w") as f:
        json.dump(json_payload, f, indent=2)

    # =========================================================
    # Save {tag}_raw_data.npz -- samples (lowT/highT, d=0, per method)
    # and loss logs (if --log_loss, at requested T-indices, d=0, GFN
    # methods only).
    # =========================================================
    npz_payload = {"T_grid": T_grid}
    for m in args.methods:
        ml = m.lower()
        for label, T_key_idx in [("lowT", 0), ("highT", -1)]:
            T_val = float(T_grid[T_key_idx])
            if m in first_samples and first_samples[m]:
                key = min(first_samples[m].keys(), key=lambda k: abs(k - T_val))
                npz_payload[f"samples_{label}_{ml}"] = np.asarray(first_samples[m][key], dtype=np.float32)
                npz_payload[f"samples_{label}_{ml}_T"] = np.float32(key)
    if args.log_loss:
        for m in gfn_methods:
            ml = m.lower()
            for idx, log in first_loss_logs.get(m, {}).items():
                if log:
                    arr = np.asarray(log, dtype=np.float64)  # shape (n_points, 2): [iteration, loss]
                    npz_payload[f"loss_{ml}_Tidx{idx}"] = arr
                    npz_payload[f"loss_{ml}_Tidx{idx}_T"] = np.float64(T_grid[idx])
    np.savez(os.path.join(args.outdir, f"{tag}_raw_data.npz"), **npz_payload)

    # =========================================================
    # Lightweight human-readable summary (NOT a plotting substitute)
    # =========================================================
    out_path = os.path.join(args.outdir, f"{tag}_tc_estimates.txt")
    with open(out_path, "w") as f:
        f.write(f"Critical temperature sweep: model={args.model}\n" + "=" * 60 + "\n")
        f.write(LITERATURE_NOTES[args.model] + "\n\n")
        f.write(f"Methods run: {', '.join(args.methods)}\n")
        f.write(f"Grid: N={args.N if args.model!='sk' else '-'}  "
                f"D={args.D if args.model=='sk' else (args.N*args.N)}  "
                f"p_af={args.p_af if args.model!='ising' else '-'}\n")
        f.write(f"T sweep: [{args.T_min}, {args.T_max}] with {args.n_T} points, "
                f"warm_start={args.warm_start}, n_disorder={n_disorder}\n")
        f.write(f"n_iter (GFN)={args.n_iter}  n_eval={args.n_eval}  "
                f"mcmc_warmup={args.mcmc_warmup}  mcmc_thin={args.mcmc_thin}  "
                f"lr_logz_mult={args.lr_logz_mult}  log_loss={args.log_loss}\n\n")

        if is_spin_glass:
            f.write("PRIMARY ESTIMATE -- Binder cumulant (U4) steepest-slope crossover:\n")
            for m in args.methods:
                f.write(f"  T* ({METHOD_LABELS[m]}):  {Tcross[m]:.4f}"
                        + ("" if Tcross_ok[m] else "  [WARNING: at grid boundary]") + "\n")
            f.write("\nSECONDARY -- susceptibility chi_SG peak:\n")
            for m in args.methods:
                f.write(f"  T_c ({METHOD_LABELS[m]}):  {Tc[m]:.4f}"
                        + ("" if Tc_ok[m] else "  [boundary]") + "\n")
        else:
            f.write("Estimated T_c (susceptibility peak):\n")
            for m in args.methods:
                f.write(f"  {METHOD_LABELS[m]}:  {Tc[m]:.4f}"
                        + ("" if Tc_ok[m] else "  [WARNING: at grid boundary]") + "\n")
        f.write("\n")

        if "MCMC" in args.methods:
            f.write("MCMC diagnostics (avg over disorder seeds):\n")
            for i, T in enumerate(T_grid):
                f.write(f"  T={T:.3f}  accept={mcmc_diag_mean['acceptance_rate'][i]:.3f}"
                        f"  cluster_frac={mcmc_diag_mean['mean_cluster_fraction'][i]:.3f}"
                        f"  tau_int={mcmc_diag_mean['tau_int'][i]:.1f}\n")
            f.write("\n")

        if "MCMC" in args.methods and gfn_methods:
            for m in gfn_methods:
                if len(flagged.get(m, [])) > 0:
                    f.write(f"WARNING: {METHOD_LABELS[m]} and MCMC disagree by "
                            f">{args.disagree_tol*100:.0f}% at {len(flagged[m])} T point(s): "
                            + ", ".join(f"{t:.3f}" for t in flagged[m]) + "\n")

    try:
        os.remove(ckpt_path)
    except OSError:
        pass

    print(f"\nDone. Wrote {tag}_chi_sweep.json, {tag}_raw_data.npz, {tag}_tc_estimates.txt "
          f"to {args.outdir}/")


if __name__ == "__main__":
    main()
