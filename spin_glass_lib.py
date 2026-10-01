"""
spin_glass_lib.py
==================
Shared physics + sampling code for the critical-temperature sweeps.

# ------------------------------------------------------------------
# DOCSTRING CORRECTED -- the previous version of this header was
# stale and described an earlier iteration of this file, not the one
# below it:
#   - It said "drops KC" -- KCGFlowNet is fully implemented below.
#   - It described Ising/EA as a fixed 5x5 grid and SK as D=5 --
#     the actual sweeps this code has been run with go up to N=10
#     (Ising) and D=100 (SK). D/N are CLI-configurable (see
#     run_tc_sweep.py), not fixed at 5.
# Left uncorrected for a while, which is exactly the kind of drift
# worth watching for elsewhere in this file too. If you find another
# copy of this module with a docstring matching the OLD text below,
# that copy is more likely to be the genuinely older version:
#   "Trimmed / adapted from the notebook prototype: drops KC, XY, and
#    exact enumeration... Ising: 5x5 periodic grid... SK: D=5
#    fully-connected..."
# ------------------------------------------------------------------

Provides:
  - SpinEnv: unified +-1 spin environment for Ising (grid), EA (grid,
    disordered), and SK (fully-connected, disordered) models.
  - TBGFlowNet / KCGFlowNet: batched, weight-tied-across-timesteps
    GFlowNet training, differing only in their loss (TB fits an
    explicit log_Z; KC eliminates it via a telescoping residual
    variance).
  - MetropolisHastings / WolffCluster + run_mcmc(): MCMC baselines
    (Wolff for uniform-J Ising only; Metropolis for EA/SK).
  - susceptibility estimators (ferromagnetic chi for Ising, spin-glass
    chi_SG for EA/SK) computed from *independent equilibrium samples*
    (the standard "two-replica overlap" trick -- no ground truth
    needed), plus binder_cumulant_sg().
  - Optional warm-start: initialize a policy at temperature T from the
    already-trained policy at the neighboring T in the sweep.

# ------------------------------------------------------------------
# LEGACY API NOTE: SpinEnv carries a second, older interface alongside
# the batched/tensor one actually used for training --
# root()/step_index()/step()/encode()/state_to_x()/energy()/
# log_reward(), a tuple-based single-trajectory representation. Cross-
# checked against every file currently known to import this module
# (run_tc_sweep.py, make_plots.py, sk_exact_enumeration_D10.py): NONE
# of them call this API. run_tc_sweep.py's actual training path goes
# through BatchedGFlowNetBase._sample_actions_batch/_replay_batch,
# which build spins/mask tensors directly and never touch
# root()/step()/encode(); reward uses energy_batch() (the vectorized
# version), never energy()/log_reward()/state_to_x().
# sk_exact_enumeration_D10.py reimplements SK coupling construction
# independently rather than calling SpinEnv at all.
# Each flagged method below is marked individually with why it looks
# unused. Marked here rather than removed, since "not called by any
# file currently known to import this module" isn't the same
# guarantee as "not called anywhere" -- an unseen notebook or script
# could still depend on it. Safe to delete once that's ruled out.
# ------------------------------------------------------------------
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------
# Hyperparameters (safe defaults; override via CLI in run_tc_sweep.py)
# ---------------------------------------------------------------
BATCH        = 16
LR_NN        = 1e-3
LR_LOGZ_MULT = 50.0   # log_Z's Adam lr = LR_NN * LR_LOGZ_MULT, see TBGFlowNet
EPS_START    = 0.8
EPS_END      = 0.05
BETA_START   = 0.3
BETA_END     = 1.0
ENTROPY_COEF = 0.05
MCMC_WARMUP  = 50_000
MCMC_THIN    = 50
LOG_PB_FIXED = 0.0


def integrated_autocorr_time(x, c=5.0):
    """
    Integrated autocorrelation time of a 1D time series x, via Sokal's
    automatic-windowing estimator: tau_int(M) = 1 + 2*sum_{t=1}^{M} rho(t),
    where rho is the normalized autocorrelation function, and M is chosen
    as the smallest window where M >= c * tau_int(M) (c=5 is a standard
    default). This is an approximate, standard diagnostic -- not a
    rigorous bound -- used here to flag whether an MCMC chain's samples
    are effectively decorrelated given the configured thinning, not as a
    precise scientific measurement in its own right.

    Returns tau_int (float, >= 1.0). If x is too short or constant
    (autocorrelation undefined), returns float('nan').
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 10:
        return float("nan")
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var < 1e-12:
        return float("nan")  # constant series (e.g., chain never moved)

    # autocorrelation via FFT (much faster than direct summation for long chains)
    size = 1
    while size < 2 * n:
        size *= 2
    fx = np.fft.rfft(x, n=size)
    acf = np.fft.irfft(fx * np.conjugate(fx), n=size)[:n].real
    acf /= (var * n)  # normalize so acf[0] == 1

    tau = 1.0
    for M in range(1, n // 2):
        tau = 1.0 + 2.0 * np.sum(acf[1:M + 1])
        if M >= c * tau:
            return max(tau, 1.0)
    return max(tau, 1.0)  # window never satisfied c*tau <= M; best available estimate


def build_neighbour_list(N):
    pairs, seen = [], set()
    for r in range(N):
        for c in range(N):
            i = r * N + c
            for nr, nc in [((r + 1) % N, c), (r, (c + 1) % N)]:
                j = nr * N + nc
                if (i, j) not in seen:
                    pairs.append((i, j))
                    seen.add((i, j))
    return pairs


def build_all_pairs(D):
    return [(i, j) for i in range(D) for j in range(i + 1, D)]


# =====================================================
#  ENVIRONMENTS
# =====================================================
class SpinEnv:
    """
    Unified +-1 spin environment.

    kind = "ising"  : N x N grid, nearest-neighbor, uniform J=+1
    kind = "ea"     : N x N grid, nearest-neighbor, bimodal J=+-1
    kind = "sk"     : D spins, fully connected, bimodal J=+-1/sqrt(D)

    LEGACY API: root(), step_index(), step(), encode(), state_to_x(),
    energy(), and log_reward() below implement an older tuple-based,
    single-trajectory state representation. See the LEGACY API NOTE at
    the top of this file -- none of the three files currently known to
    import this module call these methods; the actual batched training
    and reward path uses energy_batch() and direct tensor indexing
    instead. Kept here, unmodified, rather than removed.
    """
    def __init__(self, kind, N=None, D=None, p_af=0.5, J_seed=0):
        self.kind = kind
        if kind in ("ising", "ea"):
            assert N is not None
            self.N = N
            self.D = N * N
            self.PAIRS = build_neighbour_list(N)
        elif kind == "sk":
            assert D is not None
            self.N = None
            self.D = D
            self.PAIRS = build_all_pairs(D)
        else:
            raise ValueError(f"unknown kind={kind!r}")

        self.p_af = p_af
        if kind == "ising":
            self.J = {pair: 1.0 for pair in self.PAIRS}
        elif kind == "ea":
            rng = np.random.default_rng(J_seed)
            signs = np.where(rng.random(len(self.PAIRS)) < p_af, -1.0, 1.0)
            self.J = {pair: float(j) for pair, j in zip(self.PAIRS, signs)}
        elif kind == "sk":
            rng = np.random.default_rng(J_seed)
            signs = np.where(rng.random(len(self.PAIRS)) < p_af, -1.0, 1.0)
            vals = signs / np.sqrt(self.D)
            self.J = {pair: float(j) for pair, j in zip(self.PAIRS, vals)}

        self.n_actions = 2
        self.encode_dim = self.D

        # --- precomputed structures for vectorized/batched ops ---
        # (built lazily/cached; kept separate from the tuple-based root()/
        # step() API above so existing single-trajectory code paths, if
        # anything still uses them, are untouched)
        i_idx = np.array([p[0] for p in self.PAIRS], dtype=np.int64)
        j_idx = np.array([p[1] for p in self.PAIRS], dtype=np.int64)
        Jvals = np.array([self.J[p] for p in self.PAIRS], dtype=np.float32)
        self._pair_i_np, self._pair_j_np, self._Jvals_np = i_idx, j_idx, Jvals
        self._pair_tensors_cache = {}  # device -> (i_idx, j_idx, Jvals) tensors

        # neighbor list for O(degree) local-delta-energy MCMC instead of
        # O(D) or O(pairs) full-energy recomputation per proposed flip
        # (the latter is what the original MetropolisHastings did -- a
        # real, easy-to-fix inefficiency, not just a GFlowNet-side issue)
        self.neighbors = {i: [] for i in range(self.D)}
        for (a, b), Jab in self.J.items():
            self.neighbors[a].append((b, Jab))
            self.neighbors[b].append((a, Jab))

    def _pair_tensors(self, device):
        key = str(device)
        if key not in self._pair_tensors_cache:
            self._pair_tensors_cache[key] = (
                torch.as_tensor(self._pair_i_np, device=device),
                torch.as_tensor(self._pair_j_np, device=device),
                torch.as_tensor(self._Jvals_np, device=device),
            )
        return self._pair_tensors_cache[key]

    def energy_batch(self, X):
        """
        Vectorized energy for a BATCH of full configurations.
        X: [B, D] tensor of +-1 values (any device).
        Returns: [B] tensor of energies.
        Replaces the old per-sample Python-loop energy() for anywhere
        that used to call it once per trajectory in a loop -- this is the
        the same fix in spirit as batching the GFlowNet policy calls,
        just applied to the reward computation instead.
        """
        i_idx, j_idx, Jvals = self._pair_tensors(X.device)
        return -(Jvals.unsqueeze(0) * X[:, i_idx] * X[:, j_idx]).sum(dim=1)

    def local_delta_energy(self, x, i):
        """
        Energy change from flipping site i alone, using only that site's
        neighbors (O(degree) instead of recomputing the full energy over
        all pairs). x: numpy array of +-1 values. Standard result:
        dE = 2 * x[i] * h_i, where h_i = sum_j J_ij * x[j] over neighbors.
        """
        h_i = sum(Jij * x[j] for j, Jij in self.neighbors[i])
        return 2.0 * x[i] * h_i

    def root(self):
        """LEGACY -- see the LEGACY API NOTE at the top of this file and
        the class docstring above. Appears unused: the batched training
        path never calls this."""
        return (None,) * self.D

    def step_index(self, s):
        """LEGACY -- see root()."""
        for i, v in enumerate(s):
            if v is None:
                return i
        return self.D

    def step(self, s, a):
        """LEGACY -- see root()."""
        i = self.step_index(s)
        s = list(s)
        s[i] = -1 if a == 0 else +1
        return tuple(s)

    def energy(self, x):
        """LEGACY -- see root(). Superseded by energy_batch() (vectorized,
        actually used by the training/reward path) and energy_fast_np()
        (vectorized single-config version, used by the MCMC diagnostics
        loop). This version does a Python dict-lookup per pair and is
        slow if called in a tight loop."""
        x = np.asarray(x, dtype=np.float32)
        return -sum(self.J[(i, j)] * x[i] * x[j] for i, j in self.PAIRS)

    def energy_fast_np(self, x):
        """Vectorized numpy energy for a SINGLE configuration, using the
        same cached pair-index arrays as energy_batch (built once in
        __init__). This is the version to call from a tight loop --
        e.g. once per raw MCMC step for an autocorrelation-time
        diagnostic -- since plain energy() above does a slow per-pair
        Python dict-lookup loop (fine for one-off calls, far too slow
        to call tens of thousands of times per T-point)."""
        x = np.asarray(x, dtype=np.float32)
        return float(-np.sum(self._Jvals_np * x[self._pair_i_np] * x[self._pair_j_np]))

    def log_reward(self, terminal, T):
        """LEGACY -- see root(). Depends on energy() and state_to_x(),
        both themselves flagged legacy/unused; the actual training path
        computes log-reward inline from energy_batch() instead (see
        TBGFlowNet.loss / KCGFlowNet.loss)."""
        return -self.energy(self.state_to_x(terminal)) / T

    def encode(self, state):
        """LEGACY -- see root()."""
        return torch.tensor(
            [0.0 if v is None else float(v) for v in state], dtype=torch.float32)

    def state_to_x(self, state):
        """LEGACY -- see root()."""
        return np.array([float(v) for v in state], dtype=np.float32)


# =====================================================
#  POLICY (NN only -- tabular is infeasible at D=25 grid,
#  and we want one consistent representation across all 3 models)
# =====================================================
class StepEmbedPolicy(nn.Module):
    """
    Batched, weight-shared-across-timesteps policy.


    Input per state: (masked spin vector [D], assignment mask [D],
    learned step embedding, and -- for grid models only -- normalized
    (row, col) of the site about to be chosen, a cheap partial nod
    toward giving the policy some sense of 2D locality without a full
    conv architecture). Output: 2 logits (for -1 / +1), regardless of D.
    """
    def __init__(self, D, n_actions=2, hidden=128, step_embed_dim=16, grid_N=None):
        super().__init__()
        self.D = D
        self.grid_N = grid_N
        extra = 2 if grid_N is not None else 0
        in_dim = D + D + step_embed_dim + extra
        self.step_embed = nn.Embedding(D, step_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_actions),
        )
        if grid_N is not None:
            coords = torch.tensor(
                [[r / grid_N, c / grid_N] for r in range(grid_N) for c in range(grid_N)],
                dtype=torch.float32)
            self.register_buffer("coords", coords)  # [D, 2]

    def forward(self, spins, mask, step_t):
        """spins, mask: [B, D] tensors (same device/dtype). step_t: python
        int -- same timestep for the whole batch, since sampling advances
        one position at a time for every trajectory in lockstep (that's
        what makes the batching possible in the first place)."""
        B = spins.shape[0]
        step_idx = torch.full((B,), step_t, dtype=torch.long, device=spins.device)
        feats = [spins, mask, self.step_embed(step_idx)]
        if self.grid_N is not None:
            feats.append(self.coords[step_t].unsqueeze(0).expand(B, -1))
        return self.net(torch.cat(feats, dim=-1))  # [B, n_actions]


# =====================================================
#  Batched GFlowNet base class (shared by TB and KC)
# =====================================================
class BatchedGFlowNetBase:
    """
    Shared machinery for batched, weight-tied-across-timesteps GFlowNet
    training: policy construction, the mixture-exploration trajectory
    sampler, and the WITH-grad replay used to compute log p_F and
    entropy for a fixed action sequence.

    TBGFlowNet and KCGFlowNet differ only in loss():
      - TB fits an explicit log_Z parameter against the reward.
      - KC eliminates log_Z via a telescoping identity between pairs of
        trajectories -- the per-trajectory residual log_pf - log_reward
        should be the SAME constant (that constant is log_Z, but it's
        never fit directly) for every trajectory at convergence, so KC's
        loss is just the variance of that residual across the batch,
        and needs no log_Z parameter at all.

    Fixes relative to the original per-trajectory implementation
    (applies to both TB and KC, since both build on this base class):

    1. BATCHING (the dominant cost at large D): sampling and log_pf/
       entropy both advance one position at a time, but process the
       WHOLE batch of trajectories at that position in a single forward
       call, instead of looping over trajectories with batch-size-1
       calls. Forward passes per training step go from O(B*D) to O(D).
    2. Beta no longer scales the policy logits (previously the same
       parameters meant a different effective policy at every point in
       training, and the reward term wasn't tempered by the same beta,
       a real mismatch in the original TB residual). Beta now only
       tempers the REWARD: logR = beta * env.log_reward(...).
    3. Entropy coefficient anneals linearly to zero instead of staying
       fixed for the whole run, so training actually converges to the
       true Boltzmann target rather than a permanently-entropy-
       regularized approximation of it.
    4. Exploration samples directly from the mixture
       q = (1-eps)*pi + eps/n_actions, rather than an if/else branch --
       same behavior, batchable, and log_pf still evaluates log pi_theta
       of whatever action was drawn (which is what makes off-policy
       exploration valid here in the first place -- no importance
       weighting needed, since both TB's and KC's fixed points don't
       depend on the behavior policy as long as it has full support).
    5. Reward uses env.energy_batch (vectorized) instead of one
       env.energy call per sample in a Python loop.
    """
    def __init__(self, env, T, device="cpu", init_state_dict=None,
                 entropy_coef0=ENTROPY_COEF, entropy_anneal_frac=0.8):
        self.env = env
        self.T = T
        self.device = device
        grid_N = env.N if env.kind in ("ising", "ea") else None
        self.policy = StepEmbedPolicy(env.D, grid_N=grid_N).to(device)
        if init_state_dict is not None:
            self.policy.load_state_dict(init_state_dict)
        self.entropy_coef0 = entropy_coef0
        self.entropy_anneal_frac = entropy_anneal_frac
        self.opt = None  # subclasses build this once all params are registered

    def state_dict(self):
        return self.policy.state_dict()

    def schedule(self, it, n_iter):
        frac = min(it / n_iter, 1.0)
        beta = BETA_START + frac * (BETA_END - BETA_START)
        eps = EPS_START + frac * (EPS_END - EPS_START)
        anneal_t = max(self.entropy_anneal_frac * n_iter, 1)
        ent_coef = self.entropy_coef0 * max(0.0, 1.0 - it / anneal_t)
        return beta, eps, ent_coef

    @torch.no_grad()
    def _sample_actions_batch(self, B, eps):
        """No-grad rollout: advance B trajectories together, one position
        at a time, sampling from the mixture q = (1-eps)*pi + eps/A.
        Returns the recorded actions [B, D] (the terminal spins can be
        cheaply replayed from these via _replay_batch, which is also
        where gradients get attached for the loss)."""
        D = self.env.D
        spins = torch.zeros(B, D, device=self.device)
        mask = torch.zeros(B, D, device=self.device)
        actions = torch.zeros(B, D, dtype=torch.long, device=self.device)
        for t in range(D):
            logits = self.policy(spins, mask, t)
            pi = torch.softmax(logits, dim=-1)
            q = (1.0 - eps) * pi + eps / pi.shape[-1]
            a = torch.multinomial(q, 1).squeeze(-1)
            actions[:, t] = a
            spins[:, t] = torch.where(a == 0, -1.0, 1.0)
            mask[:, t] = 1.0
        return actions

    def _replay_batch(self, actions):
        """WITH-grad replay of a fixed action sequence: recompute logits
        batched per-timestep (same cost shape as sampling), and return
        log_pf [B], mean entropy [B], and the resulting terminal spins
        [B, D] for the reward. Splitting sampling (no-grad) from replay
        (grad) like this mirrors the original code's separate log_pf/
        entropy_bonus passes, but each is now ONE batched pass over all D
        steps instead of D*B individual calls."""
        B, D = actions.shape
        spins = torch.zeros(B, D, device=self.device)
        mask = torch.zeros(B, D, device=self.device)
        log_pf = torch.zeros(B, device=self.device)
        ent = torch.zeros(B, device=self.device)
        for t in range(D):
            logits = self.policy(spins, mask, t)
            logp = torch.log_softmax(logits, dim=-1)
            a = actions[:, t]
            log_pf = log_pf + logp.gather(1, a.unsqueeze(1)).squeeze(1)
            p = logp.exp()
            ent = ent + (-(p * logp).sum(-1))
            spins[:, t] = torch.where(a == 0, -1.0, 1.0)
            mask[:, t] = 1.0
        return log_pf, ent / D, spins

    @torch.no_grad()
    def sample(self, n=500):
        actions = self._sample_actions_batch(n, eps=0.0)
        _, _, spins = self._replay_batch(actions)
        return [spins[i].cpu().numpy() for i in range(n)]

    def loss(self, it, n_iter, batch_size):
        raise NotImplementedError  # TB and KC differ only here

    def train(self, n_iter, batch_size_fn=None, stop_check=None,
              loss_log=None, log_every=25):
        """batch_size_fn(it, n_iter) -> int, if given, lets the batch size
        change over training (e.g. ramp up) -- see make_batch_size_schedule().
        stop_check, if given, is a zero-arg callable checked before every
        iteration; if it returns True, training stops immediately and this
        returns False (instead of the usual True) to signal an INCOMPLETE
        run. Used for graceful, prompt exit on SLURM preemption -- see the
        checkpointing logic in run_tc_sweep.py, which discards (rather than
        keeps) a T-point interrupted mid-training, so no partially-trained
        result is ever silently treated as final.

        loss_log: if given, a list that (iteration, loss_value) tuples are
        appended to every `log_every` iterations, plus iteration 1 and the
        final iteration always -- lets a caller record a loss-vs-iteration
        curve without logging every single step (which would bloat saved
        output for long runs). Left as None (no logging, no overhead) by
        default -- pass a list explicitly to opt in."""
        for it in range(1, n_iter + 1):
            if stop_check is not None and stop_check():
                return False
            B = batch_size_fn(it, n_iter) if batch_size_fn is not None else BATCH
            loss = self.loss(it, n_iter, B)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            if loss_log is not None and (it == 1 or it == n_iter or it % log_every == 0):
                loss_log.append((it, float(loss.item())))
        return True


class TBGFlowNet(BatchedGFlowNetBase):
    """
    Trajectory Balance: fits an explicit log_Z parameter so that
    log_Z + log_pf(tau) matches beta*log_reward(tau) for every tau.

    log_Z gets its OWN Adam param group at a much higher learning rate
    than the policy (lr_logz_mult x LR_NN, default 50x). Rationale: log_Z
    is a single scalar that has to track an EXTENSIVE quantity -- e.g.
    O(D) for SK at large D -- while Adam's per-parameter step size is
    roughly bounded by its lr regardless of gradient magnitude. At the
    policy's lr (tuned for a many-parameter network), log_Z simply cannot
    move far enough over a realistic number of iterations to track a
    large-D partition function; it isn't that TB is fundamentally unable
    to fit log_Z at scale, it's that it was starved of the step size to
    do so. This matters especially here because the paper's central claim
    is that KC's advantage comes from *not needing* log_Z -- so TB's
    log_Z needs a fair chance to converge, or a low-lr collapse gets
    mistaken for a fundamental scaling failure.

    log_Z is also warm-startable now (init_log_z), independent of the
    policy's own warm start, for the same reason the policy is warm-
    started: retraining log_Z from the fixed init D*log(2) at every T in
    a sweep -- even when the policy itself carries over -- was a second,
    compounding way TB was effectively handicapped relative to KC (which
    has no log_Z to reset).
    """
    def __init__(self, env, T, lr_logz_mult=LR_LOGZ_MULT, init_log_z=None, **kwargs):
        super().__init__(env, T, **kwargs)
        init_val = float(init_log_z) if init_log_z is not None else float(env.D * np.log(env.n_actions))
        self.log_Z = nn.Parameter(torch.tensor(init_val, device=self.device))
        self.opt = optim.Adam([
            {"params": self.policy.parameters(), "lr": LR_NN},
            {"params": [self.log_Z], "lr": LR_NN * lr_logz_mult},
        ])

    def loss(self, it, n_iter, batch_size):
        beta, eps, ent_coef = self.schedule(it, n_iter)
        actions = self._sample_actions_batch(batch_size, eps=eps)
        log_pf, ent, spins = self._replay_batch(actions)
        logR = beta * (-self.env.energy_batch(spins) / self.T)
        residual = self.log_Z + log_pf - logR - LOG_PB_FIXED
        return (residual ** 2).mean() - ent_coef * ent.mean()


class KCGFlowNet(BatchedGFlowNetBase):
    """
    Kolmogorov-Criterion GFlowNet: no log_Z parameter. The per-trajectory
    residual delta(tau) = log_pf(tau) - beta*log_reward(tau) should be
    the SAME constant across trajectories at convergence (that constant
    is log_Z, but it cancels via telescoping and is never fit
    explicitly) -- so the loss is just the spread of that residual
    across the batch.

    Uses the FULL BATCH variance rather than disjoint pairs: Var[delta]
    is proportional to the average over ALL pairwise squared
    differences, so it uses every sample in the batch instead of only
    B/2 arbitrary pairings, for a lower-variance loss estimate at the
    same batch size (this was flagged as an improvement over the
    original disjoint-pair implementation and is essentially free to
    adopt from the start here).
    """
    def __init__(self, env, T, **kwargs):
        super().__init__(env, T, **kwargs)
        self.opt = optim.Adam(self.policy.parameters(), lr=LR_NN)

    def loss(self, it, n_iter, batch_size):
        beta, eps, ent_coef = self.schedule(it, n_iter)
        actions = self._sample_actions_batch(batch_size, eps=eps)
        log_pf, ent, spins = self._replay_batch(actions)
        logR = beta * (-self.env.energy_batch(spins) / self.T)
        residual = log_pf - logR - LOG_PB_FIXED
        return residual.var(unbiased=False) - ent_coef * ent.mean()


def make_batch_size_schedule(batch_start, batch_end):
    """
    Linearly ramp the training batch size from batch_start to batch_end
    over the course of training. Rationale: early in training the policy
    is far from converged, so smaller, cheaper, noisier batches cover
    more distinct trajectories per unit of compute; later, as the policy
    approaches the target, a larger batch gives a lower-variance gradient
    estimate, which matters more once you're trying to nail down a
    precise final distribution than it does early on (the same tradeoff
    behind "increase batch size instead of decaying the learning rate").
    This is specifically affordable now (and wasn't really before) BECAUSE
    of the batching fix above: since a whole batch is processed in one
    forward pass per timestep, increasing batch size doesn't multiply the
    number of forward CALLS the way it would have in the old per-
    trajectory loop -- it just makes each call's tensor a bit wider,
    which is close to free until you're memory-bound.
    """
    def fn(it, n_iter):
        frac = min(it / n_iter, 1.0)
        return int(round(batch_start + frac * (batch_end - batch_start)))
    return fn


# =====================================================
#  MCMC baseline (single chain, burn-in + thinning)
# =====================================================
class MetropolisHastings:
    """
    Single-spin-flip Metropolis, now using the O(degree) local-delta
    energy instead of recomputing the FULL energy (O(pairs)) on every
    single proposed flip. For a D=100 grid this is a ~50x reduction in
    work per step (4 neighbors vs ~200 pairs), on top of just being the
    textbook-correct way to implement Metropolis. This alone makes the
    N=10 MCMC baseline dramatically more affordable, independent of
    anything on the GFlowNet side.
    """
    def __init__(self, env, T):
        self.env = env
        self.T = T
        self.x = np.random.choice([-1., 1.], size=env.D).astype(np.float32)
        self.n_accepted = 0
        self.n_total = 0

    def step(self):
        i = np.random.randint(self.env.D)
        dE = self.env.local_delta_energy(self.x, i)
        self.n_total += 1
        if dE <= 0 or np.random.random() < np.exp(-dE / self.T):
            self.x[i] *= -1
            self.n_accepted += 1

    def run_chain(self, warmup, n_samples, thin, track_energy=False):
        """track_energy: if True, records the FULL-PRECISION single-spin
        energy at every raw step (not just thinned samples) during the
        post-warmup sampling phase, for the autocorrelation-time
        diagnostic. Returns (samples, energy_trace) if track_energy else
        just samples (energy_trace is None in that case)."""
        for _ in range(warmup):
            self.step()
        samples = []
        energy_trace = [] if track_energy else None
        for _ in range(n_samples):
            for _ in range(thin):
                self.step()
                if track_energy:
                    energy_trace.append(self.env.energy_fast_np(self.x))
            samples.append(self.x.copy())
        return (samples, energy_trace) if track_energy else samples

    @property
    def acceptance_rate(self):
        return self.n_accepted / self.n_total if self.n_total > 0 else float("nan")


class WolffCluster:
    """
    Wolff single-cluster algorithm, for uniform-coupling ferromagnetic
    Ising ONLY (env.kind == "ising"). Grows a cluster of aligned spins
    probabilistically and flips it as a block, which sidesteps the
    critical slowing down that single-spin Metropolis suffers from near
    T_c -- exactly the concern raised about using Metropolis as the
    "ground truth" reference for validating an N=10 Ising sampler near
    its (finite-size-shifted) transition. NOT valid for EA/SK: the
    bond-dependent acceptance probability this relies on assumes a
    single uniform coupling strength, which frustrated/disordered models
    don't have.
    """
    def __init__(self, env, T):
        assert env.kind == "ising", "Wolff cluster is only implemented for uniform-J Ising"
        self.env = env
        self.T = T
        self.J = 1.0  # uniform coupling, matches IsingEnv/SpinEnv("ising")
        self.p_add = 1.0 - np.exp(-2.0 * self.J / T)
        self.x = np.random.choice([-1., 1.], size=env.D).astype(np.float32)
        self.n_flips = 0  # cluster flips are always "accepted" by construction,
        # so we track cluster SIZE distribution instead of an accept/reject rate
        self.cluster_sizes = []

    def step(self):
        D = self.env.D
        seed = np.random.randint(D)
        cluster = {seed}
        frontier = [seed]
        s0 = self.x[seed]
        while frontier:
            i = frontier.pop()
            for j, _Jij in self.env.neighbors[i]:
                if j not in cluster and self.x[j] == s0 and np.random.random() < self.p_add:
                    cluster.add(j)
                    frontier.append(j)
        for i in cluster:
            self.x[i] *= -1
        self.n_flips += 1
        self.cluster_sizes.append(len(cluster))

    def run_chain(self, warmup, n_samples, thin, track_energy=False):
        for _ in range(warmup):
            self.step()
        samples = []
        energy_trace = [] if track_energy else None
        for _ in range(n_samples):
            for _ in range(thin):
                self.step()
                if track_energy:
                    energy_trace.append(self.env.energy_fast_np(self.x))
            samples.append(self.x.copy())
        return (samples, energy_trace) if track_energy else samples

    @property
    def acceptance_rate(self):
        """Wolff cluster moves are always accepted (never rejected) by
        construction, so 'acceptance rate' isn't a meaningful diagnostic
        here the way it is for Metropolis -- mean fractional cluster size
        (cluster size / D) is reported instead, in run_mcmc's diagnostics."""
        return float("nan")

    @property
    def mean_cluster_fraction(self):
        return float(np.mean(self.cluster_sizes)) / self.env.D if self.cluster_sizes else float("nan")


def run_mcmc(env, T, n_eval, warmup=MCMC_WARMUP, thin=MCMC_THIN, method="auto",
             return_diagnostics=False):
    """
    method: "auto" uses Wolff for Ising (recommended -- avoids critical
    slowing down near T_c) and Metropolis for EA/SK (Wolff isn't valid
    for disordered/frustrated couplings). "metropolis" or "wolff" force
    a specific sampler; forcing "wolff" on a non-Ising env raises.
    Wolff's warmup/thin are already in units of whole-cluster flips, not
    single-spin flips, so far fewer of them are needed for equivalent
    decorrelation -- if you force "wolff", consider much smaller
    --mcmc_warmup/--mcmc_thin than the single-spin defaults.

    return_diagnostics: if True, returns (samples, diagnostics) where
    diagnostics is a dict with:
      - 'acceptance_rate': fraction of proposed single-spin flips
        accepted (Metropolis only; NaN for Wolff, which always accepts
        by construction -- see 'mean_cluster_fraction' instead)
      - 'mean_cluster_fraction': mean cluster size / D per flip (Wolff
        only; NaN for Metropolis)
      - 'tau_int': integrated autocorrelation time of the single-spin-
        chain energy trace during the sampling phase (see
        integrated_autocorr_time docstring) -- large tau_int relative
        to --mcmc_thin is a sign the chain hasn't decorrelated between
        recorded samples (i.e. --mcmc_thin is too small, or the chain
        is stuck/mixing poorly, e.g. from basin-trapping at low T).
    """
    if method == "auto":
        method = "wolff" if env.kind == "ising" else "metropolis"
    cls = {"metropolis": MetropolisHastings, "wolff": WolffCluster}[method]
    chain = cls(env, T)

    if not return_diagnostics:
        return chain.run_chain(warmup, n_eval, thin)

    samples, energy_trace = chain.run_chain(warmup, n_eval, thin, track_energy=True)
    diagnostics = {
        "acceptance_rate": chain.acceptance_rate,
        "mean_cluster_fraction": (chain.mean_cluster_fraction
                                   if isinstance(chain, WolffCluster) else float("nan")),
        "tau_int": integrated_autocorr_time(energy_trace),
        "method": method,
    }
    return samples, diagnostics


# =====================================================
#  Observables / susceptibility
# =====================================================
def magnetization_susceptibility(env, samples, T):
    """
    Standard ferromagnetic susceptibility (Ising):
      chi = D * (<m^2> - <|m|>^2) / T
    """
    X = np.asarray(samples, dtype=np.float32)
    m = X.mean(axis=1)
    absm = np.abs(m)
    chi = env.D * (float((m ** 2).mean()) - float(absm.mean()) ** 2) / T
    return chi, float(absm.mean())


def spin_glass_overlap_moments(env, samples, n_pairs=None):
    """
    Two-replica overlap trick: split independent equilibrium samples into
    disjoint pairs (a, b) from the SAME disorder realization, compute
    q_ab = (1/D) sum_i s_i^a s_i^b, and return its raw overlap values
    plus 2nd/4th moments. This needs no ground truth / enumeration --
    only i.i.d. equilibrium samples, which is exactly what both MCMC and
    a trained GFlowNet give.

    Used for both:
      - chi_SG = D * <q^2>                      (spin-glass susceptibility)
      - U4     = 1 - <q^4> / (3 <q^2>^2)         (Binder cumulant)

    chi_SG saturates rather than peaking as T->0 at small D (limited by
    ground-state degeneracy), so it does NOT have a clean turnover point
    to extract T_c from at N=5 -- see binder_cumulant_sg() and the
    crossover_from_binder() estimator in run_tc_sweep.py, which is the
    standard, more robust alternative used in spin-glass simulations.
    """
    X = np.asarray(samples, dtype=np.float32)
    n = len(X)
    if n_pairs is None:
        n_pairs = n // 2
    idx = np.random.permutation(n)
    qs = np.array([float(np.mean(X[idx[2 * k]] * X[idx[2 * k + 1]]))
                   for k in range(n_pairs)])
    q2 = float(np.mean(qs ** 2))
    q4 = float(np.mean(qs ** 4))
    return qs, q2, q4


def spin_glass_susceptibility(env, samples, T, n_pairs=None):
    """chi_SG = D * <q^2>. See spin_glass_overlap_moments() for caveats."""
    qs, q2, q4 = spin_glass_overlap_moments(env, samples, n_pairs)
    chi_sg = env.D * q2
    return chi_sg, float(np.mean(np.abs(qs)))


def binder_cumulant_sg(env, samples, n_pairs=None):
    """
    Binder cumulant of the overlap distribution:
      U4 = 1 - <q^4> / (3 <q^2>^2)
    U4 -> 0 for a Gaussian (paramagnetic, disordered) overlap distribution
    and U4 -> 2/3 for a fully bimodal q=+-1 distribution (frozen, near-
    degenerate ground states). It's dimensionless and monotonic across
    the crossover, so unlike chi_SG it doesn't need a peak to exist --
    the standard indicator for locating a spin-glass transition/crossover
    when true finite-size scaling (multiple N, looking for where curves
    cross) isn't being done.
    """
    qs, q2, q4 = spin_glass_overlap_moments(env, samples, n_pairs)
    U4 = 1.0 - q4 / (3.0 * q2 ** 2 + 1e-12)
    return U4


def susceptibility(env, samples, T):
    if env.kind == "ising":
        return magnetization_susceptibility(env, samples, T)
    else:  # ea, sk
        return spin_glass_susceptibility(env, samples, T)


def order_parameter_samples_for_hist(env, samples):
    """LEGACY -- appears unused. Return a 1D array to histogram: m for
    Ising, pairwise overlap q for EA/SK. make_plots.py's
    cmd_distributions computes the same quantity with its own inline
    code rather than calling this -- its own comment says "pairwise
    overlap via disjoint-pair trick, matching spin_glass_lib", i.e. it
    was written to replicate this function's logic, not call it. Kept
    here, unmodified, rather than removed."""
    if env.kind == "ising":
        X = np.asarray(samples, dtype=np.float32)
        return X.mean(axis=1)
    else:
        qs, _, _ = spin_glass_overlap_moments(env, samples)
        return qs