"""DISCOVER-style directed goal selection for HySR data collection.

Implements the "Achievability + Novelty" variant of DISCOVER
(Diaz-Bone et al., "DISCOVER: Automated Curricula for Sparse-Reward
Reinforcement Learning", NeurIPS 2025) as a data-collection baseline for
table tennis: instead of sampling the commanded landing target uniformly
on the opponent's table half (the "baseline" exploration policy), the
target is selected each episode by scoring candidate goals with an
ensemble of value networks:

    score(g) = w_ach * standardize(V_mean(s0, g))
             + w_nov * standardize(V_std(s0, g))

The relevance term of full DISCOVER is dropped: there is no single target
goal g* in this setting (the objective is coverage of all achievable
landing points), and episodes terminate when the ball lands, so
goal-space chaining V(g, g*) is undefined. "Achievability + Novelty" is a
named ablation in the DISCOVER paper (their baseline 4, Figure 3).

Mirrors the official implementation
(https://github.com/LeanderDiazBone/discover, src/baselines/td3/td3_train.py):

* candidate goals are drawn from previously *achieved* goals (their
  G_ach); here: landing positions achieved during data collection,
* each UCB term is min-max standardized over the candidate batch
  ((x - min) / (max - min + 0.01), cf. ``standardize``),
* selection is a hard argmax over a freshly sampled candidate batch,
* the achievability weight is adapted online with a deadband of +-0.2
  around a target achievement rate of 0.5, step = midpoint /
  adaptation_rate, while the novelty weight stays fixed at 1
  (cf. ``adapt_ucb_params``, strategy "simple").

A MEGA-style mode (Pitis et al., ICML 2020; also a baseline in DISCOVER)
is included as well: the commanded goal is the candidate with the lowest
KDE density of achieved landings (cf. their strategy "MEGA").

The value ensemble is trained on (observation, return) pairs from the PPO
rollout buffer via ``DiscoverCallback`` (same integration pattern as
``RLeXploreWithOnPolicyRL``). Observations are fed raw, exactly as PPO
sees them; the commanded goal is part of the observation vector, so
V(s0, g) is obtained by patching the goal slice of the reset observation.
"""

import json
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback


class DiscoverConfig:
    """Configuration for DISCOVER goal selection, loadable from json."""

    DEFAULTS = {
        # "discover": achievability + novelty UCB; "mega": lowest-density achieved goal
        "strategy": "discover",
        # --- value ensemble ---
        "ensemble_size": 4,
        "ensemble_lr": 3e-4,
        "ensemble_epochs": 3,
        "ensemble_batch_size": 256,
        # fraction of the fit data each head sees (bootstrap, for diversity)
        "ensemble_bootstrap_fraction": 0.8,
        # randomized prior networks (Osband et al.): each head predicts
        # trainable(x) + scale * frozen_prior(x), giving persistent epistemic
        # diversity that per-round bootstrap resampling cannot provide.
        # 0 disables the priors.
        "ensemble_prior_scale": 1.0,
        # "episode_starts": fit only on episode-start observations with the
        # episode return as target -- the literal V(s0, g) DISCOVER queries,
        # avoiding the mid-episode/reset-state distribution shift.
        # "all": legacy, fit on all rollout timesteps.
        "fit_on": "episode_starts",
        # ring buffer of past fit samples (0: current rollout only)
        "fit_replay_size": 20000,
        "min_fit_samples": 64,
        "device": "cpu",
        # --- candidate goals ---
        "n_candidates": 500,
        # "achieved": sample from achieved landing positions (DISCOVER's G_ach)
        # "uniform": sample uniformly on the opponent's table half
        # "mixed": both, with uniform_candidate_fraction uniform
        "candidate_source": "mixed",
        "uniform_candidate_fraction": 0.2,
        # xy gaussian jitter (m) applied to achieved candidates (finite pool)
        "candidate_jitter": 0.05,
        "max_pool_size": 20000,
        # minimum achieved goals before non-uniform selection starts
        "min_pool_size": 10,
        # --- selection ---
        "selection": "argmax",  # or "softmax"
        "softmax_temperature": 1.0,
        "standardize_scores": True,
        "w_achievability_init": 0.5,
        "w_novelty": 1.0,
        # --- online adaptation of the achievability weight ---
        # "continuous": paper form (DISCOVER Eq. 4), w += lr * (p* - p_t),
        # no deadband, so the weight regulates instead of railing at the cap.
        # "deadband": legacy variant mirroring the official implementation
        # (adapt_ucb_params "simple": +-deadband around p*, fixed step).
        "adapt_achievability": True,
        "adaptation_mode": "continuous",
        "adaptation_lr": 0.01,
        "target_achievement_rate": 0.5,
        "adaptation_deadband": 0.2,
        "adaptation_rate": 100,
        "achievement_window": 20,
        # commanded goal counts as achieved if the achievement distance <= eps
        "achievement_eps": 0.7,
        # additional thresholds tracked for logging/calibration only
        "achievement_eps_log": [0.3, 0.5, 0.7, 1.0, 1.3],
        # "landing": achieved only if the main ball's projected landing is on
        # the goal region within eps of the commanded goal (recommended; this
        # is the quantity that predicts offline GCSL performance).
        # "min_distance": legacy slab-crossing distance, which cannot
        # distinguish an on-table landing from an overshoot past the table.
        "achievement_metric": "landing",
        # if true, only ball-hit episodes enter the achievement statistics
        # (whiffed balls say nothing about the goal choice). false = faithful
        # to the official DISCOVER implementation, which counts all episodes.
        "achievement_hit_episodes_only": False,
        # "all_balls": the achieved-goal pool is fed by the projected
        # landings of ALL hit balls (main + extras), matching DISCOVER's
        # G_ach sampled from the full replay buffer (~20x faster pool
        # growth). "main_ball": legacy, main ball's closest-approach point.
        "pool_source": "all_balls",
        # --- warmup: uniform goal sampling (= existing baseline behavior) ---
        "warmup_episodes": 50,
        "seed": None,
    }

    def __init__(self, **kwargs):
        unknown = set(kwargs) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(
                "unknown DiscoverConfig keys: {}".format(sorted(unknown))
            )
        for key, default in self.DEFAULTS.items():
            setattr(self, key, kwargs.get(key, default))

    @classmethod
    def from_json(cls, jsonpath=None):
        if jsonpath is None:
            return cls()
        if not os.path.isfile(jsonpath):
            raise FileNotFoundError(
                "failed to find discover configuration file: {}".format(jsonpath)
            )
        with open(jsonpath) as f:
            conf = json.load(f)
        return cls(**conf)


def _minmax_standardize(array):
    # cf. official implementation: (x - min) / (max - min + 0.01)
    array = array - np.min(array)
    return array / (np.max(array) + 0.01)


class ValueEnsemble(nn.Module):
    """Ensemble of small value MLPs, architecture mirroring the PPO critic.

    With ``use_layer_norm`` the per-layer structure is
    Linear(bias=False) -> LayerNorm -> ReLU, matching
    ``LayerNormFeaturesExtractor``; otherwise Linear -> ReLU.
    Trained by regression on returns; diversity comes from independent
    initialization and per-head bootstrap resampling.
    """

    def __init__(
        self,
        obs_dim,
        num_hidden=380,
        num_layers=1,
        ensemble_size=4,
        use_layer_norm=False,
        hidden_layers_bias=True,
        lr=3e-4,
        prior_scale=1.0,
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.prior_scale = prior_scale
        self.heads = nn.ModuleList(
            [
                self._make_head(
                    obs_dim, num_hidden, num_layers, use_layer_norm, hidden_layers_bias
                )
                for _ in range(ensemble_size)
            ]
        )
        # frozen randomized prior networks (Osband et al., 2018): persistent
        # per-head diversity independent of the (shared) training data
        if prior_scale > 0:
            self.priors = nn.ModuleList(
                [
                    self._make_head(
                        obs_dim,
                        num_hidden,
                        num_layers,
                        use_layer_norm,
                        hidden_layers_bias,
                    )
                    for _ in range(ensemble_size)
                ]
            )
            for prior in self.priors:
                for param in prior.parameters():
                    param.requires_grad_(False)
        else:
            self.priors = None
        self.to(self.device)
        self.optimizers = [
            torch.optim.Adam(head.parameters(), lr=lr) for head in self.heads
        ]

    def _head_output(self, index, obs_t):
        out = self.heads[index](obs_t).squeeze(-1)
        if self.priors is not None:
            with torch.no_grad():
                prior_out = self.priors[index](obs_t).squeeze(-1)
            out = out + self.prior_scale * prior_out
        return out

    @staticmethod
    def _make_head(obs_dim, num_hidden, num_layers, use_layer_norm, hidden_layers_bias):
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, num_hidden, bias=hidden_layers_bias))
            if use_layer_norm:
                layers.append(nn.LayerNorm(num_hidden))
            layers.append(nn.ReLU())
            in_dim = num_hidden
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def predict(self, obs):
        """Returns (mean, std) over ensemble heads, both shape (n,)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        values = torch.stack(
            [self._head_output(i, obs_t) for i in range(len(self.heads))]
        )
        return (
            values.mean(dim=0).cpu().numpy(),
            values.std(dim=0).cpu().numpy(),
        )

    def fit(self, obs, targets, epochs, batch_size, bootstrap_fraction, rng):
        """One fitting round on (obs, targets); returns mean MSE loss.

        With priors enabled, each trainable head learns the residual
        target - prior, so heads keep disagreeing where data is scarce.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        targets_t = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        n = len(obs_t)
        n_boot = max(1, int(n * bootstrap_fraction))
        losses = []
        for i, (head, optimizer) in enumerate(zip(self.heads, self.optimizers)):
            boot_idx = torch.as_tensor(
                rng.choice(n, size=n_boot, replace=True), device=self.device
            )
            for _ in range(epochs):
                perm = boot_idx[torch.randperm(n_boot, device=self.device)]
                for start in range(0, n_boot, batch_size):
                    idx = perm[start : start + batch_size]
                    pred = head(obs_t[idx]).squeeze(-1)
                    if self.priors is not None:
                        with torch.no_grad():
                            prior_out = self.priors[i](obs_t[idx]).squeeze(-1)
                        pred = pred + self.prior_scale * prior_out
                    loss = nn.functional.mse_loss(pred, targets_t[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")


class DiscoverGoalSelector:
    """Selects the commanded landing goal for each episode.

    The selector is owned by ``HysrDiscoverEnv`` (which calls
    ``select_goal`` at reset and ``report_outcome`` with the previous
    episode's result) while ``DiscoverCallback`` builds and trains the
    value ensemble from the PPO rollout buffer.
    """

    def __init__(self, obs_dim, goal_start_idx, goal_bounds, goal_z, config):
        """
        :param obs_dim: length of the flat observation vector
        :param goal_start_idx: start index of the 3d goal slice in the observation
        :param goal_bounds: ((x_min, x_max), (y_min, y_max)) of the goal region
            (opponent's table half, same region as HysrOneBall.sample_goal)
        :param goal_z: fixed z coordinate of goals
        :param config: DiscoverConfig
        """
        self.config = config
        self.obs_dim = obs_dim
        self.goal_start_idx = goal_start_idx
        self.goal_bounds = goal_bounds
        self.goal_z = goal_z

        self.ensemble = None  # built via build_ensemble() once arch is known
        # rng is created lazily on first use: with seed=null it derives from
        # numpy's global rng, which the run seed has initialized by the time
        # the first episode starts (but not yet during env construction)
        self._rng_obj = None

        self.achieved_pool = []  # achieved landing positions (G_ach), np arrays (3,)
        self.w_achievability = config.w_achievability_init
        # midpoint/step as in official adapt_ucb_params ("simple" strategy,
        # ensemble-exploration branch): mid = (w_ach_init0 + w_nov) / 2
        self._w_mid = (0.0 + config.w_novelty) / 2.0
        self._w_step = self._w_mid / config.adaptation_rate

        self._recent_by_eps = {
            float(eps): deque(maxlen=config.achievement_window)
            for eps in set(list(config.achievement_eps_log) + [config.achievement_eps])
        }
        self.n_episodes_selected = 0
        self.n_outcomes = 0
        self._last_stats = {}

    @property
    def _rng(self):
        if self._rng_obj is None:
            seed = self.config.seed
            # null or a negative value: derive from the (already seeded)
            # global numpy rng, i.e. from the run seed. (negative sentinel
            # exists because cluster_utils grids cannot express null)
            if seed is None or (isinstance(seed, (int, float)) and seed < 0):
                seed = int(np.random.randint(0, 2**31 - 1))
            self._rng_obj = np.random.default_rng(int(seed))
        return self._rng_obj

    # ------------------------------------------------------------------
    def build_ensemble(
        self,
        num_hidden=380,
        num_layers=1,
        use_layer_norm=False,
        hidden_layers_bias=True,
    ):
        """Idempotent; called by DiscoverCallback once the PPO arch is known."""
        if self.ensemble is None:
            self.ensemble = ValueEnsemble(
                self.obs_dim,
                num_hidden=num_hidden,
                num_layers=num_layers,
                ensemble_size=self.config.ensemble_size,
                use_layer_norm=use_layer_norm,
                hidden_layers_bias=hidden_layers_bias,
                lr=self.config.ensemble_lr,
                prior_scale=self.config.ensemble_prior_scale,
                device=self.config.device,
            )
        return self.ensemble

    # ------------------------------------------------------------------
    # goal selection
    def select_goal(self, reset_obs):
        """Returns the commanded goal (np array (3,)) for the new episode."""
        self.n_episodes_selected += 1
        if (
            self.n_episodes_selected <= self.config.warmup_episodes
            or self.ensemble is None
            or len(self.achieved_pool) < self.config.min_pool_size
        ):
            goal = self._uniform_goal()
            self._last_stats = {"selection_uniform": 1.0}
            return goal

        candidates = self._sample_candidates()
        if self.config.strategy == "mega":
            scores = self._score_mega(candidates)
        else:
            scores = self._score_discover(np.asarray(reset_obs), candidates)
        goal = candidates[self._pick(scores)]
        self._last_stats["selection_uniform"] = 0.0
        return goal

    def _uniform_goal(self):
        (x_min, x_max), (y_min, y_max) = self.goal_bounds
        return np.array(
            [
                self._rng.uniform(x_min, x_max),
                self._rng.uniform(y_min, y_max),
                self.goal_z,
            ]
        )

    def _sample_candidates(self):
        cfg = self.config
        n = cfg.n_candidates
        if cfg.candidate_source == "uniform":
            n_uniform = n
        elif cfg.candidate_source == "achieved":
            n_uniform = 0
        else:  # mixed
            n_uniform = int(n * cfg.uniform_candidate_fraction)
        n_achieved = n - n_uniform

        parts = []
        if n_achieved > 0:
            pool = np.asarray(self.achieved_pool)
            idx = self._rng.integers(0, len(pool), size=n_achieved)
            achieved = pool[idx].copy()
            if cfg.candidate_jitter > 0:
                achieved[:, :2] += self._rng.normal(
                    scale=cfg.candidate_jitter, size=(n_achieved, 2)
                )
            parts.append(achieved)
        if n_uniform > 0:
            parts.append(np.stack([self._uniform_goal() for _ in range(n_uniform)]))

        candidates = np.concatenate(parts, axis=0)
        (x_min, x_max), (y_min, y_max) = self.goal_bounds
        candidates[:, 0] = np.clip(candidates[:, 0], x_min, x_max)
        candidates[:, 1] = np.clip(candidates[:, 1], y_min, y_max)
        candidates[:, 2] = self.goal_z
        return candidates

    def _score_discover(self, reset_obs, candidates):
        cfg = self.config
        batch = np.tile(reset_obs, (len(candidates), 1))
        batch[:, self.goal_start_idx : self.goal_start_idx + 3] = candidates
        mean, std = self.ensemble.predict(batch)
        self._last_stats = {
            "value_mean": float(np.mean(mean)),
            "value_std": float(np.mean(std)),
        }
        if cfg.standardize_scores:
            mean = _minmax_standardize(mean)
            std = _minmax_standardize(std)
        return self.w_achievability * mean + cfg.w_novelty * std

    def _score_mega(self, candidates):
        # lowest KDE density of achieved landings (cf. official strategy "MEGA")
        from scipy.stats import gaussian_kde

        pool = np.asarray(self.achieved_pool)[:, :2]
        mean = pool.mean(axis=0)
        std = pool.std(axis=0) + 1e-6
        try:
            kde = gaussian_kde(((pool - mean) / std).T)
            log_density = kde.logpdf(((candidates[:, :2] - mean) / std).T)
        except np.linalg.LinAlgError:
            # degenerate pool (e.g. collinear landings): fall back to uniform scores
            log_density = self._rng.uniform(size=len(candidates))
            return -log_density
        self._last_stats = {"mega_log_density_mean": float(np.mean(log_density))}
        return -log_density

    def _pick(self, scores):
        if self.config.selection == "softmax":
            logits = scores / self.config.softmax_temperature
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            return self._rng.choice(len(scores), p=probs)
        return int(np.argmax(scores))

    # ------------------------------------------------------------------
    # outcome reporting and adaptation
    def _in_goal_region(self, position, margin=0.1):
        (x_min, x_max), (y_min, y_max) = self.goal_bounds
        return (
            x_min - margin <= position[0] <= x_max + margin
            and y_min - margin <= position[1] <= y_max + margin
        )

    def report_outcome(
        self, commanded_goal, ball_hit, landing_position, min_distance_ball_target=None
    ):
        """Report the previous episode's main-ball outcome.

        :param commanded_goal: goal that was commanded for the episode
        :param ball_hit: whether the racket made contact with the main ball
        :param landing_position: projected landing point of the main ball
            (ball_landing_data), None if unavailable
        :param min_distance_ball_target: legacy slab-crossing distance to the
            commanded goal (may be +inf if the ball never crossed the table
            plane near target height); used for achievement_metric
            "min_distance" only

        Note: this only tracks achievement and drives the adaptation. Pool
        feeding is separate (``report_landings``).
        """
        self.n_outcomes += 1

        distance = float("inf")
        if ball_hit:
            if self.config.achievement_metric == "landing":
                if landing_position is not None and self._in_goal_region(
                    landing_position
                ):
                    distance = float(
                        np.hypot(
                            landing_position[0] - commanded_goal[0],
                            landing_position[1] - commanded_goal[1],
                        )
                    )
            else:  # "min_distance" (legacy)
                if min_distance_ball_target is not None and np.isfinite(
                    min_distance_ball_target
                ):
                    distance = float(min_distance_ball_target)

        if ball_hit or not self.config.achievement_hit_episodes_only:
            for eps, window in self._recent_by_eps.items():
                window.append(distance <= eps)

        if self.config.adapt_achievability:
            self._adapt()

    def report_landings(self, positions):
        """Feed achieved landings (any balls, hit episodes only) to the pool."""
        for position in positions:
            if position is not None:
                self._add_to_pool(np.asarray(position, dtype=float))

    def _add_to_pool(self, position):
        # only positions on (or very near) the goal region qualify as
        # achieved goals; off-table landings must not become candidates
        (x_min, x_max), (y_min, y_max) = self.goal_bounds
        margin = 0.1
        if not (
            x_min - margin <= position[0] <= x_max + margin
            and y_min - margin <= position[1] <= y_max + margin
        ):
            return
        position = position.copy()
        position[0] = np.clip(position[0], x_min, x_max)
        position[1] = np.clip(position[1], y_min, y_max)
        position[2] = self.goal_z
        if len(self.achieved_pool) >= self.config.max_pool_size:
            # replace a random old entry (reservoir-style)
            self.achieved_pool[
                int(self._rng.integers(0, len(self.achieved_pool)))
            ] = position
        else:
            self.achieved_pool.append(position)

    def _adapt(self):
        cfg = self.config
        window = self._recent_by_eps[float(cfg.achievement_eps)]
        if len(window) < window.maxlen:
            return
        rate = float(np.mean(window))
        if cfg.adaptation_mode == "continuous":
            # paper form (DISCOVER Eq. 4): under-achieving -> more weight on
            # achievability (easier goals), over-achieving -> less. No
            # deadband, so the weight regulates around the target instead of
            # railing at the cap after an early low-achievement phase.
            self.w_achievability += cfg.adaptation_lr * (
                cfg.target_achievement_rate - rate
            )
        else:  # "deadband" (legacy, mirrors official adapt_ucb_params "simple")
            if rate > cfg.target_achievement_rate + cfg.adaptation_deadband:
                self.w_achievability -= self._w_step
            elif rate < cfg.target_achievement_rate - cfg.adaptation_deadband:
                self.w_achievability += self._w_step
        self.w_achievability = float(
            np.clip(self.w_achievability, 0.0, 2.0 * self._w_mid)
        )

    # ------------------------------------------------------------------
    def get_log_dict(self):
        log = {
            "w_achievability": self.w_achievability,
            "pool_size": float(len(self.achieved_pool)),
            "episodes_selected": float(self.n_episodes_selected),
        }
        for eps, window in sorted(self._recent_by_eps.items()):
            if window:
                log["achievement_rate_eps_{:g}".format(eps)] = float(np.mean(window))
        log.update(self._last_stats)
        return log


class DiscoverCallback(BaseCallback):
    """Trains the value ensemble on PPO rollout data and logs diagnostics.

    Same integration pattern as ``RLeXploreWithOnPolicyRL``: attached via
    ``model.learn(callback=...)``, acts at ``_on_rollout_end``. Fits the
    ensemble by regression on (observation, return) pairs; a small ring
    buffer of past rollouts stabilizes the fit (PPO rollouts are short).
    """

    def __init__(self, selector, ensemble_arch=None, verbose=0):
        super().__init__(verbose)
        self.selector = selector
        self.ensemble_arch = ensemble_arch or {}
        self._replay_obs = None
        self._replay_returns = None

    def _init_callback(self):
        self.selector.build_ensemble(**self.ensemble_arch)

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer
        obs = np.asarray(buffer.observations).reshape(-1, self.selector.obs_dim)
        returns = np.asarray(buffer.returns).reshape(-1)

        if self.selector.config.fit_on == "episode_starts":
            # fit only on (s0, episode return) pairs: the literal V(s0, g)
            # that goal selection queries, avoiding the mid-episode/reset
            # distribution shift
            starts = np.asarray(buffer.episode_starts).reshape(-1).astype(bool)
            if starts.any():
                obs = obs[starts]
                returns = returns[starts]

        replay_size = self.selector.config.fit_replay_size
        if replay_size > 0:
            if self._replay_obs is None:
                self._replay_obs = obs
                self._replay_returns = returns
            else:
                self._replay_obs = np.concatenate([self._replay_obs, obs])[
                    -replay_size:
                ]
                self._replay_returns = np.concatenate(
                    [self._replay_returns, returns]
                )[-replay_size:]
            fit_obs, fit_returns = self._replay_obs, self._replay_returns
        else:
            fit_obs, fit_returns = obs, returns

        if len(fit_obs) < self.selector.config.min_fit_samples:
            return

        loss = self.selector.ensemble.fit(
            fit_obs,
            fit_returns,
            epochs=self.selector.config.ensemble_epochs,
            batch_size=self.selector.config.ensemble_batch_size,
            bootstrap_fraction=self.selector.config.ensemble_bootstrap_fraction,
            rng=self.selector._rng,
        )

        self.logger.record("discover/ensemble_loss", loss)
        for key, value in self.selector.get_log_dict().items():
            self.logger.record("discover/{}".format(key), value)
