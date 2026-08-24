"""HysrManyBallEnv with DISCOVER-style commanded-goal selection.

Identical to the baseline exploration setup (goal-conditioned policy,
dense distance-to-target reward, multi-ball HySR) except that the
commanded landing target is selected by ``DiscoverGoalSelector`` instead
of uniform sampling. Everything downstream (episode logging for offline
GCSL, rewards, multi-ball handling) is inherited unchanged.
"""

import json
import os

import numpy as np

from .compat import USE_GYMNASIUM, get_obs_array
from .discover import DiscoverConfig, DiscoverGoalSelector
from .hysr_many_ball_env import HysrManyBallEnv

# table half-width / half-length as in HysrOneBall.sample_goal
# (radius < 0 branch) and HysrManyBallEnv.dump_data
_TABLE_HALF_WIDTH = 0.7625
_TABLE_HALF_LENGTH = 1.37
_NET_MARGIN = 0.1


class HysrDiscoverEnv(HysrManyBallEnv):
    def __init__(
        self,
        reward_config_file=None,
        hysr_one_ball_config_file=None,
        log_episodes=False,
        logger=None,
        stop_new_actions_after_main_ball_hit=True,
        dict_obs=False,
        discover_config_file=None,
    ):
        super().__init__(
            reward_config_file=reward_config_file,
            hysr_one_ball_config_file=hysr_one_ball_config_file,
            log_episodes=log_episodes,
            logger=logger,
            stop_new_actions_after_main_ball_hit=stop_new_actions_after_main_ball_hit,
            dict_obs=dict_obs,
        )

        if not self._goal_in_state:
            raise ValueError(
                "HysrDiscoverEnv requires the goal in the observation "
                "(target_position_sampling_radius != 0 in the hysr config)"
            )

        config = DiscoverConfig.from_json(self._resolve_config_file(discover_config_file))

        table_x, table_y = (
            self._hysr._hysr_config.table_position[0],
            self._hysr._hysr_config.table_position[1],
        )
        goal_bounds = (
            (table_x - _TABLE_HALF_WIDTH, table_x + _TABLE_HALF_WIDTH),
            (table_y + _NET_MARGIN, table_y + _TABLE_HALF_LENGTH),
        )
        goal_z = self._hysr._target_position[2]

        obs_space = (
            self.observation_space["observation"]
            if self._dict_obs
            else self.observation_space
        )
        self._goal_start_idx = self._obs_boxes.get_start_index("goal")

        self.discover_selector = DiscoverGoalSelector(
            obs_dim=obs_space.shape[0],
            goal_start_idx=self._goal_start_idx,
            goal_bounds=goal_bounds,
            goal_z=goal_z,
            config=config,
        )
        self._last_reported_episode = -1
        print(
            "---HysrDiscoverEnv: strategy={}, goal bounds={}, z={:.3f}---".format(
                config.strategy, goal_bounds, goal_z
            )
        )

    @staticmethod
    def _resolve_config_file(discover_config_file):
        """Explicit kwarg > "discover_config" key in ./config.json >
        ./config/discover_default.json > package defaults (None)."""
        if discover_config_file:
            return discover_config_file
        if os.path.isfile("config.json"):
            with open("config.json") as f:
                path = json.load(f).get("discover_config")
            if path:
                return path
        default = os.path.join("config", "discover_default.json")
        if os.path.isfile(default):
            return default
        return None

    def reset(self, *, seed=None, options=None):
        self._report_previous_outcome()

        ret = super().reset(seed=seed, options=options)
        obs = ret[0] if USE_GYMNASIUM else ret
        obs_array = get_obs_array(obs)

        goal = self.discover_selector.select_goal(obs_array.copy())
        self.set_goal([float(v) for v in goal])

        # super().reset() built the observation with the uniformly sampled
        # goal; patch the returned observation and the cached copies
        start = self._goal_start_idx
        obs_array[start : start + 3] = goal
        get_obs_array(self.previous_obs)[start : start + 3] = goal
        for extra_obs in self.previous_extra_obs:
            get_obs_array(extra_obs)[start : start + 3] = goal
        self._obs_boxes.set_values_non_norm("goal", goal)

        return ret

    def _report_previous_outcome(self):
        # report each completed episode exactly once (reset may also be
        # called initially or after an aborted episode)
        if self.n_eps == 0 or self.n_eps == self._last_reported_episode:
            return
        self._last_reported_episode = self.n_eps

        ball_status = self._hysr._ball_status
        landing_data = getattr(self._hysr, "ball_landing_data", {}) or {}

        # main ball: achievement tracking + adaptation.
        # BallStatus semantics: min_distance_ball_racket is None iff the
        # racket made contact; min_distance_ball_target is +inf (not None)
        # when the ball never crossed the table plane near target height.
        main = landing_data.get(0, {})
        main_hit = ball_status.min_distance_ball_racket is None
        main_landing = main.get("landing_position")
        self.discover_selector.report_outcome(
            np.asarray(ball_status.target_position, dtype=float),
            ball_hit=main_hit,
            landing_position=np.asarray(main_landing, dtype=float)
            if main_landing is not None
            else None,
            min_distance_ball_target=ball_status.min_distance_ball_target,
        )

        # achieved-goal pool feeding
        if self.discover_selector.config.pool_source == "all_balls":
            # projected landings of every ball whose episode involved a
            # racket hit (min_distance_ball_racket None <=> contact);
            # unhit replayed balls also get a landing projection and must
            # be excluded (they are incoming-trajectory landings)
            landings = [
                data["landing_position"]
                for data in landing_data.values()
                if data.get("landing_position") is not None
                and data.get("min_distance_ball_racket") is None
            ]
            self.discover_selector.report_landings(landings)
        else:  # "main_ball" (legacy): closest-approach point of the main ball
            if main_hit and ball_status.min_position_ball_target is not None:
                self.discover_selector.report_landings(
                    [ball_status.min_position_ball_target]
                )
