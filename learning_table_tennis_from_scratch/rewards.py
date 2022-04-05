import sys
import os
import math
import numpy
import json
import site


def _no_hit_reward(min_distance_ball_racket):
    return -min_distance_ball_racket


def _return_task_reward(min_distance_ball_target, c, rtt_cap):
    reward = 1.0 - ((min_distance_ball_target / c) ** 0.75)
    reward = max(reward, rtt_cap)
    return reward


def _smash_task_reward(min_distance_ball_target, max_ball_velocity, c, rtt_cap):
    reward = 1.0 - ((min_distance_ball_target / c) ** 0.75)
    reward = reward * max_ball_velocity
    reward = max(reward, rtt_cap)
    return reward


def _compute_reward(
    smash,
    min_distance_ball_racket,
    min_distance_ball_target,
    max_ball_velocity,
    c,
    rtt_cap,
):

    # i.e. the ball did not hit the racket,
    # so computing a reward based on the minimum
    # distance between the racket and the ball
    if min_distance_ball_racket is not None:
        return _no_hit_reward(min_distance_ball_racket)

    # the ball did hit the racket, so computing
    # a reward based on the ball / target

    if smash:
        return _smash_task_reward(
            min_distance_ball_target, max_ball_velocity, c, rtt_cap
        )

    else:
        return _return_task_reward(min_distance_ball_target, c, rtt_cap)


def _reward(
    min_distance_ball_racket, min_distance_ball_target, max_ball_velocity, c, rtt_cap
):
    return _compute_reward(
        False,
        min_distance_ball_racket,
        min_distance_ball_target,
        max_ball_velocity,
        c,
        rtt_cap,
    )


def _smash_reward(
    min_distance_ball_racket, min_distance_ball_target, max_ball_velocity, c, rtt_cap
):
    return _compute_reward(
        True,
        min_distance_ball_racket,
        min_distance_ball_target,
        max_ball_velocity,
        c,
        rtt_cap,
    )


class RewardConfig:
    def __init__(self, normalization_constant=3.0, rtt_cap=-0.2):
        self.normalization_constant = normalization_constant
        self.rtt_cap = rtt_cap


class Reward:
    def __init__(self, config):
        self.config = config

    def __call__(
        self, min_distance_ball_racket, min_distance_ball_target, max_ball_velocity
    ):
        return _reward(
            min_distance_ball_racket,
            min_distance_ball_target,
            max_ball_velocity,
            self.config.normalization_constant,
            self.config.rtt_cap,
        )


class SmashReward:
    def __init__(self, config):
        self.config = config

    def __call__(
        self, min_distance_ball_racket, min_distance_ball_target, max_ball_velocity
    ):
        return _smash_reward(
            min_distance_ball_racket,
            min_distance_ball_target,
            max_ball_velocity,
            self.config.normalization_constant,
            self.config.rtt_cap,
        )


class JsonReward:
    @staticmethod
    def get(jsonpath):
        if not os.path.isfile(jsonpath):
            raise FileNotFoundError(
                "failed to find reward configuration file: {}".format(jsonpath)
            )
        try:
            with open(jsonpath) as f:
                conf = json.load(f)
        except Exception as e:
            raise ValueError(
                "failed to parse reward json configuration file {}: {}".format(
                    jsonpath, e
                )
            )
        for attr in ("smash", "normalization_constant", "rtt_cap"):
            if attr not in conf:
                raise ValueError(
                    "failed to find the attribute {} "
                    "in the json reward configuration "
                    "file: {}".format(attr, jsonpath)
                )
        smash = conf["smash"]
        normalization_constant = conf["normalization_constant"]
        rtt_cap = conf["rtt_cap"]
        config = RewardConfig(normalization_constant, rtt_cap)
        if smash:
            return SmashReward(config)
        return Reward(config)

    @staticmethod
    def default_path():
        global_install = os.path.join(
            sys.prefix,
            "local",
            "learning_table_tennis_from_scratch_config",
            "reward_default.json",
        )
        local_install = os.path.join(
            site.USER_BASE,
            "learning_table_tennis_from_scratch_config",
            "reward_default.json",
        )

        if os.path.isfile(local_install):
            return local_install
        if os.path.isfile(global_install):
            return global_install


def compute_rewards(reward_function, target, nb_balls, observations, episode=None):

    """
    observations is a list of observations obtained via an extra_balls_frontend.
    This method returns a list of reward {extra ball index:reward}
    :param function reward_function: reward function, expected to be an instance
                                     of either Reward or SmashReward
    :param tuple target: the reward function required a 3d position target as
        input parameter
    :param int nb_balls: number of extra balls
    :param list observations: as returned by frontend.get_latest_observation
    :param int episode: if not None, filter observations to be of the given episode
    :return iterable rewards: reward score for each ball
    """

    # filtering observations based on episode number
    if episode:
        episodes = [(o, o.get_extended_state().episode) for o in observations]
        observations_ = [e[0] for e in episodes if e[1] == episode]
        # FIXME: observations is not used.  Can this be removed?  In this case
        # the `episode` argument of the function can be removed as well.

    # some reformatting of data, for balls positions and velocities
    states = [o.get_observed_states() for o in observations]
    states = [
        [obs_state.get(index) for index in range(nb_balls)] for obs_state in states
    ]
    positions_velocities = numpy.array(
        [
            [(ball.get_position(), ball.get_velocity()) for ball in entry]
            for entry in states
        ]
    )

    # some reformating of data, for robot position
    ext_states = [o.get_extended_state() for o in observations]
    robot_positions = [ext_state.robot_position for ext_state in ext_states]

    # some reformating of data, for contacts
    contacts = numpy.array(
        [
            [ext_state.contacts[index] for index in range(nb_balls)]
            for ext_state in ext_states
        ]
    ).T

    # contacts takes the format:
    # contacts = [index at which contact became true, -1 if never became true]
    def index_true(row):
        # returns index of first true value in row,
        # -1 if no true value
        for index, val in numpy.ndenumerate(row):
            if val:
                return index[0]
        return -1

    contacts = numpy.apply_along_axis(index_true, 1, contacts)

    # compute the reward of a specific ball
    def _compute_reward(index):

        # returns norm of a vector
        def _velocity_norm(velocity):
            return math.sqrt(sum([v**2 for v in velocity]))

        # returns euclidian distance between two vectors
        def _distance(p1, p2):
            return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

        # traj1 is a list of positions, so is traj2
        # returns the minimal distance between positions in traj1 and
        # traj2 that are at the same index
        def _min_distance(traj1, traj2):
            return min([_distance(p1, p2) for p1, p2 in zip(traj1, traj2)])

        pos_vels = positions_velocities[:, index]
        positions = pos_vels[:, 0]
        velocities = pos_vels[:, 1]
        contact = contacts[index]  # index of contact, -1 if no contact

        # if no contact: reward based on the minimal distance between ball and racket
        if contact < 0:
            min_distance_ball_racket = _min_distance(positions, robot_positions)
            min_distance_ball_target = None
            max_ball_velocity = None
        # if contact: reward based on the minimal distance between ball and target
        # (and possibly ball speed, if smash reward function)
        else:
            min_distance_ball_racket = None
            # trimming ball positions to steps after contact
            positions = positions[contact:]
            min_distance_ball_target = min([_distance(p, target) for p in positions])
            max_ball_velocity = max([_velocity_norm(v) for v in velocities])

        return reward_function(
            min_distance_ball_racket, min_distance_ball_target, max_ball_velocity
        )

    return map(_compute_reward, [index for index in range(nb_balls)])
