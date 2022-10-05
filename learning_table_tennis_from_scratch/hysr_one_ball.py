import math
import pathlib
import json
import os
import site
import sys
import time
import numpy as np
import math
import random

import o80
import o80_pam
import pam_interface
import pam_mujoco
import context
import frequency_monitoring
import shared_memory
from pam_mujoco import mirroring
from . import configure_mujoco
from . import robot_integrity


SEGMENT_ID_BALL = pam_mujoco.segment_ids.ball
SEGMENT_ID_GOAL = pam_mujoco.segment_ids.goal
SEGMENT_ID_HIT_POINT = pam_mujoco.segment_ids.hit_point
SEGMENT_ID_ROBOT_MIRROR = pam_mujoco.segment_ids.mirroring
SEGMENT_ID_PSEUDO_REAL_ROBOT = o80_pam.segment_ids.robot
SEGMENT_ID_EPISODE_FREQUENCY = "hysr_episode_frequency"
SEGMENT_ID_STEP_FREQUENCY = "hysr_step_frequency"


def _to_robot_type(robot_type: str) -> pam_mujoco.RobotType:
    try:
        return pam_mujoco.RobotType[robot_type.upper()]
    except KeyError:
        error = str(
            "hysr configuration robot_type should be either "
            "'pamy1' or 'pamy2' (entered value: {})"
        ).format(robot_type)
        raise ValueError(error)

def velocity_norm(velocity):
    return math.sqrt(sum([v ** 2 for v in velocity]))

def distance(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

def min_distance(traj1, traj2):
    return min([distance(p1, p2) for p1, p2 in zip(traj1, traj2)])

def sample_point_circle(center, radius):
    x = 1
    y = 1
    while x**2 + y**2>1:
        x = random.random() * 2 - 1
        y = random.random() * 2 - 1
    return  [center[0] + radius * x,
        center[1] + radius * y,
        center[2]]


class HysrOneBallConfig:

    __slots__ = (
        "real_robot",
        "robot_type",
        "o80_pam_time_step",
        "mujoco_time_step",
        "algo_time_step",
        "pam_config_file",
        "robot_position",
        "robot_orientation",
        "table_position",
        "table_orientation",
        "target_position",
        "target_position_sampling_radius",
        "reference_posture",
        "starting_pressures",
        "world_boundaries",
        "pressure_change_range",
        "action_in_state",
        "delta_p",
        "delta_p_p0_is_action",
        "delta_p_p0_value",
        "delta_u_init",
        "pd_control",
        "pd_control_T",
        "pd_control_K_p",
        "pd_control_K_d",
        "trajectory",
        "accelerated_time",
        "graphics_pseudo_real",
        "graphics_simulation",
        "graphics_extra_balls",
        "instant_reset",
        "nb_steps_per_episode",
        "extra_balls_sets",
        "extra_balls_per_set",
        "trajectory_group",
        "vel_filter_window_size",
        "frequency_monitoring_step",
        "frequency_monitoring_episode",
        "robot_integrity_check",
        "robot_integrity_threshold",
    )

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)

    def get(self):
        r = {s: getattr(self, s) for s in self.__slots__}
        return r

    @classmethod
    def from_json(cls, jsonpath):
        if not os.path.isfile(jsonpath):
            raise FileNotFoundError(
                "failed to find hysr configuration file: {}".format(jsonpath)
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
        instance = cls()
        for s in cls.__slots__:
            try:
                setattr(instance, s, conf[s])
            except Exception:
                raise ValueError(
                    "failed to find the attribute {} " "in {}".format(s, jsonpath)
                )
        # robot type given as string in json config, but
        # the rest of the code will expect a pam_mujoco.RobotType
        instance.robot_type = _to_robot_type(instance.robot_type)

        # convert paths to Path objects and expand '~'
        instance.pam_config_file = pathlib.Path(instance.pam_config_file).expanduser()

        return instance

    @staticmethod
    def default_path():
        global_install = os.path.join(
            sys.prefix,
            "local",
            "learning_table_tennis_from_scratch_config",
            "hysr_one_ball_default.json",
        )
        local_install = os.path.join(
            site.USER_BASE,
            "learning_table_tennis_from_scratch_config",
            "hysr_one_ball_default.json",
        )

        if os.path.isfile(local_install):
            return local_install
        if os.path.isfile(global_install):
            return global_install


class _BallBehavior:
    """
    HYSROneBall supports 3 ball behaviors:
    line: (3d tuple, 3d tuple, float): ball going from
          start to end position in straight line over the
          the provided duration (ms)
    index: (positive int) ball playing the pre-recorded trajectory
           corresponding to the index
    random: (True) ball playing a random pre-recorded trajectory
    """

    LINE = -1
    INDEX = -2
    RANDOM = -3

    @classmethod
    def read_trajectories(cls, group: str) -> None:
        cls._trajectory_reader = context.BallTrajectories(group)

    def __init__(self, line=False, index=False, random=False):

        if not hasattr(self.__class__, "_trajectory_reader"):
            raise UnboundLocalError(
                "_BallBehavior: the classmethod read_trajectories(group:str) "
                "has to be called before the constructor"
            )

        not_false = [a for a in (line, index, random) if a]
        if not not_false:
            raise ValueError("type of ball behavior not specified")
        if len(not_false) > 1:
            raise ValueError("type of ball behavior over-specified")
        if line:
            self.type = self.LINE
            self.value = line
        elif index:
            self.type = self.INDEX
            self.value = index
        elif random:
            self.type = self.RANDOM

    def get_trajectory(self):
        # ball behavior is a straight line, self.value is (start,end,duration ms)
        if self.type == self.LINE:
            duration_trajectory = context.ball_trajectories.duration_line_trajectory(
                *self.value
            )
            trajectory = context.ball_trajectories.to_stamped_trajectory(
                duration_trajectory
            )
            return trajectory
        # ball behavior is a specified pre-recorded trajectory
        if self.type == self.INDEX:
            trajectory = self._trajectory_reader.get_trajectory(self.value)
            return trajectory
        # ball behavior is a randomly selected pre-recorded trajectory
        if self.type == self.RANDOM:
            trajectory = self._trajectory_reader.random_trajectory()
            return trajectory

    def get(self):
        return self.value


class _ExtraBall:

    # see pam_demos/balls
    # for usage of handles and frontends
    # setid : handle
    handles = {}
    # setid: frontend to extra balls
    frontends = {}

    def __init__(self, handle, frontend, ball_status, segment_id):
        self.handle = handle  # shared between all balls of same setid
        self.frontend = frontend  # shared between all balls of same setid
        self.segment_id = segment_id
        self.ball_status = ball_status
        self.ball_behavior = None

    def reset_contact(self):
        self.handle.reset_contact(self.segment_id)

    def deactivate_contact(self):
        self.handle.deactivate_contact(self.segment_id)

    def status_reset(self):
        self.ball_status.reset()

    @classmethod
    def reset(cls):
        for handle in cls.handles.values():
            handle.reset()


def _get_extra_balls(setid, hysr_config):

    values = configure_mujoco.configure_extra_set(setid, hysr_config)

    handle = values[0]
    mujoco_id = values[1]
    extra_balls_segment_id = values[2]
    robot_segment_id = values[3]
    ball_segment_ids = values[4]
    nb_balls = hysr_config.extra_balls_per_set
    extra_balls_frontend = handle.get_extra_balls_frontend(
        configure_mujoco.get_extra_balls_segment_id(setid), nb_balls
    )

    # instance of o80_pam.o80_robot_mirroring.o80RobotMirroring,
    # to control the robot
    mirroring = handle.interfaces[robot_segment_id]
    # o80 frontend to control the balls
    # (one frontend to control all the balls,
    #  one ball 'corresponds' to one dof)
    frontend = handle.frontends[extra_balls_segment_id]

    ball_status = [
        context.BallStatus(hysr_config.target_position) for _ in range(nb_balls)
    ]

    balls = [
        _ExtraBall(handle, frontend, ball_status, segment_id)
        for ball_status, segment_id in zip(ball_status, ball_segment_ids)
    ]

    _ExtraBall.handles[setid] = handle
    _ExtraBall.frontends[setid] = frontend

    return balls, mirroring, mujoco_id, extra_balls_frontend


def _convert_pressures_in(pressures):
    # convert pressure from [ago1, antago1, ago2, antago2, ...]
    # to [(ago1, antago1), (ago2, antago2), ...]
    return list(zip(pressures[::2], pressures[1::2]))


def _convert_pressures_out(pressures_ago, pressures_antago):
    pressures = list(zip(pressures_ago, pressures_antago))
    return [p for sublist in pressures for p in sublist]


class _Observation:
    def __init__(
        self,
        joint_positions,
        joint_velocities,
        pressures,
        ball_position,
        ball_velocity,
    ):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.pressures = pressures
        self.ball_position = ball_position
        self.ball_velocity = ball_velocity


class HysrOneBall:
    def __init__(self, hysr_config, reward_function):

        self._hysr_config = hysr_config

        # we will track the episode number
        self._episode_number = -1

        # we will track the step number (reset at the start
        # of each episode)
        self._step_number = -1

        # we end an episode after a fixed number of steps
        self._nb_steps_per_episode = hysr_config.nb_steps_per_episode
        # note: if self._nb_steps_per_episode is 0 or less,
        #       an episode will end based on a threshold
        #       in the z component of the ball position
        #       (see method _episode_over)

        # this instance of HysrOneBall interacts with several
        # instances of mujoco (pseudo real robot, simulated robot,
        # possibly instances of mujoco for extra balls).
        # Listing all the corresponding mujoco_ids
        self._mujoco_ids = []

        # pam muscles configuration
        self._pam_config = pam_interface.JsonConfiguration(
            str(hysr_config.pam_config_file)
        )

        # to control pseudo-real robot (pressure control)
        if not hysr_config.real_robot:
            (
                self._real_robot_handle,
                self._real_robot_frontend,
            ) = configure_mujoco.configure_pseudo_real(
                str(hysr_config.pam_config_file),
                hysr_config.robot_type,
                graphics=hysr_config.graphics_pseudo_real,
                accelerated_time=hysr_config.accelerated_time,
            )
            self._mujoco_ids.append(self._real_robot_handle.get_mujoco_id())
        else:
            # real robot: making some sanity check that the
            # rest of the configuration is ok
            if hysr_config.instant_reset:
                raise ValueError(
                    str(
                        "HysrOneBall configured for "
                        "real robot and instant reset."
                        "Real robot does not support "
                        "instant reset."
                    )
                )
            if hysr_config.accelerated_time:
                raise ValueError(
                    str(
                        "HysrOneBall configured for "
                        "real robot and accelerated time."
                        "Real robot does not support "
                        "accelerated time."
                    )
                )

        # to control the simulated robot (joint control)
        self._simulated_robot_handle = configure_mujoco.configure_simulation(
            hysr_config
        )
        self._mujoco_ids.append(self._simulated_robot_handle.get_mujoco_id())

        # where we want to shoot the ball
        self._target_position = hysr_config.target_position
        self._target_position_sampling_radius = hysr_config.target_position_sampling_radius
        self._goal = self._simulated_robot_handle.interfaces[SEGMENT_ID_GOAL]

        # to read all recorded trajectory files
        self._trajectory_reader = context.BallTrajectories(hysr_config.trajectory_group)
        _BallBehavior.read_trajectories(hysr_config.trajectory_group)

        # if requested, logging info about the frequencies of the steps and/or the
        # episodes
        if hysr_config.frequency_monitoring_step:
            size = 1000
            self._frequency_monitoring_step = frequency_monitoring.FrequencyMonitoring(
                SEGMENT_ID_STEP_FREQUENCY, size
            )
        else:
            self._frequency_monitoring_step = None
        if hysr_config.frequency_monitoring_episode:
            size = 1000
            self._frequency_monitoring_episode = (
                frequency_monitoring.FrequencyMonitoring(
                    SEGMENT_ID_EPISODE_FREQUENCY, size
                )
            )
        else:
            self._frequency_monitoring_episode = None

        # if o80_pam (i.e. the pseudo real robot)
        # has been started in accelerated time,
        # the corresponding o80 backend will burst through
        # an algorithm time step
        self._accelerated_time = hysr_config.accelerated_time
        if self._accelerated_time:
            self._o80_time_step = hysr_config.o80_pam_time_step
            self._nb_robot_bursts = int(
                hysr_config.algo_time_step / hysr_config.o80_pam_time_step
            )

        # pam_mujoco (i.e. simulated ball and robot) should have been
        # started in accelerated time. It burst through algorithm
        # time steps
        self._mujoco_time_step = hysr_config.mujoco_time_step
        self._nb_sim_bursts = int(
            hysr_config.algo_time_step / hysr_config.mujoco_time_step
        )

        # the config sets either a zero or positive int (playing the
        # corresponding indexed pre-recorded trajectory) or a negative int
        # (playing randomly selected indexed trajectories)
        if hysr_config.trajectory >= 0:
            self._ball_behavior = _BallBehavior(index=hysr_config.trajectory)
        else:
            self._ball_behavior = _BallBehavior(random=True)

        # the robot will interpolate between current and
        # target posture over this duration
        self._period_ms = hysr_config.algo_time_step

        # reward configuration
        self._reward_function = reward_function

        # to get information regarding the ball
        # (instance of o80_pam.o80_ball.o80Ball)
        self._ball_communication = self._simulated_robot_handle.interfaces[
            SEGMENT_ID_BALL
        ]

        # to send pressure commands to the real or pseudo-real robot
        # (instance of o80_pam.o80_pressures.o80Pressures)
        # hysr_config.real robot is either false (i.e. pseudo real
        # mujoco robot) or the segment_id of the real robot backend
        if not hysr_config.real_robot:
            self._pressure_commands = self._real_robot_handle.interfaces[
                SEGMENT_ID_PSEUDO_REAL_ROBOT
            ]
        else:
            self._real_robot_frontend = o80_pam.FrontEnd(hysr_config.real_robot)
            self._pressure_commands = o80_pam.o80Pressures(
                hysr_config.real_robot, frontend=self._real_robot_frontend
            )

        # will encapsulate all information
        # about the ball (e.g. min distance with racket, etc)
        self._ball_status = context.BallStatus(hysr_config.target_position)

        # to send mirroring commands to simulated robots
        self._mirrorings = [
            self._simulated_robot_handle.interfaces[SEGMENT_ID_ROBOT_MIRROR]
        ]

        # to move the hit point marker
        # (instance of o80_pam.o80_hit_point.o80HitPoint)
        self._hit_point = self._simulated_robot_handle.interfaces[SEGMENT_ID_HIT_POINT]

        # tracking if this is the first step of the episode
        # (a call to the step function sets it to false, call to reset function sets it
        # back to true)
        self._first_episode_step = True

        # normally an episode ends when the ball z position goes
        # below a certain threshold (see method _episode_over)
        # this is to allow user to force ending an episode
        # (see force_episode_over method)
        self._force_episode_over = False

        # if false, the system will reset via execution of commands
        # if true, the system will reset by resetting the simulations
        # Only "false" is supported by the real robot
        self._instant_reset = hysr_config.instant_reset

        # adding extra balls (if any)
        if (
            hysr_config.extra_balls_sets is not None
            and hysr_config.extra_balls_sets > 0
        ):

            self._extra_balls = []

            for setid in range(hysr_config.extra_balls_sets):

                # balls: list of instances of _ExtraBalls (defined in this file)
                # mirroring : for sending mirroring command to the robot
                #             of the set (joint controlled)
                #             (instance of
                #             o80_pam.o80_robot_mirroring.o80RobotMirroring)
                balls, mirroring, mujoco_id, frontend = _get_extra_balls(
                    setid, hysr_config
                )

                self._extra_balls.extend(balls)
                self._mirrorings.append(mirroring)
                self._mujoco_ids.append(mujoco_id)
                self._extra_balls_frontend = frontend
        else:
            self._extra_balls = []
            self._extra_balls_frontend = None

        # for running all simulations (main + for extra balls)
        # in parallel (i.e. when bursting is called, all mujoco
        # instance perform step(s) in parallel)
        self._parallel_burst = pam_mujoco.mirroring.ParallelBurst(self._mirrorings)

        # if set, logging the position of the robot at the end of reset, and possibly
        # get a warning when this position drifts as the number of episodes increase
        if hysr_config.robot_integrity_check is not None:
            self._robot_integrity = robot_integrity.RobotIntegrity(
                hysr_config.robot_integrity_threshold,
                file_path=hysr_config.robot_integrity_check,
            )
        else:
            self._robot_integrity = None

        # when starting, the real robot and the virtual robot(s)
        # may not be aligned, which may result in graphical issues,
        # so aligning them
        # (get values of real robot via self._pressure_commands,
        # and set values to all simulated robot via self._mirrorings)
        # source of mirroring in pam_mujoco.mirroring.py
        pam_mujoco.mirroring.align_robots(self._pressure_commands, self._mirrorings)

    def get_starting_pressures(self):
        return self._hysr_config.starting_pressures

    def _share_episode_number(self, episode_number):
        # write the episode number in a memory shared
        # with the instances of mujoco
        for mujoco_id in self._mujoco_ids:
            shared_memory.set_long_int(mujoco_id, "episode", episode_number)

    def force_episode_over(self):
        # will trigger the method _episode_over
        # (called in the step method) to return True
        self._force_episode_over = True

    def set_ball_behavior(self, line=False, index=False, random=False):
        # overwrite the ball behavior (set to a trajectory in the constructor)
        # see comments in _BallBehavior, in this file
        self._ball_behavior = _BallBehavior(line=line, index=index, random=random)

    def set_extra_ball_behavior(
        self, ball_index, line=False, index=False, random=False
    ):
        # overwrite the ball behavior of the extra ball (set to random
        # selected pre-recorded trajectory in constructor)
        # see comments in _BallBehavior, in this file
        if ball_index < 0 or ball_index >= len(self._extra_balls):
            raise IndexError(ball_index)
        self._extra_balls[ball_index].ball_behavior = _BallBehavior(
            line=line, index=index, random=random
        )

    def _create_observation(self):
        (
            pressures_ago,
            pressures_antago,
            joint_positions,
            joint_velocities,
        ) = self._pressure_commands.read()
        ball_position, ball_velocity = self._ball_communication.get()
        observation = _Observation(
            joint_positions,
            joint_velocities,
            _convert_pressures_out(pressures_ago, pressures_antago),
            ball_position,
            ball_velocity,
        )
        return observation

    def get_robot_iteration(self):
        return self._pressure_commands.get_iteration()

    def get_ball_iteration(self):
        return self._ball_communication.get_iteration()

    def get_current_desired_pressures(self):
        (pressures_ago, pressures_antago, _, __) = self._pressure_commands.read(
            desired=True
        )
        return pressures_ago, pressures_antago

    def get_current_pressures(self):
        (pressures_ago, pressures_antago, _, __) = self._pressure_commands.read(
            desired=False
        )
        return pressures_ago, pressures_antago

    def contact_occured(self):
        return self._ball_status.contact_occured()

    def _load_main_ball(self):
        # "load" the ball means creating the o80 commands corresponding
        # to the ball behavior (set by the "set_ball_behavior" method)
        trajectory = self._ball_behavior.get_trajectory()
        iterator = context.ball_trajectories.BallTrajectories.iterate(trajectory, vel_filter_window_size = self._hysr_config.vel_filter_window_size)
        # setting the ball to the first trajectory point
        duration, state = next(iterator)
        self._ball_communication.set(state.get_position(), [0, 0, 0])
        # shooting the ball
        self._ball_communication.iterate_trajectory(iterator, overwrite=False)

    def _load_extra_balls(self):
        # load the trajectory of each extra balls, as set by their
        # ball_behavior attribute. See method set_extra_ball_behavior
        # in this file
        item3d = o80.Item3dState()
        # loading the ball behavior trajectory of each extra balls.
        # If set_extra_ball_behavior has not been called for a given
        # extra ball, this trajectory will be None
        trajectories = [
            extra_ball.ball_behavior.get_trajectory()
            if extra_ball.ball_behavior is not None
            else None
            for extra_ball in self._extra_balls
        ]
        # None trajectories (i.e. set_extra_ball_behavior uncalled) then
        # setting a random trajectory
        none_trajectory_indexes = [
            index for index, value in enumerate(trajectories) if value is None
        ]
        if none_trajectory_indexes:
            extra_trajectories = (
                self._trajectory_reader.get_different_random_trajectories(
                    len(none_trajectory_indexes)
                )
            )
            for index, trajectory in zip(none_trajectory_indexes, extra_trajectories):
                trajectories[index] = trajectory
        for index_ball, (ball, trajectory) in enumerate(
            zip(self._extra_balls, trajectories)
        ):
            iterator = context.ball_trajectories.BallTrajectories.iterate(trajectory, vel_filter_window_size = self._hysr_config.vel_filter_window_size)
            # going to first trajectory point
            _, state = next(iterator)
            item3d.set_position(state.get_position())
            item3d.set_velocity(state.get_velocity())
            ball.frontend.add_command(index_ball, item3d, o80.Mode.OVERWRITE)
            # loading full trajectory
            for duration, state in iterator:
                item3d.set_position(state.get_position())
                item3d.set_velocity(state.get_velocity())
                ball.frontend.add_command(
                    index_ball,
                    item3d,
                    o80.Duration_us.microseconds(duration),
                    o80.Mode.QUEUE,
                )
        for frontend in _ExtraBall.frontends.values():
            frontend.pulse()

    def load_ball(self):
        # loading ball: setting all trajectories points
        # to the ball controllers
        self._load_main_ball()
        if self._extra_balls:
            self._load_extra_balls()

    def reset_contact(self):
        # after contact with the racket, o80 control of the ball
        # is disabled (and customized contact happen). This
        # restore the control.
        # Also: this delete the information about the
        # contact with the racket (if any)
        self._simulated_robot_handle.reset_contact(SEGMENT_ID_BALL)
        for ball in self._extra_balls:
            ball.reset_contact()

    def deactivate_contact(self):
        self._simulated_robot_handle.deactivate_contact(SEGMENT_ID_BALL)
        for ball in self._extra_balls:
            ball.deactivate_contact()

    def _do_natural_reset(self):
        self._move_to_position(self._hysr_config.reference_posture)

    def _do_instant_reset(self):

        # "instant": reset all mujoco instances
        # to their starting state. Not applicable
        # to real robot

        self._real_robot_handle.reset()
        self._simulated_robot_handle.reset()
        for handle in _ExtraBall.handles.values():
            handle.reset()
        self._move_to_pressure(self._hysr_config.reference_posture)

    def _move_to_pressure(self, pressures):
        # moves to pseudo-real robot to desired pressure in synchronization
        # with the simulated robot(s)
        if self._accelerated_time:
            for _ in range(self._nb_robot_bursts):
                self._pressure_commands.set(pressures, burst=1)
                _, _, joint_positions, joint_velocities = self._pressure_commands.read()
                for mirroring_ in self._mirrorings:
                    mirroring_.set(joint_positions, joint_velocities)
                self._parallel_burst.burst(1)
            return
        else:
            self._pressure_commands.set(pressures, burst=False)
        time_start = self._real_robot_frontend.latest().get_time_stamp() * 1e-9
        current_time = time_start
        timeout = 0.5
        while current_time - time_start < timeout:
            current_time = self._real_robot_frontend.latest().get_time_stamp() * 1e-9
            _, _, joint_positions, joint_velocities = self._pressure_commands.read()
            for mirroring_ in self._mirrorings:
                mirroring_.set(joint_positions, joint_velocities)
            self._parallel_burst.burst(self._nb_sim_bursts)

    def _move_to_position(self, position):
        # moves the pseudo-real robot to a desired position (in radians)
        # via a position controller (i.e. compute the pressure trajectory
        # required to reach, hopefully, for the position) in
        # synchronization with the simulated robot(s).

        # configuration for the controller
        KP = [0.4309, 1.212, 0.5, .05]
        KI = [0.05629, 0.08202, .1, .1]
        KD = [0.04978, 0.1712, 0.0, 0.0]
        NDP = [.9]*4
        TIME_STEP = 0.05  # seconds
        QD_DESIRED = [math.pi*.75]*4  # radian per seconds
        EXTRA_STEPS = 20
        pi4 = math.pi/4.0
        g80 = 80/180*math.pi
        target_position = [0,+g80,0,0]
        _, _, Q_CURRENT, _ = self._pressure_commands.read()

        # configuration for HYSR
        NB_SIM_BURSTS = int((TIME_STEP / self._hysr_config.mujoco_time_step) + 0.5)

        # configuration for accelerated time
        if self._accelerated_time:
            NB_ROBOT_BURSTS = int((TIME_STEP / self._hysr_config.o80_pam_time_step) + 0.5)

        # the position controller
        position_controller_factory = o80_pam.position_control.PositionControllerFactory(
            QD_DESIRED,
            self._pam_config,
            KP,KD,KI,NDP,
            TIME_STEP,EXTRA_STEPS
        )

        def control():

            # configuration for real time
            if not self._accelerated_time:
                frequency_manager = o80.FrequencyManager(1.0 / TIME_STEP)
            
            # starting position
            _, _, q_current, _ = self._pressure_commands.read()

            controller = position_controller_factory.get(q_current,target_position)
            # rolling the controller
            while controller.has_next():
                # current position and velocity of the real robot
                _, _, q, qd = self._pressure_commands.read()
                # mirroing the simulated robot(s)
                for mirroring_ in self._mirrorings:
                    mirroring_.set(q, qd)
                self._parallel_burst.burst(NB_SIM_BURSTS)
                # applying the controller to get the pressure to set
                pressures = controller.next(q, qd)
                # setting the pressures to real robot
                if self._accelerated_time:
                    # if accelerated times, running the pseudo real robot iterations
                    # (note : o80_pam expected to have started in bursting mode)
                    pressures,_,_,_ = self._pressure_commands.read()
                    self._pressure_commands.set(pressures, burst=NB_ROBOT_BURSTS)
                else:
                    # Should start acting now in the background if not accelerated time
                    self._pressure_commands.set(pressures, burst=False)
                    frequency_manager.wait()

            _, _, q_current, _ = self._pressure_commands.read()

            error = [abs(p-t) for p,t in zip(q_current,target_position)]

            return error

        def _mirror_sleep(duration):
            nb_steps = int((duration / TIME_STEP)+0.5)
            if not self._accelerated_time:
                frequency_manager = o80.FrequencyManager(1.0 / TIME_STEP)
            for step in range(nb_steps):
                _, _, q, qd = self._pressure_commands.read()
                for mirroring_ in self._mirrorings:
                    mirroring_.set(q, qd)
                self._parallel_burst.burst(NB_SIM_BURSTS)
                if not self._accelerated_time:
                    frequency_manager.wait()
                else:
                    self._pressure_commands.set(pressures, burst=NB_ROBOT_BURSTS)
            
        error = control()
        its=0
        while  max([abs(i/math.pi*180) for i in error]) > 10 :
            if its<10:
                _mirror_sleep(2)
                print("_move_to_position: err before ",[e/math.pi*180 for e in error])
                error = control()
                print("_move_to_position: after before ",[e/math.pi*180 for e in error])
                # input("_move_to_position: Press key ... ")
                its +=1
            else:
                input("_move_to_position: STOPPING!!! Could not move to target pos with sufficient accuracy! Press key...")
                break

    def reset(self):

        # what happens during reset does not correspond
        # to any episode (-1 means: no active episode)
        self._share_episode_number(-1)

        # resetting the measure of step frequency monitoring
        if self._frequency_monitoring_step:
            self._frequency_monitoring_step.reset()

        # exporting episode frequency
        if self._frequency_monitoring_episode:
            self._frequency_monitoring_episode.ping()
            self._frequency_monitoring_episode.share()

        # in case the episode was forced to end by the
        # user (see force_episode_over method)
        self._force_episode_over = False

        # resetting first episode step
        self._first_episode_step = True

        # resetting the hit point
        self._hit_point.set([0, 0, -0.62], [0, 0, 0])

        if self._instant_reset:
            # going back to vertical position
            # on simulated robot
            self._do_instant_reset()
            mirroring.align_robots(self._pressure_commands, self._mirrorings)
        else:
            # moving to reset position
            self._do_natural_reset()

        # going to starting pressure
        self._move_to_pressure(self._hysr_config.starting_pressures)

        # moving the goal to the target position
        self._goal.set(self._target_position, [0, 0, 0])

        # setting the ball behavior
        self.load_ball()

        # control post contact was lost, restoring it
        self._simulated_robot_handle.reset_contact(SEGMENT_ID_BALL)
        self._simulated_robot_handle.deactivate_contact(SEGMENT_ID_BALL)
        for ball in self._extra_balls:
            ball.handle.reset_contact(ball.segment_id)
            ball.handle.deactivate_contact(ball.segment_id)

        # moving the ball(s) to initial position
        self._parallel_burst.burst(4)

        # resetting ball/robot contact information
        self._simulated_robot_handle.reset_contact(SEGMENT_ID_BALL)
        self._simulated_robot_handle.activate_contact(SEGMENT_ID_BALL)
        for ball in self._extra_balls:
            ball.handle.reset_contact(ball.segment_id)
            ball.handle.activate_contact(ball.segment_id)

        time.sleep(0.1)

        # resetting ball info, e.g. min distance ball/racket, target_position etc
        if self._target_position_sampling_radius>0:
            self._ball_status.target_position = sample_point_circle(self._target_position, self._target_position_sampling_radius)
        self._ball_status.reset()
        for ball in self._extra_balls:
            if self._target_position_sampling_radius>0:
                ball.ball_status.target_position = sample_point_circle(self._target_position, self._target_position_sampling_radius)
            ball.ball_status.reset()

        # resetting extra balls
        self.extra_contacts = [False]*self._hysr_config.extra_balls_per_set
        self.extra_min_distance_ball_racket = [None]*self._hysr_config.extra_balls_per_set
        self.extra_min_distance_ball_target = [None]*self._hysr_config.extra_balls_per_set
        self.extra_max_ball_velocity = [0]*self._hysr_config.extra_balls_per_set
        self.extra_dones_before = [False]*self._hysr_config.extra_balls_per_set

        # checking the position of the robot, to see if it drifts
        # as episode increase (or if it not what is expected at all).
        # raise an exception if drifted too much).
        if self._robot_integrity is not None:
            _, _, joint_positions, _ = self._pressure_commands.read()
            warning = self._robot_integrity.set(joint_positions)
            if warning:
                self._robot_integrity.close()
                self.close()
                raise robot_integrity.RobotIntegrityException(
                    self._robot_integrity, joint_positions
                )

        # a new episode starts
        self._step_number = 0
        self._episode_number += 1
        self._share_episode_number(self._episode_number)

        # returning an observation
        observation = self._create_observation()

        if self._extra_balls_frontend is not None:
            nb_balls = self._hysr_config.extra_balls_per_set

            observation_extra = self._extra_balls_frontend.latest()
            states = observation_extra.get_observed_states()
            extra_ball_positions = [states.get(index).get_position() for index in range(nb_balls)]
            extra_ball_velocities = [states.get(index).get_velocity() for index in range(nb_balls)]
            extra_observations = [ _Observation(
                                observation.joint_positions,
                                observation.joint_velocities,
                                observation.pressures,
                                extra_ball_positions[index],
                                extra_ball_velocities[index],)
                                for index in range(nb_balls)]

            #returning with extra transitions
            return observation, extra_observations
        
        return observation, []



    def _episode_over(self):

        # if self._nb_steps_per_episode is positive,
        # exiting based on the number of steps
        if self._nb_steps_per_episode > 0:
            if self._step_number >= self._nb_steps_per_episode:
                return True
            else:
                return False

        # otherwise exiting based on a threshold on the
        # z position of the ball

        # ball falled below the table
        # note : all prerecorded trajectories are added a last ball position
        # with z = -10.0, to insure this always occurs.
        # see: function reset
        if self._ball_status.ball_position[2] < 0.75 or self._ball_status.ball_position[1] < -0.5:
            return True
        # in case the user called the method
        # force_episode_over
        if self._force_episode_over:
            return True

        return False

    def get_ball_position(self):
        # returning current ball position
        ball_position, _ = self._ball_communication.get()
        return ball_position

    # action assumed to be np.array(ago1,antago1,ago2,antago2,...)
    def step(self, action):

        # reading current real (or pseudo real) robot state
        (
            pressures_ago,
            pressures_antago,
            joint_positions,
            joint_velocities,
        ) = self._pressure_commands.read()

        # getting information about simulated ball
        ball_position, ball_velocity = self._ball_communication.get()

        # getting information about extra simulated balls
        if self._extra_balls_frontend is not None:

            nb_balls = self._hysr_config.extra_balls_per_set

            observation = self._extra_balls_frontend.latest()

            robot_cartesian_position = observation.get_extended_state().robot_position
            states = observation.get_observed_states()
            contacts = observation.get_extended_state().contacts
            extra_ball_positions = [states.get(index).get_position() for index in range(nb_balls)]
            extra_ball_velocities = [states.get(index).get_velocity() for index in range(nb_balls)]

            self.extra_contacts = [self.extra_contacts[index] or contacts[index] for index in range(nb_balls) ]
            self.extra_min_distance_ball_racket = [None if self.extra_contacts[index]
                                            else distance(extra_ball_positions[index], robot_cartesian_position) if not self.extra_min_distance_ball_racket[index] 
                                            else min([distance(extra_ball_positions[index], robot_cartesian_position), self.extra_min_distance_ball_racket[index]])
                                            for index in range(nb_balls)]

            self.extra_min_distance_ball_target = [None if not self.extra_contacts[index]
                                            else distance(extra_ball_positions[index], self._target_position) if not self.extra_min_distance_ball_racket[index] 
                                            else min([distance(extra_ball_positions[index], self._target_position), self.extra_min_distance_ball_racket[index]])
                                            for index in range(nb_balls)]

            self.extra_max_ball_velocity = [None if not self.extra_contacts[index]
                                            else velocity_norm(extra_ball_velocities[index]) if not self.extra_max_ball_velocity[index] 
                                            else max([velocity_norm(extra_ball_velocities[index]), self.extra_max_ball_velocity[index]])
                                            for index in range(nb_balls)]


            extra_dones =   [(self._nb_steps_per_episode>0 and self._step_number>= self._nb_steps_per_episode)
                        or (self._nb_steps_per_episode<=0 and (states.get(index).get_position()[2] < 0.75 or states.get(index).get_position()[2] < -0.5))
                        or self.extra_dones_before[index]
                        for index in range(nb_balls)]

            self.extra_dones_before = extra_dones.copy()

            extra_rewards = [0 if not extra_dones[index]
                    else self._reward_function(self.extra_min_distance_ball_racket[index], self.extra_min_distance_ball_target[index], self.extra_max_ball_velocity[index])
                        for index in range(nb_balls)
                ]

        # convert action [ago1,antago1,ago2] to list suitable for
        # o80 ([(ago1,antago1),(),...])
        pressures = _convert_pressures_in(list(action))

        # sending action pressures to real (or pseudo real) robot.
        if self._accelerated_time:
            # if accelerated times, running the pseudo real robot iterations
            # (note : o80_pam expected to have started in bursting mode)
            self._pressure_commands.set(pressures, burst=self._nb_robot_bursts)
        else:
            # Should start acting now in the background if not accelerated time
            self._pressure_commands.set(pressures, burst=False)

        # sending mirroring state to simulated robot(s)
        for mirroring_ in self._mirrorings:
            mirroring_.set(joint_positions, joint_velocities)

        # having the simulated robot(s)/ball(s) performing the right number of
        # iterations (note: simulated expected to run accelerated time)
        self._parallel_burst.burst(self._nb_sim_bursts)

        def _update_ball_status(handle, segment_id, ball_status):
            # getting ball/racket contact information
            # note : racket_contact_information is an instance
            #        of context.ContactInformation
            racket_contact_information = handle.get_contact(segment_id)
            # updating ball status
            ball_status.update(ball_position, ball_velocity, racket_contact_information)

        # updating the status of all balls
        _update_ball_status(
            self._simulated_robot_handle, SEGMENT_ID_BALL, self._ball_status
        )
        for ball in self._extra_balls:
            _update_ball_status(ball.handle, ball.segment_id, ball.ball_status)

        # moving the hit point to the minimal observed distance
        # between ball and target (post racket hit)
        if self._ball_status.min_position_ball_target is not None:
            self._hit_point.set(self._ball_status.min_position_ball_target, [0, 0, 0])

        # observation instance
        observation = _Observation(
            joint_positions,
            joint_velocities,
            _convert_pressures_out(pressures_ago, pressures_antago),
            self._ball_status.ball_position,
            self._ball_status.ball_velocity,
        )

        # checking if episode is over
        episode_over = self._episode_over()
        reward = 0

        # if episode over, computing related reward
        if episode_over:
            reward = self._reward_function(
                self._ball_status.min_distance_ball_racket,
                self._ball_status.min_distance_ball_target,
                self._ball_status.max_ball_velocity,
            )

        # next step can not be the first one
        # (reset will set this back to True)
        self._first_episode_step = False

        # exporting step frequency
        if self._frequency_monitoring_step:
            self._frequency_monitoring_step.ping()
            self._frequency_monitoring_step.share()

        # this step is done
        self._step_number += 1

        # setting extra transitions
        extra_observations = []
        extra_transitions = []

        if self._extra_balls_frontend is not None:
            extra_observations = [ _Observation(
                                joint_positions,
                                joint_velocities,
                                _convert_pressures_out(pressures_ago, pressures_antago),
                                extra_ball_positions[index],
                                extra_ball_velocities[index],)
                                for index in range(nb_balls)]
            extra_transitions = [(extra_observations[index],
                                extra_rewards[index],
                                extra_dones[index])
                                for index in range(nb_balls)]

            #returning with extra transitions
            return observation, reward, episode_over, extra_observations, extra_rewards, extra_dones

        # returning
        if self._extra_balls_frontend is not None:
            return observation, reward, episode_over, extra_transitions
        else:
            return observation, reward, episode_over, []

    def close(self):
        if self._robot_integrity is not None:
            self._robot_integrity.close()
        self._parallel_burst.stop()
        shared_memory.clear_shared_memory(SEGMENT_ID_EPISODE_FREQUENCY)
        shared_memory.clear_shared_memory(SEGMENT_ID_STEP_FREQUENCY)
        pass
