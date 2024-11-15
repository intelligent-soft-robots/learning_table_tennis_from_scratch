import dataclasses
import logging
import os
import pathlib
import site
import sys
import time
import typing as t

import omegaconf as oc
import variconf
from scipy.spatial.transform import Rotation

import o80
import o80_pam
import pam_interface
import pam_mujoco

# import pam_vicon
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


@dataclasses.dataclass
class Boundaries3d:
    """Represents min/max boundaries in a 3-dimensional space."""

    min: t.Tuple[float, float, float]
    max: t.Tuple[float, float, float]


@dataclasses.dataclass
class HysrOneBallConfig:
    """Configuration for HysrOneBall."""

    # NOTE: Unfortunately, OmegaConf is a bit limited regarding the types.  It only
    # supports primitive types, enums and a few basic containers.  Nesting of containers
    # (e.g. list of tuples), unions or custom types are not supported.  For these cases
    # Any is used, which basically disables type checking for the corresponding
    # parameters.

    # oc.MISSING indicates that the value is mandatory (i.e. must be provided by the
    # user).

    real_robot: bool = oc.MISSING
    robot_type: pam_mujoco.RobotType = oc.MISSING
    o80_pam_time_step: float = oc.MISSING
    mujoco_time_step: float = oc.MISSING
    algo_time_step: float = oc.MISSING
    pam_config_file: pathlib.Path = oc.MISSING
    robot_position: t.List[float] = oc.MISSING
    robot_orientation: t.Any = oc.MISSING  # Rotation
    table_position: t.List[float] = oc.MISSING
    table_orientation: t.Any = oc.MISSING  # Rotation
    target_position: t.List[float] = oc.MISSING
    reference_posture: t.List[t.Any] = oc.MISSING  # t.List[t.Tuple[float, float]]
    starting_pressures: t.List[t.Any] = oc.MISSING  # t.List[t.Tuple[float, float]]
    world_boundaries: Boundaries3d = oc.MISSING
    pressure_change_range: int = oc.MISSING
    trajectory: int = oc.MISSING
    accelerated_time: bool = oc.MISSING
    save_data: bool = oc.MISSING
    save_folder: str = "/tmp/"
    graphics_pseudo_real: bool = False
    graphics_simulation: bool = False
    graphics_extra_balls: bool = False
    instant_reset: bool = oc.MISSING
    nb_steps_per_episode: int = oc.MISSING
    extra_balls_sets: int = oc.MISSING
    extra_balls_per_set: int = oc.MISSING
    trajectory_group: str = oc.MISSING
    frequency_monitoring_step: bool = oc.MISSING
    frequency_monitoring_episode: bool = oc.MISSING
    robot_integrity_check: bool = oc.MISSING
    robot_integrity_threshold: float = oc.MISSING

    use_vicon: bool = False
    """If true, get robot and table pose from the Vicon system.

    This requires a pam_vicon back end to be running, which provides the Vicon data.
    The segment id used for this can be set via :attr:`vicon_segment_id`.

    If enabled, the values of :attr:`robot_position`, :attr:`robot_orientation`,
    :attr:`table_position` and :attr:`table_orientation` are overwritten with the data
    provided by the Vicon system.
    """

    vicon_segment_id: str = "vicon"
    """Segment ID used to access Vicon data (only used if use_vicon=True)."""

    graphics: bool = oc.MISSING
    xterms: bool = oc.MISSING

    # implement __{get,set}item__ to add dictionary-like access
    def __getitem__(self, key: str) -> t.Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: t.Any) -> None:
        setattr(self, key, value)

    @staticmethod
    def from_json(jsonpath: t.Union[str, os.PathLike]) -> t.Any:
        """Construct config from JSON file."""
        wconf = variconf.WConf(HysrOneBallConfig)
        wconf.load_file(jsonpath)

        # Convert the omegaconf DictConfig to a plain HysrOneBallConfig to disable the
        # automatic type checking.  This is needed because otherwise it is not possible
        # to overwrite fields with non-primitive types (even if they are annotated as
        # Any), for example when converting orientation quaternions to Rotation
        # instances.
        cfg = t.cast(HysrOneBallConfig, oc.OmegaConf.to_object(wconf.cfg))

        # expand '~'
        cfg.pam_config_file = cfg.pam_config_file.expanduser()

        if cfg.use_vicon:
            try:
                # get Vicon data via o80 (requires back end to be running in separate
                # process)
                vicon = pam_vicon.PamVicon(cfg.vicon_segment_id)
                vicon.update()
                robot_pose = vicon.get_robot_pose()
                table_pose = vicon.get_table_pose(yaw_only=True)
                logging.info("Get robot pose from Vicon: %s", robot_pose)
                logging.info("Get table pose from Vicon: %s", table_pose)
            except Exception as e:
                msg = f"Failed to get robot/table pose from Vicon: {e}"
                raise RuntimeError(msg) from e

            cfg.robot_position = robot_pose.translation
            cfg.robot_orientation = robot_pose.rotation
            cfg.table_position = table_pose.translation
            cfg.table_orientation = table_pose.rotation

        # convert orientation to Rotation instance
        orientation_fields = ["robot_orientation", "table_orientation"]
        for field in orientation_fields:
            try:
                value = cfg[field]
                if not isinstance(value, Rotation):
                    cfg[field] = Rotation.from_quat(value)
            except ValueError as e:
                raise ValueError(
                    "Unable to parse %s from file %s.  Expect quaternion [x, y, z, w]."
                    "  Error is '%s'" % (field, jsonpath, e)
                ) from e

        logging.debug("Load config from file '%s':\n %s", jsonpath, cfg)

        return cfg

    @staticmethod
    def default_path() -> str:
        """Get path to default config file.

        Raises:
            FileNotFoundError: if no default config file is found.
        """
        global_install = os.path.join(
            sys.prefix,
            "local",
            "learning_table_tennis_from_scratch_config",
            "hysr_one_ball_default.json",
        )
        assert site.USER_BASE is not None
        local_install = os.path.join(
            site.USER_BASE,
            "learning_table_tennis_from_scratch_config",
            "hysr_one_ball_default.json",
        )

        if os.path.isfile(local_install):
            return local_install
        if os.path.isfile(global_install):
            return global_install

        raise FileNotFoundError("No default config file found.")


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

    def __init__(
        self,
        line: t.Optional[
            t.Tuple[t.Sequence[float], t.Sequence[float], float, float]
        ] = None,
        index: t.Optional[int] = None,
        random: bool = False,
    ):
        if not hasattr(self.__class__, "_trajectory_reader"):
            raise UnboundLocalError(
                "_BallBehavior: the classmethod read_trajectories(group:str) "
                "has to be called before the constructor"
            )

        not_false = [a for a in (line, index) if a is not None] + (
            [random] if random else []
        )
        if len(not_false) == 0:
            raise ValueError("type of ball behavior not specified")
        if len(not_false) > 1:
            raise ValueError("type of ball behavior over-specified")
        if line is not None:
            self.type = self.LINE
            self.value = line
        elif index is not None:
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
                hysr_config.save_data,
                hysr_config.save_folder,
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

    def _share_step_number(self, step_number):
        # write the step number in a memory shared
        # with the instances of mujoco
        for mujoco_id in self._mujoco_ids:
            shared_memory.set_long_int(mujoco_id, "step", step_number)

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
        _, ball_position, ball_velocity = self._ball_communication.get()
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
        iterator = context.ball_trajectories.BallTrajectories.iterate(trajectory)
        # setting the ball to the first trajectory point
        self._ball_communication.set(trajectory[1][0], [0, 0, 0])
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
            (
                extra_ball.ball_behavior.get_trajectory()
                if extra_ball.ball_behavior is not None
                else None
            )
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
            iterator = context.ball_trajectories.BallTrajectories.iterate(trajectory)
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
        KP = [0.8, -3.0, 1.2, -1.0]
        KI = [0.015, -0.25, 0.02, -0.05]
        KD = [0.04, -0.09, 0.09, -0.09]
        NDP = [-0.3, -0.5, -0.34, -0.48]
        TIME_STEP = 0.01  # seconds
        QD_DESIRED = [0.7, 0.7, 0.7, 0.7]  # radian per seconds
        _, _, Q_CURRENT, _ = self._pressure_commands.read()

        # configuration for HYSR
        NB_SIM_BURSTS = int((TIME_STEP / self._hysr_config.mujoco_time_step) + 0.5)

        # configuration for accelerated time
        if self._accelerated_time:
            NB_ROBOT_BURSTS = int((TIME_STEP / hysr_config.o80_pam_time_step) + 0.5)

        # configuration for real time
        if not self._accelerated_time:
            frequency_manager = o80.FrequencyManager(1.0 / TIME_STEP)

        # applying the controller twice yields better results
        for _ in range(2):
            _, _, q_current, _ = self._pressure_commands.read()

            # the position controller
            controller = o80_pam.PositionController(
                q_current,
                position,
                QD_DESIRED,
                self._pam_config,
                KP,
                KD,
                KI,
                NDP,
                TIME_STEP,
            )

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
                    self._pressure_commands.set(pressures, burst=NB_ROBOT_BURSTS)
                else:
                    # Should start acting now in the background if not accelerated time
                    self._pressure_commands.set(pressures, burst=False)
                    frequency_manager.wait()

    def reset(self):
        # what happens during reset does not correspond
        # to any step/episode (-1 means: no active step/episode)
        self._share_step_number(-1)
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

        # resetting ball info, e.g. min distance ball/racket, etc
        self._ball_status.reset()
        for ball in self._extra_balls:
            ball.ball_status.reset()

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
        self._share_step_number(self._step_number)
        self._episode_number += 1
        self._share_episode_number(self._episode_number)

        # returning an observation
        return self._create_observation()

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

        # ball fell below the table
        # note : all prerecorded trajectories are added a last ball position
        # with z = -10.0, to insure this always occurs.
        # see: function reset
        if self._ball_status.ball_position[2] < self._hysr_config.target_position[2]:
            return True
        # in case the user called the method
        # force_episode_over
        if self._force_episode_over:
            return True

        return False

    def get_ball_position(self):
        # returning current ball position
        _, ball_position, _ = self._ball_communication.get()
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
        _, ball_position, ball_velocity = self._ball_communication.get()

        # getting information about simulated balls
        def commented():
            if self._extra_balls_frontend is not None:
                observation = self._extra_balls_frontend.latest()
                # robot racket cartesian position
                robot_cartesian_position = (
                    observation.get_extended_state().robot_position
                )
                # list: for each ball, if a contact occured during this episode so far
                # (not necessarily during previous step)
                contacts = observation.get_extended_state().contacts
                # ball position and velocity
                state = observation.get_observed_states()
                ball_0_position = state.get(0).get_position()
                ball_0_velocity = state.get(0).get_velocity()
                print(
                    robot_cartesian_position,
                    contacts[0],
                    ball_0_position,
                    ball_0_velocity,
                )

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
        self._share_step_number(self._step_number)

        # returning
        return observation, reward, episode_over

    def close(self):
        if self._robot_integrity is not None:
            self._robot_integrity.close()
        self._parallel_burst.stop()
        shared_memory.clear_shared_memory(SEGMENT_ID_EPISODE_FREQUENCY)
        shared_memory.clear_shared_memory(SEGMENT_ID_STEP_FREQUENCY)
        pass
