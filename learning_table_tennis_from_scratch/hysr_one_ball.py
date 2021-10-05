import os, sys, time, math, random, json, site, threading
import o80, o80_pam, pam_mujoco, context, pam_interface, frequency_monitoring, shared_memory
import numpy as np
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


class HysrOneBallConfig:

    __slots__ = (
        "o80_pam_time_step",
        "mujoco_time_step",
        "algo_time_step",
        "pam_config_file",
        "robot_position",
        "target_position",
        "reference_posture",
        "world_boundaries",
        "pressure_change_range",
        "trajectory",
        "accelerated_time",
        "graphics_pseudo_real",
        "graphics_simulation",
        "graphics_extra_balls",
        "instant_reset",
        "nb_steps_per_episode",
        "extra_balls_sets",
        "extra_balls_per_set",
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
            except:
                raise ValueError(
                    "failed to find the attribute {} " "in {}".format(s, jsonpath)
                )
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

    _trajectory_reader = context.BallTrajectories()

    def __init__(self, line=False, index=False, random=False):
        not_false = [a for a in (line, index, random) if a != False]
        if not not_false:
            raise ValueError("type of ball behavior not specified")
        if len(not_false) > 1:
            raise ValueError("type of ball behavior over-specified")
        if line != False:
            self.type = self.LINE
            self.value = line
        elif index != False:
            self.type = self.INDEX
            self.value = index
        elif random != False:
            self.type = self.RANDOM

    def get_trajectory(self):
        # ball behavior is a straight line, self.value is (start,end,duration ms)
        if self.type == self.LINE:
            trajectory_points = context.duration_line_trajectory(*self.value)
            return trajectory_points
        # ball behavior is a specified pre-recorded trajectory
        if self.type == self.INDEX:
            trajectory_points = self._trajectory_reader.get_trajectory(self.value)
            return trajectory_points
        # ball behavior is a randomly selected pre-recorded trajectory
        if self.type == self.RANDOM:
            _, trajectory_points = self._trajectory_reader.random_trajectory()
            return trajectory_points

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


def _get_extra_balls(setid, nb_balls, robot_position, target_position, graphics):

    values = configure_mujoco.configure_extra_set(
        setid, nb_balls, robot_position, graphics
    )

    handle = values[0]
    mujoco_id = values[1]
    extra_balls_segment_id = values[2]
    robot_segment_id = values[3]
    ball_segment_ids = values[4]
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

    ball_status = [context.BallStatus(target_position) for _ in range(nb_balls)]

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

        # to control the real (or pseudo-real) robot (pressure control)
        self._real_robot_handle = configure_mujoco.configure_pseudo_real(
            hysr_config.pam_config_file,
            graphics=hysr_config.graphics_pseudo_real,
            accelerated_time=hysr_config.accelerated_time,
        )
        self._mujoco_ids.append(self._real_robot_handle.get_mujoco_id())

        # to control the simulated robot (joint control)
        self._simulated_robot_handle = configure_mujoco.configure_simulation(
            robot_position=hysr_config.robot_position,
            graphics=hysr_config.graphics_simulation,
        )
        self._mujoco_ids.append(self._simulated_robot_handle.get_mujoco_id())

        # where we want to shoot the ball
        self._target_position = hysr_config.target_position
        self._goal = self._simulated_robot_handle.interfaces[SEGMENT_ID_GOAL]

        # to read all recorded trajectory files
        self._trajectory_reader = context.BallTrajectories()

        # if requested, logging info about the frequencies of the steps and/or the
        # episodes
        if hysr_config.frequency_monitoring_step:
            segment_id = hysr_config.frequency_monitoring_step
            size = 1000
            self._frequency_monitoring_step = frequency_monitoring.FrequencyMonitoring(
                SEGMENT_ID_STEP_FREQUENCY, size
            )
        else:
            self._frequency_monitoring_step = None
        if hysr_config.frequency_monitoring_episode:
            segment_id = hysr_config.frequency_monitoring_episode
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
        self._pressure_commands = self._real_robot_handle.interfaces[
            SEGMENT_ID_PSEUDO_REAL_ROBOT
        ]

        # the posture in which the robot will reset itself
        # upon reset (may be None if no posture reset)
        self._reference_posture = hysr_config.reference_posture

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

        # will be used to move the robot to reference posture
        # between episodes
        self._max_pressures1 = [(18000, 18000)] * 4
        self._max_pressures2 = [(21500, 21500)] * 4

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
                    setid,
                    hysr_config.extra_balls_per_set,
                    hysr_config.robot_position,
                    hysr_config.target_position,
                    hysr_config.graphics_extra_balls,
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
        trajectory_points = self._ball_behavior.get_trajectory()
        # setting the ball to the first trajectory point
        self._ball_communication.set(
            trajectory_points[0].position, trajectory_points[0].velocity
        )
        self._ball_status.ball_position = trajectory_points[0].position
        self._ball_status.ball_velocity = trajectory_points[0].velocity
        # shooting the ball
        self._ball_communication.play_trajectory(trajectory_points, overwrite=False)

    def _load_extra_balls(self):
        # load the trajectory of each extra balls, as set by their
        # ball_behavior attribute. See method set_extra_ball_behavior
        # in this file
        item3d = o80.Item3dState()
        sampling_rate_ms = self._trajectory_reader.get_sampling_rate_ms()
        duration = o80.Duration_us.milliseconds(int(sampling_rate_ms))
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
            # going to first trajectory point
            item3d.set_position(trajectory[0].position)
            item3d.set_velocity(trajectory[0].velocity)
            ball.frontend.add_command(index_ball, item3d, o80.Mode.OVERWRITE)
            # loading full trajectory
            for item in trajectory[1:]:
                item3d.set_position(item.position)
                item3d.set_velocity(item.velocity)
                ball.frontend.add_command(index_ball, item3d, duration, o80.Mode.QUEUE)
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

        # "natural": can be applied on the real robot
        # (i.e. send pressures commands to get the robot
        # in a vertical position)

        # aligning the mirrored robot with
        # (pseudo) real robot
        mirroring.align_robots(self._pressure_commands, self._mirrorings)

        # resetting real robot to "vertical" position
        # tripling down to ensure reproducibility
        for (max_pressures, duration) in zip(
            (self._max_pressures1, self._max_pressures2), (0.5, 2)
        ):
            mirroring.go_to_pressure_posture(
                self._pressure_commands,
                self._mirrorings,
                max_pressures,
                duration,
                self._accelerated_time,
                parallel_burst=self._parallel_burst,
            )

    def _do_instant_reset(self):

        # "instant": reset all mujoco instances
        # to their starting state. Not applicable
        # to real robot

        self._real_robot_handle.reset()
        self._simulated_robot_handle.reset()
        for handle in _ExtraBall.handles.values():
            handle.reset()

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

        # going back to vertical position
        if self._instant_reset:
            self._do_instant_reset()
        else:
            self._do_natural_reset()

        # moving the goal to the target position
        self._goal.set(self._target_position, [0, 0, 0])

        # moving real robot back to reference posture
        if self._reference_posture:
            for duration in (0.5, 1.0):
                mirroring.go_to_pressure_posture(
                    self._pressure_commands,
                    self._mirrorings,
                    self._reference_posture,
                    duration,  # in 1 seconds
                    self._accelerated_time,
                    parallel_burst=self._parallel_burst,
                )

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

        # ball falled below the table
        # note : all prerecorded trajectories are added a last ball position
        # with z = -10.0, to insure this always occurs.
        # see: function reset
        if self._ball_status.ball_position[2] < -0.5:
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

        # returning
        return observation, reward, episode_over

    def close(self):
        if self._robot_integrity is not None:
            self._robot_integrity.close()
        self._parallel_burst.stop()
        shared_memory.clear_shared_memory(SEGMENT_ID_EPISODE_FREQUENCY)
        shared_memory.clear_shared_memory(SEGMENT_ID_STEP_FREQUENCY)
        pass
