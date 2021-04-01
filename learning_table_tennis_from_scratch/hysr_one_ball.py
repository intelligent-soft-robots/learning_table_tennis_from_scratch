import os,sys,time,math,random,json
import o80,o80_pam,pam_mujoco,context,pam_interface
from pam_mujoco import mirroring
import numpy as np

SEGMENT_ID_BALL = pam_mujoco.segment_ids.ball
SEGMENT_ID_GOAL = pam_mujoco.segment_ids.goal
SEGMENT_ID_HIT_POINT = pam_mujoco.segment_ids.hit_point
SEGMENT_ID_CONTACT_ROBOT = pam_mujoco.segment_ids.contact_robot
SEGMENT_ID_ROBOT_MIRROR = pam_mujoco.segment_ids.mirroring
SEGMENT_ID_PSEUDO_REAL_ROBOT = o80_pam.segment_ids.robot


class HysrOneBallConfig:

    __slots__ = ("o80_pam_time_step","mujoco_time_step","algo_time_step",
                 "target_position","reference_posture","world_boundaries",
                 "pressure_change_range","trajectory")

    def __init__(self):
        for s in self.__slots__:
            setattr(self,s,None)

    def get(self):
        r = {s:getattr(self,s)
             for s in self.__slots__}
        return r
        
    @classmethod
    def from_json(cls,jsonpath):
        if not os.path.isfile(jsonpath):
            raise FileNotFoundError("failed to find hysr configuration file: {}".format(jsonpath))
        try :
            with open(jsonpath) as f:
                conf=json.load(f)
        except Exception as e:
            raise ValueError("failed to parse reward json configuration file {}: {}".format(jsonpath,e))
        instance=cls()
        for s in cls.__slots__:
            try:
                setattr(instance,s,conf[s])
            except:
                raise ValueError("failed to find the attribute {} "
                                 "in {}".format(s,jsonpath))
        return instance

    @staticmethod
    def default_path():
        return os.path.join(sys.prefix,
                            "learning_table_tennis_from_scratch_config",
                            "hysr_one_ball_default.json")
    

class _BallBehavior:
    '''
    HYSROneBall supports 4 ball behavior:
    fixed: a 3d tuple, corresponds to the ball at 
           the corresponding fixed position
    line: (3d tuple, 3d tuple, float): ball going from
          start to end position in straight line at 
          the provided velocity
    index: (positive int) ball playing the pre-recorded trajectory
           corresponding to the index
    random: (True) ball playing a random pre-recorded trajectory
    '''
    POSITION = 0
    TRAJECTORY = 1
    LINE = -1
    INDEX = -2
    RANDOM = -3
    def __init__(self,
                 fixed=False,
                 line=False,
                 index=False,
                 random=False):
        # checking input
        not_false = [a for a in (fixed,line,index,random)
                      if a!=False]
        if not not_false:
            raise ValueError("type of ball behavior not specified")
        if len(not_false)>1:
            raise ValueError("type of ball behavior over-specified")
        # setting primary type (fixed position or trajectory)
        # and if trajectory, secondary type (line trajectory,
        # indexed pre-recorded trajectory or randomly selected
        # trajectory
        if fixed!=False:
            self.primary_type=self.POSITION
            self.value = fixed
        else:
            self.primary_type=self.TRAJECTORY
            if line!=False:
                self.secondary_type=self.LINE
                self.value = line
            elif index!=False:
                self.secondary_type=self.INDEX
                self.value = index
            elif random!=False:
                self.secondary_type=self.RANDOM
    def is_trajectory(self):
        return self.primary_type == self.TRAJECTORY
    def is_fixed(self):
        return self.primary_type == self.POSITION
    def get_trajectory(self):
        if not self.is_trajectory():
            raise ValueError("requesting trajectory from a non trajectory ball behavior")
        # ball behavior is a straight line, self.value is (start,end,velocity)
        if self.secondary_type==self.LINE:
            trajectory_points = context.line_trajectory(*self.value)
            return trajectory_points
        # ball behavior is a specified pre-recorded trajectory
        if self.secondary_type==self.INDEX:
            trajectory_points = context.BallTrajectories().get_trajectory(self.value)
            return trajectory_points
        if self.secondary_type==self.RANDOM:
            _,trajectory_points = context.BallTrajectories().random_trajectory()
            return trajectory_points
    def is_line(self):
        if not self.primary_type==self.TRAJECTORY:
            return False
        if self.secondary_type==self.LINE:
            return True
    def get(self):
        return self.value
        
                 

    
def _convert_pressures_in(pressures):
    # convert pressure from [ago1, antago1, ago2, antago2, ...]
    # to [(ago1, antago1), (ago2, antago2), ...]
    return list(zip(pressures[::2],pressures[1::2]))


def _convert_pressures_out(pressures_ago,pressures_antago):
    pressures = list(zip(pressures_ago,pressures_antago))
    return [p for sublist in pressures for p in sublist]


class _Observation:

    def __init__(self,
                 joint_positions,
                 joint_velocities,
                 pressures,
                 ball_position,
                 ball_velocity):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.pressures = pressures
        self.ball_position = ball_position
        self.ball_velocity = ball_velocity
    
        
class HysrOneBall:

    def __init__(self,
                 hysr_config,
                 accelerated_time,
                 reward_function):
        
        # moving the goal to the target position
        goal = o80_pam.o80Goal(SEGMENT_ID_GOAL)
        goal.set(hysr_config.target_position,[0,0,0])

        # if o80_pam (i.e. the pseudo real robot)
        # has been started in accelerated time,
        # the corresponding o80 backend will burst through
        # an algorithm time step
        self._accelerated_time = accelerated_time
        if accelerated_time:
            self._o80_time_step = hysr_config.o80_pam_time_step
            self._nb_robot_bursts = int(hysr_config.algo_time_step/hysr_config.o80_pam_time_step)

        # pam_mujoco (i.e. simulated ball and robot) should have been
        # started in accelerated time. It burst through algorithm
        # time steps
        self._mujoco_time_step = hysr_config.mujoco_time_step
        self._nb_sim_bursts = int(hysr_config.algo_time_step/hysr_config.mujoco_time_step)
        
        # the config sets either a zero or positive int (playing the corresponding indexed
        # pre-recorded trajectory) or a negative int (playing randomly selected indexed
        # trajectories)
        if hysr_config.trajectory>=0:
            self._ball_behavior = _BallBehavior(index=hysr_config.trajectory)
        else:
            self._ball_behavior = _BallBehavior(random=True)
        
        # the robot will interpolate between current and
        # target posture over this duration
        self._period_ms = hysr_config.algo_time_step
        
        # reward configuration
        self._reward_function = reward_function

        # to get information regarding the ball
        self._ball_communication = o80_pam.o80Ball(SEGMENT_ID_BALL)
        
        # to send pressure commands to the real or pseudo-real robot
        self._pressure_commands = o80_pam.o80Pressures(SEGMENT_ID_PSEUDO_REAL_ROBOT)

        # the posture in which the robot will reset itself
        # upon reset (may be None if no posture reset)
        self._reference_posture = hysr_config.reference_posture
        
        # will encapsulate all information
        # about the ball (e.g. min distance with racket, etc)
        self._ball_status = context.BallStatus(hysr_config.target_position)
        
        # to send mirroring commands to simulated robot
        self._mirroring = o80_pam.o80RobotMirroring(SEGMENT_ID_ROBOT_MIRROR)

        # to move the hit point marker
        self._hit_point = o80_pam.o80HitPoint(SEGMENT_ID_HIT_POINT)
        
        # tracking if this is the first step of the episode
        # (a step sets it to false, reset sets it back to true)
        self._first_episode_step = True

        # when starting, the real robot and the pseudo robot
        # may not be aligned, which may result in graphical issues
        mirroring.align_robots(self._pressure_commands,
                               self._mirroring)

        # will be used to move the robot to reference posture
        # between episodes
        self._max_pressures1 = [(18000,18000)]*4
        self._max_pressures2 = [(20000,20000)]*4

        # normally an episode ends when the ball z position goes
        # below a certain threshold (see method _episode_over)
        # this is to allow user to force ending an episode
        # (see force_episode_over method)
        self._force_episode_over = False
        

    def force_episode_over(self):
        # will trigger the method _episode_over
        # (called in the step method) to return True
        self._force_episode_over = True

        
    def set_ball_behavior(self,
                          fixed=False,
                          line=False,
                          index=False,
                          random=False):
        # overwrite the ball behavior (set to a trajectory in the constructor)
        # see comments in _BallBehavior, in this file
        self._ball_behavior=_BallBehavior(fixed=fixed,
                                          line=line,
                                          index=index,
                                          random=random)

        
    def _create_observation(self):
        (pressures_ago,pressures_antago,
         joint_positions,joint_velocities) = self._pressure_commands.read()
        observation = _Observation(joint_positions,joint_velocities,
                                   _convert_pressures_out(pressures_ago,
                                                          pressures_antago),
                                   self._ball_status.ball_position,
                                   self._ball_status.ball_velocity)
        return observation
    
        
    def get_robot_iteration(self):
        return self._pressure_commands.get_iteration()

    
    def get_ball_iteration(self):
        return self._ball_communication.get_iteration()


    def get_current_desired_pressures(self):
        (pressures_ago,pressures_antago,
         _,__) = self._pressure_commands.read(desired=True)
        return pressures_ago,pressures_antago

    
    def get_current_pressures(self):
        (pressures_ago,pressures_antago,
         _,__) = self._pressure_commands.read(desired=False)
        return pressures_ago,pressures_antago

    
    def contact_occured(self):
        return self._ball_status.contact_occured()

    
    def load_ball(self,burst=True):
        # "load" the ball means creating the o80 commands corresponding
        # to the ball behavior (set by the "set_ball_behavior" method) 
        if self._ball_behavior.is_trajectory():
            trajectory_points = self._ball_behavior.get_trajectory()
            # setting the last trajectory point way below the table, to be sure
            # end of episode will be detected
            last_state = context.State([0,0,-10.00],[0,0,0])
            trajectory_points.append(last_state)
            # setting the ball to the first trajectory point
            self._ball_communication.set(trajectory_points[0].position,
                                         trajectory_points[0].velocity)
            self._ball_status.ball_position = trajectory_points[0].position
            self._ball_status.ball_velocity = trajectory_points[0].velocity
            # shooting the ball
            self._ball_communication.play_trajectory(trajectory_points,overwrite=True)
            # moving the ball to initial position
            if burst:
                self._mirroring.burst(5)
        else:
            # ball behavior is a fixed point (x,y,z)
            position = self._ball_behavior.get()
            self._ball_communication.set(position,
                                         (0,0,0))
            self._ball_status.ball_position = position
            self._ball_status.ball_velocity = (0,0,0)
            # moving the ball to initial position
            if burst:
                self._mirroring.burst(5)

                
    def reset_contact(self):
        pam_mujoco.reset_contact(SEGMENT_ID_CONTACT_ROBOT)

        
    def reset(self):

        # in case the episode was forced to end by the
        # user (see force_episode_over method)
        self._force_episode_over = False
        
        # aligning the mirrored robot with
        # (pseudo) real robot
        mirroring.align_robots(self._pressure_commands,
                               self._mirroring)
        
        # resetting first episode step
        self._first_episode_step = True
        
        # resetting the hit point 
        self._hit_point.set([0,0,-0.62],[0,0,0])
        
        # resetting ball info, e.g. min distance ball/racket, etc
        self._ball_status.reset()

        # resetting ball/robot contact information
        pam_mujoco.reset_contact(SEGMENT_ID_CONTACT_ROBOT)
        time.sleep(0.1)

        # resetting real robot to "vertical" position
        # tripling down to ensure reproducibility
        for (max_pressures,duration) in zip( (self._max_pressures1,self._max_pressures2) , (0.5,2) ):
            mirroring.go_to_pressure_posture(self._pressure_commands,
                                             self._mirroring,
                                             max_pressures,
                                             duration, 
                                             self._accelerated_time)

        # moving real robot back to reference posture
        if self._reference_posture:
            for duration in (0.5,1.0):
                mirroring.go_to_pressure_posture(self._pressure_commands,
                                                 self._mirroring,
                                                 self._reference_posture,
                                                 duration, # in 1 seconds
                                                 self._accelerated_time)

        # setting the ball behavior
        self.load_ball()

        # returning an observation
        return self._create_observation()
        
            
    def _episode_over(self):
        over = False
        # ball falled below the table
        # note : all prerecorded trajectories are added a last ball position
        # with z = -10.0, to insure this always occurs.
        # see: function reset
        if self._ball_status.ball_position[2] < -0.5:
            over = True
        # in case the user called the method
        # force_episode_over
        if self._force_episode_over:
            over = True
        return over


    def get_ball_position(self):
        # returning current ball position
        ball_position,_ = self._ball_communication.get()
        return ball_position

    
    def create_observation(self):
        # reading current real (or pseudo real) robot state
        (pressures_ago,pressures_antago,
         joint_positions,joint_velocities) = self._pressure_commands.read()
        # getting information about simulated ball
        return _Observation( joint_positions,joint_velocities,
                             pressures_ago,pressures_antago,
                             self._ball_status.ball_position,
                             self._ball_status.ball_velocity )

        
    # action assumed to be np.array(ago1,antago1,ago2,antago2,...)
    def step(self,action):

        # reading current real (or pseudo real) robot state
        (pressures_ago,pressures_antago,
         joint_positions,joint_velocities) = self._pressure_commands.read()

        # getting information about simulated ball
        ball_position,ball_velocity = self._ball_communication.get()

        # convert action [ago1,antago1,ago2] to list suitable for
        # o80 ([(ago1,antago1),(),...])
        pressures = _convert_pressures_in(list(action))

        # sending action pressures to real (or pseudo real) robot.
        if self._accelerated_time:
            # if accelerated times, running the pseudo real robot iterations
            # (note : o80_pam expected to have started in bursting mode)
            self._pressure_commands.set(pressures,
                                        burst=self._nb_robot_bursts)
        else:
            # Should start acting now in the background if not accelerated time
            self._pressure_commands.set(pressures,burst=False)
        
        # sending mirroring state to simulated robot
        self._mirroring.set(joint_positions,joint_velocities)

        # having the simulated robot/ball performing the right number of iterations
        # (note: simulated expected to run accelerated time)
        self._mirroring.burst(self._nb_sim_bursts)

        # getting ball/racket contact information
        # note : racket_contact_information is an instance
        #        of context.ContactInformation
        racket_contact_information = pam_mujoco.get_contact(SEGMENT_ID_CONTACT_ROBOT)

        # updating ball status
        self._ball_status.update(ball_position,ball_velocity,
                                 racket_contact_information)

        # moving the hit point to the minimal observed distance
        # between ball and target (post racket hit)
        if self._ball_status.min_position_ball_target is not None:
            self._hit_point.set(self._ball_status.min_position_ball_target,[0,0,0])
        
        # observation instance
        observation = _Observation(joint_positions,joint_velocities,
                                   _convert_pressures_out(pressures_ago,
                                                          pressures_antago),
                                   self._ball_status.ball_position,
                                   self._ball_status.ball_velocity)

        # checking if episode is over
        episode_over = self._episode_over()
        reward = 0

        # if episode over, computing related reward
        if episode_over:
            reward = self._reward_function(self._ball_status.min_distance_ball_racket,
                                           self._ball_status.min_distance_ball_target,
                                           self._ball_status.max_ball_velocity)

        # next step can not be the first one
        # (reset will set this back to True)
        self._first_episode_step = False

            
        # returning
        return observation,reward,episode_over

    
    def close(self):
        pass

