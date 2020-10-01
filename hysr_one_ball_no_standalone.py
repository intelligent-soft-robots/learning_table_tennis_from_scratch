import time
import math
import random
import o80
import o80_pam
import pam_mujoco
import context

SEGMENT_ID_BALL = "mujoco_ball"
SEGMENT_ID_CONTACT_ROBOT = "mujoco_contact_robot"
SEGMENT_ID_PSEUDO_REAL_ROBOT = "o80_pam"


def _reward( min_distance_ball_target,
             min_distance_ball_racket, # None if ball hit racket
             max_ball_velocity, # None if should not be taken into account (return task)
             c,rtt_cap ): 

    # returning r_hit

    if min_distance_ball_racket is not None:
        return -min_distance_ball_racket
    
    # returning r_tt

    reward = 1.- c * min_distance_ball_target ** 0.75
    
    # - return task
    if max_ball_velocity is None: 
        reward = max(reward,rtt_cap)
        return reward

    # - smash task
    reward = reward * max_ball_velocity
    reward = max(reward,rtt_cap)
    return reward
            

class _Observation:

    def __init__(self,
                 pressures_ago,
                 pressures_antago,
                 ball_position,
                 ball_velocity):
        self.pressures_ago = pressures_ago
        self.pressures_antago = pressures_antago
        self.ball_position = ball_position
        self.ball_velocity = ball_velocity

    
class HysrOneBall:

    def __init__(self,
                 mujoco_id,
                 real_robot,
                 reference_posture,
                 target_position,
                 period_ms,
                 reward_normalization_constant,
                 smash_task,
                 rtt_cap=0.2,
                 trajectory_index=None):

        # if a number, the same trajectory will be played
        # all the time. If None, a random trajectory will be
        # played each time
        self._trajectory_index = trajectory_index
        
        # the posture in which the robot will reset itself
        # upon reset
        self._reference_posture = reference_posture

        # the robot will interpolate between current and
        # target posture over this duration
        self._period_ms = period_ms
        
        # reward configuration
        self._c = reward_normalization_constant 
        self._smash_task = smash_task # True or False (return task)
        self._rtt_cap = rtt_cap

        # to get information regarding the ball
        self._ball_communication = o80_pam.o80Ball(SEGMENT_ID_BALL)
        
        # to send pressure commands to the real or pseudo-real robot
        self._pressure_commands = o80_pam.o80Pressures(SEGMENT_ID_PSEUDO_REAL_ROBOT)
        
        # will encapsulate all information
        # about the ball (e.g. min distance with racket, etc)
        self._ball_status = context.BallStatus(target_position)

        
    def reset(self):

        # resetting ball info, e.g. min distance ball/racket, etc
        self._ball_status.reset()
        # resetting ball/robot contact information
        pam_mujoco.reset_contact(SEGMENT_ID_CONTACT_ROBOT)
        time.sleep(0.1)
        # moving the robot back to reference posture
        if self._reference_posture is not None:
            self._pressure_commands.set(self._reference_posture,
                                        duration_ms=1500,
                                        wait=True)
        # getting a new trajectory
        if self._trajectory_index is not None:
            trajectory_points = context.BallTrajectories().get_trajectory(self._trajectory_index)
        else:
            trajectory_index,trajectory_points = context.BallTrajectories().random_trajectory()   
        # setting the last trajectory point way below the table, to be sure
        # end of episode will be detected
        last_state = context.State([0,0,-10.00],[0,0,0])
        trajectory_points.append(last_state)
        # setting the ball to the first trajectory point
        self._ball_communication.set(trajectory_points[0].position,
                                     trajectory_points[0].velocity,
                                     wait=True)
        # shooting the ball
        self._ball_communication.play_trajectory(trajectory_points)

            
    def _episode_over(self):

        over = False
        
        # ball falled below the table
        # note : all prerecorded trajectories are added a last ball position
        # with z = -0.41, to insure this always occurs.
        # see: o80_pam / o80_ball.py
        if self._ball_status.ball_position[2] < -0.5:
            over = True

        return over

    
    # action assumed to be [(pressure ago, pressure antago), (pressure_ago, pressure_antago), ...]
    def step(self,action):

        # getting information about simulated ball
        ball_position,ball_velocity = self._ball_communication.get()

        # getting ball/racket contact information
        # note : racket_contact_information is an instance
        #        of context.ContactInformation
        racket_contact_information = pam_mujoco.get_contact(SEGMENT_ID_CONTACT_ROBOT)

        # updating ball status
        self._ball_status.update(ball_position,ball_velocity,
                                 racket_contact_information)

        # generating observation
        pressures_ago,pressures_antago,_,__ = self._pressure_commands.read()
        observation = _Observation(pressures_ago,pressures_antago,
                                   self._ball_status.ball_position,
                                   self._ball_status.ball_velocity)
        
        # sending pressures to pseudo real robot
        #self._pressure_commands.set(action,duration_ms=self._period_ms,wait=False)
        self._pressure_commands.set(action,duration_ms=None,wait=False)

        reward = 0
        episode_over = self._episode_over()

        if episode_over:
            # computing reward
            if self._smash_task:
                reward = _reward( self._ball_status.min_distance_ball_target,
                                  self._ball_status.min_distance_ball_racket,
                                  self._ball_status.max_ball_velocity,
                                  self._c,self._rtt_cap)
            else:
                reward = _reward( self._ball_status.min_distance_ball_target,
                                  self._ball_status.min_distance_ball_racket,
                                  None,
                                  self._c,self._rtt_cap)
            
            
        
        return observation,reward,episode_over

    
    def close(self):

        # stopping mirroring
        o80_pam.stop_mirroring(self._mirror_id)
        # stopping pseudo real robot
        if self._process_pressures:
            pam_mujoco.request_stop(self._mujoco_id_pressures)
        # stopping simulated robot/ball
        pam_mujoco.request_stop(self._mujoco_id_ball)
        # waiting for all corresponding processes
        # to finish
        if self._process_pressures:
            self._process_pressures.join()
        self._process_sim.join()
        self._process_mirror.join()



        

        
