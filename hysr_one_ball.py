import time
import math
import random
import o80
import o80_pam
import pam_mujoco
import context


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
                 target_position,
                 reward_normalization_constant,
                 smash_task,
                 period_ms=100,
                 rtt_cap=0.2):

        # reward configuration
        self._c = reward_normalization_constant 
        self._smash_task = smash_task # True or False (return task)
        self._rtt_cap = rtt_cap


        # o80 segment id for contacting backend running in pseudo-real robot
        self._segment_id_pressures = mujoco_id+"_pressures"
        self._mujoco_id_pressures = self._segment_id_pressures+"_mujoco"
        
        # o80 segment id for contacting backends running in simulated robot/ball
        self._segment_id_ball = mujoco_id+"_ball"
        self._segment_id_mirror_robot = mujoco_id+"_robot"
        self._segment_id_contact_robot = mujoco_id+"_contact_robot"
        self._mujoco_id_ball = self._segment_id_ball+"_mujoco"
        
        
        # start pseudo-real robot (pressure controlled)
        if not real_robot:
            self._process_pressures = pam_mujoco.start_mujoco.pseudo_real_robot(self._mujoco_id_pressures,
                                                                                self._segment_id_pressures)
        else:
            self._process_pressures = None
            
        # start simulated ball and robot
        self._process_sim = pam_mujoco.start_mujoco.ball_and_robot(self._mujoco_id_ball,
                                                                   self._segment_id_mirror_robot,
                                                                   self._segment_id_contact_robot,
                                                                   self._segment_id_ball)

        # starting a process that has the simulated robot
        # mirroring the pseudo-real robot
        # (uses o80 in the background)
        self._mirror_id = mujoco_id+"_mirror"
        mirroring_period_ms = 1
        self._process_mirror = o80_pam.start_mirroring(self._mirror_id,
                                                       self._segment_id_mirror_robot,
                                                       self._segment_id_pressures,
                                                       mirroring_period_ms)

        # to get information regarding the ball
        self._ball_communication = o80_pam.o80Ball(self._segment_id_ball)
        
        # to send pressure commands to the real or pseudo-real robot
        self._pressure_commands = o80_pam.o80Pressures(self._segment_id_pressures)
        
        # will encapsulate all information
        # about the ball (e.g. min distance with racket, etc)
        self._ball_status = context.BallStatus(target_position)

        
    def reset(self):

        # resetting ball info, e.g. min distance ball/racket, etc
        self._ball_status.reset()
        # resetting ball/robot contact information
        pam_mujoco.reset_contact(self._segment_id_contact_robot)
        # shooting a ball
        self._ball_communication.shoot()
        time.sleep(0.3)

        
    def _episode_over(self):

        over = False
        
        # ball falled below the table
        if self._ball_status.min_z < -0.4:
            over = True

        # ball passed the racket
        elif self._ball_status.max_y > 0.1:
            over = True

        return over
    
    # action assumed to be [(pressure ago, pressure antago), (pressure_ago, pressure_antago), ...]
    def step(self,action):

        # getting information about simulated ball
        ball_position,ball_velocity = self._ball_communication.get()

        # getting ball/racket contact information
        racket_contact_information = pam_mujoco.get_contact(self._segment_id_contact_robot)

        # updating ball status
        self._ball_status.update(ball_position,ball_velocity,
                                 racket_contact_information)

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
            
            
        # generating observation
        pressures_ago,pressures_antago,_,__ = self._pressure_commands.read()
        observation = _Observation(pressures_ago,pressures_antago,
                                   self._ball_status.ball_position,
                                   self._ball_status.ball_velocity)
        
        # sending pressures to pseudo real robot
        self._pressure_commands.set(action)

        return observation,reward,self._episode_over()

    
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



        

        
