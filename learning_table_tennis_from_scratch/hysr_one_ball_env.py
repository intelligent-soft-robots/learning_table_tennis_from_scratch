import o80
import math
import gym
import random
import numpy as np
from .hysr_one_ball import HysrOneBall
from collections import OrderedDict

class Config:

    def __init__(self):
        self.accelerated_time = True
        self.o80_pam_time_step = 0.002
        self.mujoco_time_step = 0.002
        self.algo_time_step = 0.01
        self.target_position = [0.45,2.7,0.17]
        self.reference_posture = [-math.pi/4.0,math.pi/3.5,
                                  math.pi/8.0,0.0]
        self.pressure_min = 7000.
        self.pressure_max = 23000.
        self.pressure_change_range = 1000
        self.reward_normalization_constant = 1.0
        self.smash_task = True
        self.rtt_cap = -0.2
        self.nb_dofs = 4
        self.world_boundaries = { "min":(0.0,-1.0,+0.17), # x,y,z
                                  "max":(1.6,3.5,+1.5) }   
    

class _ObservationSpace:

    # the model does not support gym Dict or Tuple spaces
    # which is very inconvenient. This class implements
    # something similar to a Dict space, but which can
    # be casted to a box space.

    class Box:
        def __init__(self,low,high,size):
            self.low = low
            self.high = high
            self.size = size
        def normalize(self,value):
            return (value-self.low)/(self.high-self.low)
        def denormalize(self,value):
            return self.low + value*(self.high-self.low)
        
    def __init__(self):
        self._obs_boxes = OrderedDict()
        self._values = OrderedDict()
        
    def add_box(self,name,low,high,size):
        self._obs_boxes[name]=_ObservationSpace.Box(low,high,
                                                    size)
        self._values[name]=np.zeros(size)

    def get_gym_box(self):
        size = sum([b.size for b in self._obs_boxes.values()])
        return gym.spaces.Box(low=0.0,high=1.0,
                              shape=(size,),
                              dtype=np.float)

    def set_values(self,name,values):
        normalize = self._obs_boxes[name].normalize
        values_ = np.array(list(map(normalize,values)))
        self._values[name]=values_

    def get_normalized_values(self):
        values = list(self._values.values())
        r =  np.concatenate(values)
        return r
    
        
        
class HysrOneBallEnv(gym.Env):
    
    def __init__(self,
                 config=None):

        super().__init__()

        if config is None:
            self._config = Config()
        else:
            self._config = config

        self._hysr = HysrOneBall(self._config.accelerated_time,
                                 self._config.o80_pam_time_step,
                                 self._config.mujoco_time_step,
                                 self._config.algo_time_step,
                                 self._config.target_position,
                                 self._config.reward_normalization_constant,
                                 self._config.smash_task,
                                 rtt_cap=self._config.rtt_cap,
                                 trajectory_index=None,
                                 reference_posture=self._config.reference_posture,
                                 pam_config=self._config.pam_config)
        
        self.action_space = gym.spaces.Box(low=-1.0,
                                           high=+1.0,
                                           shape=(self._config.nb_dofs*2,),
                                           dtype=np.float)

        
        self._obs_boxes = _ObservationSpace()
        
        self._obs_boxes.add_box("robot_position",-math.pi,+math.pi,
                            self._config.nb_dofs)
        self._obs_boxes.add_box("robot_velocity",0.0,10.0,
                            self._config.nb_dofs)
        self._obs_boxes.add_box("robot_pressure",
                            self._config.pressure_min,
                            self._config.pressure_max,
                            self._config.nb_dofs*2)
        
        self._obs_boxes.add_box("ball_position",
                            min(self._config.world_boundaries["min"]),
                            max(self._config.world_boundaries["max"]),
                            3)
        self._obs_boxes.add_box("ball_velocity",
                            -10.0,+10.0,3)

        self.observation_space = self._obs_boxes.get_gym_box()

        if not self._config.accelerated_time:
            self._frequency_manager = o80.FrequencyManager(1.0/self._config.algo_time_step)
        
        
    def _bound_pressure(self,value):
        return int(max(min(value,
                           self._config.pressure_max),
                       self._config.pressure_min))

    
    def _convert_observation(self,observation):
        self._obs_boxes.set_values("robot_position",observation.joint_positions)
        self._obs_boxes.set_values("robot_velocity",observation.joint_velocities)
        self._obs_boxes.set_values("robot_pressure",observation.pressures)
        self._obs_boxes.set_values("ball_position",observation.ball_position)
        self._obs_boxes.set_values("ball_velocity",observation.ball_velocity)
        return self._obs_boxes.get_normalized_values()

    
    def step(self,action):

        # casting actions from [-1,+1] to [-pressure_change_range,+pressure_change_range]
        action = [self._config.pressure_change_range*a for a in action]
        
        # current pressures
        agos,antagos = self._hysr.get_current_pressures()
        
        # final target pressure is action + current desired
        for dof in range(self._config.nb_dofs):
            action[2*dof] = self._bound_pressure(agos[dof]+action[2*dof])
            action[2*dof+1] = self._bound_pressure(antagos[dof]+action[2*dof+1])

        # hysr takes a list of int, not float, as input
        action = [int(a) for a in action]
            
        # performing a step
        observation,reward,episode_over = self._hysr.step(list(action))

        # formatting observation in a format suitable for gym
        observation = self._convert_observation(observation)

        # imposing frequency to learning agent
        if not self._config.accelerated_time:
            self._frequency_manager.wait()
        
        return observation,reward,episode_over,{}


    def reset(self):
        observation = self._hysr.reset()
        observation = self._convert_observation(observation)
        if not self._config.accelerated_time:
            self._frequency_manager = o80.FrequencyManager(1.0/self._config.algo_time_step)
        return observation
