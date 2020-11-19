import math
import gym
import numpy as np
from .hysr_one_ball import HysrOneBall
from collections import OrderedDict

class Config:

    def __init__(self):
        self.accelerated_time = False
        self.o80_pam_time_step = 0.002
        self.mujoco_time_step = 0.002
        self.algo_time_step = 0.01
        self.target_position = [1.0,4.0,-0.44]
        # for each dof : ago,antago
        self.reference_posture = [[20000,12000],[12000,22000],
                                  [15000,15000],[15000,15000]]
        self.pressure_min = 5000
        self.pressure_max = 25000
        self.pressure_change_range = (-500,500)
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
        self._obs_boxes[name]=ObservationSpace(low,high,
                                           size,dtype)
        self._values[name]=np.zeros(size)

    def get_gym_box(self):
        size = sum([b.size for b in self._obs_boxes.values()])
        return gym.spaces.Box(low=0.0,high=1.0,
                              shape=(size,),
                              dtype=np.float)

    def set_values(self,name,values):
        normalize = self._obs_boxes[name].normalize
        values = np.array(map(normalize,values))
        self._values[name]=values

    def get_normalized_values(self):
        size = sum([b.size for b in self._obs_boxes.values()])
        values = list(self._values.items())
        return np.concatenate(*values)
            
        


    
        
        
class HysrOneBallEnv(gym.GoalEnv):
    
    def __init__(self,
                 config=None):

        if config is None:
            self._config = Config()
        else:
            self._config = config
        
        self._hysr = HysrOneBall(self._config.accelerated_time,
                                 self._config.o80_pam_time_step,
                                 self._config.mujoco_time_step,
                                 self._config.algo_time_step,
                                 self._config.reference_posture,
                                 self._config.target_position,
                                 self._config.reward_normalization_constant,
                                 self._config.smash_task,
                                 rtt_cap=self._config.rtt_cap)

        self._obs_boxes = _ObservationSpace()
        
        self._obs_boxes.add_box("robot_position",-math.pi,+math.pi,
                            self._config.nb_dofs*2)
        self._obs_boxes.add_box("robot_velocity",0.0,10.0,
                            self._config.nb_dofs*2)
        self._obs_boxes.add_box("robot_pressure",
                            self._config.pressure_change_range[0],
                            self._config.pressure_change_range[1],
                            self._config.nb_dofs*2)
        
        self._obs_boxes.add_box("ball_position",
                            min(self._config.world_boundaries["min"]),
                            max(self._config.world_boundaries["max"]),
                            3)
        self._obs_boxes.add_box("ball_velocity",
                            -10.0,+10.0,3)

        self.observation_space = self._obs_boxes.get_gym_box()

        
    def _bound_pressure(self,value):
        return max(min(value,
                       self._config.pressure_max),
                   self._config.pressure_min)

    
    def step(self,action):

        # preparing action in a format suitable
        # for HysrOneBall
        self._action = [[None,None]*config.nb_dofs]
        
        # current desired pressures
        agos,antagos = self.get_current_desired_pressures()
        
        # action is a (8,1) array, each entry being a change
        # of pressure for a muscle. converting to
        # ( (ago,antago) , ...)
        for dof in range(self._config.nb_dofs):
            self._action[dof][0] = self._bound_pressure(agos[dof]+action[2*dof])
            self._action[dof][1] = self._bound_pressure(antagos[dof]+action[2*dof+1])

        # performing a step
        observation,reward,episode_over = self._hysr.step(action)

        # formatting observation in a format suitable for gym
        self._obs_boxes.set_values("robot_position",observation_.joint_positions)
        self._obs_boxes.set_values("robot_velocity",observation_.joint_velocities)
        self._obs_boxes.set_values("robot_pressure",observation_.pressure)
        self._obs_boxes.set_values("ball_position",observation_.ball_position)
        self._obs_boxes.set_values("ball_velocity",observation_.ball_velocity)
        observation = self._obs_boxes.get_normalized_values()
        
        return observation,reward,episode_over,None


    def reset(self):
        observation = self._hysr.reset()
        observation = self._convert_observation(observation)
        return observation
