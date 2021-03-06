import o80,pam_interface
import math,gym,random
import numpy as np
from .hysr_one_ball import HysrOneBall,HysrOneBallConfig
from .rewards import JsonReward
from collections import OrderedDict


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
                 pam_config_file=None,
                 reward_config_file=None,
                 hysr_one_ball_config_file=None):

        super().__init__()
       
        hysr_one_ball_config = HysrOneBallConfig.from_json(hysr_one_ball_config_file)
        reward_function = JsonReward.get(reward_config_file)

        self._config = pam_interface.JsonConfiguration(pam_config_file)
        self._nb_dofs = len(self._config.max_pressures_ago)
        self._algo_time_step = hysr_one_ball_config.algo_time_step 
        self._pressure_change_range = hysr_one_ball_config.pressure_change_range
        self._accelerated_time = hysr_one_ball_config.accelerated_time
        
        self._hysr = HysrOneBall(hysr_one_ball_config,
                                 reward_function)
        
        self.action_space = gym.spaces.Box(low=-1.0,
                                           high=+1.0,
                                           shape=(self._nb_dofs*2,),
                                           dtype=np.float)
        
        self._obs_boxes = _ObservationSpace()
        
        self._obs_boxes.add_box("robot_position",-math.pi,+math.pi,
                            self._nb_dofs)
        self._obs_boxes.add_box("robot_velocity",0.0,10.0,
                            self._nb_dofs)
        self._obs_boxes.add_box("robot_pressure",
                            self._config.min_pressure(),
                            self._config.max_pressure(),
                            self._nb_dofs*2)
        
        self._obs_boxes.add_box("ball_position",
                            min(hysr_one_ball_config.world_boundaries["min"]),
                            max(hysr_one_ball_config.world_boundaries["max"]),
                            3)
        self._obs_boxes.add_box("ball_velocity",
                            -10.0,+10.0,3)

        self.observation_space = self._obs_boxes.get_gym_box()

        if not self._accelerated_time:
            self._frequency_manager = o80.FrequencyManager(1.0/hysr_one_ball_config.algo_time_step)
        
        
    def _bound_pressure(self,dof,ago,value):
        if ago:
            return int(max(min(value,
                               self._config.max_pressures_ago[dof]),
                           self._config.min_pressures_ago[dof]))
        else:
            return int(max(min(value,
                               self._config.max_pressures_antago[dof]),
                           self._config.min_pressures_antago[dof]))

        
    def _convert_observation(self,observation):
        self._obs_boxes.set_values("robot_position",observation.joint_positions)
        self._obs_boxes.set_values("robot_velocity",observation.joint_velocities)
        self._obs_boxes.set_values("robot_pressure",observation.pressures)
        self._obs_boxes.set_values("ball_position",observation.ball_position)
        self._obs_boxes.set_values("ball_velocity",observation.ball_velocity)
        return self._obs_boxes.get_normalized_values()

    
    def step(self,action):

        # casting actions from [-1,+1] to [-pressure_change_range,+pressure_change_range]
        action = [self._pressure_change_range*a for a in action]
        
        # current pressures
        agos,antagos = self._hysr.get_current_pressures()
        
        # final target pressure is action + current desired
        for dof in range(self._nb_dofs):
            action[2*dof] = self._bound_pressure(dof,True,agos[dof]+action[2*dof])
            action[2*dof+1] = self._bound_pressure(dof,True,antagos[dof]+action[2*dof+1])

        # hysr takes a list of int, not float, as input
        action = [int(a) for a in action]
            
        # performing a step
        observation,reward,episode_over = self._hysr.step(list(action))

        # formatting observation in a format suitable for gym
        observation = self._convert_observation(observation)

        # imposing frequency to learning agent
        if not self._accelerated_time:
            self._frequency_manager.wait()
        
        return observation,reward,episode_over,{}


    def reset(self):
        observation = self._hysr.reset()
        observation = self._convert_observation(observation)
        if not self._accelerated_time:
            self._frequency_manager = o80.FrequencyManager(1.0/self._algo_time_step)
        return observation
