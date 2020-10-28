import math
import gym
from hysr_one_ball import HysrOneBall


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
        self.rtt_cap = 0.2
        self.nb_dofs = 4
        self.world_boundaries = { "min":(0.0,-1.0,+0.17), # x,y,z
                                  "max":(1.6,3.5,+1.5) }   
    

class HysrOneBallEnv(gym.GoalEnv):
    
    def __init__(self,
                 config):

        self._config = config
        
        self._hysr = HysrOneBall(config.accelerated_time,
                                 config.o80_pam_time_step,
                                 mujoco_id,
                                 config.mujoco_time_step,
                                 algo_time_step,
                                 config.reference_posture, # no reference posture to go to between episodes
                                 config.target_position,
                                 config.reward_normalization_constant,
                                 config.smash_task,
                                 rtt_cap=config.rtt_cap)
        
        self.action_space = gym.spaces.Box(low=config.min_pressure,
                                           high=config.max_pressure,
                                           shape=(config.nb_dofs*2,),
                                           dtype=np.float)

        self._robot_space = gym.spaces.Dict(
            {
                "position"=gym.spaces.Box(low=-math.pi,
                                          high=+math.pi,
                                          shape=(config.nb_dofs*2,),
                                          dtype=np.float),
                "velocity"=gym.spaces.Box(low=-10.0,
                                          high=+10.0,
                                          shape=(config.nb_dofs*2,),
                                          dtype=np.float),
                "pressures"=gym.spaces.Box(low=config.pressure_change_range[0],
                                           high=oconfig.pressure_change_range[0],
                                           shape=(config.nb_dofs*2,),
                                           dtype=np.int)
            } )
            
        self._ball_space = gym.spaces.Dict(
            {
                "position"==gym.spaces.Box(low=min(config.world_boundaries["min"]),
                                           high=max(config.world_boundaries["max"]),
                                           shape=(3,),
                                           dtype=np.float),
                "velocity"==gym.spaces.Box(low=-10.0,
                                           high=+10.0,
                                           shape=(3,),
                                           dtype=np.float),

            } )


        self.observation_space = gym.spaces.Dict(
            {
                "robot":self._robot_space,
                "ball":self._ball_space
            } )
            

       
    def _bound_pressure(self,value):
        return max(min(value,
                       self._config.pressure_max),
                   self._config.pressure_min)

    
    def _convert_observation(self,observation):
        # convert observation as returned by HysrOneBall
        # to gym observation space
        pressures = [None]*self._config.nb_dofs*2
        for dof in range(self._config.nb_dofs):
            pressures[2*dof]=observation_.pressures_ago[dof]
            pressures[2*dof+1]=observation_.pressures_antago[dof]
        return {
            "robot" : { "position":np.array(observation_.joint_positions),
                        "velocity":np.array(observation_.joint_velocities),
                        "pressures":np.array(pressure) },
            "ball" : { "position":np.array(observation_.ball_position),
                       "velocity":np.array(observation_.ball_velocity) }
        }
        
        
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
        observation = self._convert_observation(observation)
            
        return observation,reward,episode_over,None


    def reset(self):
        observation = self._hysr.reset()
        observation = self._convert_observation(observation)
        return observation
