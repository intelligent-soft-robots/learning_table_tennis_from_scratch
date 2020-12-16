import h5py

class Step:

    __slots__ = ("pressures_ago","pressures_antago",
                 "joint_positions","joint_velocities",
                 "ball_position","ball_velocity",
                 "iteration","time_stamp")
    
    def __init__(self,pressures_ago,pressures_antago,
                 joint_positions,joint_velocities,
                 ball_position,ball_velocity,
                 iteration,time_stamp):
        locals_ = locals()
        for param in self.__slots__:
            setattr(self,param,locals_[param])

class Episode:

    __slots__ = ("steps","reward")
    
    
    def __init__(self):
        self.steps = []
        self.reward = None
        
    def record(self,pressures_ago,pressures_antago,
               joint_positions,joint_velocities,
               ball_position,ball_velocity,
               iteration,time_stamp):
        locals_ = locals()
        values = [locals_[param] for param in Step.__slots__]
        self.steps.append(Step(*values))

    def nb_steps(self):
        return len(self.steps)
    

class Recorder:

    def __init__(self,
                 experiment_id,
                 h5p_path,
                 nb_dofs=4):
        self.f = h5py.File(h5p_path,"a")
        self.experiment_group = f.create_group(experiment_id)
        self.episode_group = None
        self.episode = None
        self.nb_dofs = nb_dofs
        
    def _record_episode(self):

        nb_steps = self.episode.nb_steps()
        time_stamps = np.zeros((1,nb_steps))
        iterations = np.zeros((1,nb_steps))
        pressures_ago = np.zeros((self.nb_dofs,nb_steps))
        pressures_antago = np.zeros((self.nb_dofs,nb_steps))
        joint_positions = np.zeros((self.nb_dofs,nb_steps))
        joint_velocities = np.zeros((self.nb_dofs,nb_steps))
        ball_positions = np.zeros((3,nb_steps))
        ball_velocities = np.zeros((3,nb_steps))

        locals_ = locals()
        
        for index,step in enumerate(self.episode.steps()):
            for param in Step.__slots__:
                locals_[param][step] = np.array(getattr(step,param))
            self.episode_group.create_dataset(param,locals_.shape,dtype=locals_.dtype)
        
        self.episode_group.attrs["reward"]=self.episode.reward

        self.f.flush()
        
    def next_episode(self,episode_id):
        if self.episode:
            self._record_episode()
        self.episode = Episode()
        self.episode_group = self.experiment_group.create_group(episode_id)

    def set_reward(self,reward):
        self.episode.reward = reward

    def record(self,pressures_ago,pressures_antago,
               joint_positions,joint_velocities,
               ball_position,ball_velocity,
               iteration,time_stamp):
        locals_ = locals()
        values = [locals_[param] for param in Step.__slots__]
        self.episode.record(*values)

        
    def close(self):
        self.f.close()
    
        
        
        
