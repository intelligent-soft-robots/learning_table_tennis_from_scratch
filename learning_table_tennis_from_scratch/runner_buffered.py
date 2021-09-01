import numpy as np
from baselines.common.runners import AbstractEnvRunner


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.mb_obs_temp, self.mb_rewards_temp, self.mb_actions_temp, self.mb_values_temp, self.mb_dones_temp, self.mb_neglogpacs_temp = [],[],[],[],[],[]
        self.epinfos_temp = []

        self.obs_temp = self.obs.copy()
        self.states_temp = self.states
        self.dones_temp = self.dones.copy()

        self.n_timesteps = 0
        self.n_episodes = 0

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        #mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = self.mb_obs_temp.copy(), self.mb_rewards_temp.copy(), self.mb_actions_temp.copy(), self.mb_values_temp.copy(), self.mb_dones_temp.copy(), self.mb_neglogpacs_temp.copy()
        self.mb_obs_temp, self.mb_rewards_temp, self.mb_actions_temp, self.mb_values_temp, self.mb_dones_temp, self.mb_neglogpacs_temp = [],[],[],[],[],[]

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]

        self.epinfos_temp = []
        #self.obs = self.obs_temp.copy()
        #self.states = self.states_temp
        #self.dones = self.dones_temp.copy()

        mb_states = self.states
        #epinfos = self.epinfos_temp
        epinfos = []

        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones.copy())

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

            self.n_timesteps += 1
            if self.dones[0]:
                self.n_episodes += 1

        print("******** finishing episode ***********")
        self.obs_temp = self.obs.copy()
        self.states_temp = self.states
        self.dones_temp = self.dones.copy()
        #finish episode and buffer data for next batch
        while not self.dones_temp[0]:
            actions, values, self.states_temp, neglogpacs = self.model.step(self.obs_temp, S=self.states_temp, M=self.dones_temp)
            self.mb_obs_temp.append(self.obs_temp.copy())
            self.mb_actions_temp.append(actions)
            self.mb_values_temp.append(values)
            self.mb_neglogpacs_temp.append(neglogpacs)
            self.mb_dones_temp.append(self.dones_temp.copy())

            self.obs_temp[:], rewards, self.dones_temp, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: self.epinfos_temp.append(maybeepinfo)
            self.mb_rewards_temp.append(rewards)

            self.n_timesteps += 1
            if self.dones_temp[0]:
                self.n_episodes += 1

        print("******** update ***********")

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return_list = (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)

        return return_list



# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
