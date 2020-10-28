import sys
import random
import o80
from hysr_one_ball import HysrOneBall
from lightargs import BrightArgs,Set,Range,Positive


# default frequency for real/dummy robot
# o80_pam_time_step = 0.0005
# default frequency for mujoco robot
o80_pam_time_step = 0.002

mujoco_time_step = 0.002
algo_time_step = 0.01
target_position = (1.6,3.5,+1.5)
reward_normalization_constant = 1.0
smash_task = True
rtt_cap = 0.2
nb_episodes = 3

def execute(accelerated_time):

    hysr = HysrOneBall(accelerated_time,
                       o80_pam_time_step,
                       mujoco_time_step,
                       algo_time_step,
                       None, # no reference posture to go to between episodes
                       target_position,
                       reward_normalization_constant,
                       smash_task,
                       rtt_cap=rtt_cap,
                       trajectory_index=None)

    hysr.reset()
    frequency_manager = o80.FrequencyManager(1.0/algo_time_step)

    pressures = [ [ random.randrange(14000,20000),
                    random.randrange(14000,20000) ]
                  for _ in range(4) ]

    pressure_max_diff = 300

    
    for _ in range(nb_episodes):

        running = True

        while running:

            for dof in range(4):
                for ago in range(2):
                    pressure = pressures[dof][ago]
                    diff = random.randrange(-pressure_max_diff,
                                            pressure_max_diff)
                    pressure += diff
                    pressure = max(min(pressure,20000),14000)
                    pressures[dof][ago] = pressure

            observation,reward,reset = hysr.step(pressures)

            if not accelerated_time:
                waited = frequency_manager.wait()

            running = not reset

        hysr.reset()
        frequency_manager = o80.FrequencyManager(1.0/algo_time_step)

    hysr.close()


if __name__ == "__main__":

    accelerated = False
    if("accelerated" in sys.argv):
        accelerated = True

    execute(accelerated)
