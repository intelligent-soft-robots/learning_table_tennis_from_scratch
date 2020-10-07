import time
import random
import o80
from hysr_one_ball import HysrOneBall




# default frequency for real/dummy robot
# o80_pam_time_step = 0.0005
# default frequency for mujoco robot
o80_pam_time_step = 0.002

mujoco_id = "hysr_demo"
mujoco_time_step = 0.002
algo_time_step = 0.01
target_position = [1,4,-0.44]
reward_normalization_constant = 1.0
smash_task = True
rtt_cap = 0.2
nb_episodes = 5

reference_posture = [[20000,12000],[12000,22000],[15000,15000],[15000,15000]]
swing_posture = [[14000,22000],[14000,22000],[17000,13000],[14000,16000]]

def execute(accelerated_time):

    hysr = HysrOneBall(accelerated_time,
                       o80_pam_time_step,
                       mujoco_id,
                       mujoco_time_step,
                       algo_time_step,
                       None, # no reference posture to go to between episodes
                       target_position,
                       reward_normalization_constant,
                       smash_task,
                       rtt_cap=rtt_cap,
                       trajectory_index=37)


    hysr.reset()
    frequency_manager = o80.FrequencyManager(1.0/algo_time_step)

    # time_switch: duration after episode start after which
    # the robot performs the swing motion
    # manually selected so that sometimes the racket hits the ball,
    # sometimes it does not
    ts = 0.5
    time_switches = []
    while ts<=0.8:
        time_switches.append(ts)
        ts+=0.025

    # converting time switches from seconds to nb of iterations
    iteration_switches = [(1.0/o80_pam_time_step) * ts for ts in time_switches]



    for episode,iteration_switch in enumerate(iteration_switches):

        print("EPISODE",episode,iteration_switch)

        start_iteration = hysr.get_robot_iteration()
        running = True

        while running:

            current_iteration = hysr.get_robot_iteration()

            if  (current_iteration - start_iteration) < iteration_switch :
                pressures = reference_posture
            else:
                pressures = swing_posture

            observation,reward,reset = hysr.step(pressures)

            if not accelerated_time:
                waited = frequency_manager.wait()
                if waited<0:
                    print("! warning ! failed to maintain algorithm frequency")

            if reset:
                print("\treward:",reward)

            running = not reset


        hysr.reset()
        frequency_manager = o80.FrequencyManager(1.0/algo_time_step)

    hysr.close()


if __name__ == "__main__":

    accelerated = False
    if("accelerated" in sys.argv):
        accelerated = True

    execute(accelerated)
