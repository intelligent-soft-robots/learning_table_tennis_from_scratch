import time
import random
import o80
from hysr_one_ball import HysrOneBall
from hysr_one_ball_burst import HysrOneBallBurst


mujoco_id = "hysr_demo"
real_robot = False # i.e. mujoco robot
target_position = [1,4,-0.44]
reward_normalization_constant = 1.0
smash_task = True
period_ms = 10
frequency_manager = o80.FrequencyManager(1.0/(period_ms/1000.0))

reference_posture = [[20000,12000],[12000,22000],[15000,15000],[15000,15000]]
swing_posture = [[14000,22000],[14000,22000],[17000,13000],[14000,16000]]

hysr = HysrOneBallBurst(mujoco_id,
                   real_robot,
                   reference_posture,
                   target_position,
                   period_ms,
                   reward_normalization_constant,
                   smash_task,
                   period_ms,
                   trajectory_index=37) # always playing the same trajectory
hysr.reset()

# time_switch: duration after episode start after which
# the robot performs the swing motion
# manually selected so that sometimes the racket hits the ball,
# sometimes it does not
base=0.2
time_switches = [0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425]
time_switches = [base+ts for ts in time_switches]

for episode,time_switch in enumerate(time_switches):

    print("EPISODE",episode)
    
    running = True
    time_start = time.time()
    
    while running:

        passed_time = time.time()-time_start

        if  passed_time < time_switch:
            pressures = reference_posture
        else:
            pressures = swing_posture

        observation,reward,reset = hysr.step(pressures)
        
        frequency_manager.wait()

        if reset:
            print("\treward:",reward)
        
        running = not reset


        
    hysr.reset()


hysr.close()
