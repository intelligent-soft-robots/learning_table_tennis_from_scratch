import random
import o80
from hysr_one_ball import HysrOneBall


mujoco_id = "hysr_demo"
real_robot = False # i.e. mujoco robot
target_position = [1,4,-0.44]
reward_normalization_constant = 1.0
smash_task = True
period_ms = 10
nb_episodes = 20
frequency_manager = o80.FrequencyManager(1.0/(period_ms/1000.0))

hysr = HysrOneBall(mujoco_id,
                   real_robot,
                   target_position,
                   reward_normalization_constant,
                   smash_task,
                   period_ms)
hysr.reset()

pressures = [ [ random.randrange(6000,22000),
                random.randrange(6000,22000) ]
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
                pressure = max(min(pressure,22000),6000)
                pressures[dof][ago] =  pressure

        observation,reward,reset = hysr.step(pressures)

        frequency_manager.wait()

        running = not reset

    hysr.reset()


hysr.close()
