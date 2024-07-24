import time
from pathlib import Path

import numpy as np
import context
import o80
import o80_pam
import pam_mujoco
from matplotlib import pyplot as plt

from learning_table_tennis_from_scratch import configure_mujoco
from learning_table_tennis_from_scratch.hysr_one_ball import HysrOneBallConfig, _BallBehavior

SEGMENT_ID_BALL = pam_mujoco.segment_ids.ball

if __name__ == '__main__':
    hysr_config = HysrOneBallConfig.from_json(Path(__file__).parents[1] / "config" / "hysr_config.json")
    simulated_robot_handle = configure_mujoco.configure_simulation(
        hysr_config
    )
    parallel_burst = pam_mujoco.mirroring.ParallelBurst([simulated_robot_handle.interfaces["robot"]])
    nb_sim_bursts = int(
        hysr_config.algo_time_step / hysr_config.mujoco_time_step
    )
    ball_communication = simulated_robot_handle.interfaces[SEGMENT_ID_BALL]
    #ball_frontend = o80_pam.BallFrontEnd("ball")
    #ball_communication = o80_pam.o80Ball(ball_frontend)
    print(type(ball_communication))
    ball_communication.reset()
    _BallBehavior.read_trajectories(hysr_config.trajectory_group)
    ball_behavior = _BallBehavior(index=1)
    trajectory_orig = ball_behavior.get_trajectory()

    fake_recording_timestep = 10000
    # fake_timesteps = np.arange(0, 150 * fake_recording_timestep, fake_recording_timestep)
    # fake_positions = np.repeat(np.repeat(np.arange(30), 5)[:, None], 3, axis=1) / 30
    # trajectory_orig = (fake_timesteps, fake_positions)

    iterator = context.ball_trajectories.BallTrajectories.iterate(trajectory_orig)

    # setting the ball to the first trajectory point
    duration, state = next(iterator)
    ball_communication.set(state.get_position(), [0, 0, 0])
    simulated_robot_handle.reset_contact(SEGMENT_ID_BALL)
    simulated_robot_handle.deactivate_contact(SEGMENT_ID_BALL)
    # moving the ball(s) to initial position
    parallel_burst.burst(4)

    # shooting the ball
    ball_communication.iterate_trajectory(iterator, overwrite=False)
    # timestamps_from_durations = [sum(durations[:i]) for i in range(len(durations))]

    # resetting ball/robot contact information
    simulated_robot_handle.reset_contact(SEGMENT_ID_BALL)
    simulated_robot_handle.activate_contact(SEGMENT_ID_BALL)
    trajectory = []
    trajectory_desired = []
    timestamps = []
    other_timestamps = []
    start = time.time()
    ts0 = None
    previous_t = None
    previous_iteration = None
    frequency_manager = o80.FrequencyManager(
        1.0 / hysr_config.algo_time_step
    )
    print("\n---------------------\n")
    for _ in range(150):
        iteration, freq, ts, pos, vel, pos_desired, vel_desired = ball_communication.get()
        ts = ts*1e-3
        if ts0 is None:
            ts0 = ts
        parallel_burst.burst(nb_sim_bursts)
        trajectory.append(pos)
        trajectory_desired.append(pos_desired)
        other_timestamps.append(ts-ts0)
        tnow = time.time()
        if previous_t is None:
            previous_t = tnow
        if previous_iteration is None:
            previous_iteration = iteration
        timestamps.append((tnow - start) * 1e6)
        #print(time.time())
        #print(frequency_manager.wait(),tnow-previous_t, iteration-previous_iteration)
        frequency_manager.wait()
        #print(iteration,"\t",pos[2])
        previous_t = tnow
        previous_iteration = iteration

    #timestamps = np.array(timestamps)
    trajectory = np.array(trajectory)
    trajectory_desired = np.array(trajectory_desired)

    #plt.plot(timestamps, timestamps)
    #plt.plot(timestamps,other_timestamps)
    plt.plot(timestamps, trajectory[:, 2], label="o80")
    plt.plot(timestamps, trajectory_desired[:, 2], label="o80-desired")
    plt.plot(other_timestamps, trajectory[:, 2], label="o80_mujoco_timestamp")
    plt.plot(trajectory_orig[0], trajectory_orig[1][:, 2], label="file")


    # plt.plot(timestamps_from_durations, np.array([s.get_position()[2] for s in states]), label="iterate_trajectory")
    # plt.plot()
    plt.legend()
    plt.show()

    # with open("/tmp/balltest2.txt", "w") as f:
    #     f.write(
    #         repr(
    #             (timestamps, [t[2] for t in trajectory])
    #         )
    #     )
