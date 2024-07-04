import json
import time
import glob
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_data(filename):
    # Open the file and load the data
    with open(filename, "r") as json_data:
        data = json.load(json_data)

    # Extract the data into separate variables if needed, for example:
    ob = data["ob"]
    action_orig = data["action_orig"]
    action_casted = data["action_casted"]
    prdes = data["prdes"]
    reward = data["reward"]
    episode_over = data["episode_over"]

    # Perform operations on the data or return it
    # For now, let's just return the loaded data
    return data


# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 10))  # 2,3

# match_color_pairs = {('/tmp/ep_20240131-19', 'green'), ('/tmp/ep_20240131-22', 'yellow')}

# match_color_pairs = {('/tmp/ep_20240202-230', 'green'), ('/tmp/ep_20240202-235', 'blue'), ('/tmp/ep_20240203-030', 'red'), ('/tmp/ep_20240203-104', 'violet')}

# match_color_pairs = {('/tmp/ep_20240203-11', 'green')}

match_color_pairs = [('/tmp/ep_20240625-172917', 'blue'), ('/tmp/ep_20240627-082345', 'red'), ('/tmp/ep_20240626-175308', 'green')]
# ('/tmp/ep_20240131-17', 'blue'),

for filename_match, color in match_color_pairs:
    # read n most recent files matching /tmp/ep_*
    files = sorted(glob.glob(filename_match + "*"), key=os.path.getmtime)[-5:]
    for filename in files:
        loaded_data = read_data(filename)
        positions = [loaded_data["ob"][i][16:19] for i in range(len(loaded_data["ob"]))]
        velocities = [loaded_data["ob"][i][19:22] for i in range(len(loaded_data["ob"]))]
        timestamps = [i for i in range(len(loaded_data["ob"]))]

        # Separate the position and velocity components for plotting
        pos_x, pos_y, pos_z = zip(*positions)
        vel_x, vel_y, vel_z = zip(*velocities)

        # if color=='red':
        #     d = 6
        #     # timestamps = timestamps
        #     pos_x, pos_y, pos_z = zip(*positions[d:])
        #     vel_x, vel_y, vel_z = zip(*velocities[d:])

        # if color=='violet':
        #     d = 15
        #     timestamps = timestamps[d:]
        #     pos_x, pos_y, pos_z = zip(*positions[:-d])
        #     vel_x, vel_y, vel_z = zip(*velocities[:-d])

        # # ignore y>0
        # timestamps = [timestamps[i] for i in range(len(pos_y)) if pos_y[i]<0]
        # pos_x = [pos_x[i] for i in range(len(pos_x)) if pos_y[i]<0]
        # pos_z = [pos_z[i] for i in range(len(pos_z)) if pos_y[i]<0]
        # vel_x = [vel_x[i] for i in range(len(vel_x)) if pos_y[i]<0]
        # vel_z = [vel_z[i] for i in range(len(vel_z)) if pos_y[i]<0]
        # vel_y = [vel_y[i] for i in range(len(vel_y)) if pos_y[i]<0]
        # pos_y = [pos_y[i] for i in range(len(pos_y)) if pos_y[i]<0]

        # pos_x = np.array(pos_x)
        # pos_y = np.array(pos_y)
        # pos_z = np.array(pos_z)
        # vel_x = np.array(vel_x)
        # vel_y = np.array(vel_y)
        # vel_z = np.array(vel_z)

        # # # Plot positions
        file_name = filename.split('/')[-1]
        axes[0].scatter(timestamps, pos_x, label=file_name, s=20, color=color, alpha=0.5)
        axes[0].set_title("Position X")
        axes[1].scatter(timestamps, pos_y, s=20, color=color, alpha=0.5)
        axes[1].set_title("Position Y")
        axes[2].scatter(timestamps, pos_z, s=20, color=color, alpha=0.5)
        axes[2].set_title("Position Z")

        # # # Plot velocities
        # axes[1, 0].scatter(timestamps, vel_x, label=file_name, s=20, color=color, alpha=0.5)
        # axes[1, 0].set_title("Velocity X")
        # axes[1, 1].scatter(timestamps, vel_y, s=20, color=color, alpha=0.5)
        # axes[1, 1].set_title("Velocity Y")
        # axes[1, 2].scatter(timestamps, vel_z, s=20, color=color, alpha=0.5)
        # axes[1, 2].set_title("Velocity Z")

        # # Calculate velocity only from positions
        # delta_pos_x = [pos_x[i]-pos_x[i-1] for i in range(1, len(pos_x))]
        # delta_pos_y = [pos_y[i]-pos_y[i-1] for i in range(1, len(pos_y))]
        # delta_pos_z = [pos_z[i]-pos_z[i-1] for i in range(1, len(pos_z))]
        # delta_timestamps = [0.01 for i in range(1, len(timestamps))]
        # velocities_from_pos = [np.array([delta_pos_x[i]/delta_timestamps[i], delta_pos_y[i]/delta_timestamps[i], delta_pos_z[i]/delta_timestamps[i]]) for i in range(len(delta_pos_x))]
        # velocities_from_pos = np.array(velocities_from_pos)

        # vel_x_from_pos = [velocities_from_pos[i][0] for i in range(len(velocities_from_pos))]
        # vel_y_from_pos = [velocities_from_pos[i][1] for i in range(len(velocities_from_pos))]
        # vel_z_from_pos = [velocities_from_pos[i][2] for i in range(len(velocities_from_pos))]

        # # average velocity over 10 points
        # vel_x_from_pos = [np.mean(vel_x_from_pos[i:i+10]) for i in range(len(velocities_from_pos)-10)]
        # vel_y_from_pos = [np.mean(vel_y_from_pos[i:i+10]) for i in range(len(velocities_from_pos)-10)]
        # vel_z_from_pos = [np.mean(vel_z_from_pos[i:i+10]) for i in range(len(velocities_from_pos)-10)]

        # timestamps_from_pos = [np.mean(timestamps[i:i+10]) for i in range(len(velocities_from_pos)-10)]

        # # Plot velocities from positions
        # axes[1, 0].scatter(timestamps_from_pos, vel_x_from_pos, label=file_name, s=20, color='orange', alpha=0.5)
        # axes[1, 1].scatter(timestamps_from_pos, vel_y_from_pos, s=20, color='orange', alpha=0.5)
        # axes[1, 2].scatter(timestamps_from_pos, vel_z_from_pos, s=20, color='orange', alpha=0.5)

# ball_data = h5py.File("/var/Experiments/ball_trajectories.hdf5")
# trajectory = np.array(ball_data["/aimy_049_075_035/0/trajectory"])
# timestamps = np.array(ball_data["/aimy_049_075_035/0/time_stamps"]) / 0.01 / 1e9
# ball_path = Path("/home/sguist/data/tennicam02_long/tennicam_199_filtered")
# timestamps = []
# trajectory = []
# with ball_path.open("r") as f:
#     for line in f.readlines():
#         _, timestamp, pos, vel = eval(line)
#         timestamps.append(timestamp)
#         trajectory.append(pos)
# timestamps = np.array(timestamps) / 0.01 / 1e9
# trajectory = np.array(trajectory)
# axes[0].scatter(timestamps, trajectory[:, 0], label="recorded ball (direct replay)", s=20, color="red", alpha=0.5)
# axes[0].set_title("Position X")
# axes[1].scatter(timestamps, trajectory[:, 1], s=20, color="red", alpha=0.5)
# axes[1].set_title("Position Y")
# axes[2].scatter(timestamps, trajectory[:, 2], s=20, color="red", alpha=0.5)
# axes[2].set_title("Position Z")

files = sorted(glob.glob(match_color_pairs[0][0] + "*"), key=os.path.getmtime)[-5:]
for filename in files:
    loaded_data = read_data(filename)
    positions = [loaded_data["ball_pos_unfiltered"][i] for i in range(len(loaded_data["ball_pos_unfiltered"]))]
    velocities = [loaded_data["ball_vel_unfiltered"][i] for i in range(len(loaded_data["ball_vel_unfiltered"]))]
    timestamps = [i for i in range(len(loaded_data["ball_pos_unfiltered"]))]

    # Separate the position and velocity components for plotting
    pos_x, pos_y, pos_z = zip(*positions)
    vel_x, vel_y, vel_z = zip(*velocities)

    # # # Plot positions
    file_name = filename.split('/')[-1]
    axes[0].scatter(timestamps, pos_x, label="real ball (unfiltered)", s=20, color="orange", alpha=0.5)
    axes[0].set_title("Position X")
    axes[1].scatter(timestamps, pos_y, s=20, color="orange", alpha=0.5)
    axes[1].set_title("Position Y")
    axes[2].scatter(timestamps, pos_z, s=20, color="orange", alpha=0.5)
    axes[2].set_title("Position Z")

axes[-1].legend(["real ball", "recorded ball", "recorded ball (direct replay)", "real ball (unfiltered)"])
plt.show()