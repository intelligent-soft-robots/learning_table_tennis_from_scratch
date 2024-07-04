import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("path_json")
    args = parser.parse_args()

    timestamps = []
    positions = []
    with Path(args.path).open("r") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            if len(line) == 4:
                timestamp, x, y, z = line
                timestamps.append(int(timestamp))
                positions.append([float(x), float(y), float(z)])

    with Path(args.path_json).open("r") as f:
        data = json.load(f)

    timesteps_robot_pc = np.array(data["timestamps"])
    pos_kalman = np.array(data["ball_kl_positions"])
    pos_tennicam_robot_pc = np.array(data["ball_camera_positions_direct"])
    pos_tennicam_robot_pc_unfiltered = np.array(data["ball_camera_positions"])

    positions = np.array(positions)
    timestamps = (np.array(timestamps) / 1000.0)
    fig, axes = plt.subplots(nrows=3)
    for i, ax in enumerate(axes):
        # ax.plot(timestamps, positions[:, i], label="Tennicam (server)")
        ax.plot(timesteps_robot_pc, pos_kalman[:, i], label="Kalman")
        ax.plot(timesteps_robot_pc, pos_tennicam_robot_pc[:, i], label="Tennicam (robot PC)")
        ax.plot(timesteps_robot_pc, pos_tennicam_robot_pc_unfiltered[:, i], label="Tennicam (robot PC, filtered)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
    plt.legend()
    print("Done")
    plt.show()
