import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    with Path(args.path).open("r") as f:
        data = json.load(f)

    timesteps = [t - data["timestamps"][0] for t in data["timestamps"]]
    pos_kalman = np.array(data["ball_kl_positions"])
    pos_tennicam_robot_pc = np.array(data["ball_kl_positions"])
    fig, axes = plt.subplots(nrows=3)
    for i, ax in enumerate(axes):
        ax.scatter(timesteps, pos_kalman[:, i], label="Kalman filter", s=5)
        ax.scatter(timesteps, pos_tennicam_robot_pc[:, i], label="Tennicam (robot PC)", s=5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
    plt.show()
