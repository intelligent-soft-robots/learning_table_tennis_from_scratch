import argparse
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datafile", type=Path, default=Path(__file__).parents[1] / "out" / "tcr_intervention_scales.pkl"
    )
    args = parser.parse_args()

    with args.datafile.open("rb") as f:
        data = pickle.load(f)

    intervention_stds = list(data["rewards"].keys())
    rewards_mean = [np.mean(rewards) for rewards in data["rewards"].values()]
    plt.title("Cumulative rewards for different intervention scales")
    plt.bar(intervention_stds, rewards_mean, width=intervention_stds[1] - intervention_stds[0])
    plt.xlabel("Intervention standard deviation")
    plt.ylabel("Mean cumulative reward")
    plt.show()


    success_rates = [1 - np.mean((np.isinf(distances)).astype(float)) for distances in data["distances_to_target"].values()]
    plt.title("Hit rates for different intervention scales")
    plt.bar(intervention_stds, success_rates, width=intervention_stds[1] - intervention_stds[0])
    plt.xlabel("Intervention standard deviation")
    plt.ylabel("Hit rate")
    plt.show()

    success_rates = [np.mean((np.array(distances) < 0.65).astype(float)) for distances in data["distances_to_target"].values()]
    plt.title("Success rates for different intervention scales")
    plt.bar(intervention_stds, success_rates, width=intervention_stds[1] - intervention_stds[0])
    plt.xlabel("Intervention standard deviation")
    plt.ylabel("Success rate")
    plt.show()
