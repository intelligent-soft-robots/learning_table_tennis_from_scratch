import numpy as np
from numpy import array, uint64, float32
from matplotlib import pyplot as plt

positions_o80 = []
timestamps_o80 = []
mujoco_timestamps = []
with open("/tmp/balltest.txt") as f:
    lines_o80 = f.readlines()
    for line in lines_o80:
        try:
            obj_type, mujoco_timestamp, timestamp, _, _, pos, _ = line.split(sep="\t")
            if obj_type != "ball":
                continue
            positions_o80.append(float(pos))
            timestamps_o80.append(int(timestamp) / 1000.0)
            mujoco_timestamps.append(float(mujoco_timestamp))
        except:
            pass


with open("/tmp/balltest2.txt") as f:
    content = f.read()
    trajectory = eval(content)


# plt.plot(timestamps_o80, positions_o80, label="o80")
# plt.plot(trajectory[0], trajectory[1], label="script")
# plt.legend()
# plt.show()

plt.plot(mujoco_timestamps, timestamps_o80, label="o80")
plt.legend()
plt.show()