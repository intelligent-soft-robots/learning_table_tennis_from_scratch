

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pickle


path = "/tmp/hysr_one_ball_0.pkl"

data = pickle.load(open(path,"rb"))


fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ball_x = [d.ball_position[0] for d in data.ball]
ball_y = [d.ball_position[1] for d in data.ball]
ball_z = [d.ball_position[2] for d in data.ball]
ax1.scatter(ball_x,ball_y,ball_z)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
for dof in range(4):
    pressure_ago = [d.pressures_ago[dof] for d in data.robot]
    pressure_antago = [d.pressures_ago[dof] for d in data.robot]
    iterations = list(range(len(pressure_ago)))
    ax2.scatter(pressure_ago,pressure_antago,iterations)

plt.show()
