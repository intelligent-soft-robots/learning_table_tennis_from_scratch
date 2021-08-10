from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pickle


path1 = "/tmp/hysr_one_ball_0.pkl"
data1 = pickle.load(open(path1, "rb"))

path2 = "/tmp/previous_hysr_one_ball_0.pkl"
data2 = pickle.load(open(path2, "rb"))


fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection="3d")
ball1_x = [d.ball_position[0] for d in data1.ball]
ball1_y = [d.ball_position[1] for d in data1.ball]
ball1_z = [d.ball_position[2] for d in data1.ball]
ax1.scatter(ball1_x, ball1_y, ball1_z)
ball2_x = [d.ball_position[0] for d in data2.ball]
ball2_y = [d.ball_position[1] for d in data2.ball]
ball2_z = [d.ball_position[2] for d in data2.ball]
ax1.scatter(ball2_x, ball2_y, ball2_z)


def commented():
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    for dof in range(1):
        pressure_ago_1 = [d.pressures_ago[dof] for d in data1.robot]
        pressure_antago_1 = [d.pressures_ago[dof] for d in data1.robot]
        iterations_1 = list(range(len(pressure_ago_1)))
        pressure_ago_2 = [d.pressures_ago[dof] for d in data2.robot]
        pressure_antago_2 = [d.pressures_ago[dof] for d in data2.robot]
        iterations_2 = list(range(len(pressure_ago_2)))
        ax2.scatter(pressure_ago_1, pressure_antago_1, iterations_1)
        ax2.scatter(pressure_ago_2, pressure_antago_2, iterations_2)


plt.show()
