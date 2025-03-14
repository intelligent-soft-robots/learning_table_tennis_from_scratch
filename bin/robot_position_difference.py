import numpy as np
import pam_vicon.o80
from scipy.spatial.transform import Rotation

expected_position = np.array([0.24172509, -3.38384842, -0.95054469])
expected_orientation = np.array([-0.03136496, 0.02863854, -0.0100406, 0.99904718])
expected_euler = Rotation.from_quat(expected_orientation).as_euler("xyz", degrees=True)

vicon_frontend = pam_vicon.o80.FrontEnd("vicon")

while True:
    vicon_frame = vicon_frontend.latest().get_extended_state()
    robot_base = vicon_frame.subjects[pam_vicon.o80.Subjects.ROBOT1_BASE]
    robot_position = robot_base.global_pose.translation
    robot_orientation = robot_base.global_pose.get_rotation()
    robot_euler = Rotation.from_quat(robot_orientation).as_euler("xyz", degrees=True)
    with np.printoptions(precision=2, suppress=True):
        print(f"Position diff.: {robot_position - expected_position}, Orientation diff.: {robot_euler - expected_euler}°")
