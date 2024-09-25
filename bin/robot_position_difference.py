import numpy as np
import pam_vicon.o80
from scipy.spatial.transform import Rotation

expected_position = np.array([0.23481061, -3.38250561, -0.95063758])
expected_orientation = np.array([-0.02615020044781693, 0.021959215361716973, -0.008822280253092582, 0.999377870101381])
expected_euler = Rotation.from_quat(expected_orientation).as_euler("xyz", degrees=True)

vicon_frontend = pam_vicon.o80.FrontEnd("vicon")

while True:
    vicon_frame = vicon_frontend.latest().get_extended_state()
    robot_base = vicon_frame.subjects[pam_vicon.o80.Subjects.ROBOT1_BASE]
    robot_position = robot_base.global_pose.translation
    robot_orientation = robot_base.global_pose.get_rotation()
    robot_euler = Rotation.from_quat(robot_orientation).as_euler("xyz", degrees=True)
    with np.printoptions(precision=2, suppress=True):
        print(f"Position diff.: {robot_position - expected_position}, Orientation diff.: {robot_euler - expected_euler}°", end="\r")
