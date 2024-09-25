#!/usr/bin/env python3

import time

from ball_launcher_beepy import BallLauncherClient
PHI = 0.49
THETA = 0.75
MOTOR_ACTUATION = 0.35

client = BallLauncherClient("10.42.26.171", 5555)
client.set_state(
    PHI, THETA, MOTOR_ACTUATION, MOTOR_ACTUATION, MOTOR_ACTUATION
)
time.sleep(2)
client.launch_ball()

