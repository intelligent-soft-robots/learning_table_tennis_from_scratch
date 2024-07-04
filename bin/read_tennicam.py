import json
import time
from datetime import datetime

import tennicam_client

TENNICAM_CLIENT_DEFAULT_SEGMENT_ID = "tennicam_client"

if __name__ == '__main__':
    tennicam_frontend = tennicam_client.FrontEnd(TENNICAM_CLIENT_DEFAULT_SEGMENT_ID)
    positions = []
    velocities = []
    timestamps = []
    for _ in range(10000):
        obs = tennicam_frontend.latest()
        positions.append(obs.get_position())
        velocities.append(obs.get_velocity())
        timestamps.append(datetime.now().timestamp())
        time.sleep(0.002)

    data = {
        "ball_camera_positions": positions,
        "ball_camera_velocities": velocities,
        "timestamps": timestamps,
    }

    with open("/tmp/tennicam_robot_pc.json", "w") as f:
        json.dump(data, f)
