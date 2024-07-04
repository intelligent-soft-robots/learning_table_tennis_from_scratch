while ball_id == -1 or ball_id == last_ball_id:
    iteration = self.tennicam_frontend.latest().get_iteration()
    obs = self.tennicam_frontend.read(iteration)
    time_stamp = obs.get_time_stamp() * 1e-9
    position = obs.get_position()
    velocity = obs.get_velocity()
    ball_id = obs.get_ball_id()