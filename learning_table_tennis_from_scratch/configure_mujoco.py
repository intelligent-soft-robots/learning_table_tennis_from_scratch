import o80_pam
import pam_mujoco


SEGMENT_ID_BALL = pam_mujoco.segment_ids.ball
SEGMENT_ID_GOAL = pam_mujoco.segment_ids.goal
SEGMENT_ID_HIT_POINT = pam_mujoco.segment_ids.hit_point
SEGMENT_ID_ROBOT_MIRROR = pam_mujoco.segment_ids.mirroring
SEGMENT_ID_PSEUDO_REAL_ROBOT = o80_pam.segment_ids.robot



def configure_pseudo_real(mujoco_id="pseudo-real",
                          graphics=True,
                          accelerated_time=False):

    if accelerated_time:
        burst_mode = True
    else:
        burst_mode = False

    robot = pam_mujoco.MujocoRobot(SEGMENT_ID_PSEUDO_REAL_ROBOT,
                                   control=pam_mujoco.MujocoRobot.PRESSURE_CONTROL)
    handle = pam_mujoco.MujocoHandle(mujoco_id,
                                     graphics=graphics,
                                     accelerated_time=accelerated_time,
                                     burst_mode=burst_mode,
                                     robot1=robot)
    
    return handle


def configure_simulation(mujoco_id="simulation",graphics=True):

    accelerated_time=True
    burst_mode=True

    robot = pam_mujoco.MujocoRobot(SEGMENT_ID_ROBOT_MIRROR,
                                   control=pam_mujoco.MujocoRobot.JOINT_CONTROL)
    ball = pam_mujoco.MujocoItem(SEGMENT_ID_BALL,
                                  control=pam_mujoco.MujocoItem.COMMAND_ACTIVE_CONTROL,
                                  contact_type=pam_mujoco.ContactTypes.racket1)
    hit_point = pam_mujoco.MujocoItem(SEGMENT_ID_HIT_POINT,
                                      control=pam_mujoco.MujocoItem.CONSTANT_CONTROL)
    goal = pam_mujoco.MujocoItem(SEGMENT_ID_GOAL,
                                 control=pam_mujoco.MujocoItem.CONSTANT_CONTROL)
    handle = pam_mujoco.MujocoHandle(mujoco_id,
                                     graphics=graphics,
                                     accelerated_time=accelerated_time,
                                     burst_mode=burst_mode,
                                     table=True,
                                     robot1=robot,
                                     balls=(ball,),
                                     hit_points=(hit_point,),
                                     goals=(goal,))
    
    return handle
                          
