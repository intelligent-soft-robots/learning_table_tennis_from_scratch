import o80_pam
import pam_mujoco

SEGMENT_ID_TABLE = pam_mujoco.segment_ids.table
SEGMENT_ID_BALL = pam_mujoco.segment_ids.ball
SEGMENT_ID_GOAL = pam_mujoco.segment_ids.goal
SEGMENT_ID_HIT_POINT = pam_mujoco.segment_ids.hit_point
SEGMENT_ID_ROBOT_MIRROR = pam_mujoco.segment_ids.mirroring
SEGMENT_ID_PSEUDO_REAL_ROBOT = o80_pam.segment_ids.robot

MUJOCO_ID_EXTRA_BALLS_SETS_PREFIX = "extra_balls"
SEGMENT_ID_EXTRA_BALLS_SETS_PREFIX = "extra_balls_sid"
SEGMENT_ID_EXTRA_ROBOTS_PREFIX = "extra_robot_sid"


def get_extra_balls_set_mujoco_id(
    setid, mujoco_id_prefix=MUJOCO_ID_EXTRA_BALLS_SETS_PREFIX
):
    return "_".join([mujoco_id_prefix, str(setid)])


def get_extra_robot_segment_id(
    setid, segment_id_prefix=SEGMENT_ID_EXTRA_ROBOTS_PREFIX
):
    return "_".join([segment_id_prefix, str(setid)])


def get_extra_balls_segment_id(
    setid, segment_id_prefix=SEGMENT_ID_EXTRA_BALLS_SETS_PREFIX
):
    return "_".join([segment_id_prefix, str(setid)])


def get_ball_segment_id(set_id, index):
    return "ball_" + str(set_id) + "_" + str(index)


def configure_extra_set(setid, hysr_config):
    nb_balls = hysr_config.extra_balls_per_set
    graphics = hysr_config.graphics_extra_balls
    accelerated_time = True
    burst_mode = True

    table = pam_mujoco.MujocoTable(
        SEGMENT_ID_TABLE,
        position=hysr_config.table_position,
        orientation=hysr_config.table_orientation,
    )

    robot_segment_id = get_extra_robot_segment_id(setid)
    robot = pam_mujoco.MujocoRobot(
        robot_type=hysr_config.robot_type,
        segment_id=robot_segment_id,
        position=hysr_config.robot_position,
        orientation=hysr_config.robot_orientation,
        control=pam_mujoco.MujocoRobot.JOINT_CONTROL,
    )

    extra_balls_segment_id = get_extra_balls_segment_id(setid)
    balls = pam_mujoco.MujocoItems(extra_balls_segment_id)

    ball_segment_ids = [
        get_ball_segment_id(setid, index) for index in range(nb_balls)
    ]

    for index, ball_segment_id in enumerate(ball_segment_ids):
        ball = pam_mujoco.MujocoItem(
            ball_segment_id,
            control=pam_mujoco.MujocoItem.CONSTANT_CONTROL,
            contact_type=pam_mujoco.ContactTypes.racket1,
        )
        balls.add_ball(ball)

    mujoco_id = get_extra_balls_set_mujoco_id(setid)

    handle = pam_mujoco.MujocoHandle(
        mujoco_id + job_id,
        graphics=graphics,
        accelerated_time=accelerated_time,
        burst_mode=burst_mode,
        table=table,
        robot1=robot,
        combined=balls,
        o80_backend_period=hysr_config.mujoco_time_step,
    )

    return (
        handle,
        mujoco_id,
        extra_balls_segment_id,
        robot_segment_id,
        ball_segment_ids,
    )


def configure_pseudo_real(
    pam_config_file,
    robot_type,
    save_data,
    save_folder="/tmp",
    mujoco_id="pseudo-real",
    job_id="",
    graphics=True,
    accelerated_time=False,
):
    sement_id_pseudo_real_robot = SEGMENT_ID_PSEUDO_REAL_ROBOT + job_id
    if accelerated_time:
        burst_mode = True
    else:
        burst_mode = False

    robot = pam_mujoco.MujocoRobot(
        robot_type=robot_type,
        segment_id=sement_id_pseudo_real_robot,
        control=pam_mujoco.MujocoRobot.PRESSURE_CONTROL,
        json_control_path=pam_config_file,
    )
    handle = pam_mujoco.MujocoHandle(
        mujoco_id + job_id,
        graphics=graphics,
        accelerated_time=accelerated_time,
        burst_mode=burst_mode,
        save_data=save_data,
        save_folder=save_folder,
        robot1=robot,
    )

    frontend = handle.frontends[sement_id_pseudo_real_robot]

    return handle, frontend


def configure_simulation(hysr_config, mujoco_id="simulation", job_id=""):
    segment_id_robot_mirror = SEGMENT_ID_ROBOT_MIRROR + job_id
    segment_id_table = SEGMENT_ID_TABLE + job_id
    segment_id_ball = SEGMENT_ID_BALL + job_id
    segment_id_hit_point = SEGMENT_ID_HIT_POINT + job_id
    segment_id_goal = SEGMENT_ID_GOAL + job_id
    accelerated_time = True
    burst_mode = True
    graphics = hysr_config.graphics_simulation
    save_data = hysr_config.save_data
    save_folder = hysr_config.save_folder

    table = pam_mujoco.MujocoTable(
        segment_id_table,
        position=hysr_config.table_position,
        orientation=hysr_config.table_orientation,
    )
    robot = pam_mujoco.MujocoRobot(
        robot_type=hysr_config.robot_type,
        segment_id=segment_id_robot_mirror,
        position=hysr_config.robot_position,
        orientation=hysr_config.robot_orientation,
        control=pam_mujoco.MujocoRobot.JOINT_CONTROL,
    )
    ball = pam_mujoco.MujocoItem(
        segment_id_ball,
        control=pam_mujoco.MujocoItem.COMMAND_ACTIVE_CONTROL,
        contact_type=pam_mujoco.ContactTypes.racket1,
    )
    hit_point = pam_mujoco.MujocoItem(
        segment_id_hit_point, control=pam_mujoco.MujocoItem.CONSTANT_CONTROL
    )
    goal = pam_mujoco.MujocoItem(
        segment_id_goal, control=pam_mujoco.MujocoItem.CONSTANT_CONTROL
    )
    handle = pam_mujoco.MujocoHandle(
        mujoco_id + job_id,
        graphics=graphics,
        accelerated_time=accelerated_time,
        burst_mode=burst_mode,
        save_data=save_data,
        save_folder=save_folder,
        table=table,
        robot1=robot,
        balls=(ball,),
        hit_points=(hit_point,),
        goals=(goal,),
        o80_backend_period=hysr_config.mujoco_time_step,
    )

    return handle
