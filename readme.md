

# Installing

The below assumes your default version of python3 is python3.6.*

## install pam_mujoco

see instructions [here](https://github.com/intelligent-soft-robots/intelligent-soft-robots.github.io/wiki/01_installation-from-debian)

## install learning_table_tennis_from_scratch

```
git clone https://github.com/intelligent-soft-robots/learning_table_tennis_from_scratch.git
cd learning_table_tennis_from_scratch
pip3 install .
```

# Running

## Demos

### "normal" time

```
# in a terminal
hysr_start_robots
# in another terminal
hysr_one_ball_swing # or hysr_one_ball_random
```

### "accelerated" time

```
# in a terminal
hysr_start_robots_accelerated
# in another terminal
hysr_one_ball_swing --accelerated  # or hysr_one_ball_random --accelerated
```

**known issue** : sometimes the simulated robots will not exit properly on "ctrl+c" or even on closing the terminal. Running ```killall python3``` after each run helps.

### seeing all simulated robots

The commands *hysr_start_robots* and *hysr_start_robots_accelerated* start 3 mujoco simulations, even if only one is displayed. To display all the simulated robots, run instead in 3 different terminals:

```
# starts the pressure controlled robot ("pseudo-real" robot)
o80_mujoco
```
```
# starts the position controlled robot that will 
# mirror the "pseudo-real" robot.
pam_mujoco --accelerated --bursting_mode --graphics
```
```
# (optional)
# starts a simulation that display the position
# controlled robot, but with smoother graphics
pam_visualization -mujoco_id pam_robot_mujoco
```

## learning

### starting

Same as above, but using *hysr_one_ball_ppo* instead of *hysr_one_ball_swing*.


### tensorboard

In a terminal:

```
tensorboard --logdir /tmp/ppo2
```

