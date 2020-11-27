

# installation

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

## learning

### starting

Same as above, but using *hysr_one_ball_ppo* instead of *hysr_one_ball_swing*.


### tensorboard

In a terminal:

```
tensorboard --logdir /tmp/ppo2
```
