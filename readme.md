

# Installing PAM robot software

Follow the instructions [here](http://people.tuebingen.mpg.de/mpi-is-software/pam/docs/pam_documentation/index.html).

# Installing learning_table_tennis_from_scratch

```
git clone https://github.com/intelligent-soft-robots/learning_table_tennis_from_scratch.git
cd learning_table_tennis_from_scratch
pip3 install .
```

# Running

In a first terminal:

```bash
hysr_start_robots
```

In a second terminal, optionnaly (for graphics):

```bash
hysr_visualization
```

and finally:

```bash
hysr_one_ball_ppo
```

# Configuration files

When starting *hysr_one_ball_ppo*, a dialog allows to select the configuration files:

- hysr_config_file: configuration of the mujoco simulated robot (e.g. accelerated time, graphics, etc)
- pam_config_file: configuration of the simulated muscles
- ppo_config_file: configuration of the ppo algorithm
- reward_config_file: configuration of the reward function

It is possible to start the executable without using the dialog, for example:

```bash
hysr_on_ball_ppo -hysr_config_file /path/to/config/file.json -reward_config_file /path/to/other/config/file.json 
```



# tensorboard

In a terminal:

```
tensorboard --logdir /tmp/ppo2
```

# Unit tests

in the repository root directory, run:

```bash
pip install .
python -m unittest discover .
```

# Other executables

Can be also run after *hysr_start_robots* (executabled created for debug purposes):

- hysr_one_ball_rewards: plays several scenario with different ball trajectories and compute the corresponding reward
- hysr_one_ball_swing: has the racket performing some swing motions
- hysr_one_ball_reset: has the environment performing resets
- hysr_one_ball_random: the robot makes random moves
