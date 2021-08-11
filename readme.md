

# Installing PAM robot software

Follow the instructions [here](http://people.tuebingen.mpg.de/mpi-is-software/pam/docs/pam_documentation/index.html).

# Installing learning_table_tennis_from_scratch

```
git clone https://github.com/intelligent-soft-robots/learning_table_tennis_from_scratch.git
cd learning_table_tennis_from_scratch
pip3 install .
```

# Running

## starting the robots

In a first terminal, start either of these executable:

```bash
# start the robots in new terminals
hysr_start_robots
```

or

```bash
# start the robots in current terminal
hysr_start_robots_no_xterms
```

press enter to get the prompt back.

The executable above initialize two instances of mujoco.

- the pseudo real robot: a pressure controlled robot
- the simulated robot: a joint control robot, plus a table tennis, a ball; and a visual marker

During runtime of the learning algorithm, pressures commands will be sent to the pseudo real robot;
and the simulated robot will mirror the motion of the pseudo-real robot (HYSR - HYbrid Sim to Real,
see this [publication](https://arxiv.org/pdf/2006.05935.pdf)).

After start, the mujoco simulation hangs and wait for a configuration.

## start the control executables

The executables will configure the mujoco instances, and then send control commmands to the pseudo-real robot
and / or to the simulated robot.

### list of executables

The source of the executables are in the bin folder of the repository.

Executables for testing and debug:

- hysr_one_ball_swing : the ball plays prerecorded-trajectories, and the robot performs swing motions
- hysr_one_ball_random: the ball plays prerecorded-trajectories, and the robot performs random motions
- hysr_one ball_reset: the mujoco simulations perform several resets
- hysr_one_ball_reward_tests: the robot moves, the ball performs trajectories and the corresponding rewards are computed
 
Executable for learning:

- hysr_one_ball_ppo: learning table tennis using PPO and HYSR.

### Configuring and starting the executable

The executables requires a configuration json file to be in the current folder. The configuration file
can be named anything, and just requires a *.json extension.
The content of the configuration file is a dictionary pointing to other json configuration files.

For executables for testing and debug, the json must have this content:

```json
{
    "reward_config":"/path/to/reward/json/config/file",
    "hysr_config":"/path/to/hysr/json/config/file"
}
```

The paths can be relative or absolute. You can find example of configuration files in the config folder
of the repository.

*Note*: if the ```hysr_config``` file requests the use of extra balls (```extra_balls_set``` and ```extra_balls_per_set```)
key, please refer to the ```Extra balls``` section somewhere below.

The learning executable configuration json file requires this content:

```json
{
    "reward_config":"/path/to/reward/json/config/file",
    "hysr_config":"/path/to/hysr/json/config/file",
    "pam_config":"/path/to/pam/json/config/file",
    "ppo_config":"/path/to/ppo/json/config/file",
    "ppo_common_config":"/path/to/ppo-common/json/config/file"
}
```

The bin folder has exemple of configuration files, except for the ```pam_config```
configuration file, which has an example in ```/opt/mpi-is/pam_models/```
(very possibly you will want to use ```/opt/mpi-is/pam_models/hill.json```).

Once the configuration file has been set in the current directory, the executable can be started, e.g.:

```
hysr_one_ball_ppo
```

This will trigger the start of the mujoco simulations. Whether or not the simulations open a graphical display
depends of the content of the "hysr_config" json file.

*Note*: for as long as the configuration does not change, it is possible to run several executables in
a row without restarting the robots. If the configuration is changed, then the robots need to be restarted (see below).

## Exiting the robots

In any terminal, type:

```
pam_mujoco_stop_all
```

## Extra executables

The following executables can be started after the robot:

- hysr_episode_frequency: displaying stats regarding the frequency at which episodes run
- hysr_step_frequency: displaying stats regarding the frequency at which simulation steps run
- hysr_visualization: if the robots are started with graphical displays, you may notice these displays
are laggy. It is adviced to start the robots without graphical display, and run hysr_visualization instead

## Extra balls

It is possible to start a third instance of mujoco (i.e. on top of pseudo-real and simulated robot) which,
similarly to the simulated robot, will manages a joint controlled robot that will mirror the pseudo-real robot.
This third instance will also be used to managed extra balls.

### configuration

To add the support of extra balls, the ```hysr_config```  json files has to be updated:

```json
{
	...,
	"extra_balls_sets":1,
    	"extra_balls_per_set":20,
        "graphics_extra_balls":false,
	...
}
```

- for ```extra_balls_sets```, only the values 1 (extra balls) or 0 (no extra balls) are supported
- for ```extra_balls_per_set```, only the values 3, 10 and 20 are supported.


### starting robots with extra balls

On top of ```hysr_start_robots```, a supplementary instance of pam_mujoco has to be started:

```bash
pam_mujoco extra_balls_0
```

or

```bash
pam_mujoco_no_xterms extra_balls_0
```

It is possible to start everything at the same time:

```
hysr_start_robots_no_xterms & pam_mujoco_no_xterms extra_balls_0
```

### starting executables

The executables are started in a similar way.

### accessing extra balls data

Data related to the extra balls can be accessed via python code running in parallel to
the simulated robots and to the executable.

See the file ```hysr_display_extra_balls``` in the bin folder for an example.

```hysr_display_extra_balls``` can started at any time after the executable, and from the
same folder (i.e. using the same json configuration file).


# Tensorboard

In a terminal:

```
tensorboard --logdir /tmp/ppo2
```

