

# Installing PAM robot software

Follow the instructions [here](http://people.tuebingen.mpg.de/mpi-is-software/pam/docs/).

# Installing learning_table_tennis_from_scratch

```
git clone https://github.com/intelligent-soft-robots/learning_table_tennis_from_scratch.git
cd learning_table_tennis_from_scratch
pip3 install .
```

# Running

## Prepare configuration files

The system uses at minima 3 json configuration files (5 if running an learning algorithm).

### hysr configuration file

Example:

```json
{
    "real_robot":false,
    "o80_pam_time_step":0.002,
    "mujoco_time_step":0.002,
    "algo_time_step":0.01,
    "pam_config_file":"/opt/mpi-is/pam_interface/pamy1/pam_sim.json",
    "robot_position":[0.5, 0.0, -0.44],
    "target_position":[0.45,2.7,-0.45],
    "reference_posture":[[19900,16000],[16800,19100],[18700,17300],[18000,18000]],
    "world_boundaries":{
	"min":[0.0,-1.0,0.17],
	"max":[1.6,3.5,1.5]
    },
    "pressure_change_range":18000,
    "trajectory":-1,
    "accelerated_time":false,
    "instant_reset":true,
    "nb_steps_per_episode":-1,
    "extra_balls_sets":-1,
    "extra_balls_per_set":-1,
    "graphics":true,
    "graphics_pseudo_real":false,
    "graphics_simulation":false,
    "graphics_extra_balls":false,
    "xterms":false,
    "frequency_monitoring_episode":true,
    "frequency_monitoring_step":true
}
```

- real_robot: false, or if using the real robot, the segment_id of its o80 backend (set when starting the robot, default:"real_robot")
- o80_pam_time_step: frequency at which the pressure robot is controlled.
When using a mujoco simulated pressure robot (i.e. "real_robot" set to false), the same value as the mujoco_time_step should be used.
When using the real robot, the time step corresponding to the frequency selected at startup of the robot should be used.
- mujoco_time_step: control frequency of the mujoco instances, 0.002 second is a reasonable value.
- algo_time_step: frequency at which the learning algorithm will run
- pam_config_file: configuration of the pam robot, including min and max pressures. Possibly use "/opt/mpi-is/pam_interface/pamy1/pam_sim.json"
for a mujoco simulated pressure robot, and "/opt/mpi-is/pam_interface/pamy2/pam.json" when using the real (pamy2) robot.
- robot_position: position of the simulated robot in the xy plane
- target_position: 3d position of the target, i.e. where the robot is trained to aim the ball when learning table tennis
- reference_posture: when the environment reset, the robot will first "go" to this posture, i.e. this set of pressure,
in the format [[agonist muscle pressure, agonist muscle pressure] ...]
- world_boundaries: boundaries of the 3d world (in meters)
- pressure_change_range: the action of the rl algorithm will consist of delta of pressures, the pressure change range is the maximal
delta value
- trajectory: the ball will play at each episode a pre-recorded trajectory. A negative value indicate that at each episode, a pre-recorded
trajectory will be selected randomly. A positive value will be the index of the trajectory to play (same trajectory played at each episode)
- accelerated_time: all mujoco simulations will run at maximal speed. Should not be set to true when using the real robot.
- instant_reset: between each episode, the robot will go to the reference_posture. If instant reset is true, it will "teleport" to the reference
posture, if false it will execute commands that will bring it to the posture. Should be false when using the real robot.
- nb_steps_per_episode: if a negative value, episodes will stop when the ball position z component passes a threshold (i.e. the ball has low height).
If a positive value: episodes will stop after the corresponding number of steps .
- extra_balls_sets: if negative, the environment will "host" only one ball. If a positive value, supplementary balls will be added to environment.
For the moment, only the positive value "1" is supported.
- extra_balls_per_set: number of extra balls per extra balls set. Only values supported: 1, 3, 10 and 20.
- graphics: if true, an extra mujoco simulation will open with graphics, displaying the motion of the robot, ball and extra balls
- graphics_pseudo_real, graphics_simulation, graphics_extra_balls: if true, the corresponding mujoco instance will start with graphics. Because
 of some technical reason (use of o80's [bursting mode](http://people.tuebingen.mpg.de/mpi-is-software/o80/docs/o80/doc/06.bursting.html)), they
 may be laggy. To be used only for debug. 
- xterms: if true, the controllers of all mujoco instances will start in new dedicated terminals. If false, they all start in the current terminal
(output may be more difficult to read)
- frequency_monitoring_episode: if true, it will allow the usage of the ```hysr_episode_frequencies``` executable, see documentation somewhere below
- frequency_monitoring_step: if true, it will allow the usage of the ```hysr_step_frequencies``` executable, see documentation somewhere below


## starting the robots

In a first terminal, start either of these executables:

```bash
# start the robots in new terminals
hysr_start_robots
```

press enter to get the prompt back.

The executable above initializes several instances of mujoco.

- the pseudo real robot: a pressure controlled robot
- the simulated robot: a joint control robot, plus a table tennis, a ball; and a visual marker
- if using extra balls set: an instance of mujoco per ball set

During runtime of the learning algorithm, and via a [gym](https://gym.openai.com/) environment, pressures commands will be sent to the pseudo real robot;
and the simulated robot will mirror the motion of the pseudo-real robot (HYSR - HYbrid Sim to Real,
see this [publication](https://arxiv.org/pdf/2006.05935.pdf)).

After starting, the mujoco simulations hang and wait for configuration (which occurs when one executable is called, see right below).

## start the control executables

The executables will configure the mujoco instances, and then send control commmands to the pseudo-real robot
and / or to the simulated robot (via a gym environment, if the executable run a learning algorithm).

### list of executables

The source of the executables are in the bin folder of the repository.

Executables for testing and debug:

- hysr_one_ball_swing : the ball plays prerecorded-trajectories, and the robot performs swing motions
- hysr_one_ball_random: the ball plays prerecorded-trajectories, and the robot performs random motions
- hysr_one ball_reset: the mujoco simulations perform several resets
- hysr_one_ball_reward_tests: the robot moves, the ball performs trajectories and the corresponding rewards are computed
 
Executable for learning:

- hysr_one_ball_rl: learning table tennis using RL and HYSR.

### Configuring and starting the executable

The executables requires a configuration json file to be in the current folder. The configuration file
can be named anything, and just requires a *.json extension.
The content of the configuration file is a dictionary pointing to other json configuration files.

For executables for testing and debug, the json file must have this content:

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
    "rl_config":"/path/to/rl/json/config/file",
    "rl_common_config":"/path/to/rl-common/json/config/file"
}
```

The bin folder has exemple of configuration files, except for:

- the ```pam_model``` configuration file, which has an example in ```/opt/mpi-is/pam_models/```
(possibly you will want to use ```/opt/mpi-is/pam_models/hill.json```).

- the ```pam_config``` configuration file (very possibly you will want to use ```/opt/mpi-is/pam_interface/pamy1/pam_sim.json```)

Once the configuration file has been set in the current directory, the executable can be started, e.g.:

```
hysr_one_ball_rl
```

This will trigger the start of the mujoco simulations. Whether or not the simulations open a graphical display
depends of the content of the "hysr_config" json file.

*Note*: for as long as the configuration does not change, it is possible to run several executables in
a row without restarting the robots. If the configuration is changed, then the robots need to be restarted (see below).

## Exiting the robots

In any terminal, type:

```
hysr_stop
```

## Example configuration

The `example/` directory contains an example configuration that can be used to
run the above commands.  See the README there for more information.


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

```hysr_display_extra_balls``` can be started at any time after the executable, and from the
same folder (i.e. using the same json configuration file).


# Tensorboard

In a terminal:

```
tensorboard --logdir /tmp/rl
```

