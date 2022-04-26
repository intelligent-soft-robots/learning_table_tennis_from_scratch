Hyperparameter Optimisation with cluster_utils
==============================================


This directory contains the scripts and configuration needed to run
hyperparameter optimisation using the `cluster_utils` package.


## Install cluster_utils

Cluster utils is a Python package that can be installed with pip.

The original package can be found at
https://gitlab.tuebingen.mpg.de/mrolinek/cluster_utils.
However, at the time of writing this README, some modifications have been made
on [Felix' fork](https://gitlab.tuebingen.mpg.de/felixwidmaier/cluster_utils)
which are not (yet?) merged in the upstream repo, so it is recommended to
install the "fwidmaier/displot" branch from this fork.  You can directly do
this using pip with the following command (you will be asked for your GitLab
credentials):

    pip3 install git+https://gitlab.tuebingen.mpg.de/felixwidmaier/cluster_utils.git@fwidmaier/displot


## Run Hyperparameter Optimisation

To run the hyperparameter optimisation go to the
`learning_table_tennis_from_scratch` directory and run the following command:

    python3 -m cluster.hp_optimization hp_optim/hp_optim_ppo.json

IMPORTANT: Before running it, check the config file `hp_optim_ppo.json`, you
may need to adjust paths!

When testing locally, you may also want to set `run_in_working_dir` to "true",
so it uses the local files instead of using git to get the current master.

See the following sections for configuration of the wrapper scripts that are
used here.

### Overwrite Configuration from Command Line

You can also overwrite individual settings from the JSON file by providing
values in the command line.  This can be useful, for example, to more easily set
a different output directory for different runs:

    python3 -m cluster.hp_optimization hp_optim/hp_optim_ppo.json 'optimization_procedure_name="lttfs_2021-12-15_1"'


## Configuration for run_singularity.py

`cluster_utils` on its own can only run Python scripts and doesn't provide
built-in support for Singularity.  To be able to run things in Singularity
containers nonetheless, the `run_singularity.py` wrapper script is provided.

To use it, replace

    "script_relative_path": "my_script.py",

with

    "script_relative_path": "run_singularity.py",
    "fixed_params": {
        "singularity.image": "/path/to/iamge.sif",
        "singularity.script": "my_script.py"
    },

When using this, keep the following in mind:

- `script_relative_path` is resolved relative to the working directory
  (typically the root of the git repository when using `cluster_utils`), so the
  `run_singularity.py` script needs to be added to your config.
- `singularity.image` may use `~` which will be expanded to the users home
  directory.
- `singularity.script` is resolved relatively like `script_relative_path`.
  However, it does not assume a Python script but the given script needs to be
  executable on its own (i.e. contain a shebang line and have the executable
  bit set).
- The image needs to have all dependencies as well as `cluster_utils` itself
  installed.  You can use the "learning_table_tennis_from_scratch" image from
  [pam_singularity](https://github.com/intelligent-soft-robots/pam_singularity).


## Configuration for run_learning.py

`run_learning.py` is a wrapper script around running `hysr_one_ball_rl` which
sets up the necessary configuration files based on the parameters given by
`cluster_utils`.

For this, it uses complete configuration files, which are expected to provide
the default values, and overwrites those values that are provided as
hyperparameters by `cluster_utils`.


### Specifying the configuration files

To find "template" configuration files, a single JSON file needs to be given
through the "fixed" parameter `config.config_templates`, which points to the
relevant files.  For example:

    "fixed_params": {
        "config.config_templates": "./hp_optim/config_templates_ppo.json",
    },

The file itself follows the same structure as the JSON file that is expected by
the hysr exectuables:

    {
        "reward_config": "../config/reward_default.json",
        "hysr_config": "../config/hysr_one_ball_default_sim.json",
        "pam_config": "/opt/mpi-is/pam_models/hill.json",
        "rl_config": "../config/openai_ppo_default.json",
        "rl_common_config": "../config/rl_common_default.json"
    }

Paths are resolved relative to the location of the file.


### Defining the hyperparameters

The hyperparameters are defined in the normal way in the `cluster_utils`
configuration.  The names have to follow the pattern

    config.<config_file>.<parameter_name>

E.g. to overwrite the parameters "num_hidden" in the config file that is given
for "rl_config" in the template file (see above), the name would be

    config.rl_config.num_hidden


#### Non-scalar parameters

`cluster_utils` can only handle scalar values.  For parameters that expect a list of
values, run_learning.py has some special handling.  For this add ``:<index>`` at the end
of the parameter name.  Example:

    "optimized_params": [
        {
            "param": "config.hysr_config.robot_position:0",
            "distribution": "TruncatedNormal",
            "bounds": [-0.35, 0.35]
        }
    ]

This will sample a value for index 0 of parameter ``robot_position`` in the hysr
configuration.


### Training Iterations

If the training of a model takes very long, it can be desired to split the training into
multiple cluster jobs to save cluster money (the price for a job increases exponentially
over time, so multiple short jobs are cheaper than one long one).

This can be done with the `training_iterations` parameter:

    "fixed_params": {
        "training_iterations": 3
    }

If set, the training will be run normally for the number of time steps as configured in
the training parameters of the model.  However, once finished, it is marked for
continuation.  The cluster job will be restarted and the new job will load the model
from the previous one to continue training on (again for the configured number of time
steps).
git add
Example: Instead of having one job that trains for 1,000,000 steps, one can reduce the
training steps to 100,000 and set `training_iterations` to 10.  It will then run ten
shorter trainings, each building upon the previous one, thus resulting in the same total
number of training steps for the final model.


### Configuring number of reruns and retries

Independent of the `with_restarts` option of cluster utils, the
`run_learning.py` script provides the option to rerun the learning multiple
times in each job.  The metric for the optimisation is then the mean reward of
these runs, thus reducing the variance.
This can be configured in the configuration file by specifying
`learning_runs_per_job` in the `fixed_params` section:

    "fixed_params": {
        "learning_runs_per_job": 2
    }

The default is 3.

Further failed jobs can be retried for a configurable number of times.  This can
be useful for failure cases that only happen sometimes (e.g. based on the random
seed) so that there is a high chance that the run will be successful when simply
trying again with the same settings.
This can be configured in the configuration file by specifying
`max_attempts` in the `fixed_params` section:

    "fixed_params": {
        "max_attempts": 2
    }

Set to 1 for a single attempt without retries.  The default is 3.


## Output Files

The output directory is configured in the cluster utils config through the parameters
`optimization_procedure_name` and `results_dir`.  All output files of the hyperparameter
search will be written to `results_dir/optimization_procedure_name/`.

Once finished, the files generated by the best hyperparameter configurations are saved
in subdirectories `best_jobs/directory_#`.

Inside there, a number of "run" directories is created, one for each cluster job.  They
are named with the following pattern:

    run_{learning_run}.{training_iteration}-{attempt}
