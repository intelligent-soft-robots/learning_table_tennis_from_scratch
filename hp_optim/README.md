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

`run_learning.py` is a wrapper script around running `hysr_one_ball_ppo` which
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
        "ppo_config": "../config/openai_ppo_default.json",
        "ppo_common_config": "../config/ppo_common_default.json"
    }

Paths are resolved relative to the location of the file.


### Defining the hyperparameters

The hyperparameters are defined in the normal way in the `cluster_utils`
configuration.  The names have to follow the pattern

    config.<config_file>.<parameter_name>

E.g. to overwrite the parameters "num_hidden" in the config file that is given
for "ppo_config" in the template file (see above), the name would be

    config.ppo_config.num_hidden
