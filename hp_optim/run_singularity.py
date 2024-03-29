"""Simple wrapper script to run a Python script in Singularity.

Parameters to configure Singularity have to be provided through the "fixed_params" field
of the cluster_utils config file.  The following parameters are expected:

- "singularity.image": Path to the Singularity image.  Note: The image is expected to
  have a runscript that does any environment setup if needed and then executes the
  command that is given as argument.  For example:

      %runscript
      . /setup.sh  # do setup stuff here
      exec "$@"

- "singularity.script": Path to the script that is executed in the container.  This
  needs to be executable (i.e. have a shebang line and the executable bit set).

Example:

    "fixed_params": {
        "singularity.image": "~/image.sif",
        "singularity.script": "./main.py"
    }

"""
import os
import pathlib
import subprocess
import sys

import cluster


def main():
    # get the singularity parameters (pass copy of sys.argv so that the
    # original arguments do not get modified)
    argv_copy = sys.argv.copy()
    # clear the first argument (which includes server info) to prevent this
    # script from registering exit reporting (this is already done by the
    # run_learning.py which is run by this script).
    # FIXME: This is just a pretty dirty temporary workaround, a better
    # solution probably needs some change in cluster_utils (maybe integrate
    # Singularity support directly there?)
    del argv_copy[1]
    params = cluster.read_params_from_cmdline(
        argv_copy, verbose=False, save_params=False
    )

    # some basic sanity checks of the parameters
    if "singularity" not in params:
        raise AttributeError(
            "No Singularity parameters given.  Please specify them in 'fixed_params'."
        )
    for attr in ("image", "script"):
        if not hasattr(params.singularity, attr):
            raise KeyError(
                f"Parameter 'singularity.{attr}' is not specified.  Please add it to"
                " 'fixed_params'."
            )

    singularity_image = os.path.expanduser(params.singularity.image)

    if not os.path.exists(singularity_image):
        raise FileNotFoundError(
            "Singularity image '{}' does not exist".format(singularity_image)
        )
    if not os.path.exists(params.singularity.script):
        raise FileNotFoundError(
            "Specified script '{}' does not exist".format(params.singularity.script)
        )

    # create model directory (so it can be bound into the container)
    working_dir = pathlib.Path(params.working_dir)
    working_dir.mkdir(exist_ok=True)

    # run Singularity
    cwd = os.getcwd()
    bind_dirs = ["/tmp", params.working_dir, cwd]
    cmd = [
        "singularity",
        "run",
        "--nv",
        "--containall",
        "--bind=%s" % ",".join(bind_dirs),
        "--pwd=%s" % cwd,
        singularity_image,
        params.singularity.script,
        *sys.argv[1:],
    ]

    # DBG_NAN_SNAPSHOT_DIR
    nan_snapshot_dir = working_dir / "nan_snapshots"
    nan_snapshot_dir.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["SINGULARITYENV_DBG_NAN_SNAPSHOT_DIR"] = os.fspath(nan_snapshot_dir)
    env["APPTAINERENV_DBG_NAN_SNAPSHOT_DIR"] = os.fspath(nan_snapshot_dir)

    # explicitly redirect output to file, so it is stored also when running
    # locally
    stdout_file = working_dir / "stdout.txt"
    stderr_file = working_dir / "stderr.txt"
    with open(stdout_file, "ab") as f_out, open(stderr_file, "ab") as f_err:
        result = subprocess.run(cmd, stdout=f_out, stderr=f_err, env=env)

    # forward the return code
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
