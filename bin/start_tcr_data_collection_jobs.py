import argparse
import math
from pathlib import Path
from typing import Sequence, Tuple


import htcondor
import classad


def submit_jobs(jobs: Sequence[Tuple[int, int]], env: str, train_logs: str, intervention_std: float, cpus: int, memory: float, kwargs_str: str) -> None:
    # Initialize the HTCondor submit object
    repo_path = Path(__file__).parents[1]
    submit = htcondor.Submit(
        {
            "executable": repo_path / "bin" / "tcr_data_collection_job.sh",  # The Python script to run
            "arguments": "$(ClusterId) $(Process)",  # Pass ClusterId and Process as arguments
            "output":  repo_path / "htcondor_logs" / "output" / "$(ClusterId)_$(Process).txt",  # Standard output log file
            "error": repo_path / "htcondor_logs" / "error" / "$(ClusterId)_$(Process).txt",  # Standard error log file
            "log": repo_path / "htcondor_logs" / "log" / "$(ClusterId)_$(Process).txt",  # Condor log file
            "request_cpus": "1",  # Number of CPUs requested
            "request_memory": f"{memory}GB",  # Amount of memory requested
            "universe": "vanilla",  # Type of job (standard vanilla job)
            "getenv": "True",  # Inherit environment variables
        }
    )
    
    # Create a new HTCondor schedd (scheduler) object
    schedd = htcondor.Schedd()
    schedd.submit(submit)

    job_script_path = Path(__file__).parent / "tcr_data_collection_job.sh"

    print(f"{job_script_path} {env} {train_logs} {intervention_std} {' '.join(kwargs_str)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("train_logs", type=str)
    parser.add_argument("num_interventions", type=int)
    parser.add_argument("--job-length_h", type=float, default=1.0)
    parser.add_argument("kwargs", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.env == "tabletennis":
        intervention_std = 0.5
        cpus = 2 # TODO: Check if used
        memory = 10.0 # TODO
        time_per_intervention_h = 0.05
        args.kwargs.append("--max-episode-length 200")
    elif args.env == "Pendulum-v1":
        intervention_std = 1.4
        cpus = 1
        memory = 2.0 # TODO
        time_per_intervention_h = 0.02 # TODO
    else:
        raise ValueError(f"Unkown environment {args.env}.")
    interventions_per_job = math.floor(args.job_length_h / time_per_intervention_h)
    num_jobs = math.ceil(args.num_interventions / interventions_per_job)
    num_less_intervention_jobs = num_jobs * math.floor(interventions_per_job) - args.num_interventions
    jobs = [
        (interventions_per_job - 1, num_less_intervention_jobs),
        (interventions_per_job, num_jobs - num_less_intervention_jobs)
    ]
    assert sum([i * j for i, j in jobs]) == args.num_interventions

    print(f"Starting {len(jobs)} jobs.")

    submit_jobs(jobs, args.env, args.train_logs, intervention_std, cpus, memory, args.kwargs)
