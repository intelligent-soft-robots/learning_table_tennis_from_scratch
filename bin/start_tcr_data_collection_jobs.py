import argparse
import math
from pathlib import Path
from typing import Sequence, Tuple


import htcondor
import classad


def submit_jobs(jobs: Sequence[Tuple[int, int]], env: str, train_logs: str, dataset_name: str, intervention_std: float, cpus: int, memory: float, kwargs_str: str) -> None:
    # Initialize the HTCondor submit object
    repo_path = Path(__file__).parents[1]
    job_script_path = Path(__file__).parent / "tcr_data_collection_job.sh"

    for num_interventions, num_jobs in jobs:
        if num_interventions > 0:
            submit = htcondor.Submit(
                {
                    "executable": repo_path / "bin" / "tcr_data_collection_job.sh",  # The Python script to run
                    "arguments": f"$(ClusterId)-$(Process) {env} {train_logs} {dataset_name} {intervention_std} {num_interventions} {kwargs_str}",  # Pass ClusterId and Process as arguments
                    "output":  repo_path / "htcondor_logs" / "output" / "$(ClusterId)_$(Process).txt",  # Standard output log file
                    "error": repo_path / "htcondor_logs" / "error" / "$(ClusterId)_$(Process).txt",  # Standard error log file
                    "log": repo_path / "htcondor_logs" / "log" / "$(ClusterId)_$(Process).txt",  # Condor log file
                    "request_cpus": str(cpus),  # Number of CPUs requested
                    "request_memory": f"{memory}GB",  # Amount of memory requested
                    "universe": "vanilla",  # Type of job (standard vanilla job)
                    "getenv": "True",  # Inherit environment variables
                }
            )

            # Create a new HTCondor schedd (scheduler) object
            schedd = htcondor.Schedd()
            schedd.submit(submit, count=num_jobs)
        else:
            print("Skipping jobs with count 0.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("train_logs", type=str)
    parser.add_argument("num_interventions", type=int)
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--job-length_h", type=float, default=2.0)
    parser.add_argument("kwargs", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.env == "tabletennis":
        intervention_std = 0.5
        cpus = 2 # TODO: Check if used
        memory = 15.0 # TODO
        time_per_intervention_min = 0.25
        args.kwargs.append("--max-episode-length 200")  # TODO: This should only happen if --max-episode-length is not already in kwargs
    elif args.env == "Pendulum-v1":
        intervention_std = 1.4
        cpus = 1
        memory = 2.0 # TODO
        time_per_intervention_min = 0.1 # TODO
    else:
        raise ValueError(f"Unkown environment {args.env}.")
    time_per_intervention_h = time_per_intervention_min / 60
    max_interventions_per_job = math.floor(args.job_length_h / time_per_intervention_h)
    if max_interventions_per_job < args.num_interventions:
        num_jobs = math.ceil(args.num_interventions / max_interventions_per_job)
        num_more_intervention_jobs = args.num_interventions % num_jobs
        jobs = []
        if num_more_intervention_jobs != 0:
            jobs.append((math.ceil(args.num_interventions / num_jobs), num_more_intervention_jobs))
        jobs.append((math.floor(args.num_interventions / num_jobs), num_jobs - num_more_intervention_jobs))
    else:
        jobs = [(args.num_interventions, 1)]
    assert sum([i * j for i, j in jobs]) == args.num_interventions


    total_num_jobs = sum([j[1] for j in jobs])
    jobs_str = [f"{num_jobs} jobs with {num_interventions} interventions" for num_interventions, num_jobs in jobs]
    print(f"Starting {', '.join(jobs_str)}.")

    kwargs_str = " ".join(args.kwargs)
    submit_jobs(jobs, args.env, args.train_logs, args.dataset_name, intervention_std, cpus, memory, kwargs_str)
