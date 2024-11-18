#!/bin/bash

# Arguments: job_id, env, train_logs, intervention_std, **kwargs


source /home/jschneider/isr_workspace/workspace/install/setup.bash

echo $1 $2 $3 $4 $5 $6

pids=()

if [ "$2" == "tabletennis" ]; then
    launch_pam_mujoco "simulation_$1" & pids+=($!)
    launch_pam_mujoco "pseudo-real_$1" & pids+=($!)
fi
dataset_name=$(echo "$2" | tr 'A-Z' 'a-z' | tr '-' '_')_std$4 # TODO: The file name should not contain a dot
echo "$dataset_name"
python ../bin/tcr_data_collection.py $2 $3 $4 --outdir /scratch/tcr_datasets/$dataset_name --job-id "$1" "${@:5:99}" & pids+=($!)

# monitor processes until one of them terminates or Ctrl+C is pressed
shutdown_requested=0

function sigint_handler()
{
    echo "Initiate shutdown"
    shutdown_requested=1
}
trap sigint_handler SIGINT

echo "Start monitoring processes ${pids[@]}"
while [ ${shutdown_requested} == 0 ]; do
    for pid in "${pids[@]}"; do
        if ! ps -p ${pid} > /dev/null; then
            >&2 echo "Process ${pid} has died! Terminate other processes..."
            break 2
        fi
    done

    sleep 1
done

# Reaching here means that one of the monitored processes died or shutdown has
# been requested (e.g. via Ctrl+C).  Kill the rest and exit.
echo "processes to kill: ${pids[@]}"
echo "${pids[@]}"
for pid in "${pids[@]}"; do
    if ps -p ${pid} > /dev/null; then
        echo "Kill process ${pid}"
        kill -SIGTERM ${pid} || true
    else
        echo "Process ${pid} already terminated"
    fi
done