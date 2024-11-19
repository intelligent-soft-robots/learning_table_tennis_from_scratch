#!/bin/bash

# Arguments: job_id, env, train_logs, intervention_std, num_interventions **kwargs
job_id=$1
env=$2
train_logs=$3
dataset_name=$4
intervention_std=$5
num_interventions=$6

source $HOME/isr_workspace/workspace/install/setup.bash

CONTAINER_PATH="${BASH_SOURCE[0]}"/../../../../"learning_table_tennis_from_scratch.sif"

pids=()

if [ "$2" == "tabletennis" ]; then
    apptainer exec $CONTAINER_PATH bash -c "source $HOME/isr_workspace/workspace/install/setup.bash && launch_pam_mujoco simulation_$job_id" & pids+=($!)
    apptainer exec $CONTAINER_PATH bash -c "source $HOME/isr_workspace/workspace/install/setup.bash && launch_pam_mujoco pseudo-real_$job_id" & pids+=($!)
fi
tcr_command="source $HOME/isr_workspace/workspace/install/setup.bash && python ../bin/tcr_data_collection.py $env $train_logs $intervention_std --num-interventions $num_interventions --outdir /tcr_datasets/$dataset_name --job-id $job_id ${@:7:99}"
echo $tcr_command
apptainer exec -B /fast/jschneider/tcr_datasets:/tcr_datasets $CONTAINER_PATH bash -c "$tcr_command" & pids+=($!)

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