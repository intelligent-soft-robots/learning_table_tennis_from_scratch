#!/bin/bash


pids=()

source /home/jschneider/isr_workspace/workspace/install/setup.bash
for (( i=0; i<$1; i++ ))
do
  launch_pam_mujoco "simulation$i" & pids+=($!)
  launch_pam_mujoco "pseudo-real$i" & pids+=($!)
  python ../bin/tcr_data_collection.py ../logs/master_return_default_hyperparameters_3M --outdir ../out/test --num-episodes-per-intervention 100  --num-interventions $2 --job-id "$i" & pids+=($!)
done

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