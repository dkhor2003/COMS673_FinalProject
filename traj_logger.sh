#!/bin/bash

# Number of times to run the command
NUM_TIMES=200

# Command to run
COMMAND="python deploy_mujoco.py g1.yaml"

POLICY_ITER=("policy_lstm_1_500.pt"
             "policy_lstm_1_2200.pt"
             "policy_lstm_1_2500.pt"
             "policy_lstm_1_5000.pt"
             "policy_lstm_1_7000.pt"
             "policy_lstm_1_7500.pt"
             "policy_lstm_1_10000.pt"
             "policy_lstm_1_20000.pt")


yq -i ".log_on = true" g1.yaml
yq -i ".render = true" g1.yaml

# Loop to run the command
for policy in "${POLICY_ITER[@]}"; do
    yq -i ".policy = \"$policy\"" g1.yaml
    for ((i=1; i<=NUM_TIMES; i++)); do
        echo "Collecting Trajectory for $policy under $run (Iteration $i)..."
        timeout 20s $COMMAND || echo "Python script timed out due to OpenGL errors."
    done
done