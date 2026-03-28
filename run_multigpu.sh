#!/bin/bash
export LC_ALL=C
export LANG=C
export LANGUAGE=C

get_gpu_utilization() {
    local gpu_id=$1
    if ! command -v nvidia-smi &> /dev/null; then echo "0"; return; fi
    local u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$u" ]; then echo "$u"; else echo "0"; fi
}

get_available_gpus() {
    local g=()
    if ! command -v nvidia-smi &> /dev/null; then echo "0"; return; fi
    local n=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then echo "0"; return; fi
    for ((i=0;i<n;i++)); do local u=$(get_gpu_utilization $i); if [ "$u" -lt 80 ]; then g+=($i); fi; done
    if [ ${#g[@]} -eq 0 ]; then echo "0"; else echo "${g[@]}"; fi
}

wait_for_available_gpus() {
    local m=${1:-1}
    while true; do local a=($(get_available_gpus)); if [ ${#a[@]} -ge $m ]; then echo "${a[@]}"; return 0; fi; echo "⏳ Waiting for available GPUs (current: ${#a[@]}, need: $m)..."; sleep 30; done
}

AGENT_PATH="train.py"
RUN_GROUP="antmaze0328-new"
SEEDS=(0 1 2 3 4 5 6 7)
ENV_NAMES=("antmaze-umaze-v2" "antmaze-umaze-diverse-v2" "antmaze-medium-play-v2" "antmaze-medium-diverse-v2" "antmaze-large-play-v2" "antmaze-large-diverse-v2") 
ALGO="crossq"
WANDB_MODE="online"
WANDB_PROJECT="crossq-ant"
OFFLINE_STEPS=1000000
ONLINE_STEPS=1000000

if [ $# -ge 1 ]; then LOG_DIR="$1"; else LOG_DIR="logs_crossq"; fi
mkdir -p ${LOG_DIR}
WANDB_DIR="$(date +%m%d%H%M)_crossq"
mkdir -p ${WANDB_DIR}
WANDB_SAVE_DIR=${WANDB_DIR}

generate_command() {
  local env_name=$1; local seed=$2
  local log_name="${LOG_DIR}/${ALGO}_${env_name}_sd${seed}.log"
  echo "python ${AGENT_PATH} \
-algo ${ALGO} \
-env ${env_name} \
-seed ${seed} \
-offline_timesteps ${OFFLINE_STEPS} \
-total_timesteps ${ONLINE_STEPS} \
-wandb_mode ${WANDB_MODE} \
-wandb_project ${WANDB_PROJECT} \
-wandb_group ${RUN_GROUP} \
> ${log_name} 2>&1"
}

COMMAND_LIST=()
for env in "${ENV_NAMES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    COMMAND_LIST+=("$(generate_command "$env" "$seed")")
  done
done

echo "✅ Generated ${#COMMAND_LIST[@]} experiment commands"
AVAILABLE_GPUS=($(wait_for_available_gpus 1))
GPU_COUNT=${#AVAILABLE_GPUS[@]}
echo "✅ Found ${GPU_COUNT} available GPUs: ${AVAILABLE_GPUS[*]}"

declare -a GPU_COMMANDS
for ((i=0;i<GPU_COUNT;i++)); do declare -a "GPU_COMMANDS$i=()"; done
index=0
for cmd in "${COMMAND_LIST[@]}"; do gpu_index=$((index % GPU_COUNT)); eval "GPU_COMMANDS$gpu_index+=(\"$cmd\")"; index=$((index + 1)); done

for ((i=0;i<GPU_COUNT;i++)); do
  gpu_id=${AVAILABLE_GPUS[$i]}
  echo "🚀 Starting experiments on GPU $gpu_id..."
  (
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export LC_ALL=C
    export LANG=C
    export LANGUAGE=C
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    if command -v parallel &> /dev/null; then
      eval "printf '%s\\n' \"\${GPU_COMMANDS$i[@]}\"" | parallel -j4 --lb --tag 2>/dev/null
    else
      eval "commands=(\"\${GPU_COMMANDS$i[@]}\")"; for cmd in "${commands[@]}"; do echo "Executing: $cmd"; eval "$cmd"; done
    fi
  ) &
  if [ $i -lt $((GPU_COUNT - 1)) ]; then echo "⏳ GPU $gpu_id started, waiting 10 seconds before starting next GPU..."; sleep 10; fi
done

echo "⏳ Waiting for all GPU processes to complete..."; wait
echo "🎉 All experiments completed, logs saved in ${LOG_DIR}/ directory"
