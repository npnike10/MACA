#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=21
#SBATCH --gpus-per-task=1
#SBATCH --mem=40G
#SBATCH --account=huytran1-ic # huytran1-ic huytran1-ae-eng
#SBATCH --job-name="maca-lbf-final"
#SBATCH --partition=IllinoisComputes-GPU # IllinoisComputes-GPU eng-research-gpu
#SBATCH --output=train_sbatch_outputs/maca-lbf-final.o%A_%a

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/train_sbatch_outputs}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "PROJECT_DIR does not exist: ${PROJECT_DIR}"
  exit 1
fi

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_ROOT}"

module load anaconda3/2024.10
source /sw/apps/anaconda3/2024.10/bin/activate /projects/illinois/eng/aero/huytran1/niketnp2/conda/harl

export WANDB_API_KEY=
wandb login

ALGS=(
  mappo_t
  happo
)

TASKS=(
  2s-9x9-3p-2f-coop
  2s-11x11-3p-2f-coop
  4s-11x11-3p-2f-coop
  7s-20x20-5p-3f
  20x20-5p-5f
  20x20-3p-5f
  7s-20x20-5p-3f-coop
  8s-25x25-8p-5f
  7s-30x30-7p-4f
)

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
REPLICATES="${REPLICATES:-10}"
CAMPAIGN_SEED="${CAMPAIGN_SEED:-123456789}"
RUNS_PER_JOB="${RUNS_PER_JOB:-1}"

if ! [[ "$ARRAY_TASK_ID" =~ ^[0-9]+$ && "$REPLICATES" =~ ^[1-9][0-9]*$ && "$CAMPAIGN_SEED" =~ ^[0-9]+$ && "$RUNS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be non-negative; REPLICATES and RUNS_PER_JOB must be >=1; CAMPAIGN_SEED must be non-negative."
  exit 1
fi

N_ALGS="${#ALGS[@]}"
N_TASKS="${#TASKS[@]}"
RUNS_PER_REPLICATE="$((N_ALGS * N_TASKS))"
TOTAL_COMBINATIONS="$((RUNS_PER_REPLICATE * REPLICATES))"
JOB_COUNT="$(((TOTAL_COMBINATIONS + RUNS_PER_JOB - 1) / RUNS_PER_JOB))"

if (( ARRAY_TASK_ID < 0 || ARRAY_TASK_ID >= JOB_COUNT )); then
  echo "SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID} is out of range for this packing setup (0..$((JOB_COUNT - 1)))."
  exit 1
fi

JOB_START_INDEX="$((ARRAY_TASK_ID * RUNS_PER_JOB))"
JOB_END_INDEX="$((JOB_START_INDEX + RUNS_PER_JOB))"
if (( JOB_END_INDEX > TOTAL_COMBINATIONS )); then
  JOB_END_INDEX="${TOTAL_COMBINATIONS}"
fi

if command -v scontrol >/dev/null 2>&1; then
  packed_name="maca-lbf-pack-${JOB_START_INDEX}-$((JOB_END_INDEX - 1))"
  if ! scontrol update JobId="${SLURM_JOB_ID}" JobName="${packed_name}" >/dev/null; then
    echo "Warning: could not set packed job name to ${packed_name}"
  fi
fi

echo "Packed launch: array_task_id=${ARRAY_TASK_ID} runs_per_job=${RUNS_PER_JOB} run_index_range=[${JOB_START_INDEX},$((JOB_END_INDEX - 1))] total_runs=${TOTAL_COMBINATIONS}"
echo "campaign_seed=${CAMPAIGN_SEED} replicates=${REPLICATES}"

declare -a RUN_PIDS=()
declare -a RUN_TAGS=()

launch_run() {
  local global_index="$1"
  local replicate_idx within_replicate run_seed task_idx alg_idx
  local alg_name task_spec env_key run_tag run_output_dir run_output_file

  replicate_idx="$((global_index / RUNS_PER_REPLICATE))"
  within_replicate="$((global_index % RUNS_PER_REPLICATE))"
  run_seed="$(( (CAMPAIGN_SEED + (replicate_idx + 1) * 104729) % 2147483646 + 1 ))"

  task_idx="$((within_replicate / N_ALGS))"
  alg_idx="$((within_replicate % N_ALGS))"

  alg_name="${ALGS[$alg_idx]}"
  task_spec="${TASKS[$task_idx]}"
  env_key="lbforaging:Foraging-${task_spec}-v3"
  run_tag="${alg_name}-${task_spec}-seed${run_seed}"
  run_output_dir="${OUTPUT_ROOT}/${alg_name}/${task_spec}"
  run_output_file="${run_output_dir}/${run_tag}.o${SLURM_JOB_ID}"

  mkdir -p "${run_output_dir}"

  (
    exec > >(tee -a "${run_output_file}") 2>&1
    echo "Launching ${run_tag} global_index=${global_index}/${TOTAL_COMBINATIONS} replicate=$((replicate_idx + 1))/${REPLICATES} array_task_id=${ARRAY_TASK_ID}"
    echo "log_file=${run_output_file}"
    echo "algo=${alg_name} env=${env_key} seed=${run_seed} campaign_seed=${CAMPAIGN_SEED}"
    python3 -m examples.train \
      --algo="${alg_name}" \
      --env=gym \
      --exp_name="${run_tag}" \
      --scenario="${env_key}" \
      --episode_length=50 \
      --algo_args.seed.seed="${run_seed}"
  ) &

  RUN_PIDS+=("$!")
  RUN_TAGS+=("${run_tag}")
}

for ((global_index=JOB_START_INDEX; global_index<JOB_END_INDEX; global_index++)); do
  launch_run "${global_index}"
done

failed=0
for i in "${!RUN_PIDS[@]}"; do
  pid="${RUN_PIDS[$i]}"
  tag="${RUN_TAGS[$i]}"
  if ! wait "${pid}"; then
    echo "Run failed: ${tag} (pid=${pid})"
    failed=1
  else
    echo "Run finished: ${tag} (pid=${pid})"
  fi
done

if (( failed != 0 )); then
  echo "At least one packed run failed in this job."
  exit 1
fi
