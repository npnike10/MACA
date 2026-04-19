#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/train_marl_lbf_campaign.sbatch"
MODE="${1:-all}"
DRY_RUN="${DRY_RUN:-0}"
REPLICATES="${REPLICATES:-10}"
CAMPAIGN_SEED="${CAMPAIGN_SEED:-$(( (10#$(date +%s) ^ (RANDOM << 16) ^ RANDOM) & 2147483647 ))}"
RUNS_PER_JOB="${RUNS_PER_JOB:-1}"

ALGO_COUNT=2
TASK_COUNT=9
RUNS_PER_REPLICATE="$((ALGO_COUNT * TASK_COUNT))"

IC_ACCOUNT="huytran1-ic"
IC_PARTITION="IllinoisComputes-GPU"
IC_TIME="${IC_TIME:-24:00:00}"
IC_CONCURRENCY="${IC_CONCURRENCY:-4}"

ENG_ACCOUNT="huytran1-ae-eng"
ENG_PARTITION="eng-research-gpu"
ENG_TIME="${ENG_TIME:-24:00:00}"
ENG_CONCURRENCY="${ENG_CONCURRENCY:-12}"

# Current known hard caps:
# - IllinoisComputes-GPU is under gpu4 QoS => max 4 GPUs/user.
# - eng-research-gpu has 320 CPUs and this job needs 25 CPUs, so practical
#   hard cap is floor(320/25)=12 concurrent jobs.
IC_CONCURRENCY_HARD_CAP=4
ENG_CONCURRENCY_HARD_CAP=12

if [[ ! -f "${SBATCH_FILE}" ]]; then
  echo "Could not find ${SBATCH_FILE}"
  exit 1
fi

if ! [[ "${IC_CONCURRENCY}" =~ ^[0-9]+$ && "${ENG_CONCURRENCY}" =~ ^[0-9]+$ && "${DRY_RUN}" =~ ^[0-9]+$ && "${REPLICATES}" =~ ^[1-9][0-9]*$ && "${CAMPAIGN_SEED}" =~ ^[0-9]+$ && "${RUNS_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "IC_CONCURRENCY, ENG_CONCURRENCY, DRY_RUN, REPLICATES, CAMPAIGN_SEED, and RUNS_PER_JOB must be valid integers."
  exit 1
fi

TOTAL_RUNS="$((RUNS_PER_REPLICATE * REPLICATES))"
JOB_COUNT="$(((TOTAL_RUNS + RUNS_PER_JOB - 1) / RUNS_PER_JOB))"

if (( IC_CONCURRENCY < 1 || ENG_CONCURRENCY < 1 )); then
  echo "IC_CONCURRENCY and ENG_CONCURRENCY must be >= 1."
  exit 1
fi

if (( IC_CONCURRENCY > IC_CONCURRENCY_HARD_CAP )); then
  echo "IC_CONCURRENCY=${IC_CONCURRENCY} exceeds cap ${IC_CONCURRENCY_HARD_CAP}; clamping."
  IC_CONCURRENCY="${IC_CONCURRENCY_HARD_CAP}"
fi

if (( ENG_CONCURRENCY > ENG_CONCURRENCY_HARD_CAP )); then
  echo "ENG_CONCURRENCY=${ENG_CONCURRENCY} exceeds cap ${ENG_CONCURRENCY_HARD_CAP}; clamping."
  ENG_CONCURRENCY="${ENG_CONCURRENCY_HARD_CAP}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/train_sbatch_outputs}"
mkdir -p "${OUTPUT_ROOT}"

submit_array() {
  local account="$1"
  local partition="$2"
  local walltime="$3"
  local array_spec="$4"
  local export_vars="ALL,REPLICATES=${REPLICATES},CAMPAIGN_SEED=${CAMPAIGN_SEED},RUNS_PER_JOB=${RUNS_PER_JOB},PROJECT_DIR=${SCRIPT_DIR},OUTPUT_ROOT=${OUTPUT_ROOT}"

  if (( DRY_RUN == 1 )); then
    sbatch --test-only --account="${account}" --partition="${partition}" --time="${walltime}" --array="${array_spec}" --output="${OUTPUT_ROOT}/maca-lbf-final.o%A_%a" --export="${export_vars}" "${SBATCH_FILE}"
  else
    sbatch --parsable --account="${account}" --partition="${partition}" --time="${walltime}" --array="${array_spec}" --output="${OUTPUT_ROOT}/maca-lbf-final.o%A_%a" --export="${export_vars}" "${SBATCH_FILE}"
  fi
}

ids_to_array_spec() {
  local concurrency="$1"
  shift
  local ids=("$@")
  local csv
  csv="$(IFS=,; echo "${ids[*]}")"
  echo "${csv}%${concurrency}"
}

case "${MODE}" in
  all)
    IC_IDS=()
    ENG_IDS=()

    if (( JOB_COUNT == 1 )); then
      if (( IC_CONCURRENCY >= ENG_CONCURRENCY )); then
        IC_IDS+=(0)
      else
        ENG_IDS+=(0)
      fi
    else
      total_concurrency="$((IC_CONCURRENCY + ENG_CONCURRENCY))"
      ic_target_jobs="$(( (JOB_COUNT * IC_CONCURRENCY + total_concurrency / 2) / total_concurrency ))"

      if (( ic_target_jobs < 1 )); then
        ic_target_jobs=1
      fi
      if (( ic_target_jobs > JOB_COUNT - 1 )); then
        ic_target_jobs="$((JOB_COUNT - 1))"
      fi

      prev_bucket=0
      for ((i=0; i<JOB_COUNT; i++)); do
        bucket="$(( ((i + 1) * ic_target_jobs) / JOB_COUNT ))"
        if (( bucket > prev_bucket )); then
          IC_IDS+=("${i}")
          prev_bucket="${bucket}"
        else
          ENG_IDS+=("${i}")
        fi
      done
    fi

    ic_result="not submitted"
    if (( ${#IC_IDS[@]} > 0 )); then
      ic_array_spec="$(ids_to_array_spec "${IC_CONCURRENCY}" "${IC_IDS[@]}")"
      ic_result="$(submit_array "${IC_ACCOUNT}" "${IC_PARTITION}" "${IC_TIME}" "${ic_array_spec}")"
    fi

    eng_result="not submitted"
    if (( ${#ENG_IDS[@]} > 0 )); then
      eng_array_spec="$(ids_to_array_spec "${ENG_CONCURRENCY}" "${ENG_IDS[@]}")"
      eng_result="$(submit_array "${ENG_ACCOUNT}" "${ENG_PARTITION}" "${ENG_TIME}" "${eng_array_spec}")"
    fi

    echo "Submitted split campaign."
    echo "  replicates=${REPLICATES}, campaign_seed=${CAMPAIGN_SEED}, runs_per_job=${RUNS_PER_JOB}"
    echo "  total_runs=${TOTAL_RUNS}, total_jobs=${JOB_COUNT}"
    echo "  ${IC_PARTITION}:  ${#IC_IDS[@]} jobs, concurrency=${IC_CONCURRENCY}, result=${ic_result}"
    echo "  ${ENG_PARTITION}: ${#ENG_IDS[@]} jobs, concurrency=${ENG_CONCURRENCY}, result=${eng_result}"
    ;;
  ic-only)
    array_spec="0-$((JOB_COUNT - 1))%${IC_CONCURRENCY}"
    result="$(submit_array "${IC_ACCOUNT}" "${IC_PARTITION}" "${IC_TIME}" "${array_spec}")"
    echo "Submitted full campaign to ${IC_PARTITION}: ${result} (array ${array_spec}, replicates=${REPLICATES}, campaign_seed=${CAMPAIGN_SEED}, runs_per_job=${RUNS_PER_JOB})"
    ;;
  eng-only)
    array_spec="0-$((JOB_COUNT - 1))%${ENG_CONCURRENCY}"
    result="$(submit_array "${ENG_ACCOUNT}" "${ENG_PARTITION}" "${ENG_TIME}" "${array_spec}")"
    echo "Submitted full campaign to ${ENG_PARTITION}: ${result} (array ${array_spec}, replicates=${REPLICATES}, campaign_seed=${CAMPAIGN_SEED}, runs_per_job=${RUNS_PER_JOB})"
    ;;
  first5)
    if (( RUNS_PER_JOB != 1 )); then
      echo "first5 mode requires RUNS_PER_JOB=1 to avoid boundary overlap."
      exit 1
    fi
    first_rep_count=$((REPLICATES < 5 ? REPLICATES : 5))
    array_spec="0-$((RUNS_PER_REPLICATE * first_rep_count - 1))%${IC_CONCURRENCY}"
    result="$(submit_array "${IC_ACCOUNT}" "${IC_PARTITION}" "${IC_TIME}" "${array_spec}")"
    echo "Submitted first ${first_rep_count} replicates to ${IC_PARTITION}: ${result} (array ${array_spec}, campaign_seed=${CAMPAIGN_SEED})"
    ;;
  second5)
    if (( RUNS_PER_JOB != 1 )); then
      echo "second5 mode requires RUNS_PER_JOB=1 to avoid boundary overlap."
      exit 1
    fi
    if (( REPLICATES <= 5 )); then
      echo "No second-5 segment available because REPLICATES=${REPLICATES}."
      exit 1
    fi
    second_rep_end=$((REPLICATES < 10 ? REPLICATES : 10))
    array_start="$((RUNS_PER_REPLICATE * 5))"
    array_end="$((RUNS_PER_REPLICATE * second_rep_end - 1))"
    array_spec="${array_start}-${array_end}%${IC_CONCURRENCY}"
    result="$(submit_array "${IC_ACCOUNT}" "${IC_PARTITION}" "${IC_TIME}" "${array_spec}")"
    echo "Submitted replicates 6-${second_rep_end} to ${IC_PARTITION}: ${result} (array ${array_spec}, campaign_seed=${CAMPAIGN_SEED})"
    ;;
  *)
    echo "Usage: $0 [all|ic-only|eng-only|first5|second5]"
    echo "Optional env overrides: REPLICATES, CAMPAIGN_SEED, RUNS_PER_JOB, IC_CONCURRENCY, ENG_CONCURRENCY, IC_TIME, ENG_TIME, DRY_RUN=1"
    exit 1
    ;;
esac
