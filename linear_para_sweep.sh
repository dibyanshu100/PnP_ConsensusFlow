#!/usr/bin/env bash
# Full-factorial hyperparameter sweep for PnP-Flow ADMM on linear inverse tasks.
#
# Run this script from the repository root containing main.py.
#
# Examples:
#   chmod +x linear_admm_hparam_sweep_corrected.sh
#   ./linear_admm_hparam_sweep_corrected.sh
#
#   DATASETS="celeba" ./linear_admm_hparam_sweep_corrected.sh
#   GPU=1 MAX_BATCH=10 BATCH_SIZE_IP=4 ./linear_admm_hparam_sweep_corrected.sh
#   RESUME=0 ./linear_admm_hparam_sweep_corrected.sh

set -u
set -o pipefail

# ---------------- User-adjustable settings ----------------

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

# Evaluation subset.
# Note: in the Python code, start_batch=1 skips the first dataloader batch.
START_BATCH="${START_BATCH:-1}"
MAX_BATCH="${MAX_BATCH:-5}"
BATCH_SIZE_IP="${BATCH_SIZE_IP:-5}"

SEED="${SEED:-42}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MODEL="${MODEL:-ot}"
METHOD="${METHOD:-pnp_flow_admm}"

# Space-separated lists may be overridden from the shell.
read -r -a datasets <<< "${DATASETS:-celeba afhq_cat}"
read -r -a problems <<< "${PROBLEMS:-inpainting paintbrush_inpainting random_inpainting superresolution}"

# Sweep values.
read -r -a rhos   <<< "${RHOS:-200 500 1000 1500}"
read -r -a gammas <<< "${GAMMAS:-0.2 1.0 1.8}"
read -r -a steps  <<< "${STEPS:-200 250 300}"

# RESUME=1 skips runs with an existing completion marker.
RESUME="${RESUME:-1}"
LOG_ROOT="${LOG_ROOT:-sweep_logs/linear_admm}"
DONE_ROOT="${DONE_ROOT:-${LOG_ROOT}/completed}"
FAILED_FILE="${FAILED_FILE:-${LOG_ROOT}/failed_runs.txt}"

mkdir -p "$LOG_ROOT" "$DONE_ROOT"

# ----------------------------------------------------------

# Validate that the script is being run from the repository root.
if [[ ! -f "main.py" ]]; then
  echo "ERROR: main.py was not found in the current directory."
  echo "Run this script from the repository root containing main.py."
  exit 1
fi

num_runs=$((${#datasets[@]} * ${#problems[@]} * ${#rhos[@]} * ${#gammas[@]} * ${#steps[@]}))
run_id=0
failed=0
skipped=0

printf 'ADMM linear-task hyperparameter sweep\n'
printf 'GPU: %s\n' "$GPU"
printf 'Datasets: %s\n' "${datasets[*]}"
printf 'Problems: %s\n' "${problems[*]}"
printf 'Rhos: %s\n' "${rhos[*]}"
printf 'Gammas: %s\n' "${gammas[*]}"
printf 'Steps: %s\n' "${steps[*]}"
printf 'Evaluation: start_batch=%s, max_batch=%s, batch_size_ip=%s\n' \
  "$START_BATCH" "$MAX_BATCH" "$BATCH_SIZE_IP"
printf 'Images per configuration: %d\n' "$((MAX_BATCH * BATCH_SIZE_IP))"
printf 'Total runs: %d\n\n' "$num_runs"

for dataset in "${datasets[@]}"; do
  for problem in "${problems[@]}"; do
    for rho in "${rhos[@]}"; do
      for gamma in "${gammas[@]}"; do
        for nsteps in "${steps[@]}"; do
          run_id=$((run_id + 1))

          tag="${dataset}_${problem}_rho${rho}_gamma${gamma}_steps${nsteps}"
          log_dir="${LOG_ROOT}/${dataset}/${problem}"
          log_file="${log_dir}/${tag}.log"
          done_file="${DONE_ROOT}/${tag}.done"

          mkdir -p "$log_dir"

          if [[ "$RESUME" == "1" && -f "$done_file" ]]; then
            printf '[%d/%d] SKIP %s (already completed)\n' \
              "$run_id" "$num_runs" "$tag"
            skipped=$((skipped + 1))
            continue
          fi

          printf '\n[%d/%d] RUN %s\n' "$run_id" "$num_runs" "$tag"

          python main.py --opts \
            dataset "$dataset" \
            model "$MODEL" \
            train False \
            eval True \
            eval_split "$EVAL_SPLIT" \
            method "$METHOD" \
            problem "$problem" \
            noise_type gaussian \
            seed "$SEED" \
            start_batch "$START_BATCH" \
            max_batch "$MAX_BATCH" \
            batch_size_ip "$BATCH_SIZE_IP" \
            steps_pnp "$nsteps" \
            rho_admm "$rho" \
            t_schedule_gamma "$gamma" \
            adaptive_noise True \
            deterministic_noise False \
            save_norms_admm False \
            save_results True \
            compute_time False \
            compute_memory False \
            2>&1 | tee "$log_file"

          status=${PIPESTATUS[0]}

          if [[ "$status" -ne 0 ]]; then
            printf 'FAILED (%d): %s\n' "$status" "$tag" | tee -a "$FAILED_FILE"
            failed=$((failed + 1))
            rm -f "$done_file"
          else
            touch "$done_file"
            printf 'DONE: %s\n' "$tag"
          fi
        done
      done
    done
  done
done

printf '\nSweep finished. Total=%d, skipped=%d, failed=%d\n' \
  "$num_runs" "$skipped" "$failed"
printf 'Logs: %s\n' "$LOG_ROOT"
printf 'Completion markers: %s\n' "$DONE_ROOT"

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi