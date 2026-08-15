#!/usr/bin/env bash

set -u
set -o pipefail

# ---------------- User-adjustable settings ----------------

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

# Evaluation subset.
# Note: in the Python code, start_batch=1 skips the first dataloader batch.
START_BATCH="${START_BATCH:-1}"
MAX_BATCH="${MAX_BATCH:-1}"
BATCH_SIZE_IP="${BATCH_SIZE_IP:-10}"

SEED="${SEED:-42}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MODEL="${MODEL:-ot}"
METHOD="${METHOD:-pnp_flow_admm}"

# Space-separated lists may be overridden from the shell.
read -r -a datasets <<< "${DATASETS:-afhq_cat}"    # afhq_cat

# Example:
# read -r -a problems <<< "${PROBLEMS:-inpainting random_inpainting superresolution hdr_reconstruction nonlinear_deblurring phase_retrieval}"
read -r -a problems <<< "${PROBLEMS:-phase_retrieval}"

# ---------------- Sweep values ----------------

# Example larger sweeps:
# read -r -a rhos   <<< "${RHOS:-100 500 1000 1500}"
# read -r -a gammas <<< "${GAMMAS:-0.2 1.0 1.8}"
# read -r -a steps  <<< "${STEPS:-200 250 300}"

read -r -a rhos   <<< "${RHOS:-1500}"
read -r -a gammas <<< "${GAMMAS:-1.9}"
read -r -a steps  <<< "${STEPS:-300}"
read -r -a num_denoiser_calls <<< "${NUM_DENOISER_CALLS:-5}"

# Only inverse-power schedule is tested.
read -r -a sch <<< "${SCH:-inverse_power}"
read -r -a beta_inverse <<< "${BETA_INVERSE:-2.2}"   # 0.01 1.0 1.5 1.3 2.0

LOG_ROOT="${LOG_ROOT:-sweep_logs/linear_admm}"
FAILED_FILE="${FAILED_FILE:-${LOG_ROOT}/failed_runs.txt}"

mkdir -p "$LOG_ROOT"

# ----------------------------------------------------------

# Validate that the script is being run from the repository root.
if [[ ! -f "main.py" ]]; then
    echo "ERROR: main.py was not found in the current directory."
    echo "Run this script from the repository root containing main.py."
    exit 1
fi

# Validate schedule names before starting.
for sch_value in "${sch[@]}"; do
    if [[ "$sch_value" != "inverse_power" ]]; then
        echo "ERROR: This script only supports rho_schedule_style=inverse_power."
        echo "Received: $sch_value"
        exit 1
    fi
done

# Total number of parameter combinations.
num_runs=$((
    ${#datasets[@]} *
    ${#problems[@]} *
    ${#rhos[@]} *
    ${#gammas[@]} *
    ${#steps[@]} *
    ${#num_denoiser_calls[@]} *
    ${#sch[@]} *
    ${#beta_inverse[@]}
))

run_id=0
failed=0

# Clear the old failed-runs list for this new sweep.
rm -f "$FAILED_FILE"

printf 'ADMM linear-task hyperparameter sweep\n'
printf 'GPU: %s\n' "$GPU"
printf 'Datasets: %s\n' "${datasets[*]}"
printf 'Problems: %s\n' "${problems[*]}"
printf 'Rhos: %s\n' "${rhos[*]}"
printf 'Gammas: %s\n' "${gammas[*]}"
printf 'Steps: %s\n' "${steps[*]}"
printf 'Num denoiser calls: %s\n' "${num_denoiser_calls[*]}"
printf 'Schedules: %s\n' "${sch[*]}"
printf 'Inverse-power beta: %s\n' "${beta_inverse[*]}"
printf 'Evaluation: start_batch=%s, max_batch=%s, batch_size_ip=%s\n' \
    "$START_BATCH" "$MAX_BATCH" "$BATCH_SIZE_IP"

printf 'Images per configuration: %d\n' \
    "$((MAX_BATCH * BATCH_SIZE_IP))"

printf 'Total runs: %d\n\n' "$num_runs"

# ---------------- Main sweep ----------------

for dataset in "${datasets[@]}"; do
    for problem in "${problems[@]}"; do
        for rho in "${rhos[@]}"; do
            for gamma in "${gammas[@]}"; do
                for nsteps in "${steps[@]}"; do
                    for ndenoiser in "${num_denoiser_calls[@]}"; do
                        for sch_value in "${sch[@]}"; do
                            for beta_value in "${beta_inverse[@]}"; do

                                run_id=$((run_id + 1))

                                tag="${dataset}_${problem}_rho${rho}_gamma${gamma}_steps${nsteps}_denoiser${ndenoiser}_beta${beta_value}_sch${sch_value}"

                                log_dir="${LOG_ROOT}/${dataset}/${problem}"
                                log_file="${log_dir}/${tag}.log"

                                mkdir -p "$log_dir"

                                printf '\n[%d/%d] RUN %s\n' \
                                    "$run_id" "$num_runs" "$tag"

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
                                    num_denoiser_calls "$ndenoiser" \
                                    rho_admm "$rho" \
                                    t_schedule_gamma "$gamma" \
                                    beta_rho "$beta_value" \
                                    rho_schedule_style "$sch_value" \
                                    adaptive_noise True \
                                    deterministic_noise False \
                                    save_norms_admm False \
                                    save_results True \
                                    early_stopping False \
                                    compute_time False \
                                    compute_memory False \
                                    2>&1 | tee "$log_file"

                                status=${PIPESTATUS[0]}

                                if [[ "$status" -ne 0 ]]; then
                                    printf 'FAILED (%d): %s\n' \
                                        "$status" "$tag" | tee -a "$FAILED_FILE"

                                    failed=$((failed + 1))
                                else
                                    printf 'DONE: %s\n' "$tag"
                                fi

                            done
                        done
                    done
                done
            done
        done
    done
done

# ---------------- Summary ----------------

printf '\nSweep finished. Total=%d, failed=%d\n' \
    "$num_runs" "$failed"

printf 'Logs: %s\n' "$LOG_ROOT"

if [[ "$failed" -ne 0 ]]; then
    exit 1
fi