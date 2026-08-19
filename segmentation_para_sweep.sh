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
MODEL="${MODEL:-ot}"          # segmentation's prior requires model=ot
METHOD="${METHOD:-pnp_flow_admm}"

# Segmentation requires dataset=celebahq, dim_image=128.
read -r -a datasets <<< "${DATASETS:-celebahq}"
read -r -a problems <<< "${PROBLEMS:-segmentation}"

# ---------------- Sweep values ----------------

# Shared ADMM schedule (same mechanism as the linear-task sweep).
read -r -a rhos   <<< "${RHOS:-500 1000}"       ##  1000 1500
read -r -a gammas <<< "${GAMMAS:-0.5 1.0 1.5}"       #  1.5 2.0
read -r -a steps  <<< "${STEPS:-200 500}"
read -r -a beta_inverse <<< "${BETA_INVERSE:-0.0 0.01 1.0}"          #0.01 0.1  1.5 2.0 2.5

# Segmentation-only knobs.
read -r -a seg_ce_weights   <<< "${SEG_CE_WEIGHTS:-3.0 5.0 10.0}"
read -r -a seg_x_lrs        <<< "${SEG_X_LRS:-1e-4}"
read -r -a seg_x_inner_steps <<< "${SEG_X_INNER_STEPS:-3 5 10}"

LOG_ROOT="${LOG_ROOT:-sweep_logs/segmentation_admm}"
FAILED_FILE="${FAILED_FILE:-${LOG_ROOT}/failed_runs.txt}"

mkdir -p "$LOG_ROOT"

# ----------------------------------------------------------

if [[ ! -f "main.py" ]]; then
    echo "ERROR: main.py was not found in the current directory."
    echo "Run this script from the repository root containing main.py."
    exit 1
fi

for problem in "${problems[@]}"; do
    if [[ "$problem" != "segmentation" ]]; then
        echo "ERROR: This script is for problem=segmentation only. Received: $problem"
        exit 1
    fi
done
for dataset in "${datasets[@]}"; do
    if [[ "$dataset" != "celebahq" ]]; then
        echo "ERROR: Segmentation requires dataset=celebahq. Received: $dataset"
        exit 1
    fi
done

num_runs=$((
    ${#datasets[@]} *
    ${#problems[@]} *
    ${#rhos[@]} *
    ${#gammas[@]} *
    ${#steps[@]} *
    ${#beta_inverse[@]} *
    ${#seg_ce_weights[@]} *
    ${#seg_x_lrs[@]} *
    ${#seg_x_inner_steps[@]}
))

run_id=0
failed=0

rm -f "$FAILED_FILE"

printf 'ADMM segmentation hyperparameter sweep\n'
printf 'GPU: %s\n' "$GPU"
printf 'Datasets: %s\n' "${datasets[*]}"
printf 'Rhos: %s\n' "${rhos[*]}"
printf 'Gammas: %s\n' "${gammas[*]}"
printf 'Steps: %s\n' "${steps[*]}"
printf 'Inverse-power beta: %s\n' "${beta_inverse[*]}"
printf 'CE weights: %s\n' "${seg_ce_weights[*]}"
printf 'X-update lrs: %s\n' "${seg_x_lrs[*]}"
printf 'X-update inner steps: %s\n' "${seg_x_inner_steps[*]}"
printf 'Evaluation: start_batch=%s, max_batch=%s, batch_size_ip=%s\n' \
    "$START_BATCH" "$MAX_BATCH" "$BATCH_SIZE_IP"
printf 'Images per configuration: %d\n' "$((MAX_BATCH * BATCH_SIZE_IP))"
printf 'Total runs: %d\n\n' "$num_runs"

# ---------------- Main sweep ----------------

for dataset in "${datasets[@]}"; do
    for problem in "${problems[@]}"; do
        for rho in "${rhos[@]}"; do
            for gamma in "${gammas[@]}"; do
                for nsteps in "${steps[@]}"; do
                    for beta_value in "${beta_inverse[@]}"; do
                        for ce_weight in "${seg_ce_weights[@]}"; do
                            for x_lr in "${seg_x_lrs[@]}"; do
                                for x_inner in "${seg_x_inner_steps[@]}"; do

                                    run_id=$((run_id + 1))

                                    tag="${dataset}_${problem}_rho${rho}_gamma${gamma}_steps${nsteps}_beta${beta_value}_ce${ce_weight}_xlr${x_lr}_xinner${x_inner}"

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
                                        rho_admm "$rho" \
                                        t_schedule_gamma "$gamma" \
                                        beta_rho "$beta_value" \
                                        seg_ce_weight "$ce_weight" \
                                        seg_x_lr "$x_lr" \
                                        seg_x_inner_steps "$x_inner" \
                                        save_norms_admm False \
                                        save_results True \
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
done

# ---------------- Summary ----------------

printf '\nSweep finished. Total=%d, failed=%d\n' "$num_runs" "$failed"
printf 'Logs: %s\n' "$LOG_ROOT"

if [[ "$failed" -ne 0 ]]; then
    exit 1
fi