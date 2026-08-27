# #!/usr/bin/env bash

# set -u
# set -o pipefail

# # ============================================================
# # Hyperparameter sweep for NONLINEAR tasks only:
# #
# #   1. PnP-Flow / PnP-FBS: sweep ONLY alpha
# #   2. FlowDPS:            sweep ONLY grad_step_size
# #
# # Datasets:
# #   - celeba
# #   - afhq_cat
# #
# # Nonlinear tasks from the reference script:
# #   - phase_retrieval
# #   - hdr_reconstruction
# #   - nonlinear_deblurring
# #
# # Evaluation is restricted to exactly 2 batches.
# # All other method settings are kept fixed.
# # ============================================================

# GPU="${GPU:-0}"
# export CUDA_VISIBLE_DEVICES="$GPU"

# START_BATCH="${START_BATCH:-0}"
# MAX_BATCH=2
# BATCH_SIZE_IP="${BATCH_SIZE_IP:-4}"

# SEED="${SEED:-42}"
# EVAL_SPLIT="${EVAL_SPLIT:-test}"

# LOG_ROOT="${LOG_ROOT:-sweep_logs/nonlinear_pnp_fbs_flow_dps}"
# mkdir -p "$LOG_ROOT"

# # ------------------------------------------------------------
# # Hyperparameter grids
# # ------------------------------------------------------------

# # PnP-FBS: sweep ONLY alpha.
# ALPHAS=(
#   0.01
#   0.05
#   0.1
#   0.3
#   0.5
#   0.8
#   1.0
# )

# # FlowDPS: sweep ONLY grad_step_size.
# GRAD_STEP_SIZES=(
#   0.01
#   0.05
#   0.1
#   0.25
#   0.5
#   1.0
#   2.0
# )

# DATASETS=(
#   celeba
#   afhq_cat
# )

# PROBLEMS=(
#   phase_retrieval
#   hdr_reconstruction
#   nonlinear_deblurring
# )

# TOTAL=$(( ${#DATASETS[@]} * ${#PROBLEMS[@]} * ( ${#ALPHAS[@]} + ${#GRAD_STEP_SIZES[@]} ) ))

# RUN_NUM=0
# FAILED=0
# FAILED_LABELS=()

# # ------------------------------------------------------------
# # Helper: run one experiment and save its terminal output.
# # ------------------------------------------------------------
# run() {
#   local label="$1"
#   shift

#   RUN_NUM=$((RUN_NUM + 1))
#   local remaining=$((TOTAL - RUN_NUM))

#   local safe_label
#   safe_label=$(printf '%s' "$label" | tr ' /[]()=' '_________')
#   local log_file="${LOG_ROOT}/${safe_label}.log"

#   printf '\n============================================================\n'
#   printf '[%d/%d] %s  (%d left after this)\n' \
#     "$RUN_NUM" "$TOTAL" "$label" "$remaining"
#   printf 'Log: %s\n' "$log_file"
#   printf '============================================================\n'

#   local start_ts elapsed status
#   start_ts=$(date +%s)

#   "$@" 2>&1 | tee "$log_file"
#   status=${PIPESTATUS[0]}

#   elapsed=$(( $(date +%s) - start_ts ))

#   if [[ "$status" -ne 0 ]]; then
#     printf '\n>>> FAILED (%d): %s  [%ds]\n' \
#       "$status" "$label" "$elapsed"
#     FAILED=$((FAILED + 1))
#     FAILED_LABELS+=("$label")
#   else
#     printf '\n>>> DONE: %s  [%ds]\n' "$label" "$elapsed"
#   fi
# }


# # ------------------------------------------------------------
# # Fixed PnP-FBS step counts from the reference script.
# # Only alpha is swept.
# # ------------------------------------------------------------
# get_fbs_steps() {
#   local dataset="$1"
#   local problem="$2"

#   case "${dataset}:${problem}" in

#     # ---------------- CelebA ----------------
#     celeba:phase_retrieval)
#       echo 100
#       ;;
#     celeba:hdr_reconstruction)
#       echo 100
#       ;;
#     celeba:nonlinear_deblurring)
#       echo 500
#       ;;

#     # ---------------- AFHQ-Cat ----------------
#     afhq_cat:phase_retrieval)
#       echo 300
#       ;;
#     afhq_cat:hdr_reconstruction)
#       echo 100
#       ;;
#     afhq_cat:nonlinear_deblurring)
#       echo 500
#       ;;

#     *)
#       printf 'ERROR: Unknown PnP-FBS dataset/problem: %s/%s\n' \
#         "$dataset" "$problem" >&2
#       return 1
#       ;;
#   esac
# }


# # ------------------------------------------------------------
# # Fixed FlowDPS step counts from the reference script.
# # Only grad_step_size is swept.
# # ------------------------------------------------------------
# get_flow_dps_steps() {
#   local dataset="$1"
#   local problem="$2"

#   case "${dataset}:${problem}" in

#     # ---------------- CelebA ----------------
#     celeba:phase_retrieval)
#       echo 100
#       ;;
#     celeba:hdr_reconstruction)
#       echo 100
#       ;;
#     celeba:nonlinear_deblurring)
#       echo 500
#       ;;

#     # ---------------- AFHQ-Cat ----------------
#     afhq_cat:phase_retrieval)
#       echo 300
#       ;;
#     afhq_cat:hdr_reconstruction)
#       echo 100
#       ;;
#     afhq_cat:nonlinear_deblurring)
#       echo 500
#       ;;

#     *)
#       printf 'ERROR: Unknown FlowDPS dataset/problem: %s/%s\n' \
#         "$dataset" "$problem" >&2
#       return 1
#       ;;
#   esac
# }


# # ============================================================
# # Sweep
# # ============================================================

# for dataset in "${DATASETS[@]}"; do

#   printf '\n\n############################################################\n'
#   printf 'DATASET: %s\n' "$dataset"
#   printf '############################################################\n'

#   for problem in "${PROBLEMS[@]}"; do

#     printf '\n------------------------------------------------------------\n'
#     printf 'TASK: %s / %s\n' "$dataset" "$problem"
#     printf '------------------------------------------------------------\n'

#     fbs_steps=$(get_fbs_steps "$dataset" "$problem") || exit 1
#     flow_dps_steps=$(get_flow_dps_steps "$dataset" "$problem") || exit 1


#     # ========================================================
#     # PnP-Flow / PnP-FBS
#     # Sweep ONLY alpha.
#     #
#     # Fixed:
#     #   lr_pnp      = 1.0
#     #   num_samples = 5
#     # ========================================================

#     for alpha in "${ALPHAS[@]}"; do

#       run "${dataset} / ${problem} / pnp_flow_fbs / alpha=${alpha}" \
#         python main.py --opts \
#           method pnp_flow_fbs \
#           dataset "$dataset" \
#           problem "$problem" \
#           steps_pnp "$fbs_steps" \
#           lr_pnp 1.0 \
#           alpha "$alpha" \
#           num_samples 5 \
#           model ot \
#           train False \
#           eval True \
#           eval_split "$EVAL_SPLIT" \
#           noise_type gaussian \
#           seed "$SEED" \
#           start_batch "$START_BATCH" \
#           max_batch "$MAX_BATCH" \
#           batch_size_ip "$BATCH_SIZE_IP" \
#           compute_time True \
#           compute_memory True \
#           save_results True

#     done


#     # ========================================================
#     # FlowDPS
#     # Sweep ONLY grad_step_size.
#     #
#     # Fixed:
#     #   grad_steps           = 3
#     #   use_stochastic_noise = True
#     # ========================================================

#     for grad_step_size in "${GRAD_STEP_SIZES[@]}"; do

#       run "${dataset} / ${problem} / flow_dps / grad_step_size=${grad_step_size}" \
#         python main.py --opts \
#           method flow_dps \
#           dataset "$dataset" \
#           problem "$problem" \
#           steps_ode "$flow_dps_steps" \
#           grad_steps 3 \
#           grad_step_size "$grad_step_size" \
#           use_stochastic_noise True \
#           model ot \
#           train False \
#           eval True \
#           eval_split "$EVAL_SPLIT" \
#           noise_type gaussian \
#           seed "$SEED" \
#           start_batch "$START_BATCH" \
#           max_batch "$MAX_BATCH" \
#           batch_size_ip "$BATCH_SIZE_IP" \
#           compute_time True \
#           compute_memory True \
#           save_results True

#     done

#   done
# done


# # ============================================================
# # Final summary
# # ============================================================

# printf '\n============================================================\n'
# printf 'All nonlinear sweep runs finished.\n'
# printf 'Total runs:  %d\n' "$TOTAL"
# printf 'Failed runs: %d\n' "$FAILED"
# printf 'Logs saved:  %s\n' "$LOG_ROOT"
# printf '============================================================\n'

# if [[ "$FAILED" -gt 0 ]]; then
#   printf '\nFailed runs:\n'
#   for label in "${FAILED_LABELS[@]}"; do
#     printf '  - %s\n' "$label"
#   done
#   exit 1
# fi



#!/usr/bin/env bash

set -u
set -o pipefail

# ============================================================
# FlowDPS hyperparameter sweep for LINEAR tasks only
#
# Sweep ONLY:
#   - grad_step_size
#
# Datasets:
#   - celeba
#   - afhq_cat
#
# Linear tasks:
#   - superresolution
#   - random_inpainting
#   - inpainting
#
# Evaluation:
#   - exactly 2 batches
#
# Fixed FlowDPS settings:
#   - steps_ode = 100
#   - grad_steps = 3
#   - use_stochastic_noise = True
# ============================================================

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

START_BATCH="${START_BATCH:-0}"
MAX_BATCH=2
BATCH_SIZE_IP="${BATCH_SIZE_IP:-4}"

SEED="${SEED:-42}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

LOG_ROOT="${LOG_ROOT:-sweep_logs/linear_flow_dps}"
mkdir -p "$LOG_ROOT"

# ------------------------------------------------------------
# Sweep ONLY grad_step_size
# ------------------------------------------------------------
GRAD_STEP_SIZES=(
  2.5
  3.0
  4.0
)

DATASETS=(
  celeba
  afhq_cat
)

LINEAR_PROBLEMS=(
  superresolution
  random_inpainting
  inpainting
)

# 2 datasets × 3 tasks × 7 grad_step_size values = 42 runs
TOTAL=$(( ${#DATASETS[@]} * ${#LINEAR_PROBLEMS[@]} * ${#GRAD_STEP_SIZES[@]} ))

RUN_NUM=0
FAILED=0
FAILED_LABELS=()

# ------------------------------------------------------------
# Helper: run one experiment and save its terminal output.
# ------------------------------------------------------------
run() {
  local label="$1"
  shift

  RUN_NUM=$((RUN_NUM + 1))
  local remaining=$((TOTAL - RUN_NUM))

  local safe_label
  safe_label=$(printf '%s' "$label" | tr ' /[]()=' '_________')
  local log_file="${LOG_ROOT}/${safe_label}.log"

  printf '\n============================================================\n'
  printf '[%d/%d] %s  (%d left after this)\n' \
    "$RUN_NUM" "$TOTAL" "$label" "$remaining"
  printf 'Log: %s\n' "$log_file"
  printf '============================================================\n'

  local start_ts elapsed status
  start_ts=$(date +%s)

  "$@" 2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}

  elapsed=$(( $(date +%s) - start_ts ))

  if [[ "$status" -ne 0 ]]; then
    printf '\n>>> FAILED (%d): %s  [%ds]\n' \
      "$status" "$label" "$elapsed"
    FAILED=$((FAILED + 1))
    FAILED_LABELS+=("$label")
  else
    printf '\n>>> DONE: %s  [%ds]\n' "$label" "$elapsed"
  fi
}


# ============================================================
# ACTIVE SECTION:
# FlowDPS sweep on LINEAR inverse problems
#
# Fixed:
#   steps_ode            = 100
#   grad_steps           = 3
#   use_stochastic_noise = True
#
# Swept:
#   grad_step_size
# ============================================================

for dataset in "${DATASETS[@]}"; do

  printf '\n\n############################################################\n'
  printf 'DATASET: %s\n' "$dataset"
  printf '############################################################\n'

  for problem in "${LINEAR_PROBLEMS[@]}"; do

    printf '\n------------------------------------------------------------\n'
    printf 'TASK: %s / %s\n' "$dataset" "$problem"
    printf '------------------------------------------------------------\n'

    for grad_step_size in "${GRAD_STEP_SIZES[@]}"; do

      run "${dataset} / ${problem} / flow_dps / grad_step_size=${grad_step_size}" \
        python main.py --opts \
          method flow_dps \
          dataset "$dataset" \
          problem "$problem" \
          steps_ode 100 \
          grad_steps 3 \
          grad_step_size "$grad_step_size" \
          use_stochastic_noise True \
          model ot \
          train False \
          eval True \
          eval_split "$EVAL_SPLIT" \
          noise_type gaussian \
          seed "$SEED" \
          start_batch "$START_BATCH" \
          max_batch "$MAX_BATCH" \
          batch_size_ip "$BATCH_SIZE_IP" \
          compute_time True \
          compute_memory True \
          save_results True

    done

  done
done


# ============================================================
# Final summary
# ============================================================

printf '\n============================================================\n'
printf 'All LINEAR FlowDPS sweep runs finished.\n'
printf 'Total runs:  %d\n' "$TOTAL"
printf 'Failed runs: %d\n' "$FAILED"
printf 'Logs saved:  %s\n' "$LOG_ROOT"
printf '============================================================\n'

if [[ "$FAILED" -gt 0 ]]; then
  printf '\nFailed runs:\n'
  for label in "${FAILED_LABELS[@]}"; do
    printf '  - %s\n' "$label"
  done
  exit 1
fi