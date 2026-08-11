#!/usr/bin/env bash

set -u
set -o pipefail

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

START_BATCH="${START_BATCH:-1}"
MAX_BATCH="${MAX_BATCH:-2}"
BATCH_SIZE_IP="${BATCH_SIZE_IP:-4}"

SEED="${SEED:-42}"
NUM_RESTARTS_PHASE_RETRIEVAL="${NUM_RESTARTS_PHASE_RETRIEVAL:-2}"

LR_PNP="${LR_PNP:-1.0}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"

TOTAL=26
RUN_NUM=0
FAILED=0
FAILED_LABELS=()

# Wraps a single run: prints "[n/TOTAL] label", times it, reports pass/fail.
# Usage: run "label" python main.py --opts ...
run() {
  local label="$1"
  shift
  RUN_NUM=$((RUN_NUM + 1))
  local remaining=$((TOTAL - RUN_NUM))

  printf '\n============================================================\n'
  printf '[%d/%d] %s  (%d left after this)\n' "$RUN_NUM" "$TOTAL" "$label" "$remaining"
  printf '============================================================\n'

  local start_ts elapsed
  start_ts=$(date +%s)

  "$@"
  local status=$?

  elapsed=$(( $(date +%s) - start_ts ))
  if [[ "$status" -ne 0 ]]; then
    printf '\n>>> FAILED (%d): %s  [%ds]\n' "$status" "$label" "$elapsed"
    FAILED=$((FAILED + 1))
    FAILED_LABELS+=("$label")
  else
    printf '\n>>> DONE: %s  [%ds]\n' "$label" "$elapsed"
  fi
}

# CelebA
run "celeba / inpainting" \
  python main.py --opts method pnp_flow_admm dataset celeba model ot train False eval True eval_split test problem inpainting noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 100 rho_admm 50 t_schedule_gamma 1.0 beta_rho 2.5 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5

run "celeba / superresolution" \
  python main.py --opts method pnp_flow_admm dataset celeba model ot train False eval True eval_split test problem superresolution noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 100 rho_admm 50 t_schedule_gamma 1.0 beta_rho 0.003 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5

run "celeba / random_inpainting" \
  python main.py --opts method pnp_flow_admm dataset celeba model ot train False eval True eval_split test problem random_inpainting noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 100 rho_admm 100 t_schedule_gamma 1.0 beta_rho 2.5 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5

# run "celeba / paintbrush_inpainting" \
#   python main.py --opts method pnp_flow_admm dataset celeba model ot train False eval True eval_split test problem paintbrush_inpainting noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 100 rho_admm 200 t_schedule_gamma 1.0 beta_rho 1.9 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5

# run "celeba / denoising" \
#   python main.py --opts method pnp_flow_admm dataset celeba model ot train False eval True eval_split test problem denoising noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 500 rho_admm 500 t_schedule_gamma 1.0 beta_rho 0.009 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping True compute_time True compute_memory True num_denoiser_calls 5

# run "celeba / gaussian_deblurring_FFT" \
#   python main.py --opts method pnp_flow_admm dataset celeba model ot train False eval True eval_split test problem gaussian_deblurring_FFT noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 100 rho_admm 500 t_schedule_gamma 0.2 beta_rho 1.0 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5

# run "celeba / nonlinear_deblurring" \
#   python main.py --opts method pnp_flow_admm dataset celeba model ot train False eval True eval_split test problem nonlinear_deblurring noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 100 rho_admm 1000 t_schedule_gamma 1.0 beta_rho 1.0 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5 admm_gd_steps 5

# run "celeba / phase_retrieval" \
#   python main.py --opts method pnp_flow_admm dataset celeba model ot train False eval True eval_split test problem phase_retrieval noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 200 rho_admm 1000 t_schedule_gamma 1.8 beta_rho 1.0 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5 admm_gd_steps 10 num_restarts "$NUM_RESTARTS_PHASE_RETRIEVAL"


# # AFHQ-Cat
# run "afhq_cat / inpainting" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat model ot train False eval True eval_split test problem inpainting noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 100 rho_admm 50 t_schedule_gamma 1.5 beta_rho 2.5 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5

# run "afhq_cat / superresolution" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat model ot train False eval True eval_split test problem superresolution noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 100 rho_admm 50 t_schedule_gamma 1.0 beta_rho 0.003 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5

run "afhq_cat / random_inpainting" \
  python main.py --opts method pnp_flow_admm dataset afhq_cat model ot train False eval True eval_split test problem random_inpainting noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 100 rho_admm 100 t_schedule_gamma 1.0 beta_rho 2.5 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5

# run "afhq_cat / paintbrush_inpainting" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat model ot train False eval True eval_split test problem paintbrush_inpainting noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 100 rho_admm 500 t_schedule_gamma 1.0 beta_rho 1.3 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5

# run "afhq_cat / denoising" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat model ot train False eval True eval_split test problem denoising noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 500 rho_admm 500 t_schedule_gamma 1.0 beta_rho 0.009 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping True compute_time True compute_memory True num_denoiser_calls 5

# run "afhq_cat / gaussian_deblurring_FFT" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat model ot train False eval True eval_split test problem gaussian_deblurring_FFT noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 100 rho_admm 500 t_schedule_gamma 0.2 beta_rho 0.5 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5

# run "afhq_cat / nonlinear_deblurring" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat model ot train False eval True eval_split test problem nonlinear_deblurring noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 100 rho_admm 1000 t_schedule_gamma 1.0 beta_rho 1.0 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5 admm_gd_steps 5

# run "afhq_cat / phase_retrieval" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat model ot train False eval True eval_split test problem phase_retrieval noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 500 rho_admm 1000 t_schedule_gamma 1.0 beta_rho 0.5 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5 admm_gd_steps 20 num_restarts "$NUM_RESTARTS_PHASE_RETRIEVAL"


# =========================
# FBS: CelebA
# =========================

# run "celeba / FBS denoising" \
#   python main.py --opts dataset celeba eval_split test train False eval True seed "$SEED" problem denoising method pnp_flow_fbs start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" lr_pnp "$LR_PNP" alpha 0.8 num_samples "$NUM_SAMPLES" steps_pnp 100

# run "celeba / FBS gaussian_deblurring_FFT" \
#   python main.py --opts dataset celeba eval_split test train False eval True seed "$SEED" problem gaussian_deblurring_FFT method pnp_flow_fbs start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" lr_pnp "$LR_PNP" alpha 0.01 num_samples "$NUM_SAMPLES" steps_pnp 100

# run "celeba / FBS superresolution" \
#   python main.py --opts dataset celeba eval_split test train False eval True seed "$SEED" problem superresolution method pnp_flow_fbs start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" lr_pnp "$LR_PNP" alpha 0.3 num_samples "$NUM_SAMPLES" steps_pnp 100

run "celeba / FBS random_inpainting" \
  python main.py --opts dataset celeba eval_split test train False eval True seed "$SEED" problem random_inpainting method pnp_flow_fbs start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" lr_pnp "$LR_PNP" alpha 0.01 num_samples "$NUM_SAMPLES" steps_pnp 100

# run "celeba / FBS inpainting" \
#   python main.py --opts dataset celeba eval_split test train False eval True seed "$SEED" problem inpainting method pnp_flow_fbs start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" lr_pnp "$LR_PNP" alpha 0.5 num_samples "$NUM_SAMPLES" steps_pnp 100


# =========================
# FBS: AFHQ-Cat
# =========================

# run "afhq_cat / FBS denoising" \
#   python main.py --opts dataset afhq_cat eval_split test train False eval True seed "$SEED" problem denoising method pnp_flow_fbs start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" lr_pnp "$LR_PNP" alpha 0.8 num_samples "$NUM_SAMPLES" steps_pnp 100

# run "afhq_cat / FBS gaussian_deblurring_FFT" \
#   python main.py --opts dataset afhq_cat eval_split test train False eval True seed "$SEED" problem gaussian_deblurring_FFT method pnp_flow_fbs start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" lr_pnp "$LR_PNP" alpha 0.01 num_samples "$NUM_SAMPLES" steps_pnp 100

# run "afhq_cat / FBS superresolution" \
#   python main.py --opts dataset afhq_cat eval_split test train False eval True seed "$SEED" problem superresolution method pnp_flow_fbs start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" lr_pnp "$LR_PNP" alpha 0.01 num_samples "$NUM_SAMPLES" steps_pnp 100

run "afhq_cat / FBS random_inpainting" \
  python main.py --opts dataset afhq_cat eval_split test train False eval True seed "$SEED" problem random_inpainting method pnp_flow_fbs start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" lr_pnp "$LR_PNP" alpha 0.01 num_samples "$NUM_SAMPLES" steps_pnp 100

# run "afhq_cat / FBS inpainting" \
#   python main.py --opts dataset afhq_cat eval_split test train False eval True seed "$SEED" problem inpainting method pnp_flow_fbs start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" lr_pnp "$LR_PNP" alpha 0.5 num_samples "$NUM_SAMPLES" steps_pnp 100


printf '\n============================================================\n'
printf 'All runs finished. Total=%d Failed=%d\n' "$TOTAL" "$FAILED"
if [[ "$FAILED" -gt 0 ]]; then
  printf 'Failed runs:\n'
  for label in "${FAILED_LABELS[@]}"; do
    printf '  - %s\n' "$label"
  done
  exit 1
fi



# python main.py --opts method pnp_flow_admm dataset celeba model ot train False eval True eval_split test problem blur_then_downsample noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" steps_pnp 300 rho_admm 200 t_schedule_gamma 0.2 beta_rho 1.7 rho_schedule_style inverse_power adaptive_noise True deterministic_noise False save_norms_admm False save_results True early_stopping False compute_time True compute_memory True num_denoiser_calls 5