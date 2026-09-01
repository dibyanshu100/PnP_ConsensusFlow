#!/usr/bin/env bash
# Runs pnp_flow_admm with save_norms_admm=True (adaptive-noise variant) across
# every problem/dataset pair from run_all.sh's pnp_flow_admm block, plus the
# segmentation config, so norms_batch_*.npz/png convergence diagnostics get
# written for each. Hyperparameters are copied as-is from run_all.sh; only
# save_norms_admm was flipped to True and the batch controls below were added.

set -u
set -o pipefail

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

START_BATCH="${START_BATCH:-0}"
MAX_BATCH="${MAX_BATCH:-5}"
BATCH_SIZE_IP="${BATCH_SIZE_IP:-4}"

SEED="${SEED:-42}"

TOTAL=13
RUN_NUM=0
FAILED=0
FAILED_LABELS=()

# Wraps a single run: prints "[n/TOTAL] label", times it, reports pass/fail.
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


# ===== celeba =====
run "celeba / inpainting / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset celeba problem inpainting steps_pnp 100 rho_admm 50 beta_rho 2.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm True early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "celeba / random_inpainting / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset celeba problem random_inpainting steps_pnp 100 rho_admm 100 beta_rho 2.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm True early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "celeba / superresolution / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset celeba problem superresolution steps_pnp 100 rho_admm 50 beta_rho 1.3 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm True early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "celeba / phase_retrieval / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset celeba problem phase_retrieval steps_pnp 100 rho_admm 1500 beta_rho 1.8 t_schedule_gamma 1.5 num_denoiser_calls 5 save_norms_admm True early_stopping False admm_gd_steps 10 adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "celeba / hdr_reconstruction / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset celeba problem hdr_reconstruction steps_pnp 100 rho_admm 100 beta_rho 1.5 t_schedule_gamma 0.2 num_denoiser_calls 5 save_norms_admm True early_stopping False admm_gd_steps 10 adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "celeba / nonlinear_deblurring / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset celeba problem nonlinear_deblurring steps_pnp 500 rho_admm 100 beta_rho 1.5 t_schedule_gamma 1.5 num_denoiser_calls 1 save_norms_admm True early_stopping True admm_gd_steps 10 adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# ===== afhq_cat =====
run "afhq_cat / inpainting / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset afhq_cat problem inpainting steps_pnp 100 rho_admm 50 beta_rho 2.5 t_schedule_gamma 1.5 num_denoiser_calls 5 save_norms_admm True early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "afhq_cat / random_inpainting / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset afhq_cat problem random_inpainting steps_pnp 100 rho_admm 100 beta_rho 2.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm True early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "afhq_cat / superresolution / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset afhq_cat problem superresolution steps_pnp 500 rho_admm 50 beta_rho 1.3 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm True early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "afhq_cat / phase_retrieval / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset afhq_cat problem phase_retrieval steps_pnp 300 rho_admm 1500 beta_rho 2.2 t_schedule_gamma 1.9 num_denoiser_calls 5 save_norms_admm True early_stopping False admm_gd_steps 10 adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "afhq_cat / hdr_reconstruction / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset afhq_cat problem hdr_reconstruction steps_pnp 100 rho_admm 500 beta_rho 1.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm True early_stopping False admm_gd_steps 10 adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "afhq_cat / nonlinear_deblurring / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset afhq_cat problem nonlinear_deblurring steps_pnp 500 rho_admm 100 beta_rho 1.5 t_schedule_gamma 1.5 num_denoiser_calls 1 save_norms_admm True early_stopping True admm_gd_steps 10 adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# ===== celebahq / segmentation =====
run "celebahq / segmentation / pnp_flow_admm" \
  python main.py --opts dataset celebahq dim_image 128 problem segmentation method pnp_flow_admm steps_pnp 500 rho_admm 500 beta_rho 0.0 t_schedule_gamma 1.0 num_denoiser_calls 1 save_norms_admm True early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP"


printf '\n============================================================\n'
printf 'All runs finished. Total=%d Failed=%d\n' "$TOTAL" "$FAILED"
if [[ "$FAILED" -gt 0 ]]; then
  printf 'Failed runs:\n'
  for label in "${FAILED_LABELS[@]}"; do
    printf '  - %s\n' "$label"
  done
  exit 1
fi
