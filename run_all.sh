#!/usr/bin/env bash

set -u
set -o pipefail

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

START_BATCH="${START_BATCH:-0}"
MAX_BATCH="${MAX_BATCH:-25}"
BATCH_SIZE_IP="${BATCH_SIZE_IP:-4}"

SEED="${SEED:-42}"

TOTAL=84
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


# # ===== celeba / inpainting =====
# run "celeba / inpainting / pnp_flow_admm[adaptive]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem inpainting steps_pnp 100 rho_admm 50 beta_rho 2.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / inpainting / pnp_flow_admm[stochastic]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem inpainting steps_pnp 100 rho_admm 50 beta_rho 2.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise False deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / inpainting / pnp_flow_admm[deterministic]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem inpainting steps_pnp 100 rho_admm 50 beta_rho 2.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise False deterministic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / inpainting / pnp_flow_fbs" \
#   python main.py --opts method pnp_flow_fbs dataset celeba problem inpainting steps_pnp 100 lr_pnp 1.0 alpha 0.5 num_samples 5 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / inpainting / ot_ode" \
#   python main.py --opts method ot_ode dataset celeba problem inpainting steps_ode 100 start_time 0.1 gamma gamma_t model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / inpainting / flow_priors" \
#   python main.py --opts method flow_priors dataset celeba problem inpainting lmbda 10000 eta 0.01 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / inpainting / d_flow" \
#   python main.py --opts method d_flow dataset celeba problem inpainting lmbda 0.001 alpha 0.1 max_iter 9 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / inpainting / flow_dps" \
#   python main.py --opts method flow_dps dataset celeba problem inpainting steps_ode 100 grad_steps 3 grad_step_size 1.0 use_stochastic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# # ===== celeba / random_inpainting =====
# run "celeba / random_inpainting / pnp_flow_admm[adaptive]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem random_inpainting steps_pnp 100 rho_admm 100 beta_rho 2.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / random_inpainting / pnp_flow_admm[stochastic]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem random_inpainting steps_pnp 100 rho_admm 100 beta_rho 2.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise False deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / random_inpainting / pnp_flow_admm[deterministic]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem random_inpainting steps_pnp 100 rho_admm 100 beta_rho 2.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise False deterministic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / random_inpainting / pnp_flow_fbs" \
#   python main.py --opts method pnp_flow_fbs dataset celeba problem random_inpainting steps_pnp 100 lr_pnp 1.0 alpha 0.01 num_samples 5 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / random_inpainting / ot_ode" \
#   python main.py --opts method ot_ode dataset celeba problem random_inpainting steps_ode 100 start_time 0.1 gamma constant model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / random_inpainting / flow_priors" \
#   python main.py --opts method flow_priors dataset celeba problem random_inpainting lmbda 10000 eta 0.01 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / random_inpainting / d_flow" \
#   python main.py --opts method d_flow dataset celeba problem random_inpainting lmbda 0.01 alpha 0.1 max_iter 20 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / random_inpainting / flow_dps" \
#   python main.py --opts method flow_dps dataset celeba problem random_inpainting steps_ode 100 grad_steps 3 grad_step_size 1.0 use_stochastic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# # ===== celeba / superresolution =====
# run "celeba / superresolution / pnp_flow_admm[adaptive]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem superresolution steps_pnp 100 rho_admm 50 beta_rho 1.3 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / superresolution / pnp_flow_admm[stochastic]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem superresolution steps_pnp 100 rho_admm 50 beta_rho 1.3 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise False deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / superresolution / pnp_flow_admm[deterministic]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem superresolution steps_pnp 100 rho_admm 50 beta_rho 1.3 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise False deterministic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / superresolution / pnp_flow_fbs" \
#   python main.py --opts method pnp_flow_fbs dataset celeba problem superresolution steps_pnp 100 lr_pnp 1.0 alpha 0.3 num_samples 5 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / superresolution / ot_ode" \
#   python main.py --opts method ot_ode dataset celeba problem superresolution steps_ode 100 start_time 0.1 gamma constant model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / superresolution / flow_priors" \
#   python main.py --opts method flow_priors dataset celeba problem superresolution lmbda 10000 eta 0.1 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / superresolution / d_flow" \
#   python main.py --opts method d_flow dataset celeba problem superresolution lmbda 0.001 alpha 0.1 max_iter 10 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / superresolution / flow_dps" \
#   python main.py --opts method flow_dps dataset celeba problem superresolution steps_ode 100 grad_steps 3 grad_step_size 1.0 use_stochastic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# # ===== celeba / phase_retrieval =====
# run "celeba / phase_retrieval / pnp_flow_admm[adaptive]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem phase_retrieval steps_pnp 100 rho_admm 1500 beta_rho 1.8 t_schedule_gamma 1.5 num_denoiser_calls 5 save_norms_admm False early_stopping False admm_gd_steps 10 adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / phase_retrieval / pnp_flow_admm[stochastic]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem phase_retrieval steps_pnp 100 rho_admm 1500 beta_rho 1.8 t_schedule_gamma 1.5 num_denoiser_calls 5 save_norms_admm False early_stopping False admm_gd_steps 10 adaptive_noise False deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / phase_retrieval / pnp_flow_admm[deterministic]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem phase_retrieval steps_pnp 100 rho_admm 1500 beta_rho 1.8 t_schedule_gamma 1.5 num_denoiser_calls 5 save_norms_admm False early_stopping False admm_gd_steps 10 adaptive_noise False deterministic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / phase_retrieval / pnp_flow_fbs" \
#   python main.py --opts method pnp_flow_fbs dataset celeba problem phase_retrieval steps_pnp 100 lr_pnp 1.0 alpha 0.01 num_samples 5 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / phase_retrieval / d_flow" \
#   python main.py --opts method d_flow dataset celeba problem phase_retrieval lmbda 0.01 alpha 0.1 max_iter 10 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / phase_retrieval / flow_dps" \
#   python main.py --opts method flow_dps dataset celeba problem phase_retrieval steps_ode 100 grad_steps 3 grad_step_size 1.0 use_stochastic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# # ===== celeba / hdr_reconstruction =====
# run "celeba / hdr_reconstruction / pnp_flow_admm[adaptive]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem hdr_reconstruction steps_pnp 100 rho_admm 100 beta_rho 1.5 t_schedule_gamma 0.2 num_denoiser_calls 5 save_norms_admm False early_stopping False admm_gd_steps 10 adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / hdr_reconstruction / pnp_flow_admm[stochastic]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem hdr_reconstruction steps_pnp 100 rho_admm 100 beta_rho 1.5 t_schedule_gamma 0.2 num_denoiser_calls 5 save_norms_admm False early_stopping False admm_gd_steps 10 adaptive_noise False deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / hdr_reconstruction / pnp_flow_admm[deterministic]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem hdr_reconstruction steps_pnp 100 rho_admm 100 beta_rho 1.5 t_schedule_gamma 0.2 num_denoiser_calls 5 save_norms_admm False early_stopping False admm_gd_steps 10 adaptive_noise False deterministic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / hdr_reconstruction / pnp_flow_fbs" \
#   python main.py --opts method pnp_flow_fbs dataset celeba problem hdr_reconstruction steps_pnp 100 lr_pnp 1.0 alpha 0.5 num_samples 5 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / hdr_reconstruction / d_flow [yaml-default, unverified]" \
#   python main.py --opts method d_flow dataset celeba problem hdr_reconstruction lmbda 0.01 alpha 0.1 max_iter 10 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / hdr_reconstruction / flow_dps" \
#   python main.py --opts method flow_dps dataset celeba problem hdr_reconstruction steps_ode 100 grad_steps 3 grad_step_size 1.0 use_stochastic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# # ===== celeba / nonlinear_deblurring =====
# run "celeba / nonlinear_deblurring / pnp_flow_admm[adaptive]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem nonlinear_deblurring steps_pnp 500 rho_admm 100 beta_rho 1.5 t_schedule_gamma 1.5 num_denoiser_calls 1 save_norms_admm False early_stopping True admm_gd_steps 10 adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / nonlinear_deblurring / pnp_flow_admm[stochastic]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem nonlinear_deblurring steps_pnp 500 rho_admm 100 beta_rho 1.5 t_schedule_gamma 1.5 num_denoiser_calls 1 save_norms_admm False early_stopping True admm_gd_steps 10 adaptive_noise False deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / nonlinear_deblurring / pnp_flow_admm[deterministic]" \
#   python main.py --opts method pnp_flow_admm dataset celeba problem nonlinear_deblurring steps_pnp 500 rho_admm 100 beta_rho 1.5 t_schedule_gamma 1.5 num_denoiser_calls 1 save_norms_admm False early_stopping True admm_gd_steps 10 adaptive_noise False deterministic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / nonlinear_deblurring / pnp_flow_fbs" \
#   python main.py --opts method pnp_flow_fbs dataset celeba problem nonlinear_deblurring steps_pnp 500 lr_pnp 1.0 alpha 0.5 num_samples 5 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / nonlinear_deblurring / d_flow [yaml-default, unverified]" \
#   python main.py --opts method d_flow dataset celeba problem nonlinear_deblurring lmbda 0.01 alpha 0.1 max_iter 10 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "celeba / nonlinear_deblurring / flow_dps" \
#   python main.py --opts method flow_dps dataset celeba problem nonlinear_deblurring steps_ode 500 grad_steps 3 grad_step_size 1.0 use_stochastic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# # ===== afhq_cat / inpainting =====
# run "afhq_cat / inpainting / pnp_flow_admm[adaptive]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem inpainting steps_pnp 100 rho_admm 50 beta_rho 2.5 t_schedule_gamma 1.5 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / inpainting / pnp_flow_admm[stochastic]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem inpainting steps_pnp 100 rho_admm 50 beta_rho 2.5 t_schedule_gamma 1.5 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise False deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / inpainting / pnp_flow_admm[deterministic]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem inpainting steps_pnp 100 rho_admm 50 beta_rho 2.5 t_schedule_gamma 1.5 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise False deterministic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / inpainting / pnp_flow_fbs" \
#   python main.py --opts method pnp_flow_fbs dataset afhq_cat problem inpainting steps_pnp 100 lr_pnp 1.0 alpha 0.5 num_samples 5 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / inpainting / ot_ode" \
#   python main.py --opts method ot_ode dataset afhq_cat problem inpainting steps_ode 100 start_time 0.1 gamma gamma_t model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / inpainting / flow_priors" \
#   python main.py --opts method flow_priors dataset afhq_cat problem inpainting lmbda 10000 eta 0.01 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / inpainting / d_flow" \
#   python main.py --opts method d_flow dataset afhq_cat problem inpainting lmbda 0.01 alpha 0.1 max_iter 9 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / inpainting / flow_dps" \
#   python main.py --opts method flow_dps dataset afhq_cat problem inpainting steps_ode 100 grad_steps 3 grad_step_size 1.0 use_stochastic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# # ===== afhq_cat / random_inpainting =====
# run "afhq_cat / random_inpainting / pnp_flow_admm[adaptive]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem random_inpainting steps_pnp 100 rho_admm 100 beta_rho 2.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / random_inpainting / pnp_flow_admm[stochastic]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem random_inpainting steps_pnp 100 rho_admm 100 beta_rho 2.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise False deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / random_inpainting / pnp_flow_admm[deterministic]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem random_inpainting steps_pnp 100 rho_admm 100 beta_rho 2.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise False deterministic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / random_inpainting / pnp_flow_fbs" \
#   python main.py --opts method pnp_flow_fbs dataset afhq_cat problem random_inpainting steps_pnp 100 lr_pnp 1.0 alpha 0.01 num_samples 5 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / random_inpainting / ot_ode" \
#   python main.py --opts method ot_ode dataset afhq_cat problem random_inpainting steps_ode 100 start_time 0.1 gamma constant model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / random_inpainting / flow_priors" \
#   python main.py --opts method flow_priors dataset afhq_cat problem random_inpainting lmbda 10000 eta 0.01 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / random_inpainting / d_flow" \
#   python main.py --opts method d_flow dataset afhq_cat problem random_inpainting lmbda 0.001 alpha 0.1 max_iter 20 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / random_inpainting / flow_dps" \
#   python main.py --opts method flow_dps dataset afhq_cat problem random_inpainting steps_ode 100 grad_steps 3 grad_step_size 1.0 use_stochastic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# # ===== afhq_cat / superresolution =====
# run "afhq_cat / superresolution / pnp_flow_admm[adaptive]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem superresolution steps_pnp 100 rho_admm 50 beta_rho 1.3 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / superresolution / pnp_flow_admm[stochastic]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem superresolution steps_pnp 100 rho_admm 50 beta_rho 1.3 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise False deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / superresolution / pnp_flow_admm[deterministic]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem superresolution steps_pnp 100 rho_admm 50 beta_rho 1.3 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False adaptive_noise False deterministic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / superresolution / pnp_flow_fbs" \
#   python main.py --opts method pnp_flow_fbs dataset afhq_cat problem superresolution steps_pnp 100 lr_pnp 1.0 alpha 0.01 num_samples 5 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / superresolution / ot_ode" \
#   python main.py --opts method ot_ode dataset afhq_cat problem superresolution steps_ode 100 start_time 0.1 gamma constant model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / superresolution / flow_priors" \
#   python main.py --opts method flow_priors dataset afhq_cat problem superresolution lmbda 10000 eta 0.1 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / superresolution / d_flow" \
#   python main.py --opts method d_flow dataset afhq_cat problem superresolution lmbda 0.001 alpha 0.1 max_iter 20 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / superresolution / flow_dps" \
#   python main.py --opts method flow_dps dataset afhq_cat problem superresolution steps_ode 100 grad_steps 3 grad_step_size 1.0 use_stochastic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# # ===== afhq_cat / phase_retrieval =====
run "afhq_cat / phase_retrieval / pnp_flow_admm[adaptive]" \
  python main.py --opts method pnp_flow_admm dataset afhq_cat problem phase_retrieval steps_pnp 300 rho_admm 1500 beta_rho 2.2 t_schedule_gamma 1.9 num_denoiser_calls 5 save_norms_admm False early_stopping False admm_gd_steps 10 adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "afhq_cat / phase_retrieval / pnp_flow_admm[stochastic]" \
  python main.py --opts method pnp_flow_admm dataset afhq_cat problem phase_retrieval steps_pnp 300 rho_admm 1500 beta_rho 2.2 t_schedule_gamma 1.9 num_denoiser_calls 5 save_norms_admm False early_stopping False admm_gd_steps 10 adaptive_noise False deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "afhq_cat / phase_retrieval / pnp_flow_admm[deterministic]" \
  python main.py --opts method pnp_flow_admm dataset afhq_cat problem phase_retrieval steps_pnp 300 rho_admm 1500 beta_rho 2.2 t_schedule_gamma 1.9 num_denoiser_calls 5 save_norms_admm False early_stopping False admm_gd_steps 10 adaptive_noise False deterministic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "afhq_cat / phase_retrieval / pnp_flow_fbs" \
  python main.py --opts method pnp_flow_fbs dataset afhq_cat problem phase_retrieval steps_pnp 300 lr_pnp 1.0 alpha 0.01 num_samples 5 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / phase_retrieval / d_flow [yaml-default, unverified]" \
#   python main.py --opts method d_flow dataset afhq_cat problem phase_retrieval lmbda 0.01 alpha 0.1 max_iter 10 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

run "afhq_cat / phase_retrieval / flow_dps" \
  python main.py --opts method flow_dps dataset afhq_cat problem phase_retrieval steps_ode 300 grad_steps 3 grad_step_size 1.0 use_stochastic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# # ===== afhq_cat / hdr_reconstruction =====
# run "afhq_cat / hdr_reconstruction / pnp_flow_admm[adaptive]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem hdr_reconstruction steps_pnp 100 rho_admm 500 beta_rho 1.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False admm_gd_steps 10 adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / hdr_reconstruction / pnp_flow_admm[stochastic]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem hdr_reconstruction steps_pnp 100 rho_admm 500 beta_rho 1.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False admm_gd_steps 10 adaptive_noise False deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / hdr_reconstruction / pnp_flow_admm[deterministic]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem hdr_reconstruction steps_pnp 100 rho_admm 500 beta_rho 1.5 t_schedule_gamma 1.0 num_denoiser_calls 5 save_norms_admm False early_stopping False admm_gd_steps 10 adaptive_noise False deterministic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / hdr_reconstruction / pnp_flow_fbs" \
#   python main.py --opts method pnp_flow_fbs dataset afhq_cat problem hdr_reconstruction steps_pnp 100 lr_pnp 1.0 alpha 0.5 num_samples 5 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / hdr_reconstruction / d_flow [yaml-default, unverified]" \
#   python main.py --opts method d_flow dataset afhq_cat problem hdr_reconstruction lmbda 0.01 alpha 0.1 max_iter 10 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / hdr_reconstruction / flow_dps" \
#   python main.py --opts method flow_dps dataset afhq_cat problem hdr_reconstruction steps_ode 100 grad_steps 3 grad_step_size 1.0 use_stochastic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True


# # ===== afhq_cat / nonlinear_deblurring =====
# run "afhq_cat / nonlinear_deblurring / pnp_flow_admm[adaptive]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem nonlinear_deblurring steps_pnp 500 rho_admm 100 beta_rho 1.5 t_schedule_gamma 1.5 num_denoiser_calls 1 save_norms_admm False early_stopping True admm_gd_steps 10 adaptive_noise True deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / nonlinear_deblurring / pnp_flow_admm[stochastic]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem nonlinear_deblurring steps_pnp 500 rho_admm 100 beta_rho 1.5 t_schedule_gamma 1.5 num_denoiser_calls 1 save_norms_admm False early_stopping True admm_gd_steps 10 adaptive_noise False deterministic_noise False model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / nonlinear_deblurring / pnp_flow_admm[deterministic]" \
#   python main.py --opts method pnp_flow_admm dataset afhq_cat problem nonlinear_deblurring steps_pnp 500 rho_admm 100 beta_rho 1.5 t_schedule_gamma 1.5 num_denoiser_calls 1 save_norms_admm False early_stopping True admm_gd_steps 10 adaptive_noise False deterministic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / nonlinear_deblurring / pnp_flow_fbs" \
#   python main.py --opts method pnp_flow_fbs dataset afhq_cat problem nonlinear_deblurring steps_pnp 500 lr_pnp 1.0 alpha 0.5 num_samples 5 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / nonlinear_deblurring / d_flow [yaml-default, unverified]" \
#   python main.py --opts method d_flow dataset afhq_cat problem nonlinear_deblurring lmbda 0.01 alpha 0.1 max_iter 10 model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

# run "afhq_cat / nonlinear_deblurring / flow_dps" \
#   python main.py --opts method flow_dps dataset afhq_cat problem nonlinear_deblurring steps_ode 500 grad_steps 3 grad_step_size 1.0 use_stochastic_noise True model ot train False eval True eval_split test noise_type gaussian seed "$SEED" start_batch "$START_BATCH" max_batch "$MAX_BATCH" batch_size_ip "$BATCH_SIZE_IP" compute_time True compute_memory True save_results True

printf '\n============================================================\n'
printf 'All runs finished. Total=%d Failed=%d\n' "$TOTAL" "$FAILED"
if [[ "$FAILED" -gt 0 ]]; then
  printf 'Failed runs:\n'
  for label in "${FAILED_LABELS[@]}"; do
    printf '  - %s\n' "$label"
  done
  exit 1
fi