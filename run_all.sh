# RANDOM=42

# nohup bash run_all.sh > output.log 2>&1 &

#!/bin/bash

# RANDOM = 42

# dataset="celeba"
# method="pnp_flow_admm"

# gammas=(0.2 1.0 2.0)

# problems=(
#     superresolution
#     random_inpainting
#     gaussian_deblurring_FFT
#     inpainting
#     denoising
#     paintbrush_inpainting
# )

# start_batch=1

# for problem in "${problems[@]}"; do

#     echo "===================================="
#     echo "Problem: $problem  start_batch=$start_batch"
#     echo "===================================="

#     for gamma in "${gammas[@]}"; do

#         echo "Running gamma=$gamma"

#         python main.py --opts \
#             dataset "$dataset" \
#             train False \
#             eval True \
#             problem "$problem" \
#             start_batch "$start_batch" \
#             max_batch 5 \
#             batch_size_ip 5 \
#             method "$method" \
#             t_schedule_gamma "$gamma"

#     done
# done

# echo "All experiments completed successfully!"


# #!/bin/bash
# #!/bin/bash
# set -e

# dataset="afhq_cat"
# method="pnp_flow_admm"

# # Five independent runs for reporting mean ± standard deviation
# seeds=(42 123 2024 3407 7777)

# # Representative subset requested by the reviewer
# problems=(
#     superresolution
#     random_inpainting
#     gaussian_deblurring_FFT
#     inpainting
#     denoising
#     paintbrush_inpainting
# )

# # Main-paper configuration: linear timestep schedule
# gamma=1.0

# # Use the same test subset for every seed
# start_batch=1
# max_batch=5
# batch_size=5

# results_dir="results/multiseed"
# mkdir -p "$results_dir"

# for seed in "${seeds[@]}"; do
#     echo "============================================"
#     echo "Seed: $seed"
#     echo "============================================"

#     for problem in "${problems[@]}"; do
#         echo "--------------------------------------------"
#         echo "Dataset: $dataset"
#         echo "Problem: $problem"
#         echo "Seed: $seed"
#         echo "Start batch: $start_batch"
#         echo "--------------------------------------------"

#         log_file="${results_dir}/${dataset}_${problem}_seed${seed}.log"

#         python main.py --opts \
#             dataset "$dataset" \
#             train False \
#             eval True \
#             problem "$problem" \
#             start_batch "$start_batch" \
#             max_batch "$max_batch" \
#             batch_size_ip "$batch_size" \
#             method "$method" \
#             t_schedule_gamma "$gamma" \
#             seed "$seed" \
#             | tee "$log_file"
#     done
# done

# echo "All multi-seed experiments completed successfully!"
# echo "Logs saved to ${results_dir}/ for aggregation."





# python main.py --opts dataset celeba train False eval True problem phase_retrieval method pnp_flow_admm rho_admm 1000 steps_pnp 200 admm_gd_steps 10 t_schedule_gamma 1.0
# python main.py --opts dataset celeba train False eval True problem phase_retrieval method pnp_flow_fbs
# python main.py --opts dataset celeba train False eval True problem phase_retrieval method d_flow 


#!/usr/bin/env bash

set -u
set -o pipefail

# Seed Bash's pseudo-random generator.
RANDOM=42

datasets=(
  celeba
  afhq_cat
)

problems=(
  denoising
  gaussian_deblurring_FFT
  superresolution
  random_inpainting
  inpainting
)

# Use "pnp_flow" instead if that is the exact name expected by your config.
method="pnp_flow_fbs"

lr_pnp=1.0
num_samples=5

eval_split="test"
max_batch=2
batch_size_ip=4

for dataset in "${datasets[@]}"; do
  echo "============================================================"
  echo "Dataset: $dataset"
  echo "============================================================"

  for problem in "${problems[@]}"; do

    # Dataset- and task-specific PnP-Flow hyperparameters.
    case "${dataset}:${problem}" in

      # ---------------- CelebA ----------------
      celeba:superresolution)
        alpha=0.3
        steps_pnp=100
        ;;

      celeba:random_inpainting)
        alpha=0.01
        steps_pnp=100
        ;;

      celeba:denoising)
        alpha=0.8
        steps_pnp=100
        ;;

      celeba:gaussian_deblurring_FFT)
        alpha=0.01
        steps_pnp=100
        ;;

      celeba:inpainting)
        # Corresponds to box inpainting in the table.
        alpha=0.5
        steps_pnp=100
        ;;

      # ---------------- AFHQ-Cat ----------------
      afhq_cat:denoising)
        alpha=0.8
        steps_pnp=100
        ;;

      afhq_cat:gaussian_deblurring_FFT)
        alpha=0.01
        steps_pnp=500
        ;;

      afhq_cat:superresolution)
        alpha=0.01
        steps_pnp=500
        ;;

      afhq_cat:random_inpainting)
        alpha=0.01
        steps_pnp=200
        ;;

      afhq_cat:inpainting)
        # Corresponds to box inpainting in the table.
        alpha=0.5
        steps_pnp=100
        ;;

      *)
        echo "No PnP-Flow parameters defined for:"
        echo "  dataset=$dataset"
        echo "  problem=$problem"
        exit 1
        ;;
    esac

    start_batch=$((RANDOM % 10 + 1))

    echo
    echo "------------------------------------------------------------"
    echo "Running PnP-Flow"
    echo "  dataset:       $dataset"
    echo "  problem:       $problem"
    echo "  alpha:         $alpha"
    echo "  steps_pnp:     $steps_pnp"
    echo "  num_samples:   $num_samples"
    echo "  start_batch:   $start_batch"
    echo "------------------------------------------------------------"

    python main.py --opts \
      dataset "$dataset" \
      eval_split "$eval_split" \
      train False \
      eval True \
      problem "$problem" \
      method "$method" \
      start_batch "$start_batch" \
      max_batch "$max_batch" \
      batch_size_ip "$batch_size_ip" \
      lr_pnp "$lr_pnp" \
      alpha "$alpha" \
      num_samples "$num_samples" \
      steps_pnp "$steps_pnp"

  done
done