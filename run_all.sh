# RANDOM=42

# nohup bash run_all.sh > output.log 2>&1 &

#!/bin/bash

RANDOM = 42

dataset="celeba"
method="pnp_flow_admm"

gammas=(0.2 1.0 2.0)

problems=(
    superresolution
    random_inpainting
    gaussian_deblurring_FFT
    inpainting
    denoising
    paintbrush_inpainting
)

start_batch=1

for problem in "${problems[@]}"; do

    echo "===================================="
    echo "Problem: $problem  start_batch=$start_batch"
    echo "===================================="

    for gamma in "${gammas[@]}"; do

        echo "Running gamma=$gamma"

        python main.py --opts \
            dataset "$dataset" \
            train False \
            eval True \
            problem "$problem" \
            start_batch "$start_batch" \
            max_batch 5 \
            batch_size_ip 5 \
            method "$method" \
            t_schedule_gamma "$gamma"

    done
done

echo "All experiments completed successfully!"


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