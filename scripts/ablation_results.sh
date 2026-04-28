#!/bin/bash
RANDOM=42
datasets=("celeba" "afhq_cat")

problems=(
  denoising
  inpainting
  superresolution
  random_inpainting
  gaussian_deblurring_FFT
  paintbrush_inpainting
)

methods=("pnp_flow_fbs" "pnp_flow_admm")
adaptive_options=(False True)
deterministic_options=(True False)

for dataset in "${datasets[@]}"; do
  echo "Running dataset: $dataset"

  for problem in "${problems[@]}"; do
    start_batch=$((RANDOM % 10 + 1))
    echo "Running $problem with start_batch=$start_batch"

    for method in "${methods[@]}"; do

      # set ADMM-specific hyperparameters
      extra_args=()
      if [[ "$method" == "pnp_flow_admm" ]]; then
        if [[ "$dataset" == "celeba" ]]; then
          extra_args+=(steps_pnp 200 rho_admm 1000)
        elif [[ "$dataset" == "afhq_cat" ]]; then
          extra_args+=(steps_pnp 250 rho_admm 500)
        fi
        echo "ADMM ${extra_args[*]}"
      else
        echo "FBS"
      fi

      for adaptive in "${adaptive_options[@]}"; do

        if [[ "$method" == "pnp_flow_admm" && "$adaptive" == "False" ]]; then
          for deterministic in "${deterministic_options[@]}"; do

            python main.py --opts \
              dataset "$dataset" train False eval True \
              problem "$problem" start_batch "$start_batch" \
              max_batch 5 batch_size_ip 5 \
              method "$method" \
              adaptive_noise "$adaptive" \
              deterministic_noise "$deterministic" \
              "${extra_args[@]}"

          done

        else

          python main.py --opts \
            dataset "$dataset" train False eval True \
            problem "$problem" start_batch "$start_batch" \
            max_batch 5 batch_size_ip 5 \
            method "$method" \
            adaptive_noise "$adaptive" \
            "${extra_args[@]}"

        fi

      done
    done
  done
done