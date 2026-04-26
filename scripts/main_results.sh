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

methods_batch=(
  pnp_flow_fbs
  pnp_flow_admm
  ot_ode
  flow_priors
  d_flow
)

for dataset in "${datasets[@]}"; do
  echo "Running dataset: $dataset"

  for problem in "${problems[@]}"; do
    start_batch=$((RANDOM % 10 + 1))
    echo "Running $problem on $dataset with start_batch=$start_batch"

    for method in "${methods_batch[@]}"; do

      extra_args=()

      if [[ "$method" == "pnp_flow_admm" ]]; then
        if [[ "$dataset" == "celeba" ]]; then
          extra_args+=(steps_pnp 200 rho_admm 1000)
        elif [[ "$dataset" == "afhq_cat" ]]; then
          extra_args+=(steps_pnp 250 rho_admm 500)
        fi
      fi
      echo "Method: $method ${extra_args[*]}"
      python main.py --opts \
        dataset "$dataset" train False eval True \
        problem "$problem" start_batch "$start_batch" \
        max_batch 25 batch_size_ip 4 method "$method" \
        "${extra_args[@]}"

    done
  done
done