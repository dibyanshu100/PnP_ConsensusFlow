#!/usr/bin/env bash
set -euo pipefail

python main.py --opts \
    dataset celebahq dim_image 128 problem segmentation method pnp_flow_admm \
    steps_pnp 500 rho_admm 500 beta_rho 0.0 t_schedule_gamma 1.0 \
    num_denoiser_calls 1 save_norms_admm False early_stopping False \
    adaptive_noise True deterministic_noise False model ot train False \
    eval True eval_split test noise_type gaussian \
    batch_size_ip 15 max_batch 100 start_batch 0 save_results True

RECON_DIR=$(find results/celebahq/ot/segmentation/pnp_flow_admm/test \
    -type d -path '*/fid_kid_images/recon' | head -n 1)

if [ -z "${RECON_DIR}" ]; then
    echo "Could not find a fid_kid_images/recon directory under " \
         "results/celebahq/ot/segmentation/pnp_flow_admm/test -- " \
         "did step 1 run with save_results True?" >&2
    exit 1
fi

echo "Using recon dir: ${RECON_DIR}"

NUM_RECON=$(ls "${RECON_DIR}" | wc -l)
echo "Found ${NUM_RECON} reconstructed images."

python scripts/segmentation_fid_kid.py \
    --recon_dir "${RECON_DIR}" \
    --real_data_dir data/celebahq/test \
    --real_ids_file data/celebahq/splits/test_ids.txt \
    --subset_size 50