#!/usr/bin/env bash
set -euo pipefail

DATASET_ARGS="dataset celebahq dim_image 128 problem segmentation model ot \
train False eval True eval_split test noise_type gaussian \
batch_size_ip 4 max_batch 25 start_batch 0 save_results True"

run_method () {
    local method="$1"
    shift
    local method_args="$*"

    echo "=============================================="
    echo "Running method: ${method}"
    echo "=============================================="
    python main.py --opts ${DATASET_ARGS} method "${method}" ${method_args}

    local recon_dir
    recon_dir=$(find "results/celebahq/ot/segmentation/${method}/test" \
        -type d -path '*/fid_kid_images/recon' | head -n 1)

    if [ -z "${recon_dir}" ]; then
        echo "Could not find a fid_kid_images/recon directory under " \
             "results/celebahq/ot/segmentation/${method}/test -- " \
             "did the run above finish with save_results True?" >&2
        return 1
    fi

    echo "Using recon dir: ${recon_dir}"
    local num_recon
    num_recon=$(ls "${recon_dir}" | wc -l)
    echo "Found ${num_recon} reconstructed images for ${method}."

    python scripts/segmentation_fid_kid.py \
        --recon_dir "${recon_dir}" \
        --real_data_dir data/celebahq/test \
        --real_ids_file data/celebahq/splits/test_ids.txt \
        --subset_size 50
}

# ---- pnp_flow_admm ----------------------------------------------------
run_method pnp_flow_admm \
    steps_pnp 500 rho_admm 500 beta_rho 0.0 t_schedule_gamma 1.0 \
    num_denoiser_calls 1 save_norms_admm False early_stopping False \
    adaptive_noise True deterministic_noise False

# ---- pnp_flow_fbs -------------------------------------------------------
run_method pnp_flow_fbs steps_pnp 500

# ---- d_flow --------------------------------------------------------------
run_method d_flow

# ---- flow_dps --------------------------------------------------------------
run_method flow_dps

echo "=============================================="
echo "Done. FID/KID/LPIPS results written under each method's "
echo "results/celebahq/ot/segmentation/<method>/test/.../fid_kid_images/fid_kid.txt"
echo "=============================================="