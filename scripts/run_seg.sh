#!/usr/bin/env bash
set -euo pipefail

DATASET_ARGS="dataset celebahq dim_image 128 problem segmentation model ot \
train False eval True eval_split test noise_type gaussian \
batch_size_ip 4 max_batch 25 start_batch 0 save_results True"


run_method () {
    local method="$1"
    shift
    local method_args=("$@")

    echo "=============================================="
    echo "Running method: ${method}"
    echo "=============================================="

    # Run reconstruction/evaluation.
    # Explicitly catch failure so this method stops, but the whole script doesn't.
    if ! python main.py \
        --opts ${DATASET_ARGS} \
        method "${method}" \
        "${method_args[@]}"; then

        echo "ERROR: main.py failed for ${method}" >&2
        return 1
    fi

    # Find the reconstruction directory.
    # -print -quit avoids the find | head issue with pipefail.
    local recon_dir
    recon_dir=$(
        find "results/celebahq/ot/segmentation/${method}/test" \
            -type d \
            -path '*/fid_kid_images/recon' \
            -print \
            -quit
    ) || {
        echo "ERROR: Failed while searching for recon directory for ${method}" >&2
        return 1
    }

    if [ -z "${recon_dir}" ]; then
        echo "ERROR: Could not find fid_kid_images/recon under:" >&2
        echo "results/celebahq/ot/segmentation/${method}/test" >&2
        echo "Did the run finish with save_results=True?" >&2
        return 1
    fi

    echo "Using recon dir: ${recon_dir}"

    local num_recon
    num_recon=$(
        find "${recon_dir}" \
            -maxdepth 1 \
            -type f \
            | wc -l
    )

    echo "Found ${num_recon} reconstructed images for ${method}."

    # Compute FID/KID/LPIPS.
    if ! python scripts/segmentation_fid_kid.py \
        --recon_dir "${recon_dir}" \
        --real_data_dir data/celebahq/test \
        --real_ids_file data/celebahq/splits/test_ids.txt \
        --subset_size 50; then

        echo "ERROR: FID/KID/LPIPS computation failed for ${method}" >&2
        return 1
    fi

    echo "SUCCESS: ${method}"
    return 0
}


safe_run () {
    local method="$1"
    shift

    if run_method "${method}" "$@"; then
        echo "=============================================="
        echo "${method} completed successfully."
        echo "=============================================="
    else
        echo "==============================================" >&2
        echo "WARNING: ${method} failed. Continuing to next method..." >&2
        echo "==============================================" >&2
    fi

    echo
}


# ---- pnp_cf ----------------------------------------------------

safe_run pnp_cf \
    steps_pnp 500 \
    rho_admm 500 \
    beta_rho 0.0 \
    t_schedule_gamma 1.0 \
    num_denoiser_calls 1 \
    save_norms_admm False \
    early_stopping False \
    adaptive_noise True \
    deterministic_noise False


# ---- pnp_flow_fbs -----------------------------------------------------

safe_run pnp_flow_fbs \
    steps_pnp 500


# ---- d_flow -----------------------------------------------------------

safe_run d_flow


# ---- flow_dps ---------------------------------------------------------

safe_run flow_dps


echo "=============================================="
echo "All methods attempted."
echo "FID/KID/LPIPS results for successful methods are under:"
echo "results/celebahq/ot/segmentation/<method>/test/.../fid_kid_images/fid_kid.txt"
echo "=============================================="