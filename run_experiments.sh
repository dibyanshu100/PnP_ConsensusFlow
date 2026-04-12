#!/bin/bash
# jobs

# nohup bash run_experiments.sh > output.log 2>&1 &

# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 method d_flow

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 method d_flow


# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 method d_flow


# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 method d_flow

# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 method d_flow

# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 method d_flow

########################################################################################################################################
# RUN PNP ADMMM HP tuning pho parameter

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 steps_pnp 300 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 steps_pnp 300 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 steps_pnp 300 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 steps_pnp 300 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 steps_pnp 300 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 steps_pnp 300 rho_admm 500 method pnp_flow_admm


# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 steps_pnp 300 rho_admm 700 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 steps_pnp 300 rho_admm 700 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 steps_pnp 300 rho_admm 700 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 steps_pnp 300 rho_admm 700 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 steps_pnp 300 rho_admm 700 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 steps_pnp 300 rho_admm 700 method pnp_flow_admm


# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 steps_pnp 300 rho_admm 900 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 steps_pnp 300 rho_admm 900 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 steps_pnp 300 rho_admm 900 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 steps_pnp 300 rho_admm 900 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 steps_pnp 300 rho_admm 900 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 steps_pnp 300 rho_admm 900 method pnp_flow_admm


########################################################################################################################################

# RUN PNP ADMMM HP tuning pho steps (300 steps better)

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 steps_pnp 500 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 steps_pnp 500 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 steps_pnp 500 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 steps_pnp 500 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 steps_pnp 500 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 steps_pnp 500 rho_admm 500 method pnp_flow_admm




########################################################################################################################################

# Experiments for grap for rho

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 100 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 steps_pnp 300 rho_admm 100 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 max_batch 1 steps_pnp 300 rho_admm 100 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 100 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 100 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 max_batch 1 steps_pnp 300 rho_admm 100 method pnp_flow_admm

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 200 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 steps_pnp 300 rho_admm 200 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 max_batch 1 steps_pnp 300 rho_admm 200 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 200 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 200 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 max_batch 1 steps_pnp 300 rho_admm 200 method pnp_flow_admm

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 300 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 steps_pnp 300 rho_admm 300 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 max_batch 1 steps_pnp 300 rho_admm 300 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 300 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 300 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 max_batch 1 steps_pnp 300 rho_admm 300 method pnp_flow_admm

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 400 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 steps_pnp 300 rho_admm 400 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 max_batch 1 steps_pnp 300 rho_admm 400 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 400 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 400 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 max_batch 1 steps_pnp 300 rho_admm 400 method pnp_flow_admm

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 steps_pnp 300 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 max_batch 1 steps_pnp 300 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 max_batch 1 steps_pnp 300 rho_admm 500 method pnp_flow_admm

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 600 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 steps_pnp 300 rho_admm 600 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 max_batch 1 steps_pnp 300 rho_admm 600 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 600 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 600 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 max_batch 1 steps_pnp 300 rho_admm 600 method pnp_flow_admm

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 700 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 steps_pnp 300 rho_admm 700 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 max_batch 1 steps_pnp 300 rho_admm 700 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 700 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 700 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 max_batch 1 steps_pnp 300 rho_admm 700 method pnp_flow_admm

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 800 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 steps_pnp 300 rho_admm 800 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 max_batch 1 steps_pnp 300 rho_admm 800 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 800 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 800 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 max_batch 1 steps_pnp 300 rho_admm 800 method pnp_flow_admm

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 900 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 steps_pnp 300 rho_admm 900 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 max_batch 1 steps_pnp 300 rho_admm 900 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 900 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 900 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 max_batch 1 steps_pnp 300 rho_admm 900 method pnp_flow_admm

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 1000 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 steps_pnp 300 rho_admm 1000 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 max_batch 1 steps_pnp 300 rho_admm 1000 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 1000 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 max_batch 1 steps_pnp 300 rho_admm 1000 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 max_batch 1 steps_pnp 300 rho_admm 1000 method pnp_flow_admm

########################################################################################################################################

# Experiments for flow priors
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 method flow_priors

# Experiments for d-flow
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 method d_flow

########################################################################################################################################
# For 1 sample

# echo "CAT"

# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 max_batch 1 batch_size_ip 1 method d_flow

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 7 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 7 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 7 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 7 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 7 max_batch 1 batch_size_ip 1 method d_flow


# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 11 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 11 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 11 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 11 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 11 max_batch 1 batch_size_ip 1 method d_flow


# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 15 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 15 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 15 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 15 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 15 max_batch 1 batch_size_ip 1 method d_flow

# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 1 batch_size_ip 1 method d_flow

# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 21 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 21 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 21 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 21 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 21 max_batch 1 batch_size_ip 1 method d_flow



# echo "Celeba"

# python main.py --opts dataset celeba train False eval True problem denoising start_batch 4 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset celeba train False eval True problem denoising start_batch 4 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset celeba train False eval True problem denoising start_batch 4 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset celeba train False eval True problem denoising start_batch 4 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset celeba train False eval True problem denoising start_batch 4 max_batch 1 batch_size_ip 1 method d_flow

# python main.py --opts dataset celeba train False eval True problem inpainting start_batch 7 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset celeba train False eval True problem inpainting start_batch 7 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset celeba train False eval True problem inpainting start_batch 7 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset celeba train False eval True problem inpainting start_batch 7 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset celeba train False eval True problem inpainting start_batch 7 max_batch 1 batch_size_ip 1 method d_flow


# python main.py --opts dataset celeba train False eval True problem superresolution start_batch 11 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset celeba train False eval True problem superresolution start_batch 11 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset celeba train False eval True problem superresolution start_batch 11 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset celeba train False eval True problem superresolution start_batch 11 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset celeba train False eval True problem superresolution start_batch 11 max_batch 1 batch_size_ip 1 method d_flow


# python main.py --opts dataset celeba train False eval True problem random_inpainting start_batch 15 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset celeba train False eval True problem random_inpainting start_batch 15 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset celeba train False eval True problem random_inpainting start_batch 15 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset celeba train False eval True problem random_inpainting start_batch 15 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset celeba train False eval True problem random_inpainting start_batch 15 max_batch 1 batch_size_ip 1 method d_flow

# python main.py --opts dataset celeba train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset celeba train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset celeba train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset celeba train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset celeba train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 1 batch_size_ip 1 method d_flow

# python main.py --opts dataset celeba train False eval True problem paintbrush_inpainting start_batch 21 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset celeba train False eval True problem paintbrush_inpainting start_batch 21 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset celeba train False eval True problem paintbrush_inpainting start_batch 21 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset celeba train False eval True problem paintbrush_inpainting start_batch 21 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset celeba train False eval True problem paintbrush_inpainting start_batch 21 max_batch 1 batch_size_ip 1 method d_flow

########################################################################################################################################

# For comparing ADMM vs FBS on FBS trained with adaptive interpolation

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 30 max_batch 1 batch_size_ip 1 method pnp_flow_fbs random_noise False
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 31 max_batch 1 batch_size_ip 1 method pnp_flow_fbs random_noise False
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 32 max_batch 1 batch_size_ip 1 method pnp_flow_fbs random_noise False
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 33 max_batch 1 batch_size_ip 1 method pnp_flow_fbs random_noise False
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 30 max_batch 1 batch_size_ip 1 method pnp_flow_fbs random_noise False
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 30 max_batch 1 batch_size_ip 1 method pnp_flow_fbs random_noise False


python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 30 max_batch 10 batch_size_ip 4 method pnp_flow_fbs random_noise False
python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 35 max_batch 10 batch_size_ip 4 method pnp_flow_fbs random_noise False
python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 40 max_batch 10 batch_size_ip 4 method pnp_flow_fbs random_noise False
python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 45 max_batch 10 batch_size_ip 4 method pnp_flow_fbs random_noise False
python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 50 max_batch 10 batch_size_ip 4 method pnp_flow_fbs random_noise False
python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 55 max_batch 10 batch_size_ip 4 method pnp_flow_fbs random_noise False

python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 30 max_batch 10 batch_size_ip 4 method pnp_flow_fbs random_noise True
python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 35 max_batch 10 batch_size_ip 4 method pnp_flow_fbs random_noise True
python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 40 max_batch 10 batch_size_ip 4 method pnp_flow_fbs random_noise True
python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 45 max_batch 10 batch_size_ip 4 method pnp_flow_fbs random_noise True
python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 50 max_batch 10 batch_size_ip 4 method pnp_flow_fbs random_noise True
python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 55 max_batch 10 batch_size_ip 4 method pnp_flow_fbs random_noise True

python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 30 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise False
python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 35 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise False
python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 40 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise False
python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 45 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise False
python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 50 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise False
python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 55 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise False

python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 30 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise True
python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 35 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise True
python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 40 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise True
python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 45 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise True
python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 50 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise True
python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 55 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise True


echo "All experiments completed successfully!"