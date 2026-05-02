#!/bin/bash
# jobs
RANDOM=42

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

# RUN PNP ADMMM HP tuning rho steps (300 steps better)

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 4 steps_pnp 500 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 steps_pnp 500 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 4 steps_pnp 500 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 4 steps_pnp 500 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 4 steps_pnp 500 rho_admm 500 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 4 steps_pnp 500 rho_admm 500 method pnp_flow_admm




########################################################################################################################################

# # Experiments for graph for rho cat

# for rho in 100 200 300 400 500 600 700 800 900 1000; do
#   for problem in inpainting denoising superresolution random_inpainting paintbrush_inpainting gaussian_deblurring_FFT; do
    
#     python main.py --opts dataset afhq_cat train False eval True \
#       problem $problem \
#       start_batch 4 \
#       max_batch 4 \
#       batch_size_ip 10 \
#       steps_pnp 300 \
#       rho_admm $rho \
#       method pnp_flow_admm

#   done
# done


# ########################################################################################################################################

# # Experiments for graph for rho celeba

# for rho in 100 200 300 400 500 600 700 800 900 1000; do
#   for problem in inpainting denoising superresolution random_inpainting paintbrush_inpainting gaussian_deblurring_FFT; do
    
#     python main.py --opts dataset celeba train False eval True \
#       problem $problem \
#       start_batch 4 \
#       max_batch 4 \
#       batch_size_ip 10 \
#       steps_pnp 300 \
#       rho_admm $rho \
#       method pnp_flow_admm

#   done

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

# echo "FBS"

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 30 max_batch 10 batch_size_ip 4 method pnp_flow_fbs adaptive_noise False
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 35 max_batch 10 batch_size_ip 4 method pnp_flow_fbs adaptive_noise False
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 40 max_batch 10 batch_size_ip 4 method pnp_flow_fbs adaptive_noise False
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 45 max_batch 10 batch_size_ip 4 method pnp_flow_fbs adaptive_noise False
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 50 max_batch 10 batch_size_ip 4 method pnp_flow_fbs adaptive_noise False
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 55 max_batch 10 batch_size_ip 4 method pnp_flow_fbs adaptive_noise False

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 30 max_batch 10 batch_size_ip 4 method pnp_flow_fbs adaptive_noise True
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 35 max_batch 10 batch_size_ip 4 method pnp_flow_fbs adaptive_noise True
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 40 max_batch 10 batch_size_ip 4 method pnp_flow_fbs adaptive_noise True
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 45 max_batch 10 batch_size_ip 4 method pnp_flow_fbs adaptive_noise True
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 50 max_batch 10 batch_size_ip 4 method pnp_flow_fbs adaptive_noise True
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 55 max_batch 10 batch_size_ip 4 method pnp_flow_fbs adaptive_noise True

# echo "ADMM"

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 30 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise False
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 35 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise False
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 40 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise False
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 45 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise False
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 50 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise False
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 55 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise False

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 30 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise True
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 35 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise True
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 40 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise True
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 45 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise True
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 50 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise True
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 55 max_batch 10 batch_size_ip 4 method pnp_flow_admm adaptive_noise True

########################################################################################################################################

# important for paper showing zoomed version of inpainting  

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 75 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 75 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 75 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 75 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 75 max_batch 1 batch_size_ip 1 method d_flow


# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 90 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 90 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 90 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 90 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 90 max_batch 1 batch_size_ip 1 method d_flow


# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 85 max_batch 1 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 85 max_batch 1 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 85 max_batch 1 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 85 max_batch 1 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 85 max_batch 1 batch_size_ip 1 method d_flow

########################################################################################################################################


# echo "Celeba"
dataset="afhq_cat"  #"afhq_cat"

# list of problems (no fixed values anymore)
# problems=(
#   denoising
#   inpainting
#   superresolution
#   random_inpainting
#   gaussian_deblurring_FFT
#   paintbrush_inpainting
# )

problems=(
  denoising
)

########################################################################################################################################
# # important for paper 100 data result

# # methods grouped by config
# methods_batch4=("pnp_flow_fbs" "pnp_flow_admm" "ot_ode")
# methods_batch1=("flow_priors" "d_flow")

# for problem in "${problems[@]}"; do
#   # assign random number between 1 and 100
#   start_batch=$((RANDOM % 10 + 1))

#   echo "Running $problem with start_batch=$start_batch"

#   # methods with batch_size_ip=4
#   for method in "${methods_batch4[@]}"; do
#     python main.py --opts \
#       dataset "$dataset" train False eval True \
#       problem "$problem" start_batch "$start_batch" \
#       max_batch 10 batch_size_ip 10 method "$method"      # max_batch 10 batch_size_ip 10 method # max_batch 5 batch_size_ip 4 method "$method"
#   done

#   # methods with batch_size_ip=1
#   for method in "${methods_batch1[@]}"; do
#     python main.py --opts \
#       dataset "$dataset" train False eval True \
#       problem "$problem" start_batch "$start_batch" \
#       max_batch 50 batch_size_ip 2 method "$method"       # max_batch 50 batch_size_ip 2 method # max_batch 5 batch_size_ip 4 method "$method"
#   done
# done


########################################################################################################################################

# for only single
method="pnp_flow_admm"   #"pnp_flow_admm"

for problem in "${problems[@]}"; do
  # assign random number between 1 and 10
  start_batch=$((RANDOM % 10 + 1))
  echo "Running $problem with start_batch=$start_batch"
  python main.py --opts \
    dataset "$dataset" train False eval True \
    problem "$problem" start_batch "$start_batch" \
    max_batch 5 batch_size_ip 10 method "$method"
done


########################################################################################################################################
# # for ablation studies

# methods=("pnp_flow_fbs" "pnp_flow_admm")
# adaptive_options=(False True)
# deterministic_options=(True False)

# for problem in "${problems[@]}"; do
#   start_batch=$((RANDOM % 10 + 1))
#   echo "Running $problem with start_batch=$start_batch"

#   for method in "${methods[@]}"; do
#     if [[ "$method" == "pnp_flow_fbs" ]]; then
#       echo "FBS"
#     else
#       echo "ADMM"
#     fi

#     for adaptive in "${adaptive_options[@]}"; do
#       if [[ "$method" == "pnp_flow_admm" && "$adaptive" == "False" ]]; then
#         for deterministic in "${deterministic_options[@]}"; do
#           python main.py --opts dataset $dataset train False eval True problem $problem start_batch $start_batch max_batch 5 batch_size_ip 5 method $method adaptive_noise $adaptive deterministic_noise $deterministic
#         done
#       else
#         python main.py --opts dataset $dataset train False eval True problem $problem start_batch $start_batch max_batch 5 batch_size_ip 5 method $method adaptive_noise $adaptive
#       fi
#     done
#   done
# done

# dataset="afhq_cat"
# methods=("pnp_flow_fbs" "pnp_flow_admm")
# adaptive_options=(False True)
# deterministic_options=(True False)

# problems=(
#   inpainting
#   random_inpainting
#   denoising
#   paintbrush_inpainting
#   superresolution
#   gaussian_deblurring_FFT
# )

# start_batches=(30 35 40 45 50 55)

# for i in "${!problems[@]}"; do
#   problem="${problems[$i]}"
#   start_batch="${start_batches[$i]}"

#   echo "Running $problem with start_batch=$start_batch"

#   for method in "${methods[@]}"; do
#     if [[ "$method" == "pnp_flow_fbs" ]]; then
#       echo "FBS"
#     else
#       echo "ADMM"
#     fi

#     for adaptive in "${adaptive_options[@]}"; do
#       if [[ "$method" == "pnp_flow_admm" && "$adaptive" == "False" ]]; then
#         for deterministic in "${deterministic_options[@]}"; do
#           python main.py --opts \
#             dataset "$dataset" \
#             train False \
#             eval True \
#             problem "$problem" \
#             start_batch "$start_batch" \
#             max_batch 10 \
#             batch_size_ip 4 \
#             method "$method" \
#             adaptive_noise "$adaptive" \
#             deterministic_noise "$deterministic" \
#             steps_pnp 300
#         done
#       elif [[ "$method" == "pnp_flow_admm" ]]; then
#         python main.py --opts \
#           dataset "$dataset" \
#           train False \
#           eval True \
#           problem "$problem" \
#           start_batch "$start_batch" \
#           max_batch 10 \
#           batch_size_ip 4 \
#           method "$method" \
#           adaptive_noise "$adaptive" \
#           steps_pnp 300
#       else
#         python main.py --opts \
#           dataset "$dataset" \
#           train False \
#           eval True \
#           problem "$problem" \
#           start_batch "$start_batch" \
#           max_batch 10 \
#           batch_size_ip 4 \
#           method "$method" \
#           adaptive_noise "$adaptive"
#       fi
#     done
#   done
# done




# echo "CAT"

# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 5 max_batch 100 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 5 max_batch 100 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 5 max_batch 100 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 5 max_batch 100 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 5 max_batch 100 batch_size_ip 1 method d_flow

# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 75 max_batch 100 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 75 max_batch 100 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 75 max_batch 100 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 75 max_batch 100 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem inpainting start_batch 75 max_batch 100 batch_size_ip 1 method d_flow

# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 90 max_batch 100 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 90 max_batch 100 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 90 max_batch 100 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 90 max_batch 100 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem superresolution start_batch 90 max_batch 100 batch_size_ip 1 method d_flow

# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 85 max_batch 100 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 85 max_batch 100 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 85 max_batch 100 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 85 max_batch 100 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem random_inpainting start_batch 85 max_batch 100 batch_size_ip 1 method d_flow

# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 100 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 100 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 100 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 100 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem gaussian_deblurring_FFT start_batch 18 max_batch 100 batch_size_ip 1 method d_flow

# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 21 max_batch 100 batch_size_ip 1 method pnp_flow_fbs
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 21 max_batch 100 batch_size_ip 1 method pnp_flow_admm
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 21 max_batch 100 batch_size_ip 1 method flow_priors
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 21 max_batch 100 batch_size_ip 1 method ot_ode
# python main.py --opts dataset afhq_cat train False eval True problem paintbrush_inpainting start_batch 21 max_batch 100 batch_size_ip 1 method d_flow


echo "All experiments completed successfully!"