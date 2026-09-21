![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)

# PnP-CF: A Consensus Flow-Matching Solver for General Inverse Problems

This repository contains the implementation of **PnP-CF**, a flow-matching plug-and-play solver that combines ADMM-like consensus updates with a time-adaptive stochastic–deterministic pullback, enabling strong reconstruction performance across both linear and nonlinear inverse problems.

Our implementation is adapted from the PnP-Flow (ICLR 2025) codebase, which is based on a Forward-Backward Splitting (FBS) formulation.

---

## Overview

Inverse problems aim to recover signals from degraded measurements and are typically ill-posed. Plug-and-Play (PnP) methods address this by integrating pretrained generative priors, with recent work leveraging efficient continuous-time generative models such as flow matching. However, the reconstruction quality depends critically on the interplay and consensus between the inverse-problem solver and the generative sampler.

We propose **PnP-CF (Consensus Flow)**, a flow-matching PnP framework designed to improve this interaction and maintain consistency with the learned flow trajectory.

**Our contributions include:**
* **ADMM-like dual-variable updates** to promote consensus between the data-consistency and generative-prior steps
* **Time-adaptive trajectory pullback** onto intermediate flow states using complementary stochastic and deterministic components
* **Support for both linear and nonlinear inverse problems**, including forward operators implemented by pretrained neural networks

<p align="center">
  <img src="al_comparisons_paper.png" width="800"/>
</p>

---

## Requirements

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install all the necessary Python packages required to run the project.


## Download Datasets and Pretrained Models

We provide a `download.sh` script to download datasets and pretrained flow-matching checkpoints from Google Drive.

Install dependency (if needed):

```bash
pip install gdown
```

---

### Usage

```bash
bash download.sh <option>
```

Available options:

```bash
celeba-dataset
afhq-cat-dataset
celebahq-dataset
pretrained-network-celeba
pretrained-network-afhq-cat
pretrained-network-celebahq
deblur_model
```

---

### Output Structure

* **CelebA**

  * Dataset: `data/celeba/`
  * Model: `model/celeba/ot/model_final.pt`

* **AFHQ-Cat**

  * Dataset: `data/afhq_cat/`
  * Model: `model/afhq_cat/ot/model_final.pt`

---

## Solving Inverse Problems

This project supports the following inverse problems:

* **Denoising** - `denoising`
* **Gaussian Deblurring** - `gaussian_deblurring_FFT`
* **Super-resolution** - `superresolution`
* **Box Inpainting** - `inpainting`
* **Random Inpainting** - `random_inpainting`
* **Paintbrush Inpainting** - `paintbrush_inpainting`

---

### Available Methods

To run a specific method, you must pass its name using the `method` argument.

* `pnp_cf` — **Proposed method**
* `pnp_flow_fbs`
* `ot_ode`
* `d_flow`
* `flow_priors`

---

### Example Run

```bash
python main.py --opts \
    dataset celeba \
    train False \
    eval True \
    problem inpainting \
    method pnp_cf
```

---

You can replace `dataset celeba` with:

* `dataset afhq_cat` (for AFHQ-Cat)

Replace `problem` and `method` with the desired options from the lists above.

---

## Hyperparameters (PnP-ADMM)

The key hyperparameters for the `pnp_cf` method are:

* `steps_pnp` — Number of ADMM iterations (AFHQ-Cat: 250 and CelebA: 200)
* `rho_admm` — ADMM penalty parameter controlling data–prior balance (AFHQ-Cat: 500 and CelebA: 1000)
* `max_cgiter` — Maximum number of conjugate gradient iterations (default: 10)
* `cg_tol` — Tolerance for the CG solver (default: 1e-5)
* `save_norms_admm` — Store ADMM residual norms (default: False)

### Projection Settings

* `adaptive_noise=True` — **Adaptive projection (recommended)**
* `adaptive_noise=False`, `deterministic_noise=True` — Deterministic projections
* `adaptive_noise=False`, `deterministic_noise=False` — Stochastic projections

---

## Evaluation

We provide a script to reproduce the main results, including **quantitative evaluation, runtime, and memory measurements**.

Experiments are run across:

* Datasets: `celeba`, `afhq_cat`
* Inverse problems: denoising, deblurring, super-resolution, inpainting, random-inpainting and paintbrush-inpainting
* Methods: OT-ODE, Flow Prior, D-Flow, PNP-Flow-FBS, PNP-Flow-ADMM

Each run processes batches of size 4 and evaluates performance over 25 batches.

---

### Run All Experiments

```bash
bash scripts/main_results.sh
```

---

### Experiment Details

* Random starting batch per run
* Evaluation over **25 batches**, each batch containing 4 images
* Runtime and GPU memory recorded
* Results averaged across batches
* Same setup across all methods
* Reported time/memory metrics are computed using this script

---

### Visualization and Analysis

We provide the jupyter notebook `graphs.ipynb` for visualizing results, plotting graphs, and summarizing metrics 
in tabular form.

The notebook can be used to:

* Plot reconstruction results
* Compare methods across datasets and inverse problems
* Generate tables for quantitative evaluation

Ensure that the evaluation scripts have been run beforehand so that results are available in the `results/` directory.

---

### Appendix Results

Appendix results can be reproduced using `scripts/ablation_results.sh`.

---

<!-- ## Project Structure

```
├── config/          # Configuration files
├── data/            # Datasets
├── model/           # Pretrained and trained models
├── results/         # Output results
├── scripts/         # Bash scripts for experiments
├── main.py          # Main entry point
```

--- -->

<!-- ## License

This project is licensed under the **BSD 3-Clause License**.
This repository builds upon the PnP-Flow codebase.
See the `LICENSE` file for details.

--- -->

## Acknowledgements

This repository builds upon the following works:

* PnP-Flow (ICLR 2025)
* GSPnP
* DeepInv
* Rectified Flow

We thank the authors for making their code publicly available.
