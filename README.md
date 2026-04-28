# PnP-Flow-ADMM: Plug-and-Play Flow Matching with ADMM

This repository contains the implementation of **PnP-Flow-ADMM**, a method for solving inverse problems using **Plug-and-Play (PnP) priors with flow matching models**, optimized via **ADMM (Alternating Direction Method of Multipliers)**.

Our approach extends the PnP-Flow framework (ICLR 2025), which uses Forward-Backward Splitting (FBS), by introducing:

* **ADMM-based optimization** for improved convergence stability
* **Reduced computational cost** via fewer denoiser evaluations
* **Adaptive projection strategies** for improved reconstruction quality

---

## Overview

Plug-and-Play (PnP) methods solve inverse problems by combining:

* a **data-fidelity term** (forward model)
* a **learned prior** (denoiser / generative model)

PnP-Flow integrates **flow matching models** as priors.
In this work, we replace the FBS scheme with an **ADMM formulation**, leading to:

* better optimization behavior
* improved performance across inverse problems
* efficient use of pretrained flow models

<p align="center">
  <img src="assets/teaser.png" width="800"/>
</p>

---
## 1.1 Requirements
* Python ≥ 3.9
* torch ≥ 1.13.1
* torchvision
* numpy
* scipy
* tqdm
* pandas
* pyyaml
* torchdiffeq
* deepinv
---

## 1.2 Download Datasets and Pretrained Models

We provide a `download.sh` script to download the datasets and pretrained flow-matching checkpoints used in our experiments. The files are downloaded from Google Drive using `gdown`.

First install `gdown` if needed:

```bash
pip install gdown
```

### CelebA

To download and prepare the CelebA dataset:

```bash
bash download.sh celeba-dataset
```

This will download the dataset, extract all images, and create an 80/10/10 train/validation/test split.

The final directory structure will be:

```text
data/celeba/
├── img_align_celeba/
├── train/
├── val/
├── test/
└── list_eval_partition.csv
```

To download the pretrained flow-matching model for CelebA:

```bash
bash download.sh pretrained-network-celeba
```

The checkpoint will be stored at:

```text
model/celeba/ot/model_final.pt
```

### AFHQ-Cat

To download and prepare the AFHQ-Cat dataset:

```bash
bash download.sh afhq-cat-dataset
```

The dataset will be extracted into:

```text
data/afhq_cat/
```

To download the pretrained flow-matching model for AFHQ-Cat:

```bash
bash download.sh pretrained-network-afhq-cat
```

The checkpoint will be stored at:

```text
model/afhq_cat/ot/model_final.pt
```

### Available Download Arguments

The download script supports the following arguments:

```bash
bash download.sh celeba-dataset
bash download.sh afhq-cat-dataset
bash download.sh pretrained-network-celeba
bash download.sh pretrained-network-afhq-cat
```


## 2. Training Flow Matching Models

You can train your own flow matching model using:

```bash
python main.py --opts dataset celeba train True eval False batch_size 128 num_epoch 100
```

Models are saved in:

```
./model/<dataset>/<prior_type>/ot/
```

Generated samples are saved in:

```
./results/<dataset>/<prior_type>/ot/
```

---

## 3. Solving Inverse Problems

This project support the following inverse problems:

* **Denoising**
* **Gaussian Deblurring**
* **Super-resolution**
* **Inpainting**

  * Box inpainting
  * Random inpainting
  * Free-form (paintbrush) inpainting

---

### Available Methods

* `pnp_flow_admm` — **Proposed method (ADMM-based)**
* `pnp_flow_fbs` — PnP-Flow (FBS-based)
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
    method pnp_flow_admm
```

---

### 3.1 Hyperparameter Tuning

To find optimal parameters:

```bash
bash scripts/script_val.sh
```

Important parameters for ADMM:

* `rho` — penalty parameter
* `num_steps` — number of iterations
* `adaptive` — enable adaptive projections

---

### 3.2 Evaluation

```bash
bash scripts/script_test.sh
```

Results are saved in:

```
./results/<dataset>/<task>/
```

---

## Computational Complexity

Compared to FBS-based PnP-Flow:

* FBS requires **multiple denoiser calls per iteration**
* ADMM uses **a single denoiser call per iteration**

This leads to:

* Reduced computational cost
* Faster runtime in practice
* Improved scalability

---

## Key Contributions

* ADMM-based Plug-and-Play framework for flow matching priors
* Improved convergence compared to FBS
* Reduced number of denoiser evaluations
* Adaptive projection strategy for enhanced reconstruction
* Extensive evaluation across inverse problems

---

## Project Structure

```
├── config/          # Configuration files
├── data/            # Datasets
├── model/           # Pretrained and trained models
├── results/         # Output results
├── scripts/         # Bash scripts for experiments
├── main.py          # Main entry point
```

---

## License

This project is licensed under the **BSD 3-Clause License**.

This repository builds upon the PnP-Flow codebase.
Please refer to the `LICENSE` file for details.

---

## Acknowledgements

This repository builds upon the following works:

* PnP-Flow (ICLR 2025)
* GSPnP
* DeepInv library
* Rectified Flow

We thank the authors for making their code publicly available.
