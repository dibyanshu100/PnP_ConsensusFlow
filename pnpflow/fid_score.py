"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

Code apapted from https://github.com/mseitzer/pytorch-fid
"""

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from pnpflow.models import InceptionV3


def get_activations(images, model, batch_size=50, dims=2048, device='cpu'):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Images
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(images):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(images)

    dl = torch.utils.data.DataLoader(images, batch_size=batch_size,
                                     drop_last=False)

    pred_arr = np.empty((len(images), dims))

    start_idx = 0

    for batch in dl:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(
        images,
        model,
        batch_size=50,
        dims=2048,
        device='cpu'):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Images
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1.0):
    """(gamma * X @ Y^T + coef0) ** degree -- the kernel used by the
    standard KID estimator (Binkowski et al., 2018)."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (gamma * X.dot(Y.T) + coef0) ** degree


def _mmd2_unbiased(K_XX, K_YY, K_XY):
    """Unbiased estimator of the squared Maximum Mean Discrepancy from
    precomputed kernel (Gram) matrices, excluding the diagonal terms of
    K_XX/K_YY as is standard for the unbiased estimator."""
    m = K_XX.shape[0]
    n = K_YY.shape[0]

    sum_XX = (K_XX.sum() - np.trace(K_XX)) / (m * (m - 1))
    sum_YY = (K_YY.sum() - np.trace(K_YY)) / (n * (n - 1))
    sum_XY = K_XY.sum() / (m * n)

    return sum_XX + sum_YY - 2 * sum_XY


def calculate_kid(act1, act2, subset_size=1000, num_subsets=100, degree=3,
                   gamma=None, coef0=1.0, seed=0):
    """Kernel Inception Distance (Binkowski et al., 2018), computed on
    precomputed Inception activations. Unlike FID, KID makes no Gaussian
    assumption on the activations and is estimated by repeatedly
    subsampling both activation sets and averaging the unbiased
    polynomial-kernel MMD^2 over the subsets.

    Params:
    -- act1/act2   : (N, dims) arrays of Inception activations for the two
                     distributions being compared (order doesn't matter).
    -- subset_size : Size of each random subset. Capped to the smaller of
                     len(act1), len(act2) if either has fewer samples.
    -- num_subsets : Number of subsets to average over.
    -- degree, gamma, coef0 : Polynomial kernel hyperparameters.

    Returns:
    -- kid_mean, kid_std : Mean and standard deviation of the per-subset
       unbiased MMD^2 estimates.
    """
    rng = np.random.RandomState(seed)
    n = min(len(act1), len(act2), subset_size)
    if n < 2:
        raise ValueError(
            'calculate_kid needs at least 2 samples per subset; got '
            f'{len(act1)} and {len(act2)} activations available.')

    mmd2_estimates = np.empty(num_subsets)
    for i in range(num_subsets):
        idx1 = rng.choice(len(act1), n, replace=False)
        idx2 = rng.choice(len(act2), n, replace=False)
        x = act1[idx1]
        y = act2[idx2]

        K_XX = _polynomial_kernel(x, x, degree, gamma, coef0)
        K_YY = _polynomial_kernel(y, y, degree, gamma, coef0)
        K_XY = _polynomial_kernel(x, y, degree, gamma, coef0)

        mmd2_estimates[i] = _mmd2_unbiased(K_XX, K_YY, K_XY)

    return float(mmd2_estimates.mean()), float(mmd2_estimates.std())


def evaluate_fid_kid(images1, images2, batch_size=50, subset_size=1000,
                      num_subsets=100):
    """Calculation of both FID and KID from two sets of images, reusing a
    single pass of Inception activations for both metrics.

    Params:
    -- images1/2   : Images of shape (N, H, W, C), values in [0, 1].
    -- batch_size  : Batch size for the Inception forward passes.
    -- subset_size, num_subsets : See calculate_kid.

    Returns:
    -- fid_value           : The FID score.
    -- (kid_mean, kid_std) : The KID score (mean and std over subsets).
    """
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)

    if images1.shape[-1] == 1:
        images1 = np.concatenate([images1, images1, images1], axis=-1)
    if images2.shape[-1] == 1:
        images2 = np.concatenate([images2, images2, images2], axis=-1)

    if np.max(images1) > 1 or np.min(images1) < 0 or np.max(
            images2) > 1 or np.min(images2) < 0:
        import warnings
        warnings.warn(
            'FID/KID score: the values of images should be in range [0,1].')

    images1 = np.transpose(images1, axes=(0, 3, 1, 2))
    images2 = np.transpose(images2, axes=(0, 3, 1, 2))

    act1 = get_activations(images1, model, batch_size, 2048, device)
    act2 = get_activations(images2, model, batch_size, 2048, device)

    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    kid_mean, kid_std = calculate_kid(
        act1, act2, subset_size=subset_size, num_subsets=num_subsets)

    return fid_value, (kid_mean, kid_std)


def evaluate_fid_score(images1, images2, batch_size=50):
    """Calculation of FID.
    Params:
    -- images1/2   : Images of shape (N, H, W, C)
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.

    Returns:
    -- fid_value   : The FID score.
    """

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

    model = InceptionV3([block_idx]).to(device)

    if images1.shape[-1] == 1:
        images1 = np.concatenate([images1, images1, images1], axis=-1)
    if images2.shape[-1] == 1:
        images2 = np.concatenate([images2, images2, images2], axis=-1)

    if np.max(images1) > 1 or np.min(images1) < 0 or np.max(
            images2) > 1 or np.min(images2) < 0:
        import warnings
        warnings.warn(
            'FID score: the values of images should be in range [0,1].')

    images1 = np.transpose(images1, axes=(0, 3, 1, 2))
    images2 = np.transpose(images2, axes=(0, 3, 1, 2))

    m1, s1 = calculate_activation_statistics(
        images1, model, batch_size, 2048, device)
    m2, s2 = calculate_activation_statistics(
        images2, model, batch_size, 2048, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value