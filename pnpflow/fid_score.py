import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from pnpflow.models import InceptionV3


def get_activations(images, model, batch_size=50, dims=2048, device='cpu'):
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
    act = get_activations(images, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1.0):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (gamma * X.dot(Y.T) + coef0) ** degree


def _mmd2_unbiased(K_XX, K_YY, K_XY):
    m = K_XX.shape[0]
    n = K_YY.shape[0]

    sum_XX = (K_XX.sum() - np.trace(K_XX)) / (m * (m - 1))
    sum_YY = (K_YY.sum() - np.trace(K_YY)) / (n * (n - 1))
    sum_XY = K_XY.sum() / (m * n)

    return sum_XX + sum_YY - 2 * sum_XY


def calculate_kid(act1, act2, subset_size=1000, num_subsets=100, degree=3,
                   gamma=None, coef0=1.0, seed=0):
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