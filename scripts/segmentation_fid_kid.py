import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pnpflow.fid_score import evaluate_fid_kid


def load_images_as_array(image_dir, dim_image, ids_file=None, max_images=None):
    all_files = sorted(os.listdir(image_dir))

    if ids_file is not None:
        with open(ids_file) as f:
            allowed_ids = {line.strip() for line in f if line.strip()}
        files = [fn for fn in all_files
                 if os.path.splitext(fn)[0] in allowed_ids]
        if len(files) == 0:
            raise ValueError(
                f"No files in '{image_dir}' matched any id in '{ids_file}'.")
    else:
        files = all_files

    if max_images is not None:
        files = files[:max_images]

    transform = transforms.Compose([
        transforms.Resize((dim_image, dim_image)),
        transforms.ToTensor(),  # -> float32 in [0, 1]
    ])

    imgs = np.empty((len(files), dim_image, dim_image, 3), dtype=np.float32)
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(image_dir, fn)).convert('RGB')
        imgs[i] = transform(img).permute(1, 2, 0).numpy()

    return imgs


def compute_paired_lpips(recon_dir, real_dir, dim_image, net='alex',
                          batch_size=50):
    import lpips

    recon_files = sorted(os.listdir(recon_dir))
    real_files = sorted(os.listdir(real_dir))
    if recon_files != real_files:
        missing_in_real = set(recon_files) - set(real_files)
        missing_in_recon = set(real_files) - set(recon_files)
        raise ValueError(
            "recon_dir and real_dir don't contain the same filenames -- "
            "LPIPS pairing would be wrong. "
            f"In recon but not real: {sorted(missing_in_real)[:5]}... "
            f"In real but not recon: {sorted(missing_in_recon)[:5]}...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = lpips.LPIPS(net=net).to(device)

    transform = transforms.Compose([
        transforms.Resize((dim_image, dim_image)),
        transforms.ToTensor(),  # -> float32 in [0, 1]
    ])

    per_image_lpips = []
    with torch.no_grad():
        for start in range(0, len(recon_files), batch_size):
            batch_files = recon_files[start:start + batch_size]
            recon_batch = torch.stack([
                transform(Image.open(os.path.join(recon_dir, fn)).convert('RGB'))
                for fn in batch_files]).to(device)
            real_batch = torch.stack([
                transform(Image.open(os.path.join(real_dir, fn)).convert('RGB'))
                for fn in batch_files]).to(device)

            # loss_fn expects [0, 1] input when normalize=True (it does the
            # [-1, 1] rescaling internally).
            dists = loss_fn(real_batch, recon_batch, normalize=True)
            per_image_lpips.extend(dists.flatten().cpu().tolist())

    per_image_lpips = np.array(per_image_lpips)
    return float(per_image_lpips.mean()), float(per_image_lpips.std())


def main():
    parser = argparse.ArgumentParser(
        description='Compute FID/KID/LPIPS for segmentation-task reconstructions.')
    parser.add_argument(
        '--recon_dir', type=str, required=True,
        help="Directory of reconstructed test images, e.g. "
             "'<save_path_ip>/fid_kid_images/recon' (produced by "
             "utils.save_recon_for_fid during solve_ip_segmentation).")
    parser.add_argument(
        '--real_data_dir', type=str, default='data/celebahq/test',
        help='Directory containing the full CelebAMask-HQ image set '
             '(train + test), used as the FID/KID real/reference distribution.')
    parser.add_argument(
        '--real_ids_file', type=str, default=None,
        help='Optional id-list file to restrict the FID/KID real reference '
             'set (e.g. only test_ids.txt). Default: use every image in '
             'real_data_dir (train + test), per FID/KID best practice.')
    parser.add_argument(
        '--paired_real_dir', type=str, default=None,
        help="Directory of ground-truth images paired 1:1 with recon_dir "
             "for LPIPS. Default: '<recon_dir>/../real' (the sibling "
             "folder utils.save_recon_for_fid writes automatically).")
    parser.add_argument('--dim_image', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--subset_size', type=int, default=1000,
                         help='KID subset size (capped to the smaller of '
                              'the two sample counts if needed).')
    parser.add_argument('--num_subsets', type=int, default=100)
    parser.add_argument('--lpips_net', type=str, default='alex',
                         choices=['alex', 'vgg', 'squeeze'])
    parser.add_argument('--skip_lpips', action='store_true',
                         help='Skip LPIPS and only compute FID/KID.')
    parser.add_argument('--output', type=str, default=None,
                         help='Where to write the results. Defaults to '
                              '<recon_dir>/../fid_kid.txt')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.normpath(args.recon_dir)), 'fid_kid.txt')
    if args.paired_real_dir is None:
        args.paired_real_dir = os.path.join(
            os.path.dirname(os.path.normpath(args.recon_dir)), 'real')

    print(f'Loading reconstructed images from {args.recon_dir} ...')
    recon_imgs = load_images_as_array(args.recon_dir, args.dim_image)
    print(f'  -> {recon_imgs.shape[0]} images')

    print(f'Loading real reference images from {args.real_data_dir} '
          f'{"(filtered by " + args.real_ids_file + ")" if args.real_ids_file else "(full dataset)"} ...')
    real_imgs = load_images_as_array(
        args.real_data_dir, args.dim_image, ids_file=args.real_ids_file)
    print(f'  -> {real_imgs.shape[0]} images')

    print('Computing FID/KID ...')
    fid_value, (kid_mean, kid_std) = evaluate_fid_kid(
        real_imgs, recon_imgs, batch_size=args.batch_size,
        subset_size=args.subset_size, num_subsets=args.num_subsets)

    result = (f'FID: {fid_value:.4f}\n'
              f'KID: {kid_mean:.6f} +/- {kid_std:.6f}\n'
              f'  (num_real={real_imgs.shape[0]}, num_recon={recon_imgs.shape[0]}, '
              f'subset_size={min(args.subset_size, real_imgs.shape[0], recon_imgs.shape[0])}, '
              f'num_subsets={args.num_subsets})\n')

    if not args.skip_lpips:
        print(f'Computing paired LPIPS ({args.lpips_net}) against '
              f'{args.paired_real_dir} ...')
        lpips_mean, lpips_std = compute_paired_lpips(
            args.recon_dir, args.paired_real_dir, args.dim_image,
            net=args.lpips_net, batch_size=args.batch_size)
        result += f'LPIPS ({args.lpips_net}): {lpips_mean:.4f} +/- {lpips_std:.4f}\n'

    print(result)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(result)
    print(f'Wrote results to {args.output}')


if __name__ == '__main__':
    main()