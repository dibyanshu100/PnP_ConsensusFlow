import os
import sys
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["WORLD_SIZE"] = "1"

pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--gpu", type=str, default="0")
pre_args, _ = pre_parser.parse_known_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = pre_args.gpu
os.environ["WORLD_SIZE"] = "1"
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from pnpflow.utils import load_cfg_from_cfg_file, merge_cfg_from_list
from pnpflow.degradations import *
from pnpflow.dataloaders import DataLoaders
from pnpflow.train_flow_matching import FLOW_MATCHING
from pnpflow.train_denoiser import GRADIENT_STEP_DENOISER
from pnpflow.compute_metric import ComputeMetric
from pnpflow.methods.pnp_cf import PNP_CF
from pnpflow.methods.pnp_flow_fbs import PNP_FLOW_FBS
from pnpflow.methods.d_flow import D_FLOW
from pnpflow.methods.ot_ode import OT_ODE
from pnpflow.methods.flow_priors import FLOW_PRIORS
from pnpflow.methods.flow_dps import FLOW_DPS
from pnpflow.methods.flower import FLOWER
from pnpflow.utils import gaussian_blur, define_model, load_model
import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["WORLD_SIZE"] = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')
    cfg = load_cfg_from_cfg_file('./' + 'config/main_config.yaml')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    dataset_config = cfg.root + \
        'config/dataset_config/{}.yaml'.format(
            cfg.dataset)
    cfg.update(load_cfg_from_cfg_file(dataset_config))

    method_config_file = cfg.root + \
        'config/method_config/{}.yaml'.format(
            cfg.method)
    cfg.update(load_cfg_from_cfg_file(method_config_file))

    if args.opts is not None:
        # override config with command line input
        cfg = merge_cfg_from_list(cfg, args.opts)

    # for all keys in the method config file, create a dictionary {key: value} in the cfg object cfg.dict_cfg_method
    method_cfg = load_cfg_from_cfg_file(method_config_file)
    cfg.dict_cfg_method = {}
    for key in method_cfg.keys():
        cfg.dict_cfg_method[key] = cfg[key]
    return cfg


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    if torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("Logical GPU index   =", torch.cuda.current_device())
        print("GPU name            =", torch.cuda.get_device_name(torch.cuda.current_device()))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    (model, state) = define_model(args)

    if args.train:
        args.batch_size = args.batch_size_train
        print('Training...')
        data_loaders = DataLoaders(
            args.dataset, args.batch_size_train, args.batch_size_train,
            celebahq_train_ids=getattr(args, 'celebahq_train_ids', None),
            celebahq_test_ids=getattr(args, 'celebahq_test_ids', None),
            dim_image=getattr(args, 'dim_image', None)).load_data()
        if args.model == "ot":
            generative_method = FLOW_MATCHING(model, device, args)
        elif args.model == "gradient_step":
            generative_method = GRADIENT_STEP_DENOISER(model, device, args)
        else:
            raise ValueError(
                "Model not implemented yet: you can choose between 'ot' and 'gradient_step'")
        generative_method.train(data_loaders)
        print('Training done!')

    if args.eval:

        if args.problem != "segmentation":
            if args.model == "ot" or args.model == "gradient_step":
                model_path = args.root + \
                    'model/{}/{}/model_final.pt'.format(
                        args.dataset, args.model)
                load_model(args.model, model, state, download=False,
                           checkpoint_path=model_path, dataset=None,  device=device)
                model.eval()

            elif args.model == "rectified":
                model_path = args.root + 'model/{}/{}/model_final.pth'.format(
                    args.dataset, args.model)
                load_model(args.model, model, state, download=False,
                           checkpoint_path=model_path, dataset=None, device=device)
                model.eval()

            elif args.model == "diffusion":
                model.eval()

        if args.model == "gradient_step":
            generative_method = GRADIENT_STEP_DENOISER(model, device, args)
        else:
            generative_method = FLOW_MATCHING(model, device, args)

        if args.compute_metrics:
            print('Computing metrics...')
            data_loaders = DataLoaders(args.dataset, 5000, 5000).load_data()
            metric = ComputeMetric(
                data_loaders, generative_method, device, args)
            metric.compute_metrics(5000)
            print('Computing metrics done!')

        if args.problem == "denoising":
            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.3
            degradation = Denoising()

        elif args.problem == "inpainting":
            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.05
            if hasattr(args, 'box_mask_half_size'):
                half_size_mask = int(args.box_mask_half_size)
            elif args.dim_image == 128:
                half_size_mask = 30   #30 #20
            elif args.dim_image == 256:
                half_size_mask = 60   #60 #40
            degradation = BoxInpainting(half_size_mask)

        elif args.problem == "paintbrush_inpainting":
            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.05
            degradation = PaintbrushInpainting()

        elif args.problem == "random_inpainting":
            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.01
            p = float(getattr(args, 'random_inpainting_p', 0.85))     # 0.7 0.85
            degradation = RandomInpainting(p)

        elif args.problem == "superresolution":
            if hasattr(args, 'sr_scale_factor'):
                sf = int(args.sr_scale_factor)
            elif args.dim_image == 128:
                sf = 4  #4 #2
            elif args.dim_image == 256:
                sf = 8  #8 #4
            print('Superresolution with scale factor {}'.format(sf))
            if args.noise_type == 'laplace':
                sigma_noise = 0.3

            elif args.noise_type == 'gaussian':
                sigma_noise = 0.05
            degradation = Superresolution(sf, args.dim_image)

        elif args.problem == "gaussian_deblurring_FFT":
            if args.dim_image == 128:
                sigma_blur = 1.0
            elif args.dim_image == 256:
                sigma_blur = 3.0

            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.05
            kernel_size = 61
            degradation = GaussianDeblurring(
                sigma_blur, kernel_size, "fft", args.num_channels, args.dim_image, device)

        elif args.problem == "nonlinear_deblurring":
            if args.noise_type == 'gaussian':
                sigma_noise = 0.05
            else:
                raise ValueError(
                    'Nonlinear deblurring currently only supports gaussian noise')
            degradation = NonlinearBlur(
                num_channels=args.num_channels, dim_image=args.dim_image,
                checkpoint_path=getattr(args, 'nonlinear_blur_checkpoint', 'model/deblur_model/GOPRO_wVAE.pth'),
                device=device)

        elif args.problem == "phase_retrieval":
            if args.noise_type == 'gaussian':
                sigma_noise = 0.01  
            else:
                raise ValueError(
                    'Phase retrieval currently only supports gaussian noise')
            degradation = PhaseRetrieval(
                oversample=getattr(args, 'pr_oversample', 2.0),
                dim_image=args.dim_image, num_channels=args.num_channels,
                device=device)
            if not hasattr(args, 'num_restarts'):
                args.num_restarts = 2

        elif args.problem == "hdr_reconstruction":
            if args.noise_type == 'gaussian':
                sigma_noise = 0.05
            else:
                raise ValueError(
                    'HDR reconstruction currently only supports gaussian noise')
            degradation = HDRReconstruction(
                scale=getattr(args, 'hdr_scale', 2.0))

        elif args.problem == "segmentation":
            if args.dataset != 'celebahq':
                raise ValueError(
                    "Segmentation requires dataset celebahq")
            if args.dim_image != 128:
                raise ValueError(
                    'Segmentation currently only supports dim_image=128 (CelebAMask-HQ)')
            sigma_noise = 0.0  # unused: segmentation's data term is CE, not Gaussian
            seg_checkpoint = getattr(args, 'segmentation_checkpoint', 'model/celeba_hq/segmentation/celebahq_segmentator.pt')
            degradation = Segmentation(
                checkpoint_path=seg_checkpoint,
                num_classes=getattr(args, 'seg_num_classes', 19),
                dim_image=args.dim_image,
                base_channels=getattr(args, 'seg_base_channels', 64),
                device=device)

            prior_checkpoint = getattr(args, 'segmentation_prior_checkpoint', 'model/celeba_hq/segmentation/prior_model_final_checkpoint.pt')
            from pnpflow.nonlinear_ops.segmentation_task.prior_net import UNet as _SegPriorUNet
            model = _SegPriorUNet(
                input_channels=3, input_height=args.dim_image, ch=32,
                ch_mult=(1, 2, 4, 8), num_res_blocks=6, attn_resolutions=(16, 8),
                resamp_with_conv=True).to(device)
            _ckpt = torch.load(prior_checkpoint, map_location=device)
            if isinstance(_ckpt, dict) and "ema" in _ckpt:
                model.load_state_dict(_ckpt["ema"])
            elif isinstance(_ckpt, dict) and "model" in _ckpt:
                model.load_state_dict(_ckpt["model"])
            else:
                model.load_state_dict(_ckpt)
            model.eval()
            if args.model != "ot":
                raise ValueError(
                    "Segmentation's prior uses the 'ot' calling convention "
                    "(model(x, t)) -- run with --opts model ot")

        else:
            raise ValueError(
                "Unknown problem '{}'".format(args.problem))

        print('Solving the {} inverse problem with the method {}...'.format(
            args.problem, args.method))
        print('sigma_noise', sigma_noise)
        data_loaders = DataLoaders(
            args.dataset, args.batch_size_ip, args.batch_size_ip,
            celebahq_train_ids=getattr(args, 'celebahq_train_ids', None),
            celebahq_test_ids=getattr(args, 'celebahq_test_ids', 'pnpflow/nonlinear_ops/segmentation_task/splits/test_ids.txt'),
            dim_image=getattr(args, 'dim_image', None)).load_data()
        if args.noise_type == 'laplace':
            args.save_path = os.path.join(
                args.root, 'results_laplace', args.dataset, args.model, args.problem, args.method, args.eval_split)
        elif args.noise_type == 'gaussian':
            args.save_path = os.path.join(
                args.root, 'results', args.dataset, args.model, args.problem, args.method, args.eval_split)
        try:
            os.makedirs(args.save_path)
        except FileExistsError:
            pass
        
        if args.method == 'pnp_cf':
            method = PNP_CF(model, device, args)
        elif args.method == 'pnp_flow_fbs':
            method = PNP_FLOW_FBS(model, device, args)
        elif args.method == 'd_flow':
            method = D_FLOW(model, device, args)
        elif args.method == 'ot_ode':
            method = OT_ODE(model, device, args)
        elif args.method == 'flow_priors':
            method = FLOW_PRIORS(model, device, args)
        elif args.method == 'flow_dps':
            method = FLOW_DPS(model, device, args)
        elif args.method == 'flower':
            method = FLOWER(model, device, args)

        else:
            raise ValueError("The method your entered does not exist")

        if not degradation.is_linear and args.method in ('ot_ode', 'flow_priors', 'flower'):
            raise ValueError(
                f"Method '{args.method}' is only implemented for linear forward "
                f"operators and cannot be run on problem '{args.problem}'. "
                f"Use pnp_cf, pnp_flow_fbs,d_flow or flow_dps instead.")

        if getattr(degradation, 'is_segmentation', False) and args.method not in (
                'pnp_cf', 'pnp_flow_fbs', 'd_flow', 'flow_dps'):
            raise ValueError(
                "Segmentation is currently only implemented for methods "
                "'pnp_cf', 'pnp_flow_fbs', 'd_flow', and 'flow_dps'.")

        method.run_method(data_loaders, degradation, sigma_noise)

# python main.py --opts dataset afhq_cat train False eval True problem denoising start_batch 4 method pnp_cf
# python main.py --opts dataset afhq_cat train False eval True problem hdr_reconstruction start_batch 4 method pnp_cf
if __name__ == "__main__":
    main()