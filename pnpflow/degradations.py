import math
import torch
import torch.nn.functional as F
from pnpflow.utils import square_mask, random_mask, paintbrush_mask, gaussian_blur, gaussian_2d_kernel, downsample, upsample, bicubic_filter, create_downsampling_matrix


class Degradation:

    # Whether H is linear. Linear operators get an exact adjoint (H_adj) used
    # both for gradients/data-consistency and for display/initialization.
    # Nonlinear operators (see below) override H_adj with a display/init-only
    # stand-in and rely on `vjp` (vector-Jacobian product) for gradients.
    is_linear = True

    def H(self, x):
        raise NotImplementedError()

    def H_adj(self, x):
        raise NotImplementedError()

    def vjp(self, x, v):
        """Returns v^T J_H(x), the vector-Jacobian product of H at x.

        For linear H this is exactly H_adj(v) (cheap, exact). For nonlinear H
        it is computed via autograd, so it is correct regardless of the outer
        `torch.no_grad()` context the PnP/ADMM loops run in.
        """
        if self.is_linear:
            return self.H_adj(v)
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            out = self.H(x)
            grad = torch.autograd.grad(out, x, grad_outputs=v)[0]
        return grad.detach()


class Denoising(Degradation):
    def H(self, x):
        return x

    def H_adj(self, x):
        return x


class BoxInpainting(Degradation):
    def __init__(self, half_size_mask):
        super().__init__()
        self.half_size_mask = half_size_mask

    def H(self, x):
        return square_mask(x, self.half_size_mask)

    def H_adj(self, x):
        return square_mask(x, self.half_size_mask)


class RandomInpainting(Degradation):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def H(self, x):
        return random_mask(x, self.p)

    def H_adj(self, x):
        return random_mask(x, self.p)


class PaintbrushInpainting(Degradation):
    def H(self, x):
        return paintbrush_mask(x)

    def H_adj(self, x):
        return paintbrush_mask(x)


class GaussianDeblurring(Degradation):
    def __init__(self, sigma_blur, kernel_size,  mode="fft", num_channels=3, dim_image=128, device="cuda") -> None:
        super().__init__()
        self.mode = mode
        self.sigma = sigma_blur
        self.kernel_size = kernel_size
        self.kernel = gaussian_2d_kernel(sigma_blur, kernel_size).to(device)
        filter = torch.zeros(
            (1, num_channels) + (dim_image, dim_image), device=device
        )

        filter[..., : kernel_size, : kernel_size] = self.kernel
        self.filter = torch.roll(
            filter, shifts=(-(kernel_size-1)//2, -(kernel_size-1)//2), dims=(2, 3))
        self.device = device

    def H(self, x):
        if self.mode != "fft":
            kernel = self.kernel.view(
                1, 1, self.kernel_size,  self.kernel_size)
            kernel = self.kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, stride=1, padding='same', groups=x.shape[1])
        else:
            return torch.real(torch.fft.ifft2(
                torch.fft.fft2(x.to(self.device)) * torch.fft.fft2(self.filter)))

    def H_adj(self, x):
        if self.mode != "fft":
            kernel = self.kernel.view(
                1, 1, self.kernel_size,  self.kernel_size)
            kernel = self.kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, stride=1, padding='same', groups=x.shape[1])
        else:
            return torch.real(torch.fft.ifft2(
                torch.fft.fft2(x.to(self.device)) * torch.conj(torch.fft.fft2(self.filter))))


class Superresolution(Degradation):
    def __init__(self, sf, dim_image, mode=None, device="cuda") -> None:
        super().__init__()
        self.sf = sf
        self.mode = mode
        if mode == "bicubic":
            self.filter = torch.nn.Parameter(
                bicubic_filter(sf), requires_grad=False
            ).to(device)
            # Move batch dim of the input into channels

            filter = torch.zeros(
                (1, 3) + (dim_image, dim_image), device=device)

            filter[..., : self.filter.shape[-1],
                   : self.filter.shape[-1]] = self.filter
            self.filter = torch.roll(
                filter, shifts=(-(self.filter.shape[-1]-1)//2, -(self.filter.shape[-1]-1)//2), dims=(2, 3))
        self.downsampling_matrix = create_downsampling_matrix(
            dim_image, dim_image, sf, device)

    def H(self, x):

        if self.mode == None:
            return downsample(x, self.sf)
        elif self.mode == "bicubic":
            x_ = torch.real(torch.fft.ifft2(
                torch.fft.fft2(x) * torch.fft.fft2(self.filter)))
            return downsample(x_, self.sf)

    def H_adj(self, x):
        if self.mode == None:
            return upsample(x, self.sf)
        elif self.mode == "bicubic":
            x_ = upsample(x, self.sf)
            return torch.real(torch.fft.ifft2(torch.fft.fft2(x_) * torch.conj(torch.fft.fft2(self.filter))))


# Architecture hyperparameters for the "GOPRO wVAE" checkpoint, copied from
# BKSE's own options/generate_blur/default.yml (the exact config DPS's
# configs/nonlinear_deblur_config.yaml points at). Only used to build the
# KernelWizard module; unrelated to anything in the rest of this codebase.
_KERNEL_WIZARD_OPT = {
    "input_nc": 3,
    "nf": 64,
    "front_RBs": 10,
    "back_RBs": 20,
    "N_frames": 1,
    "kernel_dim": 512,
    "use_vae": True,
    "KernelExtractor": {
        "norm": "none",
        "use_sharp": True,
        "n_blocks": 4,
        "padding_type": "reflect",
        "use_dropout": False,
    },
    "Adapter": {
        "norm": "none",
        "use_dropout": False,
    },
}


def _load_kernel_wizard():
    """Imports the vendored BKSE KernelWizard class.

    KernelWizard's own source uses absolute imports (`import models.arch_util`),
    so the vendored `bkse/` directory -- not its parent -- must be on
    sys.path, exactly as required when using the official BKSE repo directly.
    """
    import os
    import sys
    bkse_root = os.path.join(os.path.dirname(__file__), "nonlinear_ops", "bkse")
    if bkse_root not in sys.path:
        sys.path.insert(0, bkse_root)
    from models.kernel_encoding.kernel_wizard import KernelWizard
    return KernelWizard


class NonlinearBlur(Degradation):
    """Nonlinear blur forward operator: KernelWizard's non-blind reblurring
    path (Tran et al., 2021, "Explore Image Deblurring via Encoded Blur
    Kernel Space", CVPR'21), used as-is by DPS (Chung et al., 2022) for their
    nonlinear-deblurring baseline.

    H(x) = KernelWizard.adaptKernel(x, kernel) for a kernel code sampled once
    at construction time (kept fixed thereafter, so H is a deterministic
    function of x -- the official DPS operator instead resamples the kernel
    on every call, which is fine for their single-shot diffusion guidance but
    would make H a moving target for the iterative solvers in this codebase).

    Requires the official pretrained weights ("GOPRO_wVAE.pth" from BKSE's
    model zoo, https://github.com/VinAIResearch/blur-kernel-space-exploring)
    passed via `checkpoint_path`; without it, this falls back to a randomly-
    initialized (frozen) KernelWizard so the pipeline is still runnable
    end-to-end, but a loud warning is printed since the resulting blur is not
    meaningful for anything you'd report.

    H is genuinely nonlinear (not a fixed convolution), so there is no exact
    adjoint. H_adj is only a display/initialization stand-in (identity, since
    input and output live in the same pixel space); actual gradients of the
    data-fidelity term go through `vjp` instead.
    """

    is_linear = False

    def __init__(self, num_channels=3, dim_image=128, checkpoint_path=None, device="cuda"):
        super().__init__()
        self.device = device
        KernelWizard = _load_kernel_wizard()
        opt = dict(_KERNEL_WIZARD_OPT)
        opt["input_nc"] = num_channels
        self.net = KernelWizard(opt).to(device)
        self.net.eval()

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=device)
            state = state.get('state_dict', state)
            # strict=True on purpose: with the real architecture in place, a
            # key mismatch means something is actually wrong (wrong
            # checkpoint, wrong opt dict) and should fail loudly rather than
            # silently falling back to partially-random weights.
            self.net.load_state_dict(state, strict=True)
            print(f"[NonlinearBlur] loaded official KernelWizard weights from {checkpoint_path}")
        else:
            print("[NonlinearBlur] WARNING: no checkpoint_path given -- running "
                  "KernelWizard with random weights. This is only useful for "
                  "smoke-testing the pipeline; pass the official GOPRO_wVAE.pth "
                  "(see BKSE's model zoo) for any result you'd actually report.")

        for p in self.net.parameters():
            p.requires_grad_(False)

        # Fixed kernel code (same scale as DPS's own random_kernel) so that H
        # is a deterministic function of x -- see class docstring. Its
        # spatial size must match KernelAdapter's innermost feature map: the
        # feature extractor downsamples by 4x, then the 5-level UNet in
        # KernelAdapter downsamples by a further 32x, i.e. total 128x. DPS
        # hardcodes this to 2x2 because they only ever use 256x256 images
        # (256/128=2); it must be recomputed for other image sizes (e.g. our
        # 128x128 celeba config gives 1x1) or KernelAdapter's internal
        # torch.cat of the kernel with the downsampled features shape-mismatches.
        kernel_spatial = max(dim_image // 128, 1)
        if dim_image % 128 != 0:
            raise ValueError(
                f"NonlinearBlur (KernelWizard) requires dim_image to be a "
                f"multiple of 128 (got {dim_image}); the feature extractor "
                f"and KernelAdapter together downsample by exactly 128x.")
        gen = torch.Generator(device=device).manual_seed(0)
        self.kernel = torch.randn(1, opt["kernel_dim"], kernel_spatial, kernel_spatial,
                                   device=device, generator=gen) * 1.2

    def H(self, x):
        x = x.to(self.device)
        kernel = self.kernel.expand(x.shape[0], -1, -1, -1)
        # KernelWizard was trained on images in [0, 1]; this codebase uses
        # [-1, 1] (see pnpflow/dataloaders.py's Normalize(0.5, 0.5)), so
        # convert in and out, matching DPS's own operator exactly.
        x01 = (x + 1.0) / 2.0
        out01 = self.net.adaptKernel(x01, kernel=kernel)
        return (out01 * 2.0 - 1.0).clamp(-1, 1)

    def H_adj(self, x):
        # Display/initialization stand-in only -- see class docstring.
        return x


class PhaseRetrieval(Degradation):
    """Fourier phase retrieval: y = |FFT(pad(x))|, as used e.g. in DPS
    (Chung et al., 2022). Padding (oversampling) the image before the FFT is
    the standard convention in the phase-retrieval literature (commonly a
    factor of 2) so that the magnitude-only measurement is informative enough
    for the (still highly non-convex) problem to be solvable.

    Uses an orthonormal FFT (norm='ortho'), matching DPS's own operator
    (their fft2_m -> fft2c_new(..., norm="ortho")) -- this is not optional:
    the unnormalized FFT scales |FFT(x)| up by the padded pixel count (tens
    of thousands for typical image sizes), which silently breaks every
    sigma_noise/rho/step-size default in this codebase, all of which assume
    H(x) and y live on the same [-1, 1] scale as the image.

    H is non-differentiable-invertible and has no adjoint. H_adj is only a
    display/initialization stand-in: it applies a random phase to the
    magnitude and inverts, then crops back to the image size -- the standard
    initialization used by classical phase-retrieval algorithms (e.g. HIO).
    """

    is_linear = False

    def __init__(self, oversample=2.0, dim_image=128, num_channels=3, device="cuda"):
        super().__init__()
        self.device = device
        self.dim_image = dim_image
        self.pad_size = int(dim_image * oversample)
        self.pad = (self.pad_size - dim_image) // 2

    def H(self, x):
        x = x.to(self.device)
        x_pad = F.pad(x, [self.pad, self.pad, self.pad, self.pad])
        return torch.abs(torch.fft.fft2(x_pad, norm='ortho'))

    def H_adj(self, y):
        # Display/initialization stand-in only -- see class docstring.
        gen = torch.Generator(device=y.device).manual_seed(0)
        phase = torch.exp(1j * torch.rand(y.shape, device=y.device, generator=gen) * 2 * math.pi)
        x_pad = torch.real(torch.fft.ifft2(y.to(torch.complex64) * phase, norm='ortho'))
        c = self.pad
        return x_pad[..., c:c + self.dim_image, c:c + self.dim_image]
