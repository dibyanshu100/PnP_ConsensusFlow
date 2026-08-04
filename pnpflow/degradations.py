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


class _NonlinearBlurNet(torch.nn.Module):
    """Small conv encoder-decoder producing a spatially-varying, nonlinear blur.

    This plays the role of the pretrained network used in Tran et al. (2021)
    ("Explore Image Deblurring via Encoded Blur Kernel Space") and adopted as
    the nonlinear-deblurring forward model in DPS (Chung et al., 2022): a
    smoothing operation modulated by a random "kernel code" and passed through
    a saturating nonlinearity, so H is genuinely nonlinear (not a fixed
    convolution) and depends on x through more than a linear filter.

    If the official pretrained weights are available, load them into this
    module via `NonlinearBlur(checkpoint_path=...)`; the architecture below is
    otherwise a compact, from-scratch stand-in (fixed/frozen, untrained beyond
    random init) so the operator is still well-defined and differentiable.
    """

    def __init__(self, num_channels=3, code_channels=8, hidden=32):
        super().__init__()
        self.code_channels = code_channels
        self.enc = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels + code_channels, hidden, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, hidden, 3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.dec = torch.nn.Sequential(
            torch.nn.Conv2d(hidden, hidden, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, num_channels, 3, padding=1),
        )

    def forward(self, x, code):
        code_up = F.interpolate(code, size=x.shape[-2:], mode='nearest')
        h = self.enc(torch.cat([x, code_up], dim=1))
        residual = self.dec(h)
        blurred = F.avg_pool2d(x, 5, stride=1, padding=2)
        # Saturating nonlinearity (akin to sensor tone-mapping) is what makes
        # this operator genuinely nonlinear, on top of the spatially-varying
        # residual predicted from the image itself.
        return torch.tanh(blurred + 0.1 * residual)


class NonlinearBlur(Degradation):
    """Nonlinear blur forward operator (Tran et al., 2021 / DPS-style).

    Unlike GaussianDeblurring, H here is not a fixed convolution: it is a
    learned, spatially-varying, nonlinear function of x, so there is no exact
    adjoint. H_adj is therefore only a display/initialization stand-in
    (identity, since input and output live in the same pixel space); actual
    gradients of the data-fidelity term must go through `vjp` instead.
    """

    is_linear = False

    def __init__(self, num_channels=3, dim_image=128, checkpoint_path=None, device="cuda"):
        super().__init__()
        self.device = device
        self.net = _NonlinearBlurNet(num_channels).to(device)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        if checkpoint_path is not None:
            try:
                state = torch.load(checkpoint_path, map_location=device)
                state = state.get('state_dict', state)
                missing = self.net.load_state_dict(state, strict=False)
                print(f"[NonlinearBlur] loaded checkpoint from {checkpoint_path} ({missing})")
            except Exception as e:
                print(f"[NonlinearBlur] could not load checkpoint at {checkpoint_path} "
                      f"({e}); using randomly-initialized (frozen) weights instead.")
        # Fixed kernel code so that H is a deterministic function of x alone
        # (reproducible across runs / methods being compared).
        gen = torch.Generator(device=device).manual_seed(0)
        code_size = max(dim_image // 8, 1)
        self.code = torch.randn(1, 8, code_size, code_size, device=device, generator=gen)

    def H(self, x):
        x = x.to(self.device)
        code = self.code.repeat(x.shape[0], 1, 1, 1)
        return self.net(x, code)

    def H_adj(self, x):
        # Display/initialization stand-in only -- see class docstring.
        return x


class PhaseRetrieval(Degradation):
    """Fourier phase retrieval: y = |FFT(pad(x))|, as used e.g. in DPS
    (Chung et al., 2022). Padding (oversampling) the image before the FFT is
    the standard convention in the phase-retrieval literature (commonly a
    factor of 2) so that the magnitude-only measurement is informative enough
    for the (still highly non-convex) problem to be solvable.

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
        return torch.abs(torch.fft.fft2(x_pad))

    def H_adj(self, y):
        # Display/initialization stand-in only -- see class docstring.
        gen = torch.Generator(device=y.device).manual_seed(0)
        phase = torch.exp(1j * torch.rand(y.shape, device=y.device, generator=gen) * 2 * math.pi)
        x_pad = torch.real(torch.fft.ifft2(y.to(torch.complex64) * phase))
        c = self.pad
        return x_pad[..., c:c + self.dim_image, c:c + self.dim_image]
