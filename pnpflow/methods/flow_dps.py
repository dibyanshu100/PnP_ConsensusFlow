import torch
import os
from time import perf_counter
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils


class FLOW_DPS(object):
    """Flow-Driven Posterior Sampling"""

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method

    def model_forward(self, x, t):
        if self.args.model == "ot":
            return self.model(x, t)

        elif self.args.model == "rectified":
            model_fn = mutils.get_model_fn(self.model, train=False)
            v = model_fn(x.type(torch.float), t * 999)
            return v

    def grad_datafit(self, z, y, degradation):
        # Same normalization/convention as pnp_flow_fbs.py's grad_datafit:
        H = degradation.H
        if getattr(degradation, 'is_segmentation', False):
            # Cross-entropy data term instead of squared error
            with torch.enable_grad():
                z_req = z.detach().requires_grad_(True)
                log_probs = torch.nn.functional.log_softmax(H(z_req), dim=1)
                ce = torch.nn.functional.nll_loss(log_probs, y, reduction='sum')
                grad = torch.autograd.grad(ce, z_req)[0]
            return grad.detach()
        if self.args.noise_type == 'gaussian':
            return degradation.vjp(z, H(z) - y) / (self.args.sigma_noise**2)
        elif self.args.noise_type == 'laplace':
            return degradation.vjp(
                z, 2 * torch.heaviside(H(z) - y, torch.zeros_like(H(z) - y)) - 1
            ) / self.args.sigma_noise
        else:
            raise ValueError('Noise type not supported')

    def data_consistency_step(self, data_est, noisy_img, degradation):
        """K steps of gradient descent on ||y - H(z)||^2 (or the Laplace
        equivalent) starting from the current data estimate, following the
        authors' practical implementation
        """
        z = data_est.detach().clone()
        step_size = self.args.grad_step_size * (self.args.sigma_noise ** 2)
        for _ in range(int(self.args.grad_steps)):
            z = z - step_size * self.grad_datafit(z, noisy_img, degradation)
        return z.detach()

    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        self.args.sigma_noise = sigma_noise
        H = degradation.H
        H_adj = degradation.H_adj

        steps, delta = self.args.steps_ode, 1 / self.args.steps_ode
        start_time = self.args.start_time

        loader = iter(test_loader)
        for _ in range(self.args.start_batch):
            next(loader)
        for batch in range(self.args.max_batch):
            (clean_img, labels) = next(loader)
            self.args.batch = batch

            if getattr(degradation, 'is_segmentation', False):
                with torch.no_grad():
                    noisy_img = H(clean_img.clone().to(self.device)).argmax(dim=1)
            else:
                noisy_img = H(clean_img.clone().to(self.device))
                torch.manual_seed(batch)
                if self.args.noise_type == 'gaussian':
                    noisy_img += torch.randn_like(noisy_img) * sigma_noise
                elif self.args.noise_type == 'laplace':
                    noise = torch.distributions.laplace.Laplace(
                        torch.zeros_like(noisy_img), sigma_noise * torch.ones_like(noisy_img)).sample()
                    noisy_img += noise.to(self.device)
                else:
                    raise ValueError('Noise type not supported')
            noisy_img = noisy_img.to(self.device)
            clean_img = clean_img.to('cpu')
            x_shape_ref = H_adj(torch.zeros_like(noisy_img)).to(self.device)

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            num_restarts = int(getattr(self.args, 'num_restarts', 1))
            best_residual = None
            best_x = None

            for restart in range(num_restarts):
                if num_restarts > 1:
                    torch.manual_seed(1000 * batch + restart)
                    print(f"  [restart {restart}/{num_restarts}]")

                x = torch.randn_like(x_shape_ref)

                for count, iteration in enumerate(range(int(steps * start_time), int(steps))):
                    if self.args.compute_time:
                        time_counter_1 = perf_counter()

                    t1 = torch.ones(len(x), device=self.device) * delta * iteration
                    t_next = min(1.0, delta * (iteration + 1))
                    t_next_t = torch.ones(len(x), device=self.device) * t_next

                    with torch.no_grad():
                        v = self.model_forward(x, t1)
                        t_ = t1.view(-1, 1, 1, 1)
                        data_est = x + (1 - t_) * v      # x_hat_0|tau in the paper's notation
                        noise_est = x - t_ * v            # x_hat_1|tau in the paper's notation

                    # --- 1. Likelihood (data-consistency) gradient step ---
                    data_est_y = self.data_consistency_step(data_est, noisy_img, degradation)
                    gamma = (1 - t_)  # gamma(t) = 1 - t, see class docstring
                    data_tilde = gamma * data_est_y + (1 - gamma) * data_est

                    # --- 2. Stochasticity ---
                    if self.args.use_stochastic_noise:
                        eta = t_next_t.view(-1, 1, 1, 1)
                        eps = torch.randn_like(noise_est)
                        noise_tilde = torch.sqrt(torch.clamp(1 - eta, min=0.0)) * noise_est + \
                            torch.sqrt(torch.clamp(eta, min=0.0)) * eps
                    else:
                        noise_tilde = noise_est

                    # --- 3. Euler / DDIM-style update ---
                    t_next_ = t_next_t.view(-1, 1, 1, 1)
                    with torch.no_grad():
                        x = t_next_ * data_tilde + (1 - t_next_) * noise_tilde

                    if self.args.save_results and restart == 0:
                        if iteration % 10 == 0 or self.should_save_image(iteration, steps):
                            restored_img = x.detach().clone()
                            utils.compute_psnr(clean_img, noisy_img,
                                               restored_img, self.args, H_adj, iter=iteration)
                            utils.compute_ssim(clean_img, noisy_img,
                                               restored_img, self.args, H_adj, iter=iteration)
                            utils.compute_lpips(clean_img, noisy_img,
                                                restored_img, self.args, H_adj, iter=iteration)

                    if self.args.compute_time:
                        torch.cuda.synchronize()
                        time_counter_2 = perf_counter()
                        time_per_batch += time_counter_2 - time_counter_1

                with torch.no_grad():
                    if getattr(degradation, 'is_segmentation', False):
                        residual = torch.nn.functional.cross_entropy(
                            H(x), noisy_img, reduction='none'
                        ).flatten(1).mean(dim=1)
                    else:
                        residual = torch.norm(
                            (H(x) - noisy_img).reshape(noisy_img.shape[0], -1), dim=1)
                if best_residual is None:
                    best_residual = residual
                    best_x = x.clone()
                else:
                    improved = residual < best_residual
                    best_residual = torch.where(improved, residual, best_residual)
                    improved_view = improved.view(-1, *([1] * (x.dim() - 1)))
                    best_x = torch.where(improved_view, x, best_x)
                if num_restarts > 1:
                    print(f"  [restart {restart}] per-image ||H(x)-y||: {residual.tolist()}")

            x = best_x

            if self.args.compute_memory:
                dict_memory = {}
                dict_memory["batch"] = batch
                dict_memory["max_allocated"] = torch.cuda.max_memory_allocated(self.device)
                utils.save_memory_use(dict_memory, self.args)

            if self.args.compute_time:
                dict_time = {}
                dict_time["batch"] = batch
                dict_time["time_per_batch"] = time_per_batch
                utils.save_time_use(dict_time, self.args)

            if self.args.save_results:
                restored_img = x.detach().clone()
                utils.save_images(clean_img, noisy_img, restored_img,
                                  self.args, H_adj, iter='final')
                utils.compute_psnr(clean_img, noisy_img,
                                   restored_img, self.args, H_adj, iter=iteration)
                utils.compute_ssim(
                    clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                utils.compute_lpips(clean_img, noisy_img,
                                    restored_img, self.args, H_adj, iter=iteration)

            if self.args.save_results and getattr(degradation, 'is_segmentation', False):
                utils.save_recon_for_fid(
                    clean_img, restored_img, self.args,
                    batch_offset=batch * self.args.batch_size_ip)

        if self.args.save_results:
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
            utils.compute_average_lpips(self.args)
        if self.args.compute_memory:
            utils.compute_average_memory(self.args)
        if self.args.compute_time:
            utils.compute_average_time(self.args)

    def should_save_image(self, iteration, steps):
        return iteration % (steps // 10) == 0

    def run_method(self, data_loaders, degradation, sigma_noise, H_funcs=None):
        # Construct the save path for results
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)

        # Create the directory if it doesn't exist
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # Solve the inverse problem
        self.solve_ip(
            data_loaders[self.args.eval_split], degradation, sigma_noise, H_funcs)