import torch
import torch
import numpy as np
import os
from time import perf_counter
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils


class PNP_FLOW(object):

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method
    
    def conjugate_gradient(self, A, b, x0, max_iter=10, tol=1e-6):
        x = x0.clone()
        r = b-A(x)
        d = r
        eps = 1e-12
        for _ in range(max_iter):
            z = A(d)
            rr = torch.sum(r**2)
            alpha = rr/(torch.sum(d*z) +eps)
            x += alpha*d
            r -= alpha*z
            if torch.norm(r)/(torch.norm(b)+eps) < tol:
                break
            beta = torch.sum(r**2)/(rr+eps)
            d = r + beta*d        
        return x
    
    
    def admm_x_update(self, x_init, v, u, y, H, H_adj, rho, sigma_noise, max_cgiter=50, cg_tol=1e-6):
        """
        Solve:
            (1/sigma^2 H^T H + rho I)x = 1/sigma^2 H^T y + rho (v-u)
        using conjugate gradient
        """
        sigma2_inv = 1.0 / (sigma_noise ** 2)

        def A(x):
            return sigma2_inv * H_adj(H(x)) + rho * x

        b = sigma2_inv * H_adj(y) + rho * (v - u)

        x = self.conjugate_gradient(A, b, x_init, max_iter=max_cgiter, tol=cg_tol)
        return x

    def model_forward(self, x, t):
        if self.args.model == "ot":
            return self.model(x, t)

        elif self.args.model == "rectified":
            model_fn = mutils.get_model_fn(self.model, train=False)
            t_ = t[:, None, None, None]
            v = model_fn(x.type(torch.float), t * 999)
            return v

    # The controversial interpolation step
    def interpolation_step(self, x, t, noise):
        return t * x + noise*(1-t)  #torch.randn_like(x) * (1 - t)
        
    
    # The final Denoiser step
    def denoiser(self, x, t):
        v = self.model_forward(x, t)                     # Run the model
        return x + (1 - t.view(-1, 1, 1, 1)) * v         # Convert that into an image update


     # H: forward degradation operator (maps image to measurement space) 
     # H_adj: adjoint operator (maps measurement to image space)
    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        H = degradation.H
        H_adj = degradation.H_adj
        self.args.sigma_noise = sigma_noise
        steps = self.args.steps_pnp
        delta = 1 / steps

        if self.args.noise_type != 'gaussian':
            raise ValueError('Current ADMM implementation only supports gaussian noise')

        rho = float(self.args.rho_admm)
        max_cgiter = int(self.args.max_cgiter)
        cg_tol = float(self.args.cg_tol)

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):

            (clean_img, labels) = next(loader)
            self.args.batch = batch
            print(clean_img.shape)

            clean_img = clean_img.to(self.device)

            noisy_img = H(clean_img.clone())
            torch.manual_seed(batch)
            noisy_img = noisy_img + torch.randn_like(noisy_img) * sigma_noise

            # Keep clean_img on CPU for metrics if your utils expect that
            noisy_img, clean_img_cpu = noisy_img.to(self.device), clean_img.to('cpu')

            # ADMM initialization: better than random init for inverse problems
            # x = H_adj(noisy_img).to(self.device)
            x = torch.randn(clean_img.shape, device=self.device)
            v = x.clone()
            u = torch.zeros_like(x)
            gauss_noise = x.clone()
            # gauss_noise = torch.randn(clean_img.shape, device=self.device)

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)
            
            # t1 = torch.zeros(len(x), device=self.device)    
            # v = self.denoiser(v, t1)
            with torch.no_grad():
                for iteration in range(int(steps)):
                    if self.args.compute_time:
                        time_counter_1 = perf_counter()

                    t1 = torch.ones(len(x), device=self.device) * (delta * iteration)
                    t_view = t1.view(-1, 1, 1, 1)

                    # x-update: data-consistency proximal step
                    x = self.admm_x_update(
                        x_init=x,
                        v=v,
                        u=u,
                        y=noisy_img,
                        H=H,
                        H_adj=H_adj,
                        rho=rho,
                        sigma_noise=sigma_noise,
                        max_cgiter=max_cgiter,
                        cg_tol=cg_tol
                    )

                    # v-update: flow prior / denoiser step
                    v_input = x + u
                    v_tilde = self.interpolation_step(v_input, t_view, gauss_noise)
                    v = self.denoiser(v_tilde, t1)
                    gauss_noise_1 = (v_tilde - v*t_view)/ ((1-t_view) + 1e-12)
                    gauss_noise_2 = torch.randn_like(x)
                    w1 = t_view
                    w2 = 1 - t_view
                    gauss_noise = (w2 * gauss_noise_2 + w1 * gauss_noise_1) / torch.sqrt(w1**2 + w2**2 + 1e-12)
                                        
                    # u-update: dual variable
                    u = u + (x-v) #u + (x - v) 
                    
                    u_norm = torch.norm(u)
                    x_norm = torch.norm(x)
                    rel_u = u_norm / (x_norm + 1e-12)

                    print(f"Iter {iteration}: ||u||={u_norm.item():.4f}, ||x||={x_norm.item():.4f}, ||u||/||x||={rel_u.item():.6f}")

                    if self.args.compute_time:
                        torch.cuda.synchronize()
                        time_counter_2 = perf_counter()
                        time_per_batch += time_counter_2 - time_counter_1

                    if self.args.save_results:
                        if iteration % 50 == 0 or self.should_save_image(iteration, steps):
                            restored_img = v.detach().clone()
                            # utils.save_images(clean_img_cpu, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                            utils.compute_psnr(
                                clean_img_cpu, noisy_img, restored_img, self.args, H_adj, iter=iteration
                            )
                            utils.compute_ssim(
                                clean_img_cpu, noisy_img, restored_img, self.args, H_adj, iter=iteration
                            )
                            utils.compute_lpips(
                                clean_img_cpu, noisy_img, restored_img, self.args, H_adj, iter=iteration
                            )

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
                restored_img = v.detach().clone()
                utils.save_images(
                    clean_img_cpu, noisy_img, restored_img, self.args, H_adj, iter='final'
                )
                utils.compute_psnr(
                    clean_img_cpu, noisy_img, restored_img, self.args, H_adj, iter=steps
                )
                utils.compute_ssim(
                    clean_img_cpu, noisy_img, restored_img, self.args, H_adj, iter=steps
                )
                utils.compute_lpips(
                    clean_img_cpu, noisy_img, restored_img, self.args, H_adj, iter=steps
                )

        if self.args.save_results:
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
            utils.compute_average_lpips(self.args)
        if self.args.compute_memory:
            utils.compute_average_memory(self.args)
        if self.args.compute_time:
            utils.compute_average_time(self.args)

    def should_save_image(self, iteration, steps):
        gap = max(steps // 10, 1)
        return iteration % gap == 0

    def run_method(self, data_loaders, degradation, sigma_noise, H_funcs=None):

        # Construct the save path for results
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)

        # Create the directory if it doesn't exist
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # Solve the inverse problem
        self.solve_ip(
            data_loaders[self.args.eval_split], degradation, sigma_noise, H_funcs)
