import torch
import numpy as np
import os
from time import perf_counter
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils
import matplotlib.pyplot as plt


class PNP_FLOW_ADMM(object):

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
            l = A(d)
            rr = torch.sum(r**2)
            alpha = rr/(torch.sum(d*l) +eps)
            x += alpha*d
            r -= alpha*l
            if torch.norm(r)/(torch.norm(b)+eps) < tol:
                break
            beta = torch.sum(r**2)/(rr+eps)
            d = r + beta*d        
        return x
    
    
    def admm_x_update(self, x_init, z, u, y, H, H_adj, rho, sigma_noise, max_cgiter=50, cg_tol=1e-6):
        """
        Solve:
            (1/sigma^2 H^T H + rho I)x = 1/sigma^2 H^T y + rho (z-u)
        using conjugate gradient
        """
        sigma2_inv = 1.0 / (sigma_noise ** 2)

        def A(x):
            return sigma2_inv * H_adj(H(x)) + rho * x

        b = sigma2_inv * H_adj(y) + rho * (z - u)

        x = self.conjugate_gradient(A, b, x_init, max_iter=max_cgiter, tol=cg_tol)
        return x

    def model_forward(self, x, t):
        if self.args.model == "ot":
            return self.model(x, t)

        elif self.args.model == "rectified":
            model_fn = mutils.get_model_fn(self.model, train=False)
            z = model_fn(x.type(torch.float), t * 999)
            return z

    # projection step
    def projection_step(self, x, t, noise):
        return t * x + noise*(1-t)  
        
    
    # Denoiser step
    def denoiser(self, x, t):
        z = self.model_forward(x, t)                     # Run the model
        return x + (1 - t.view(-1, 1, 1, 1)) * z         # Convert that into an image update
    

    def save_norms(self, primal_res_list, dual_res_list, u_inc_list, batch, z_inc_list=None):
        norms_dir = os.path.join(self.args.save_path_ip, "norms")
        os.makedirs(norms_dir, exist_ok=True)

        primal_res_array = np.array(primal_res_list, dtype=np.float32)
        dual_res_array = np.array(dual_res_list, dtype=np.float32)
        u_inc_array = np.array(u_inc_list, dtype=np.float32)

        savez_kwargs = dict(
            primal_res=primal_res_array,
            dual_res=dual_res_array,
            u_inc=u_inc_array,
        )
        if z_inc_list is not None:
            savez_kwargs["z_inc"] = np.array(z_inc_list, dtype=np.float32)

        np.savez(
            os.path.join(norms_dir, f"norms_batch_{batch}.npz"),
            **savez_kwargs,
        )

        plt.figure()
        plt.plot(primal_res_array, label=r"$\|x^k - z^k\|_2$")
        plt.plot(dual_res_array, label=r"$\rho\|z^k - z^{k-1}\|_2$")
        plt.plot(u_inc_array, label=r"$\|u^{k+1} - u^k\|_2$")
        if z_inc_list is not None:
            plt.plot(savez_kwargs["z_inc"], label=r"$\|z^k - z^{k-1}\|_2$")
        plt.xlabel("Iteration")
        plt.ylabel("Norm value")
        plt.title(f"ADMM Convergence Diagnostics (Batch {batch})")
        plt.yscale("log")  
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(norms_dir, f"norms_batch_{batch}.png"))
        plt.close()


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
        # skip first N batches
        for _ in range(self.args.start_batch):
            next(loader)
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

            # ADMM initialization
            x = torch.randn(clean_img.shape, device=self.device)
            z = x.clone()
            u = torch.zeros_like(x)
            gauss_noise = x.clone()
            # Early stopping for denoising
            dual_prev = None
            dual_increase_count = 0
            max_dual_increases = 4

            # Optional norm tracking
            if self.args.save_norms_admm:
                u_norm_list = []
                res_norm_list = []
                dual_res_list = []
                z_inc_list = []

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)
            
            with torch.no_grad():
                for iteration in range(int(steps)):
                    if self.args.compute_time:
                        time_counter_1 = perf_counter()

                    t1 = torch.ones(len(x), device=self.device) * (delta * iteration)
                    t_view = t1.view(-1, 1, 1, 1)

                    # Save previous values for diagnostics
                    z_prev = z.clone()
                    u_prev = u.clone()

                    # x-update: data-consistency step
                    x = self.admm_x_update(
                        x_init=x,
                        z=z,
                        u=u,
                        y=noisy_img,
                        H=H,
                        H_adj=H_adj,
                        rho=rho,
                        sigma_noise=sigma_noise,
                        max_cgiter=max_cgiter,
                        cg_tol=cg_tol
                    )

                    # z-update: flow prior / denoiser step
                    z_input = x + u
                    z_tilde = self.projection_step(z_input, t_view, gauss_noise)
                    z = self.denoiser(z_tilde, t1)
                    gauss_noise_1 = (z_tilde - z*t_view)/ ((1-t_view) + 1e-12)
                    gauss_noise_2 = torch.randn_like(x)
                    w1 = t_view
                    w2 = 1 - t_view

                    if self.args.adaptive_noise:
                        # print("PnP-ADMM with adaptive projection")
                        gauss_noise = (w2 * gauss_noise_2 + w1 * gauss_noise_1) / torch.sqrt(w1**2 + w2**2 + 1e-12) 
                    else:
                        if self.args.deterministic_noise:
                            # print("PnP-ADMM with deterministic projection")
                            gauss_noise = gauss_noise_1
                        else:
                            # print("PnP-ADMM with stochastic projection")
                            gauss_noise = gauss_noise_2
                                        
                    # u-update: dual variable
                    u = u + (x-z) 
                    
                    # Store norms
                    # Compute residuals
                    primal_res = torch.norm(x - z)
                    z_inc = torch.norm(z - z_prev)
                    dual_res = rho * z_inc
                    u_inc = torch.norm(u - u_prev)

                    # Early stopping for denoising:
                    if self.args.problem == "denoising":
                        max_denoising_steps = getattr(self.args, "max_denoising_steps", steps-0.15*steps) # early stopping
                        if iteration >= max_denoising_steps:
                            print(f"Stopping denoising at iter {iteration}")
                            break

                    # Store norms
                    if self.args.save_norms_admm:
                        res_norm_list.append(primal_res.item())
                        dual_res_list.append(dual_res.item())
                        u_norm_list.append(u_inc.item())
                        z_inc_list.append(z_inc.item())

                        print(
                            f"Iter {iteration}: "
                            f"||x-z||={primal_res.item():.4f}, "
                            f"rho||z-z_prev||={dual_res.item():.4f}, "
                            f"||u-u_prev||={u_inc.item():.4f}"
                        )

                    if self.args.compute_time:
                        torch.cuda.synchronize()
                        time_counter_2 = perf_counter()
                        time_per_batch += time_counter_2 - time_counter_1

                    if self.args.save_results:
                        if iteration % 50 == 0 or self.should_save_image(iteration, steps):
                            restored_img = z.detach().clone()
                            utils.compute_psnr(
                                clean_img_cpu, noisy_img, restored_img, self.args, H_adj, iter=iteration
                            )
                            utils.compute_ssim(
                                clean_img_cpu, noisy_img, restored_img, self.args, H_adj, iter=iteration
                            )
                            utils.compute_lpips(
                                clean_img_cpu, noisy_img, restored_img, self.args, H_adj, iter=iteration
                            )

            if  self.args.save_norms_admm:
                self.save_norms(res_norm_list, dual_res_list, u_norm_list, batch, z_inc_list=z_inc_list)

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
                restored_img = z.detach().clone()
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

