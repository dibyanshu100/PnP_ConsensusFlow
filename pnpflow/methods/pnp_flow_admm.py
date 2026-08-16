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

    def rho_schedule(self, rho, t_val):
        beta = float(getattr(self.args, 'beta_rho'))
        return rho / (max(1.0 - t_val, 1e-3) ** beta)

    def admm_x_update(self, x_init, z, u, y, degradation, rho, sigma_noise, max_cgiter=50, cg_tol=1e-6):
        """
        For linear H, solves in closed form via CG:
            (1/sigma^2 H^T H + rho I)x = 1/sigma^2 H^T y + rho (z-u)

        For nonlinear H there is no such closed form, so instead we take a few
        proximal-gradient steps on the (still convex-in-the-quadratic-part)
        x-subproblem
            1/(2 sigma^2) ||H(x)-y||^2 + rho/2 ||x-(z-u)||^2
        using degradation.vjp for the data-fidelity gradient. The step size
        1/rho is safe because the proximal term's curvature (rho) dominates
        near the ADMM fixed point, where rho is typically large relative to
        1/sigma^2.
        """
        H = degradation.H
        sigma2_inv = 1.0 / (sigma_noise ** 2)

        if degradation.is_linear:
            H_adj = degradation.H_adj

            def A(x):
                return sigma2_inv * H_adj(H(x)) + rho * x

            b = sigma2_inv * H_adj(y) + rho * (z - u)

            x = self.conjugate_gradient(A, b, x_init, max_iter=max_cgiter, tol=cg_tol)
            return x
        else:
            # Fixed-size steps are unsafe here: the data term's curvature
            # scales with 1/sigma^2 (which can dwarf rho), so a step size
            # that is safe for one (rho, sigma) pair can diverge for another.
            x = x_init.clone()
            target = z - u

            def obj(xv):
                return sigma2_inv * 0.5 * torch.sum((H(xv) - y) ** 2) \
                    + rho * 0.5 * torch.sum((xv - target) ** 2)

            gd_steps = int(getattr(self.args, 'admm_gd_steps'))
            lr = 1.0 / rho #float(getattr(self.args, 'admm_gd_lr_init', 1.0 / rho))
            for _ in range(gd_steps):
                grad = sigma2_inv * degradation.vjp(x, H(x) - y) + rho * (x - target)
                grad_norm_sq = torch.sum(grad ** 2)
                f0 = obj(x)
                step = lr
                for _ in range(20):
                    x_try = x - step * grad
                    if obj(x_try) <= f0 - 0.5 * step * grad_norm_sq or step < 1e-8:
                        break
                    step *= 0.5
                x = x_try
                lr = min(step * 2, 1.0 / rho)  # warm-start next iter, capped
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
    

    def save_norms(self, primal_res_list, dual_res_list, u_inc_list, batch):
        norms_dir = os.path.join(self.args.save_path_ip, "norms")
        os.makedirs(norms_dir, exist_ok=True)

        primal_res_array = np.array(primal_res_list, dtype=np.float32)
        dual_res_array = np.array(dual_res_list, dtype=np.float32)
        u_inc_array = np.array(u_inc_list, dtype=np.float32)

        np.savez(
            os.path.join(norms_dir, f"norms_batch_{batch}.npz"),
            primal_res=primal_res_array,
            dual_res=dual_res_array,
            u_inc=u_inc_array
        )

        plt.figure()
        plt.plot(primal_res_array, label=r"$\|x^k - z^k\|_2$")
        plt.plot(dual_res_array, label=r"$\rho_t\|z^k - z^{k-1}\|_2$")
        plt.plot(u_inc_array, label=r"$\|u^{k+1} - u^k\|_2$")
        plt.xlabel("Iteration")
        plt.ylabel("Norm value")
        plt.title(f"ADMM Convergence Diagnostics (Batch {batch})")
        plt.yscale("log")  
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(norms_dir, f"norms_batch_{batch}.png"))
        plt.close()


    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        if getattr(degradation, 'is_segmentation', False):
            # segmentation branch
            return self.solve_ip_segmentation(test_loader, degradation)

        num_samples = int(getattr(self.args, 'num_denoiser_calls', 5))  
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

        # Multiple restarts for phase retrieval
        num_restarts = int(getattr(self.args, 'num_restarts', 1))

        # early stopping for denoising and deblurring
        early_stopping = bool(getattr(self.args, 'early_stopping', False))
        max_steps = steps - 0.15 * steps

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

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            best_residual = None
            best_z = None

            for restart in range(num_restarts):
                if num_restarts > 1:
                    # Only reseed in multi-restart mode: this keeps the
                    # num_restarts=1 path's RNG sequence (and therefore its
                    # numerical results) identical to before this feature
                    # existed -- see this method's docstring note above.
                    torch.manual_seed(1000 * batch + restart)
                    print(f"  [restart {restart}/{num_restarts}]")

                # ADMM initialization
                x = torch.randn(clean_img.shape, device=self.device)
                # x = H_adj(torch.ones_like(noisy_img)).to(self.device)
                z = x.clone()
                u = torch.zeros_like(x)
                gauss_noise = x.clone()

                # Optional norm tracking (restart 0 only, to keep the
                # diagnostics plot/file readable when num_restarts > 1)
                if self.args.save_norms_admm and restart == 0:
                    u_norm_list = []
                    res_norm_list = []
                    dual_res_list = []

                with torch.no_grad():
                    for iteration in range(int(steps)):
                        if self.args.compute_time:
                            time_counter_1 = perf_counter()

                        # t1 = torch.ones(len(x), device=self.device) * (delta * iteration)
                        t_linear = delta * iteration  # k/N, in [0,1)
                        gamma = float(self.args.t_schedule_gamma)
                        t_val = t_linear ** gamma
                        t1 = torch.ones(len(x), device=self.device) * t_val

                        t_view = t1.view(-1, 1, 1, 1)

                        # Optional t-dependent penalty parameter, analogous
                        # to FBS's gamma_t = (1-t)^alpha schedule. See
                        # rho_schedule() docstring. Defaults to plain
                        # constant rho unless rho_schedule_style is set.
                        rho_t = self.rho_schedule(rho, t_val)

                        # Save previous values for diagnostics
                        z_prev = z.clone()
                        u_prev = u.clone()

                        # x-update: data-consistency step
                        x = self.admm_x_update(
                            x_init=x,
                            z=z,
                            u=u,
                            y=noisy_img,
                            degradation=degradation,
                            rho=rho_t,
                            sigma_noise=sigma_noise,
                            max_cgiter=max_cgiter,
                            cg_tol=cg_tol
                        )

                        # z-update: flow prior / denoiser step
                        z_input = x + u
                        z_new = torch.zeros_like(x)
                        w1 = t_view
                        w2 = 1 - t_view
                        for _ in range(num_samples):
                            z_tilde = self.projection_step(z_input, t_view, gauss_noise)
                            z = self.denoiser(z_tilde, t1)
                            gauss_noise_1 = (z_tilde - z*t_view)/ ((1-t_view) + 1e-12)
                            gauss_noise_2 = torch.randn_like(x)
                            if self.args.adaptive_noise:
                                gauss_noise = (w2 * gauss_noise_2 + w1 * gauss_noise_1) / torch.sqrt(w1**2 + w2**2 + 1e-12)
                            else:
                                if self.args.deterministic_noise:
                                    gauss_noise = gauss_noise_1
                                else:
                                    gauss_noise = gauss_noise_2
                            z_new = z_new + z
                        z_new /= num_samples
                        z = z_new

                        # u-update: dual variable
                        u = u + (x-z) 

                        # Store norms
                        if self.args.save_norms_admm and restart == 0:
                            primal_res = torch.norm(x - z)
                            dual_res = rho_t * torch.norm(z - z_prev)
                            u_inc = torch.norm(u - u_prev)
                            res_norm_list.append(primal_res.item())
                            dual_res_list.append(dual_res.item())
                            u_norm_list.append(u_inc.item())
                            print(f"Iter {iteration}: rho_t={rho_t:.4f}, ||x-z||={primal_res.item():.4f}, rho_t||z-z_precv|| = {dual_res.item():.4f}), ||u-u_prev|| = {u_inc.item():.4f}")

                        if self.args.compute_time:
                            torch.cuda.synchronize()
                            time_counter_2 = perf_counter()
                            time_per_batch += time_counter_2 - time_counter_1

                        if self.args.save_results and restart == 0:
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

                        if early_stopping:
                            if iteration >= max_steps:
                                print(f"Stopping at iter {iteration}")
                                break

                # Score this restart by its final data residual, per image,
                # and keep whichever restart is best for each image so far.
                with torch.no_grad():
                    residual = torch.norm(
                        (H(z) - noisy_img).reshape(noisy_img.shape[0], -1), dim=1)
                if best_residual is None:
                    best_residual = residual
                    best_z = z.clone()
                else:
                    improved = residual < best_residual
                    best_residual = torch.where(improved, residual, best_residual)
                    improved_view = improved.view(-1, *([1] * (z.dim() - 1)))
                    best_z = torch.where(improved_view, z, best_z)
                if num_restarts > 1:
                    print(f"  [restart {restart}] per-image ||H(x)-y||: {residual.tolist()}")

            z = best_z

            if  self.args.save_norms_admm:
                self.save_norms(res_norm_list, dual_res_list, u_norm_list, batch)

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
    # =====================================================================
    # Segmentation problem: ported from the user's own PNP_segmentation-main
    # project (pnp.py: SegmentationForwardModel.admm_x_update_gd and
    # NonlinearPNPFlowADMM.solve()). Cross-entropy data term instead of
    # squared error, so it doesn't reuse admm_x_update/grad_datafit at all.
    # All hyperparameters below default to that project's own argparse
    # defaults and are overridable via `--opts seg_<name> <value>`, kept
    # under a `seg_` prefix specifically so they don't collide with
    # rho_admm/steps_pnp/etc., which stay at their (very different-scale)
    # defaults tuned for the other, unrelated problems in this file.
    # =====================================================================

    @staticmethod
    def _seg_build_neighbor_matrix(num_classes, class_neighbors):
        matrix = torch.zeros(num_classes, num_classes, dtype=torch.float32)
        for class_id in range(num_classes):
            neighbors = [n for n in class_neighbors.get(class_id, [])
                         if 0 <= n < num_classes and n != class_id]
            if not neighbors:
                matrix[class_id, class_id] = 1.0
            else:
                matrix[class_id, neighbors] = (1.0 / len(neighbors))
        return matrix

    @torch.no_grad()
    def _seg_make_local_boundary_soft_target(self, target_mask, alpha, num_classes,
                                              kernel_size, interior_scale):
        target_mask = target_mask.detach().long()
        one_hot = torch.nn.functional.one_hot(target_mask, num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float().contiguous()
        if alpha <= 0.0:
            return one_hot
        k = kernel_size
        local_classes = torch.nn.functional.avg_pool2d(
            one_hot, kernel_size=k, stride=1, padding=k // 2)
        neighbors = local_classes * (1.0 - one_hot)
        neighbor_sum = neighbors.sum(dim=1, keepdim=True)
        neighbor_distribution = neighbors / neighbor_sum.clamp_min(1e-12)
        boundary = (neighbor_sum > 0).float()
        random_distribution = torch.rand_like(one_hot)
        random_distribution = random_distribution / \
            random_distribution.sum(dim=1, keepdim=True).clamp_min(1e-12)
        noise_distribution = (boundary * neighbor_distribution
                              + (1.0 - boundary) * random_distribution)
        alpha_interior = min(max(alpha * interior_scale, 0.0), 1.0)
        alpha_map = boundary * alpha + (1.0 - boundary) * alpha_interior
        soft_target = (1.0 - alpha_map) * one_hot + alpha_map * noise_distribution
        return soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _seg_get_mask_noise_weight(progress, mask_noise_start, mask_noise_schedule):
        progress = min(max(float(progress), 0.0), 1.0)
        if mask_noise_schedule == "constant":
            alpha = mask_noise_start
        elif mask_noise_schedule == "linear":
            alpha = mask_noise_start * (1.0 - progress)
        elif mask_noise_schedule == "quadratic":
            alpha = mask_noise_start * (1.0 - progress) ** 2
        elif mask_noise_schedule == "cosine":
            alpha = mask_noise_start * 0.5 * (1.0 + np.cos(np.pi * progress))
        elif mask_noise_schedule == "early_drop":
            alpha = mask_noise_start * (1.0 - progress) ** 3
        elif mask_noise_schedule == "increasing":
            alpha = mask_noise_start * progress
        else:
            raise ValueError(f"Unknown seg_mask_noise_schedule: {mask_noise_schedule}")
        return float(alpha)

    @staticmethod
    def _seg_get_ce_weight(progress, ce_weight_start, ce_weight_end, ce_weight_schedule):
        progress = min(max(float(progress), 0.0), 1.0)
        start, end = ce_weight_start, ce_weight_end
        if ce_weight_schedule == "constant":
            return start
        elif ce_weight_schedule == "linear":
            return start + (end - start) * progress
        elif ce_weight_schedule == "quadratic":
            return start + (end - start) * (progress ** 2)
        elif ce_weight_schedule == "cosine":
            return start + (end - start) * 0.5 * (1.0 - np.cos(np.pi * progress))
        elif ce_weight_schedule == "early_drop":
            return end + (start - end) * (1.0 - progress) ** 3
        else:
            raise ValueError(f"Unknown seg_ce_weight_schedule: {ce_weight_schedule}")

    @staticmethod
    def _seg_get_active_det_weight(iteration, steps_pnp, warmup_iters, cooldown_iters, start, end):
        active_start = warmup_iters
        active_end = steps_pnp - cooldown_iters
        span = max(active_end - active_start - 1, 1)
        active_progress = min(max((iteration - active_start) / span, 0.0), 1.0)
        return start + (end - start) * active_progress

    @staticmethod
    def _seg_update_noise(z_tilde, z, t_view, old_shape_like, det_weight):
        # Only "blend" mode is ported (the only mode the original project's
        # own main.py actually drives via CLI) -- see pnp.py's
        # NonlinearPNPFlowADMM._update_noise for the other (unused) modes.
        gauss_noise_1 = (z_tilde - z * t_view) / ((1.0 - t_view) + 1e-12)
        gauss_noise_2 = torch.randn_like(old_shape_like)
        d = float(det_weight)
        w_det, w_rand = d, 1.0 - d
        norm = (w_rand ** 2 + w_det ** 2) ** 0.5 + 1e-12
        return (w_rand * gauss_noise_2 + w_det * gauss_noise_1) / norm

    def admm_x_update_segmentation(
        self, degradation, x_init, target_distribution, z, u, rho, lr,
        n_inner_steps=5, clamp_normalized=(-1.0, 1.0), max_step_norm=None,
        use_gradient_projection=False, eta=0.3, eps=1e-8, ce_weight=1.0,
        class_weights=None,
    ):
        """Ported from SegmentationForwardModel.admm_x_update_gd: CE + rho
        penalty gradient descent, with optional conflict-aware CE-gradient
        projection and a per-sample step-norm cap.
        """
        x_current = x_init.detach().clone()
        target_distribution = target_distribution.detach()
        z, u = z.detach(), u.detach()
        admm_target = z - u

        def project(v0, v1):
            v1_unit = v1 / v1.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(eps)
            parallel = (v0 * v1_unit).sum(dim=(1, 2, 3), keepdim=True) * v1_unit
            return parallel, v0 - parallel

        for _ in range(n_inner_steps):
            x_req = x_current.detach().requires_grad_(True)
            log_probs = torch.nn.functional.log_softmax(degradation.H(x_req), dim=1)

            if class_weights is not None:
                w = class_weights.to(device=x_req.device, dtype=x_req.dtype).view(1, -1, 1, 1)
                ce = -ce_weight * (target_distribution * w * log_probs).sum()
            else:
                ce = -ce_weight * (target_distribution * log_probs).sum()

            pen = 0.5 * rho * ((x_req - admm_target) ** 2).sum()

            grad_ce = torch.autograd.grad(ce, x_req, retain_graph=True)[0]
            grad_pen = torch.autograd.grad(pen, x_req)[0]

            conflict = (grad_ce * grad_pen).sum(dim=(1, 2, 3), keepdim=True) < 0
            if use_gradient_projection:
                ce_parallel, ce_orthogonal = project(grad_ce, grad_pen)
                grad_ce = torch.where(conflict, eta * ce_orthogonal + ce_parallel, grad_ce)
            step = lr * (grad_ce + grad_pen)

            if max_step_norm is not None:
                step_norm = step.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(eps)
                scale = (max_step_norm / step_norm).clamp(max=1.0)
                step = step * scale

            with torch.no_grad():
                x_current = (x_req - step).clamp(*clamp_normalized)

        return x_current.detach()

    def solve_ip_segmentation(self, test_loader, degradation):
        """On the measurement:`target_mask` is built as `degradation.H(clean_img).argmax(1)` (the
        frozen segmentator's own prediction on the clean image), matching
        how every other nonlinear operator in this codebase builds its
        measurement.
        """
        import torch.nn.functional as Fs
        from pnpflow.nonlinear_ops.segmentation_task.seg_helpers import (
            CLASS_NEIGHBORS, build_class_weights)

        a = self.args
        rho = float(getattr(a, 'seg_rho', 500.0))
        x_lr = float(getattr(a, 'seg_x_lr', 1e-4))
        x_lr_decay = float(getattr(a, 'seg_x_lr_decay', 1.0))
        x_inner_steps_min = int(getattr(a, 'seg_x_inner_steps_min', 15))
        x_inner_steps_max = int(getattr(a, 'seg_x_inner_steps_max', 5))
        steps_pnp = int(getattr(a, 'seg_steps_pnp', 500))
        mask_noise_start = float(getattr(a, 'seg_mask_noise_start', 0.3))
        mask_noise_schedule = str(getattr(a, 'seg_mask_noise_schedule', 'early_drop'))
        mask_noise_kernel_size = int(getattr(a, 'seg_mask_noise_kernel_size', 5))
        mask_noise_interior_scale = float(getattr(a, 'seg_mask_noise_interior_scale', 0.0))
        ce_weight_start = float(getattr(a, 'seg_ce_weight_start', 10.0))
        ce_weight_end = float(getattr(a, 'seg_ce_weight_end', 2.0))
        ce_weight_schedule = str(getattr(a, 'seg_ce_weight_schedule', 'linear'))
        x_update_warmup_iters = int(getattr(a, 'seg_x_update_warmup_iters', 5))
        x_update_cooldown_iters = int(getattr(a, 'seg_x_update_cooldown_iters', 0))
        x_update_frequency = max(int(getattr(a, 'seg_x_update_frequency', 1)), 1)
        max_step_norm_raw = float(getattr(a, 'seg_max_step_norm', 0.0))
        max_step_norm = None if max_step_norm_raw <= 0 else max_step_norm_raw
        use_gradient_projection = bool(getattr(a, 'seg_use_gradient_projection', 0))
        eta = float(getattr(a, 'seg_eta', 1.0))
        warmup_det_weight = float(getattr(a, 'seg_warmup_det_weight', 0.1))
        active_det_weight_start = float(getattr(a, 'seg_active_det_weight_start', 0.65))
        active_det_weight_end = float(getattr(a, 'seg_active_det_weight_end', 0.9))
        cooldown_det_weight = float(getattr(a, 'seg_cooldown_det_weight', 0.5))
        clamp_normalized = (float(getattr(a, 'seg_clamp_min', -1.0)),
                            float(getattr(a, 'seg_clamp_max', 1.0)))
        num_restarts = int(getattr(a, 'num_restarts', 1))

        class_weights = build_class_weights(degradation.num_classes).to(self.device)
        neighbor_matrix = self._seg_build_neighbor_matrix(
            degradation.num_classes, CLASS_NEIGHBORS).to(self.device)

        loader = iter(test_loader)
        for _ in range(a.start_batch):
            next(loader)
        for batch in range(a.max_batch):
            (clean_img, labels) = next(loader)
            a.batch = batch
            print(clean_img.shape)
            clean_img = clean_img.to(self.device)

            with torch.no_grad():
                target_mask = degradation.H(clean_img).argmax(dim=1)

            best_ce = None
            best_z = None

            for restart in range(num_restarts):
                if num_restarts > 1:
                    torch.manual_seed(1000 * batch + restart)
                    print(f"  [restart {restart}/{num_restarts}]")

                x = torch.randn_like(clean_img)
                z = x.clone()
                u = torch.zeros_like(x)
                gauss_noise = torch.randn_like(x)
                delta = 1.0 / max(steps_pnp, 1)

                for iteration in range(steps_pnp):
                    progress = iteration / max(steps_pnp - 1, 1)
                    mask_noise_weight = self._seg_get_mask_noise_weight(
                        progress, mask_noise_start, mask_noise_schedule)
                    target_distribution = self._seg_make_local_boundary_soft_target(
                        target_mask=target_mask, alpha=mask_noise_weight,
                        num_classes=degradation.num_classes,
                        kernel_size=mask_noise_kernel_size,
                        interior_scale=mask_noise_interior_scale)

                    inner_steps = int(round(
                        x_inner_steps_min + progress * (x_inner_steps_max - x_inner_steps_min)))
                    in_warmup = iteration < x_update_warmup_iters
                    in_cooldown = iteration >= steps_pnp - x_update_cooldown_iters
                    if in_warmup or in_cooldown:
                        inner_steps = 0
                    elif (iteration - x_update_warmup_iters) % x_update_frequency != 0:
                        inner_steps = 0
                    x_was_updated = inner_steps > 0

                    if in_warmup:
                        current_det_weight = warmup_det_weight
                    elif in_cooldown:
                        current_det_weight = cooldown_det_weight
                    else:
                        current_det_weight = self._seg_get_active_det_weight(
                            iteration, steps_pnp, x_update_warmup_iters,
                            x_update_cooldown_iters, active_det_weight_start,
                            active_det_weight_end)

                    t = torch.ones(x.shape[0], device=self.device) * (delta * iteration)
                    t_view = t.view(-1, 1, 1, 1)
                    z_prev = z.clone()
                    u_prev = u.clone()

                    current_lr = x_lr * (x_lr_decay ** iteration)
                    current_ce_weight = self._seg_get_ce_weight(
                        progress, ce_weight_start, ce_weight_end, ce_weight_schedule)

                    x = self.admm_x_update_segmentation(
                        degradation=degradation, x_init=x,
                        target_distribution=target_distribution, z=z, u=u,
                        rho=rho, lr=current_lr, n_inner_steps=inner_steps,
                        clamp_normalized=clamp_normalized,
                        max_step_norm=max_step_norm,
                        use_gradient_projection=use_gradient_projection,
                        eta=eta, ce_weight=current_ce_weight,
                        class_weights=class_weights)

                    with torch.no_grad():
                        z_input = x + u
                        z_tilde = self.projection_step(z_input, t_view, gauss_noise)
                        z = self.denoiser(z_tilde, t).clamp(*clamp_normalized)
                        gauss_noise = self._seg_update_noise(
                            z_tilde=z_tilde, z=z, t_view=t_view,
                            old_shape_like=x, det_weight=current_det_weight)

                        if x_was_updated:
                            u = u + (x - z)

                        primal_res = torch.norm(x - z).item()
                        dual_res = (rho * torch.norm(z - z_prev)).item()
                        u_inc = torch.norm(u - u_prev).item()

                        if iteration % 50 == 0 or iteration == steps_pnp - 1:
                            print(f"Iter {iteration}: ||x-z||={primal_res:.4f}, "
                                  f"rho||z-z_prev||={dual_res:.4f}, "
                                  f"||u-u_prev||={u_inc:.4f}")

                        if a.save_results and restart == 0:
                            if iteration % 50 == 0 or self.should_save_image(iteration, steps_pnp):
                                clean_img_cpu = clean_img.to('cpu')
                                restored_img = z.detach().clone()
                                utils.compute_psnr(
                                    clean_img_cpu, target_mask, restored_img,
                                    a, degradation.H_adj, iter=iteration)
                                utils.compute_ssim(
                                    clean_img_cpu, target_mask, restored_img,
                                    a, degradation.H_adj, iter=iteration)
                                utils.compute_lpips(
                                    clean_img_cpu, target_mask, restored_img,
                                    a, degradation.H_adj, iter=iteration)

                # Score this restart by final CE against the (hard-label)
                # target -- the classification analogue of the ||H(x)-y||
                # residual restarts are scored by for every other problem.
                with torch.no_grad():
                    ce_final = Fs.cross_entropy(
                        degradation.H(z), target_mask, reduction='none'
                    ).flatten(1).mean(dim=1)
                if best_ce is None:
                    best_ce = ce_final
                    best_z = z.clone()
                else:
                    improved = ce_final < best_ce
                    best_ce = torch.where(improved, ce_final, best_ce)
                    improved_view = improved.view(-1, *([1] * (z.dim() - 1)))
                    best_z = torch.where(improved_view, z, best_z)
                if num_restarts > 1:
                    print(f"  [restart {restart}] per-image final CE: {ce_final.tolist()}")

            z = best_z

            if a.save_results:
                clean_img_cpu = clean_img.to('cpu')
                restored_img = z.detach().clone()
                utils.save_images(clean_img_cpu, target_mask, restored_img,
                                  a, degradation.H_adj, iter='final')
                utils.compute_psnr(clean_img_cpu, target_mask, restored_img,
                                   a, degradation.H_adj, iter='final')
                utils.compute_ssim(clean_img_cpu, target_mask, restored_img,
                                   a, degradation.H_adj, iter='final')
                utils.compute_lpips(clean_img_cpu, target_mask, restored_img,
                                    a, degradation.H_adj, iter='final')

        if a.save_results:
            utils.compute_average_psnr(a)
            utils.compute_average_ssim(a)
            utils.compute_average_lpips(a)
