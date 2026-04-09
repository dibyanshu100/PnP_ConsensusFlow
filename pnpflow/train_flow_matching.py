# Code adapted from
#
# Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., ... & Bengio, Y. (2023).
# Improving and generalizing flow-based generative models with minibatch optimal transport.
# arXiv preprint arXiv:2302.00482.
# (https://github.com/atong01/conditional-flow-matching)
#
# Chemseddine, J., Hagemann, P., Wald, C., & Steidl, G. (2024).
# Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching.
# arXiv preprint arXiv:2403.18705.
# (https://github.com/JChemseddine/Conditional_Wasserstein_Distances/blob/main/utils/utils_FID.py)

import os
import numpy as np
import torch
import skimage.io as io
from tqdm import tqdm
from matplotlib import pyplot as plt
import ot
from torchdiffeq import odeint_adjoint as odeint

from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from pnpflow.dataloaders import DataLoaders
import pnpflow.utils as utils
from pnpflow.dataloaders import CelebADataset
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder


img_dir_celeba = './data/celeba/img_align_celeba/'
partition_csv_celeba = './data/celeba/list_eval_partition.csv'
img_dir_afhq = './data/afhq_cat/test/cat'   # fixed path


class FLOW_MATCHING(object):

    def __init__(self, model, device, args):
        self.d = args.dim_image
        self.num_channels = args.num_channels
        self.device = device
        self.args = args
        self.lr = args.lr
        self.model = model.to(device)

    def _build_fid_dataset(self):
        if self.args.dataset == "celeba":
            return CelebADataset(
                img_dir_celeba,
                partition_csv_celeba,
                partition=2,
                transform=transforms.Compose([
                    transforms.CenterCrop(178),
                    transforms.Resize([self.args.dim_image, self.args.dim_image]),
                    transforms.ToTensor(),
                ])
            )

        elif self.args.dataset == "afhq_cat":
            # Expects directory structure like:
            # ./data/afhq_cat/test/cat/*.jpg
            return ImageFolder(
                root=img_dir_afhq,
                transform=transforms.Compose([
                    transforms.Resize((self.args.dim_image, self.args.dim_image)),
                    transforms.ToTensor(),
                ])
            )

        else:
            raise ValueError(f"Unknown dataset {self.args.dataset}")

    def train_FM_model(self, train_loader, opt, num_epoch):
        fid_dataset = self._build_fid_dataset()

        tq = tqdm(range(num_epoch), desc='loss')
        for ep in tq:
            for iteration, (x, labels) in enumerate(train_loader):
                if x.size(0) == 0:
                    continue
                if iteration > 20:
                    break

                print(f'Epoch: {ep}, iter: {iteration}')
                x = x.to(self.device)

                z = torch.randn(
                    x.shape[0],
                    self.num_channels,
                    self.d,
                    self.d,
                    device=self.device,
                    requires_grad=True
                )

                t1 = torch.rand(x.shape[0], 1, 1, 1, device=self.device)

                # compute coupling
                x0 = z.clone()
                x1 = x.clone()
                a, b = np.ones(len(x0)) / len(x0), np.ones(len(x0)) / len(x0)

                M = ot.dist(
                    x0.view(len(x0), -1).cpu().data.numpy(),
                    x1.view(len(x1), -1).cpu().data.numpy()
                )
                plan = ot.emd(a, b, M)
                p = plan.flatten()
                p = p / p.sum()
                choices = np.random.choice(
                    plan.shape[0] * plan.shape[1],
                    p=p,
                    size=len(x0),
                    replace=True
                )
                i, j = np.divmod(choices, plan.shape[1])
                x0 = x0[i]
                x1 = x1[j]

                xt = t1 * x1 + (1 - t1) * x0
                loss = torch.sum((self.model(xt, t1.squeeze()) - (x1 - x0)) ** 2) / x.shape[0]

                opt.zero_grad()
                loss.backward()
                opt.step()

                with open(self.save_path + 'loss_training.txt', 'a') as file:
                    file.write(f'Epoch: {ep}, iter: {iteration}, Loss: {loss.item()}\n')

            # save samples, plot them, and compute FID on small dataset
            self.sample_plot(x, ep)

            if ep % 5 == 0:
                torch.save(self.model.state_dict(), self.model_path + f'model_{ep}.pt')

                print("Computing FID 5K")
                num_gen = 5_000
                fid_value = self.compute_fid(
                    num_images_fid=num_gen,
                    fid_dataset=fid_dataset,
                    batch_size=124,
                    integration_method="euler",
                    integration_steps=10,
                    epoch=ep
                )

                with open(self.save_path + f'FID_{(num_gen // 1000)}k.txt', 'a') as file:
                    file.write(f'Epoch: {ep}, FID: {fid_value}\n')

    def apply_flow_matching(self, NO_samples):
        self.model.eval()
        with torch.no_grad():
            model_class = cnf(self.model)
            latent = torch.randn(
                NO_samples,
                self.num_channels,
                self.d,
                self.d,
                device=self.device,
                requires_grad=False
            )
            z_t = odeint(
                model_class,
                latent,
                torch.tensor([0.0, 1.0]).to(self.device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )
            x = z_t[-1].detach()
        self.model.train()
        return x

    def sample_plot(self, x, ep=None):
        try:
            os.makedirs(self.save_path + 'results_samplings/')
        except BaseException:
            pass

        reco = utils.postprocess(self.apply_flow_matching(16), self.args)
        reco = utils.postprocess(reco, self.args)
        utils.save_samples(
            reco.detach().cpu(),
            x[:16].cpu(),
            self.save_path + 'results_samplings/' + f'samplings_ep_{ep}.pdf',
            self.args
        )

        if ep == 0:
            gt = x[:16]
            gt = utils.postprocess(gt, self.args)
            utils.save_samples(
                gt.detach().cpu(),
                gt.detach().cpu(),
                self.save_path + 'results_samplings/' + f'train_samples_ep_{ep}.pdf',
                self.args
            )

    def generate_samples(self, integration_method="dopri5", tol=1e-5,
                         n_samples=1028, batch_size=None, num_channels=3,
                         integration_steps=100, tmax=1):
        """
        Generate samples from the trained flow model.
        """

        if batch_size is None:
            batch_size = n_samples

        images_list = []
        batches = [batch_size] * (n_samples // batch_size)
        if n_samples % batch_size:
            batches += [n_samples % batch_size]

        with torch.no_grad():
            for _, batch in enumerate(tqdm(batches)):
                time_points = torch.linspace(
                    0, tmax, int(tmax * integration_steps), device=self.device
                )

                x0 = torch.randn(batch, num_channels, self.d, self.d, device=self.device)
                model_class = cnf(self.model)
                traj = odeint(
                    model_class,
                    x0,
                    time_points,
                    rtol=tol,
                    atol=tol,
                    method=integration_method
                )
                images_list.append(traj[-1, :])

        images = torch.cat(images_list, dim=0)
        return images

    def compute_fid(self, num_images_fid, fid_dataset, batch_size=512,
                    integration_method="dopri5", integration_steps=100, epoch='final'):
        """
        Compute FID using torchmetrics on raw images.
        Expects images in [0,1] or [-1,1]; converts to uint8 [0,255].
        """

        self.model.eval()

        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(self.device)

        # Load real images
        real_loader = DataLoader(
            fid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        for real_batch in tqdm(real_loader, desc="FID real images"):
            if isinstance(real_batch, (list, tuple)):
                real_imgs = real_batch[0]
            else:
                real_imgs = real_batch

            # Convert [0,1] -> uint8 [0,255]
            real_imgs = (real_imgs.clamp(0, 1) * 255.0).to(torch.uint8).to(self.device)
            fid.update(real_imgs, real=True)

        # Generate fake images
        gen_images = self.generate_samples(
            integration_method=integration_method,
            tol=1e-4,
            n_samples=num_images_fid,
            batch_size=batch_size,
            integration_steps=integration_steps,
            num_channels=self.num_channels,
        )

        # Save preview
        os.makedirs(f"training_images/{self.args.dataset}", exist_ok=True)
        preview = gen_images[:16]
        save_image(preview, f"training_images/{self.args.dataset}/gen_images_epoch{epoch}.png")

        # Convert fake images from [-1,1] -> uint8 [0,255]
        fake_imgs = ((gen_images.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)

        # Update FID in chunks
        for start in tqdm(range(0, fake_imgs.size(0), batch_size), desc="FID fake images"):
            fake_batch = fake_imgs[start:start + batch_size].to(self.device)
            fid.update(fake_batch, real=False)

        fid_val = fid.compute().item()

        self.model.train()
        return fid_val

    def train(self, data_loaders):
        self.save_path = self.args.root + 'results/{}/ot/'.format(self.args.dataset)
        try:
            os.makedirs(self.save_path)
        except BaseException:
            pass

        self.model_path = self.args.root + 'model/{}/ot/'.format(self.args.dataset)
        try:
            os.makedirs(self.model_path)
        except BaseException:
            pass

        train_loader = data_loaders['train']

        with open(self.save_path + 'model_info.txt', 'w') as file:
            file.write('PARAMETERS\n')
            file.write(f'Number of parameters: {sum(p.numel() for p in self.model.parameters())}\n')
            file.write(f'Number of epochs: {self.args.num_epoch}\n')
            file.write(f'Batch size: {self.args.batch_size_train}\n')
            file.write(f'Learning rate: {self.lr}\n')

        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_FM_model(train_loader, opt, num_epoch=self.args.num_epoch)

        torch.save(self.model.state_dict(), self.model_path + 'model_final.pt')


class cnf(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        with torch.no_grad():
            z = self.model(x, t.repeat(x.shape[0]))
        return z