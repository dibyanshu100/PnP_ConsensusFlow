import torch
import torchvision
import torchvision.transforms as v2
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import pandas as pd
import os
import warnings
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pickle
import logging


class DataLoaders:
    def __init__(self, dataset_name, batch_size_train, batch_size_test,
                 celebahq_train_ids=None, celebahq_test_ids=None, dim_image=None):
        self.dataset_name = dataset_name
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        # Optional paths to id-list files (see CelebAHQDataset) enforcing a 
        # train/test split for CelebAMask-HQ -- unused for every other
        # dataset. 
        self.celebahq_train_ids = celebahq_train_ids
        self.celebahq_test_ids = celebahq_test_ids
        self.dim_image = dim_image

    def load_data(self):

        if self.dataset_name == 'celeba':
            transform = v2.Compose([
                v2.CenterCrop(178),
                v2.Resize((128, 128)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            # Paths
            img_dir = './data/celeba/img_align_celeba/'
            partition_csv = './data/celeba/list_eval_partition.csv'

            # Datasets
            train_dataset = CelebADataset(
                img_dir, partition_csv, partition=0, transform=transform)
            val_dataset = CelebADataset(
                img_dir, partition_csv, partition=1, transform=transform)
            test_dataset = CelebADataset(
                img_dir, partition_csv, partition=2, transform=transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        elif self.dataset_name == 'celebahq':

            transform = v2.Compose([
                v2.Resize(self.dim_image or 256),
                v2.ToTensor(),         # Convert images to PyTorch tensor
            ])

            test_dir = './data/celebahq/test/'
            test_dataset = CelebAHQDataset(
                test_dir, batchsize=self.batch_size_test, transform=transform,
                split_ids_path=self.celebahq_test_ids)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

            if self.celebahq_train_ids is not None:
                train_dataset = CelebAHQDataset(
                    test_dir, batchsize=self.batch_size_train, transform=transform,
                    split_ids_path=self.celebahq_train_ids)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size_train,
                    shuffle=True,
                    collate_fn=custom_collate)
            else:
                train_loader = None
            val_loader = None

        elif self.dataset_name == 'afhq_cat':
            # transform should include a linear transform 2x - 1
            transform = v2.Compose([
                v2.Resize((256, 256)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            # transform = False
            img_dir_test = './data/afhq_cat/test/cat/'
            img_dir_val = './data/afhq_cat/val/cat/'
            img_dir_train = './data/afhq_cat/train/cat/'
            test_dataset = AFHQDataset(
                img_dir_test, batchsize=self.batch_size_test, transform=transform)
            val_dataset = AFHQDataset(
                img_dir_val, batchsize=self.batch_size_test, transform=transform)
            train_dataset = AFHQDataset(
                img_dir_train, batchsize=self.batch_size_test, transform=transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate, drop_last=True)

        else:
            raise ValueError("The dataset your entered does not exist")

        data_loaders = {'train': train_loader,
                        'test': test_loader, 'val': val_loader}

        return data_loaders


class CelebADataset(Dataset):
    def __init__(self, img_dir, partition_csv, partition, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.partition = partition

        # Load the partition file correctly
        partition_df = pd.read_csv(
            partition_csv, header=0, names=[
                'image', 'partition'], skiprows=1)
        self.img_names = partition_df[partition_df['partition']
                                      == partition]['image'].values

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


class CelebAHQDataset(Dataset):
    def __init__(self, data_dir, batchsize, transform=None, split_ids_path=None):
        all_files = sorted(os.listdir(data_dir))
        if split_ids_path is not None:
            with open(split_ids_path) as f:
                allowed_ids = {line.strip() for line in f if line.strip()}
            self.files = [fn for fn in all_files
                         if os.path.splitext(fn)[0] in allowed_ids]
            if len(self.files) == 0:
                raise ValueError(
                    f"No files in '{data_dir}' matched any id in "
                    f"'{split_ids_path}' -- check that filenames look like "
                    f"'<id>.jpg' (or similar) and that the ids in that file "
                    f"are correct.")
        else:
            self.files = all_files
        self.root_dir = data_dir
        self.num_imgs = len(self.files)
        self.transform = transform
        self.batchsize = batchsize

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            image = 2 * image - 1
        image = image.float()

        return image, 0


class AFHQDataset(Dataset):
    """AFHQ Cat dataset."""

    def __init__(self, img_dir, batchsize, category='cat', transform=None):
        self.files = sorted(os.listdir(img_dir))
        self.num_imgs = len(self.files)
        self.batchsize = batchsize
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


def custom_collate(batch):
    # Filter out None values

    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data._utils.collate.default_collate(batch)


logging.basicConfig(level=logging.INFO)
