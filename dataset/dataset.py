import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from glob import glob
# from .augmentation import create_train_transforms, create_val_transforms

import argparse
import random
from glob import glob
import os
import subprocess
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import cv2
from tqdm import tqdm
from .augmentation.augmentation import dataset_augmentation_worker, create_train_transforms, dataset_worker


def sample(df, size, random_state=42):
    oversample = False if len(df) > size else True
    return df.sample(n=size, random_state=random_state, replace=oversample)


class DeepFakeClassificationDataset(Dataset):
    def __init__(self, data_path, samples=32, transform=None, mode='train', balance=False):
        self.data = glob(f"{data_path}/*.parquet")
        self.transform = transform
        self.samples = samples
        self.mode = mode
        self.balance = balance

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]

        df = pd.read_parquet(file)
        df = df.drop(df[df.diff_shape.apply(lambda x: min(x) < 55)].index)

        real = df.loc[df["label"] == 'real']
        fake = df.loc[df["label"] == 'fake']

        if self.balance:
            real_samples = self.samples//2
            fake_samples = self.samples//2
        else:
            real_samples = random.randint(1,self.samples-1)
            fake_samples = self.samples - real_samples

        real = sample(real,real_samples)
        fake = sample(fake, fake_samples)

        df = pd.concat([fake, real])
        if self.mode == 'train':
            df['image'] = df.apply(dataset_augmentation_worker, axis=1)
        else:
            df['image'] = df.apply(dataset_worker, axis=1)
        df['y'] = df['label'].apply(lambda x:torch.tensor(0) if x == 'fake' else torch.tensor(1))

        if self.transform:
            df['image'] = df['image'].apply(lambda x: self.transform(image=x)["image"])

        df = df.sample(frac=1).reset_index(drop=True)

        return {
            "x": torch.stack(df['image'].tolist()),
            "y": torch.from_numpy(df['y'].to_numpy().astype(np.float16))
        }
