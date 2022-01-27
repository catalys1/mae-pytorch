from typing import Optional

import fgvcdata
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms


__all__ = [
    'CubDataModule',
    'DogsDataModule',
    'StanfordCarsDataModule',
    'AircraftDataModule',
    'Cifar10DataModule',
    'Cifar100DataModule',
]


to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])


class DataRepeater(object):
    def __init__(self, dataset, size=None):
        self.data = dataset
        self.size = size
        if self.size is None:
            self.size = len(dataset)
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        i = idx % len(self.data)
        return self.data[i]


class _BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        size: int = 224,
        augment: bool = True,
        num_samples: Optional[int] = None,
    ):
        super().__init__()

        self.augment = augment
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.num_samples = num_samples

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset = self.data_train,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            shuffle = True,
            drop_last = True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset = self.data_val,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            shuffle = False,
            drop_last = False
        )


class _FGVCDataModule(_BaseDataModule):
    def transforms(self, crop_scale=(0.2, 1), val=False):
        if not val:
            tform = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=crop_scale),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.25, 0.25, 0.25),
                to_tensor,
            ])
        else:
            tform = transforms.Compose([
                transforms.Resize([int(round(8 * x / 7)) for x in self.size]),
                transforms.CenterCrop(self.size),
                to_tensor,
            ])
        return tform

    def setup(self, stage=None):
        self.data_train = self.dataclass(
            root = f'{self.data_dir}/train',
            transform = self.transforms(val=not self.augment)
        )
        if self.num_samples is not None:
            self.data_train = DataRepeater(self.data_train, self.num_samples)

        self.data_val = self.dataclass(
            root = f'{self.data_dir}/val',
            transform = self.transforms(val=True)
        )


class CubDataModule(_FGVCDataModule):
    dataclass = fgvcdata.CUB
    num_class = 200


class DogsDataModule(_FGVCDataModule):
    dataclass = fgvcdata.StanfordDogs
    num_class = 120


class StanfordCarsDataModule(_FGVCDataModule):
    dataclass = fgvcdata.StanfordCars
    num_class = 196

    def transforms(self, crop_scale=(0.25, 1), val=False):
        return super().transforms(crop_scale, val)


class AircraftDataModule(_FGVCDataModule):
    dataclass = fgvcdata.Aircraft
    num_class = 100

    def transforms(self, crop_scale=(0.25, 1), val=False):
        return super().transforms(crop_scale, val)


class _CifarDataModule(_BaseDataModule):
    def prepare_data(self):
        self.dataclass(self.data_dir, download=True)

    def transforms(self, val=False):
        if not val:
            tform = transforms.Compose([
                transforms.Resize(self.size),
                transforms.Pad(self.size[0] // 8, padding_mode='reflect'),
                transforms.RandomAffine((-10, 10), (0, 1/8), (1, 1.2)),
                transforms.CenterCrop(self.size),
                transforms.RandomHorizontalFlip(0.5),
                to_tensor(self.normalize),
            ])
        else:
            tform = transforms.Compose([
                transforms.Resize(self.size),
                to_tensor(self.normalize),
            ])
        return tform

    def setup(self, stage=None):
        self.data_train = self.dataclass(
            root = self.data_dir,
            train = True, 
            transform = self.transforms(not self.augment)
        )
        self.data_val = self.dataclass(
            root = self.data_dir,
            train = False, 
            transform = self.transforms(True)
        )


class Cifar10DataModule(_CifarDataModule):
    num_class = 10
    dataclass = CIFAR10


class Cifar100DataModule(_CifarDataModule):
    num_class = 100
    dataclass = CIFAR100
