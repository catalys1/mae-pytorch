import math
from datamodule import STL10ReadoutDataModule
import os
from typing import Any, Tuple, Union
from mae import MAE_linear_probing, MAE
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import timm
import torch
from torch import distributed
import torchvision
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
# Load ViT model from ckpt
ckpt_model = torch.load('last.ckpt')  # upload checkpoint to aws
model = MAE_linear_probing(ckpt_model)
# model = MAE()

# Prepare trainer with correct data splits
stl10 = STL10ReadoutDataModule(data_dir='/scratch-shared/matt1/data', batch_size=64, num_workers=17, pin_memory=True, size=224, augment=True, num_samples=None)
trainer = Trainer()
trainer.fit(model, datamodule=stl10)
pass
