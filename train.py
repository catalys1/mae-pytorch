from pytorch_lightning.cli import LightningCLI
import torch
import os


if __name__ == '__main__':
    # load environment variables from ./.env
 
    os.environ["DATADIR"] = "/scratch-shared/matt1/data"
    torch.set_float32_matmul_precision('medium')
    cli = LightningCLI(parser_kwargs={'parser_mode': 'omegaconf'})
