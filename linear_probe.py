from pytorch_lightning.cli import LightningCLI
from dotenv import load_dotenv
import torch
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv('.env')
    torch.set_float32_matmul_precision('medium')
    cli = LightningCLI(parser_kwargs={'parser_mode': 'omegaconf'}, run=False)
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)


