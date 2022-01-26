from pytorch_lightning.utilities.cli import LightningCLI, DATAMODULE_REGISTRY, MODEL_REGISTRY

from dotenv import load_dotenv

import datamodule
import mae


# load environment variables from ./.env
load_dotenv('.env')

# register classes with LightningCLI
DATAMODULE_REGISTRY.register_classes(datamodule, datamodule._BaseDataModule)
MODEL_REGISTRY.register_classes(mae, mae.pl.LightningModule)


# customize LightningCLI with some argument linking
class CLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # compute the number of training steps from the number of samples, batch size, and
        # number of GPUs (used by the learning rate scheduler)
        parser.link_arguments(
            ('data.init_args.num_samples', 'data.init_args.batch_size', 'trainer.gpus'),
            'model.init_args.train_steps',
            lambda a, b, c: a // b // c,
            'parse'
        )

cli = CLightningCLI(parser_kwargs={'parser_mode': 'omegaconf'})
