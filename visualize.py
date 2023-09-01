import dotenv
from omegaconf import OmegaConf
from pathlib import Path
from pytorch_lightning import seed_everything
import random
import torch
import torchvision

import datamodule
import mae

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', help='Path to logging directory for a trained model')
    parser.add_argument('-n', '--num_imgs', type=int, default=8, help='Number of images to display')
    parser.add_argument('-s', '--seed', type=int, default=987654321, help='Random seed')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device type: "cpu" or "cuda"')

    args = parser.parse_args()

    dotenv.load_dotenv('.env')
    seed_everything(args.seed)
    device = args.device

    root = Path(args.logdir)
    config = OmegaConf.load(root.joinpath('config.yaml'))

    ### data setup
    print('Data... ', end='')
    dm_class = config.data.class_path.rsplit('.', 1)[-1]
    data_dir = config.data.init_args.data_dir
    dm = getattr(datamodule, dm_class)(data_dir)
    dm.setup()
    data = dm.data_val
    print(data)

    ### model setup
    print('Model... ', end='')
    ckpt_path = root.joinpath('checkpoints', 'last.ckpt')
    model = mae.MAE.load_from_checkpoint(ckpt_path, map_location='cpu')
    model = model.mae.to(device)
    print(model.__class__.__name__)

    ### get model predictions
    print('Getting predictions...', end='')
    img_indices = random.choices(range(len(data)), k=args.num_imgs)
    imgs = torch.stack([data[i][0] for i in img_indices], 0).to(device)

    preds = model.tokens_as_image(model(imgs))
    masked = model.masked_image(imgs)
    print('done')

    ### create visualization
    viz = torchvision.utils.make_grid(
        torch.cat([imgs, masked, preds], 0).clamp(-1, 1),
        nrow=args.num_imgs,
        normalize=True,
        value_range=(-1, 1),
    )
    viz = viz.mul_(255).byte()

    torchvision.io.write_png(viz, str(root.joinpath('samples.png')))