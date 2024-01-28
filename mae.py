import math
import os
from typing import Any, Tuple, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import timm
import torch
from torch import distributed
import torchvision
import torch.nn as nn


##############################################################################
# Masked Autoencoder (MAE)
##############################################################################

# Encoder:
# Use the timm VisionTransformer, but all we need is the "blocks" and the final "norm" submodules
# Add a fixed positional encoding at the beginning (sin-cos, original transformer style)
# Add a linear projection on the output to match the decoder dimension

# Decoder:
# Use the timm VisionTransformer, as in the encoder
# Position embeddings are added to the decoder input (sin-cos); note that they are different than
#   the encoder's, because the dimension is different
# There is a shared, learnable [MASK] token that is used at every masked position
# A classification token can be included, but it should work similarly without (using average pooling,
#   according to the paper); we don't include a classification token here

# The loss is MSE computed only on the masked patches, as in the paper


class ViTBlocks(torch.nn.Module):
    '''The main processing blocks of ViT. Excludes things like patch embedding and classificaton
    layer.

    Args:
        width: size of the feature dimension.
        depth: number of blocks in the network.
        end_norm: whether to end with LayerNorm or not.
    '''
    def __init__(
        self,
        width: int = 768,
        depth: int = 12,
        end_norm: bool = True,
    ):
        super().__init__()

        # transformer blocks from ViT
        ViT = timm.models.vision_transformer.VisionTransformer
        vit = ViT(embed_dim=width, depth=depth)
        self.layers = vit.blocks
        if end_norm:
            # final normalization
            self.layers.add_module('norm', vit.norm)

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    
class MaskedAutoencoder(torch.nn.Module):
    '''Masked Autoencoder for visual representation learning.

    Args:
        image_size: (height, width) of the input images.
        patch_size: side length of a patch.
        keep: percentage of tokens to process in the encoder. (1 - keep) is the percentage of masked tokens.
        enc_width: width (feature dimension) of the encoder.
        dec_width: width (feature dimension) of the decoder. If a float, it is interpreted as a percentage
            of enc_width.
        enc_depth: depth (number of blocks) of the encoder
        dec_depth: depth (number of blocks) of the decoder
    '''
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        keep: float = 0.25,
        enc_width: int = 768,
        dec_width: Union[int, float] = 0.25,
        enc_depth: int = 12,
        dec_depth: int = 4,
    ):
        super().__init__()

        assert image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.keep = keep
        self.n = (image_size[0] * image_size[1]) // patch_size**2  # number of patches

        if isinstance(dec_width, float) and dec_width > 0 and dec_width < 1:
            dec_width = int(dec_width * enc_width)
        else:
            dec_width = int(dec_width)
        self.enc_width = enc_width
        self.dec_width = dec_width
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth

        # linear patch embedding
        self.embed_conv = torch.nn.Conv2d(3, enc_width, patch_size, patch_size)

        # mask token and position encoding
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, dec_width, requires_grad=True))
        self.register_buffer('pos_encoder', self.pos_encoding(self.n, enc_width).requires_grad_(False))
        self.register_buffer('pos_decoder', self.pos_encoding(self.n, dec_width).requires_grad_(False))

        # encoder
        self.encoder = ViTBlocks(width=enc_width, depth=enc_depth)

        # linear projection from enc_width to dec_width
        self.project = torch.nn.Linear(enc_width, dec_width)

        # decoder
        self.decoder = ViTBlocks(width=dec_width, depth=dec_depth, end_norm=False)

        # linear projection to pixel dimensions
        self.pixel_project = torch.nn.Linear(dec_width, 3 * patch_size**2)

        self.freeze_mask = False  # set to True to reuse the same mask multiple times

    @property
    def freeze_mask(self):
        '''When True, the previously computed mask will be used on new inputs, instead of creating a new one.'''
        return self._freeze_mask

    @freeze_mask.setter
    def freeze_mask(self, val: bool):
        self._freeze_mask = val

    @staticmethod
    def pos_encoding(n: int, d: int, k: int=10000):
        '''Create sine-cosine positional embeddings.

        Args:
            n: the number of embedding vectors, corresponding to the number of tokens (patches) in the image.
            d: the dimension of the embeddings
            k: value that determines the maximum frequency (10,000 by default)
        
        Returns:
            (n, d) tensor of position encoding vectors
        '''
        x = torch.meshgrid(
            torch.arange(n, dtype=torch.float32),
            torch.arange(d, dtype=torch.float32),
            indexing='ij'
        )
        pos = torch.zeros_like(x[0])
        pos[:, ::2] = x[0][:, ::2].div(torch.pow(k, x[1][:, ::2].div(d // 2))).sin_()
        pos[:, 1::2] = x[0][:,1::2].div(torch.pow(k, x[1][:,1::2].div(d // 2))).cos_()
        return pos

    @staticmethod
    def generate_mask_index(bs: int, n_tok: int, device: str='cpu'):
        '''Create a randomly permuted token-index tensor for determining which tokens to mask.

        Args:
            bs: batch size
            n_tok: number of tokens per image
            device: the device where the tensors should be created

        Returns:
            (bs, 1) tensor of batch indices [0, 1, ..., bs - 1]^T
            (bs, n_tok) tensor of token indices, randomly permuted
        '''
        idx = torch.rand(bs, n_tok, device=device).argsort(dim=1)
        return idx

    @staticmethod
    def select_tokens(x: torch.Tensor, idx: torch.Tensor):
        '''Return the tokens from `x` corresponding to the indices in `idx`.
        '''
        idx = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return x.gather(dim=1, index=idx)

    def image_as_tokens(self, x: torch.Tensor):
        '''Reshape an image of shape (b, c, h, w) to a set of vectorized patches
        of shape (b, h*w/p^2, c*p^2). In other words, the set of non-overlapping
        patches of size (3, p, p) in the image are turned into vectors (tokens); 
        dimension 1 of the output indexes each patch.
        '''
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(b, (h * w) // p**2, c * p * p)
        return x

    def tokens_as_image(self, x: torch.Tensor):
        '''Reshape a set of token vectors into an image. This is the reverse operation
        of `image_as_tokens`.
        '''
        b = x.shape[0]
        im, p = self.image_size, self.patch_size
        hh, ww = im[0] // p, im[1] // p
        x = x.reshape(b, hh, ww, 3, p, p).permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(b, 3, p * hh, p * ww)
        return x

    def masked_image(self, x: torch.Tensor):
        '''Return a copy of the image batch, with the masked patches set to 0. Used
        for visualization.
        '''
        x = self.image_as_tokens(x).clone()
        bidx = torch.arange(x.shape[0], device=x.device)[:, None]
        x[bidx, self.idx[:, int(self.keep * self.n):]] = 0
        return self.tokens_as_image(x)

    def embed(self, x: torch.Tensor):
        return self.embed_conv(x).flatten(2).transpose(1, 2)

    def mask_input(self, x: torch.Tensor):
        '''Mask the image patches uniformly at random, as described in the paper: the patch tokens are
        randomly permuted (per image), and the first N are returned, where N corresponds to percentage
        of patches kept (not masked).
        
        Returns the masked (truncated) tokens. The mask indices are saved as `self.bidx` and `self.idx`.
        '''
        # create a new mask if self.freeze_mask is False, or if no mask has been created yet
        if not hasattr(self, 'idx') or not self.freeze_mask:
            self.idx = self.generate_mask_index(x.shape[0], x.shape[1], x.device)

        k = int(self.keep * self.n)
        x = self.select_tokens(x, self.idx[:, :k])
        return x

    def forward_features(self, x: torch.Tensor):
        x = self.embed(x)
        x = x + self.pos_encoder
        x = self.mask_input(x)
        x = self.encoder(x)

        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        x = self.project(x)

        k = self.n - x.shape[1]  # number of masked tokens
        mask_toks = self.mask_token.expand(x.shape[0], k, -1)
        x = torch.cat([x, mask_toks], 1)
        x = self.select_tokens(x, self.idx.argsort(1))
        x = x + self.pos_decoder
        x = self.decoder(x)
        x = self.pixel_project(x)

        return x


class MAE(pl.LightningModule):
    '''Masked Autoencoder LightningModule.

    Args:
        image_size: (height, width) of the input images.
        patch_size: size of the image patches.
        keep: percentage of tokens to keep. (1 - keep) is the percentage of masked tokens.
        enc_width: width of the encoder features.
        dec_width: width of the decoder features.
        enc_depth: depth of the encoder.
        dec_depth: depth of the decoder.
        lr: learning rate
        save_imgs_every: save some reconstructions every nth epoch.
        num_save_immgs: number of reconstructed images to save.
    '''
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        keep: float = 0.25,
        enc_width: int = 768,
        dec_width: Union[int, float] = 0.5,
        enc_depth: int = 12,
        dec_depth: int = 6,
        lr: float = 1.5e-4,
        base_batch_size: int = 256,
        normalize_for_loss: bool = False,
        save_imgs_every: int = 1,
        num_save_imgs: int = 36,
    ):
        super().__init__()

        self.mae = MaskedAutoencoder(
            image_size=image_size,
            patch_size=patch_size,
            keep=keep,
            enc_width=enc_width,
            enc_depth=enc_depth,
            dec_width=dec_width,
            dec_depth=dec_depth,
        )

        self.keep = keep
        self.n = self.mae.n
        self.lr = lr
        self.base_batch_size = base_batch_size
        self.normalize_for_loss = normalize_for_loss
        self.save_imgs_every = save_imgs_every
        self.num_save_imgs = num_save_imgs

        self.saved_imgs_list = []

    def on_train_batch_end(self, *args, **kwargs):
        if self.trainer.global_step == 2 and self.trainer.is_global_zero:
            # print GPU memory usage once at beginning of training
            avail, total = torch.cuda.mem_get_info()
            mem_used = 100 * (1 - (avail / total))
            gb = 1024**3
            self.print(f'GPU memory used: {(total-avail)/gb:.2f} of {total/gb:.2f} GB ({mem_used:.2f}%)')
        if self.trainer.num_nodes > 1 or self.trainer.num_devices > 1:
            distributed.barrier()

    def training_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        x, _ = batch
        pred = self.mae(x)
        loss = self.masked_mse_loss(x, pred)
        self.log('train/loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        x, _ = batch
        pred = self.mae(x)
        loss = self.masked_mse_loss(x, pred)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

        if self.save_imgs_every:
            p = int(self.save_imgs_every)
            if self.trainer.current_epoch % p == 0:
                nb = self.trainer.num_val_batches[0]
                ns = self.num_save_imgs
                per_batch = math.ceil(ns / nb)
                self.saved_imgs_list.append(pred[:per_batch])

        return {'loss': loss}

    def on_validation_epoch_end(self):
        if self.save_imgs_every:
            if self.trainer.is_global_zero:
                imgs = torch.cat(self.saved_imgs_list, 0)
                self.saved_imgs_list.clear()
                self.save_imgs(imgs[:self.num_save_imgs])
            if self.trainer.num_nodes > 1 or self.trainer.num_devices > 1:
                distributed.barrier()

    # @pl.utilities.rank_zero_only
    def save_imgs(self, imgs: torch.Tensor):
        with torch.no_grad():
            r = int(imgs.shape[0]**0.5)
            imgs = self.mae.tokens_as_image(imgs.detach())
            imgs = imgs.add_(1).mul_(127.5).clamp_(0, 255).byte()
            imgs = torchvision.utils.make_grid(imgs, r).cpu()
            epoch = self.trainer.current_epoch
            dir = os.path.join(self.trainer.log_dir, 'imgs')
            os.makedirs(dir, exist_ok=True)
            torchvision.io.write_png(imgs, os.path.join(dir, f'epoch_{epoch}_imgs.png'))

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches
        devices, nodes = self.trainer.num_devices, self.trainer.num_nodes
        batch_size = self.trainer.train_dataloader.batch_size
        lr_scale = devices * nodes * batch_size / self.base_batch_size
        lr = self.lr * lr_scale

        optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=(.9, .95), weight_decay=0.05)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            cycle_momentum=False,
        )
        return {
            'optimizer': optim, 
            'lr_scheduler': {'scheduler': schedule, 'interval': 'step'}
        }

    def masked_mse_loss(self, img: torch.Tensor, recon: torch.Tensor):
        # turn the image into patch-vectors for comparison to model output
        x = self.mae.image_as_tokens(img)
        if self.normalize_for_loss:
            std, mean = torch.std_mean(x, dim=-1, keepdim=True)
            x = x.sub(mean).div(std + 1e-5)
        # only compute on the mask token outputs, which is everything after the first (n * keep)
        idx = self.mae.idx[:, int(self.keep * self.n):]
        x = self.mae.select_tokens(x, idx)
        y = self.mae.select_tokens(recon, idx)
        return torch.nn.functional.mse_loss(x, y)
    
class MAE_linear_probe(pl.LightningModule):
    '''Frozen MAE encoder with trainable linear readout to class labels
    https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html

    '''
    def __init__(
            self, 
            ckpt_path: str,
            ):
        super().__init__()
        mae_module = MAE()
        mae_module.load_state_dict(torch.load(ckpt_path)['state_dict'])
        self.mae = mae_module.mae

        self.feature_extractor = self.mae.encoder

        self.classifier = torch.nn.Linear(self.mae.enc_width, 10)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        x = self.mae.embed(x)
        x = x + self.mae.pos_encoder
        self.feature_extractor.eval()
        with torch.no_grad():
            x = self.feature_extractor(x)
            x = x.mean(dim=1)  # average pool over the patch dimension
        x = self.classifier(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        x, labels = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, labels)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return {'loss': loss}
    
    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        x, labels = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, labels)
        _, predicted = torch.max(pred, 1)
        correct = (predicted == labels).sum().item()
        self.log('val/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val/acc', correct / len(labels), prog_bar=True, on_step=False, sync_dist=True, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        x, labels = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, labels)
        _, predicted = torch.max(pred, 1)
        correct = (predicted == labels).sum().item()
        self.log('test/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test/acc', correct / len(labels), prog_bar=True, on_step=False, sync_dist=True, on_epoch=True)

    def loss_fn(self, x, y):
        fn = torch.nn.CrossEntropyLoss()
        return fn(x, y)
    