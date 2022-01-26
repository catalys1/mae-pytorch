<div align=center>
<h1>Masked Autoencoders in PyTorch</h1>

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>


</div>

A simple, unofficial implementation of MAE ([Masked Autoencoders are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)) using  [pytorch-lightning](https://www.pytorchlightning.ai/).

Currently implements training on [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [StanfordCars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), but is easily extensible to any other image dataset.

## Setup

```bash
# Clone the repository
git clone https://github.com/catalys1/mae-pytorch.git
cd mae-pytorch

# Install required libraries (inside a virtual environment preferably)
pip install -r requirements.txt

# Set up .env for path to data
echo "DATADIR=/path/to/data" > .env
```

## Usage

### MAE training

Training options are provided through configuration files, handled by [LightningCLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html). See `configs/` for examples.

Train an MAE model on the CUB dataset:
```bash
python train.py fit --config=configs/mae.yaml --config=configs/data/cub_mae.yaml
```

Using multiple GPUs:
```bash
python train.py fit --config=configs/mae.yaml --config=configs/data/cub_mae.yaml --config=configs/multigpu.yaml
```

### Fine-tuning

Not yet implemented.

## Implementation

The default model uses ViT-Base for the encoder, and a small ViT (`depth=4`, `width=192`) for the decoder. This is smaller than the model used in the paper.

## Dependencies

- Configuration and training is handled completely by [pytorch-lightning](https://pytorchlightning.ai).
- The MAE model uses the VisionTransformer from [timm](https://github.com/rwightman/pytorch-image-models).
- Interface to FGVC datasets through [fgvcdata](https://github.com/catalys1/fgvc-data-pytorch).
- Configurable environment variables through [python-dotenv](https://pypi.org/project/python-dotenv/).

## Results

Image reconstructions of CUB validation set images after training with the following command:
```bash
python train.py fit --config=configs/mae.yaml --config=configs/data/cub_mae.yaml --config=configs/multigpu.yaml
```

![Bird Reconstructions](samples/bird-samples.png)