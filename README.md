**English** | [Русский](README_ru.md)

# AlphaGAN for Image Matting

Unofficial implementation of the **AlphaGAN** training pipeline for **image matting** (recovering an alpha matte from an image and a trimap), based on the paper:  
[AlphaGAN: Generative adversarial networks for natural image matting](https://arxiv.org/pdf/1807.10088)

## Contents
- [Project overview](#project-overview)
- [Repository structure](#repository-structure)
- [Dependency installation](#dependency-installation)
- [Data preparation](#data-preparation)
- [Configs](#configs)
  - [Config example](#config-example)
  - [Config field descriptions](#config-field-descriptions)
- [Training](#training)
- [Checkpoints and logging](#checkpoints-and-logging)
- [References](#references)
- [License](#license)

---

## Project overview

**AlphaGAN** targets alpha matte recovery (transparency) in the unknown trimap region using GAN components and matting specific losses.

This repository includes:
- model architectures (generator, discriminator, and building blocks)
- losses and metrics
- training pipeline (train and test)
- dataset pipeline and transforms
- TensorBoard logging and checkpoints

---

## Repository structure

```
.
├── configs/                 # YAML configs for training and testing
├── losses/                  # Implementations of all losses and metrics
├── models/                  # Model architectures and components
├── train_pipeline/          # Train/test steps and epoch loop
├── transforms/
│   ├── models/              # Custom transforms
│   └── trans_pipeline.py    # Transform pipeline for train/test data
├── cfg_loader.py            # Config loading from configs/
├── dataset.py               # Dataset, archive unpacking, CSV label generation
├── main.py                  # Entry point: initialization and training start
├── schemas.py               # Dataclasses for batch structures, losses, metrics, and training components
└── utils.py                 # Seed, logging, helpers, checkpoint save/load
```

---

## Dependency installation

The project uses `python 3.13` and the `pytorch` framework.

Example installation with `uv`:

If `uv` is not installed yet, follow the instructions [here](https://docs.astral.sh/uv/)

```bash
uv sync
```

---

Example installation with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data preparation

Datasets used for training:
- [AIM-500](https://github.com/JizhiziLi/AIM) (foreground, masks, and trimaps)
- [BG20K](https://www.kaggle.com/datasets/nguyenquocdungk16hl/bg-20o) (background images)

`dataset.py` contains:
- dataset class for train/test
- archive unpacking helpers
- `dataset_labels.csv` generation with paths to `original`, `trimap`, `mask`, `background` and the `split` field (train/test)

### Expected directory layout

```text
<root>/
  dataset/
    AIM-500/              # fg_path
      mask/
        <id>.(png|jpg|jpeg)
      trimap/
        <id>.(png|jpg|jpeg)
      original/
        <id>.(png|jpg|jpeg)

    BG20K/                # bg_path
      **/*.(png|jpg|jpeg) # nested folders are allowed
```

Requirements:
- `AIM-500` must contain `mask`, `trimap`, and `original` folders
- each `<id>` must exist in all three folders with the same filename stem
- `BG20K` can be nested, images are searched recursively by `png/jpg/jpeg`

### Usage example

1) Unpack archives:

```python
from pathlib import Path
from dataset import unpack_archives

fg_zip_path = Path(__file__).parent / "dataset" / "AIM-500-20251030T115928Z-1-001.zip"
bg_zip_path = Path(__file__).parent / "dataset" / "archive.zip"
dst_path = Path(__file__).parent / "dataset"

unpack_archives(fg_zip_path, dst_path)
unpack_archives(bg_zip_path, dst_path)
```

2) Generate `dataset_labels.csv`:

```python
from pathlib import Path
from dataset import prepare_labels

fg_path = Path(__file__).parent / "dataset" / "AIM-500"
bg_path = Path(__file__).parent / "dataset" / "BG20K"
output_path = Path(__file__).parent / "dataset" / "dataset_labels.csv"

prepare_labels(fg_path, bg_path, output_path)
```

---

## Configs

Configs use the `.yaml` format.

### Config example

```yaml
general:
  random_seed: 1669
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  batch_size: 5
  checkpoints_dir: checkpoints/
  log_dir: tb_logger/
  colab:
    use_colab: 0
    best_chkp_name: best_chkp
    last_chkp_name: last_chkp

train:
  use_gan_loss: 1
  save_chkp_n_epoches: 1
  crop_size: 256
  D:
    update_n_batches: 1
    scheduler:
      start_lr: 1e-5
      end_lr: 1.5e-4
      step_size_up: 4000
  G:
    scheduler:
      start_lr: 1e-3
      end_lr: 3e-4
      step_size_up: 4000
  optimizer:
    weight_decay: 5e-4
  logging:
    log_io_n_batches: 10
    log_lr_n_batches: 50
    log_curr_loss_n_batches: 1
  epoches: 5000
  losses:
    lambda_gan_g: 0.15
    lambda_alpha_g: 1.0
    lambda_comp_g: 1.0
  amp:
    use_amp: 1
    dtype: bf16
    use_grad_scaler: 0

test:
  resize_size: 256
  logging:
    log_curr_mets_n_batches: 1
    log_io_n_batches: 10
```

### Config field descriptions

#### `general`
- `random_seed`  
  Seed for reproducibility.
- `mean`, `std`  
  Input normalization (ImageNet values by default).
- `batch_size`  
  Batch size.
- `checkpoints_dir`  
  Directory for saving checkpoints.
- `colab.use_colab`  
  Enables a Colab and Google Drive friendly mode (useful when disk space is limited).
- `colab.best_chkp_name`, `colab.last_chkp_name`  
  Base filenames for best and last checkpoints.

#### `train`
- `use_gan_loss`  
  Enable the GAN loss term. If `0`, the adversarial component is disabled.
- `save_chkp_n_epoches`  
  Save a checkpoint every `N` epochs.
- `crop_size`  
  Crop size around the unknown (gray) trimap region.
- `D.update_n_batches`  
  Update the discriminator once per `N` batches.
- `D.scheduler`, `G.scheduler`  
  Scheduler parameters: `start_lr`, `end_lr`, `step_size_up`.
- `optimizer.weight_decay`  
  Weight decay for the optimizer.
- `logging`  
  TensorBoard logging settings.
  - `log_dir`  
    Log directory.
  - `log_io_n_batches`  
    Log input/output examples every `N` batches.
  - `log_lr_n_batches`  
    Log learning rate every `N` batches.
  - `log_curr_loss_n_batches`  
    Log current losses every `N` batches.
- `epoches`  
  Number of training epochs.
- `losses`  
  Weights for the generator loss terms:
  - `lambda_gan_g` GAN term weight for G
  - `lambda_alpha_g` alpha loss weight (error between GT and predicted alpha)
  - `lambda_comp_g` composition loss weight (error between GT composite and composite built with predicted alpha)
- `amp`  
  Mixed precision (automatic mixed precision) settings.
  - `use_amp`  
    Enable AMP. If `0`, training runs in fp32.
  - `dtype`  
    Autocast dtype, `bf16` or `fp16`.
  - `use_grad_scaler`  
    Use gradient scaling. Usually needed for `fp16`, often optional for `bf16`.

#### `test`
- `resize_size`  
  Resize size during testing.
- `logging.log_curr_mets_n_batches`  
  Metric logging frequency during testing: once per `N` batches.
- `logging.klog_io_n_batches`  
    Log input/output examples every `N` batches.

Notes:
- Training uses `AdamW` and `CyclicLR`.
- The config path is set in `cfg_loader.py`.

---

## Training

In `main.py`, set the path to the CSV labels file:

```python
csv_path = Path(__file__).parent / "dataset" / "dataset_labels.csv"
main(csv_path)
```

Run:

```bash
python main.py
```

---

## Checkpoints and logging

- checkpoints are saved to `general.checkpoints_dir`
- TensorBoard logs are written to `train.logging.log_dir`

Run TensorBoard:

```bash
tensorboard --logdir=./tb_logger --bind_all --samples_per_plugin "images=1000, scalars=100000"
```

---

## References
- Sebastian Lutz, Konstantinos Amplianitis, Aljosa Smolic. "AlphaGAN: Generative adversarial networks for natural image matting." arXiv:1807.10088, 2018.

---

## License
This project is licensed under the MIT License. See `LICENSE`.
