# Cycle-Crack PyTorch Implementation

This repository contains an unofficial PyTorch implementation of **Cycle-Crack: Weakly Supervised Pavement Crack Segmentation via Dual-Cycle Generative Adversarial Learning**.

## Project Structure
```
cycle_crack/
├── models/
├── losses/
├── data/
├── checkpoints/
├── config.yaml
├── train.py
├── inference.py
└── evaluate.py
```

## Setup
1. Switch to this directory and activate your python environment.
2. Install dependencies (make sure `torch` is compatible with your setup):
```bash
pip install torch torchvision numpy opencv-python pyyaml tqdm tensorboard
```
3. Prepare your dataset in `data/cracked/` and `data/crack_free/`.

## Training
To train the model from scratch with mixed precision, run:

```bash
python train.py --config config.yaml
```

Training hyperparameters, data paths, and loss weights can be modified in `config.yaml`. Training metrics and progress can be monitored via TensorBoard:

```bash
tensorboard --logdir checkpoints/logs
```

## Inference
To segment a cracked image using the trained Generator (GE):

```bash
python inference.py --image_path path/to/image.jpg --checkpoint checkpoints/GE_epoch_200.pth --output_dir outputs/
```

The script will output:
1. Final binary mask (`mask_{img_name}`)
2. Normalized error map (`error_{img_name}`)

## Evaluation
Use the predefined metrics inside `evaluate.py` (`calculate_metrics`) to compute Precision, Recall, F1, IoU, and Boundary F1 (mBF1) when comparing the generated mask against a ground truth mask.
# cycle_crack
# cycle_crack
