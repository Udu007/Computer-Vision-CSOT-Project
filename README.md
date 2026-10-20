# Computer Vision Project: From MLPs to Transfer Learning

A three-week, hands-on progression through image classification in **PyTorch**. It starts with a plain fully connected network and ends with ResNet-18 transfer learning on satellite imagery. Each week's notebook trains a model, saves it, and then reloads it in a separate cell to evaluate it.

| Week | Notebook | Dataset | Models |
|------|----------|---------|--------|
| 1 | [`Week 1.ipynb`](Week%201.ipynb) | Fashion-MNIST (28×28 grayscale, 10 classes) | Fully connected network vs. a small CNN |
| 2 | [`Week 2.ipynb`](Week%202.ipynb) | EuroSAT (64×64 RGB satellite images, 10 land-use classes) | Custom residual CNN with batch norm and data augmentation |
| 3 | [`Week 3.ipynb`](Week%203.ipynb) | EuroSAT, resized to 224×224 | ResNet-18 (ImageNet weights): full fine-tuning and frozen backbone |

---

## Week 1: Fully connected network vs. CNN on Fashion-MNIST

Two baselines are trained with plain SGD (learning rate 0.02, batch size 64) and cross-entropy loss.

- **Fully connected network (MLP):** `784 → 512 → 512 → 10` with ReLU. Trained for 25 epochs and saved as `Build1.pth`.
- **CNN:** four 3×3 conv layers (16 → 32 → 64 → 128 channels) with two max-pool stages, then a `6272 → 512 → 256 → 10` classifier. Trained for 60 epochs and saved as `Build2.pth`.

| Model | Test accuracy (Fashion-MNIST test set) |
|-------|------|
| MLP | **87.8%** |
| CNN | **91.9%** |

The CNN's convolutions share weights across the image, and that gives a clear gain over flattening the pixels. The CNN's training loss drops close to zero while its test loss rises, which is a textbook sign of overfitting. That motivated the regularisation used in Week 2.

## Week 2: A residual CNN on EuroSAT

This is a custom network written from scratch with:
- two convolutional stages (32 → 64 and 128 → 256 channels), each with **batch normalisation**, ReLU and max pooling
- **residual (skip) connections** around each stage, using strided 1×1 convolutions so the shapes match
- adaptive max pooling and a `512 → 256 → 10` head with batch norm

Training setup:
- **TrivialAugmentWide** data augmentation, with normalisation statistics computed for EuroSAT
- SGD with momentum 0.9, weight decay 1e-4 and a **cosine-annealing** learning-rate schedule over 100 epochs
- a random 80/20 train/test split

## Week 3: Transfer learning with ResNet-18

Images are resized to 224×224 and normalised with ImageNet statistics, with the same TrivialAugmentWide augmentation and an 80/20 split.

1. **Full fine-tuning:** an ImageNet-pretrained ResNet-18 whose final layer is replaced by a `512 → 128 → 10` head with batch norm. All weights are trained with SGD (momentum 0.9) for 60 epochs, using cosine annealing and mixed-precision (AMP) training.
2. **Frozen backbone:** the ResNet-18 convolutional layers are frozen. A new residual block (512 → 1024 channels) and a small classifier are trained on top for 80 epochs. This reaches **96.4% accuracy on the held-out 20% split**.

---

## Getting started

```bash
pip install torch torchvision
jupyter notebook "Week 1.ipynb"
```

The datasets download automatically through `torchvision.datasets` into `./data`. A GPU is strongly recommended for Weeks 2 and 3. The notebooks were run on Google Colab.

Each notebook has a **training** cell, which trains and saves a `.pth` model, and a **prediction** cell, which reloads the saved model and evaluates it. Run the training cell first.

## Notes and possible improvements

- The prediction cells in Weeks 2 and 3 evaluate on the **whole** EuroSAT dataset, which includes the images used for training. Their headline numbers (99.5% and 99.8%) are therefore optimistic. A fair evaluation would reuse the held-out split from training, with the same `torch.manual_seed`.
- The Week 3 fine-tuning cell saves `Build1.pth`, but its prediction cell loads `Build 2 Week 3.pth`. Rename one of them so the two match.
- Possible next steps:
  - early stopping, or saving the best checkpoint (the code for this is already present but commented out)
  - a validation set separate from the test set
  - confusion matrices to see which land-use classes get confused

## What I learned

- Why convolutions beat fully connected layers on images, and how to spot overfitting from the train and test curves.
- How batch normalisation, residual connections, data augmentation and learning-rate schedules work together in a deeper network.
- When to fine-tune a whole pretrained network and when to freeze its features and train only a new head.
