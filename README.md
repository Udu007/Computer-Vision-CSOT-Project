# Computer Vision Project: From Image Classification to Multi-Object Tracking

A hands-on progression through computer vision in **PyTorch**. It runs from a plain fully connected classifier up to a **custom multi-object tracker** that combines a YOLO detector, a Kalman filter, Hungarian matching and Transformer-based re-identification. Each stage builds on the one before.

| Stage | Notebook | Task | What's inside |
|-------|----------|------|---------------|
| Week 1 | [`Week 1.ipynb`](Week%201.ipynb) | Image classification on Fashion-MNIST | Fully connected network vs. a small CNN |
| Week 2 | [`Week 2.ipynb`](Week%202.ipynb) | Image classification on EuroSAT (satellite images) | Residual CNN from scratch with batch norm and data augmentation |
| Week 3 | [`Week 3.ipynb`](Week%203.ipynb) | Transfer learning on EuroSAT | ResNet-18: full fine-tuning vs. frozen backbone |
| Week 4 | [`Week 4.ipynb`](Week%204.ipynb) | Object detection and tracking | YOLOv8 on video, ByteTrack, and a MOT evaluator written from scratch |
| Final project | [`Final_Project.ipynb`](Final_Project.ipynb) | Multi-object tracking of pedestrians | A **BoT-SORT-style tracker built from scratch**, with YOLO detections, a Kalman filter, two-stage matching and Swin Transformer Re-ID |

---

## Part 1: Image classification (Weeks 1–3)

### Week 1: Fully connected network vs. CNN on Fashion-MNIST
Both models are trained with plain SGD (learning rate 0.02, batch size 64) and cross-entropy loss.
- **MLP:** `784 → 512 → 512 → 10` with ReLU, trained for 25 epochs.
- **CNN:** four 3×3 conv layers (16 → 32 → 64 → 128 channels) with two max-pool stages, then a `6272 → 512 → 256 → 10` classifier, trained for 60 epochs.

| Model | Test accuracy (Fashion-MNIST test set) |
|-------|------|
| MLP | **87.8%** |
| CNN | **91.9%** |

Convolutions share weights across the image, which gives a clear gain over flattening the pixels. The CNN's training loss falls almost to zero while its test loss rises, a textbook sign of overfitting. That motivated the regularisation in Week 2.

### Week 2: A residual CNN on EuroSAT
This is a network written from scratch for 64×64 satellite images in 10 land-use classes. It has:
- two convolutional stages (32 → 64 and 128 → 256 channels), each with **batch normalisation**, ReLU and max pooling
- **residual connections** around each stage, using strided 1×1 convolutions so the shapes match
- **TrivialAugmentWide** augmentation, SGD with momentum and weight decay, and a **cosine-annealing** learning-rate schedule
- a random 80/20 train/test split

### Week 3: Transfer learning with ResNet-18
Images are resized to 224×224 and normalised with ImageNet statistics.
1. **Full fine-tuning:** an ImageNet-pretrained ResNet-18 gets a new `512 → 128 → 10` head, and the whole network is trained for 60 epochs (SGD with momentum, cosine annealing, mixed precision).
2. **Frozen backbone:** the ResNet-18 features are frozen. A new residual block (512 → 1024 channels) and a small classifier are trained on top for 80 epochs. This reaches **96.4% on the held-out 20% split**.

---

## Part 2: Detection and multi-object tracking (Week 4 and the final project)

The goal is to follow every pedestrian through a video sequence and keep the **same ID** on each person from frame to frame. Results are written in the standard **MOTChallenge format** (`frame, id, x, y, w, h, …`) and scored against ground truth.

### Week 4: Detection, ByteTrack and evaluation
- **Video detection:** a pretrained YOLOv8n runs on a video clip with a confidence threshold of 0.4 and saves the annotated output.
- **ByteTrack with YOLOv8x:** a tuned tracker configuration with:
  - high and low detection thresholds of 0.35 and 0.10
  - a match threshold of 0.75
  - a 90-frame track buffer, so tracks survive short occlusions
  - fused detector-confidence and IoU scores

  It runs at 1280-pixel input with test-time augmentation and filters out tiny or weak boxes.
- **A MOT evaluator written from scratch:** it matches ground truth to predictions in each frame with the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) on an IoU cost, using an IoU threshold of 0.5. It then computes precision, recall, **MOTA**, ID switches, IDP, IDR and **IDF1**.

### Final project: A custom BoT-SORT-style tracker
The core tracking logic is written from scratch, following the ideas of SORT, ByteTrack and BoT-SORT:

- **Detector:** YOLO (`yolo26x`) restricted to the *person* class, at 1280-pixel input with a low confidence floor (0.05), so weak detections can still be used in a second matching stage.
- **Motion model:** a **Kalman filter** implemented by hand. Its state holds box centre, size, velocity and acceleration (10 dimensions), with explicit predict and update steps.
- **Appearance model (Re-ID):** an ImageNet-pretrained **Swin Transformer (Swin-S)** with its classification head removed. It turns each detected person crop into a 768-dimensional embedding. Each track keeps an exponentially smoothed appearance feature, updated only from confident detections.
- **Two-stage association, as in ByteTrack:**
  1. High-confidence detections are matched first. The cost combines IoU distance with **gated cosine distance between Re-ID features**, following BoT-SORT: appearance only counts when both the appearance and the overlap are close enough.
  2. Tracks left unmatched get a second chance against low-confidence detections, using IoU alone. This recovers people who are partly hidden.
- **Track management:** a track is only reported after it has been seen at least twice. Tracks are kept alive for up to 30 frames without a detection.
- **Evaluation** with the standard [`motmetrics`](https://github.com/cheind/py-motmetrics) library (patched for NumPy 2.0).

### Results

| Tracker | MOTA | IDF1 | Precision | Recall | ID switches |
|---------|------|------|-----------|--------|-------------|
| ByteTrack (YOLOv8x), with my own evaluator | 24.9% | 42.6% | 91.2% | 27.8% | 43 |
| Custom BoT-SORT + Swin Re-ID (YOLO26x), with `motmetrics` | 25.7% | 25.0% | 76.1% | 39.9% | 481 |

Both runs are scored against the same ground truth (29,193 boxes). In both, **recall** limits performance, because many pedestrians in crowded frames are never detected.
- The custom tracker lowers the detection threshold and recovers weak detections. That raises recall and MOTA slightly, at the cost of more false positives and ID switches.
- ByteTrack keeps identities more consistently, which shows in its higher IDF1.

The two runs used different evaluators, so the numbers are indicative rather than a strict head-to-head comparison.

---

## Getting started

```bash
pip install torch torchvision ultralytics opencv-python scipy pyyaml motmetrics
```

- **Weeks 1–3:** the datasets download automatically through `torchvision.datasets`. Run each notebook's training cell first, then its prediction cell.
- **Week 4 and the final project:** put the sequence frames as `.jpg` files in an image folder (`Images4/` by default) and the ground truth in `ref/gt.txt`, both in MOTChallenge format. The trackers write `ref_bytetrack_submission.txt` and `ref_botsort_submission.txt`, which the evaluation cells then score.

A GPU is strongly recommended from Week 2 onwards. The notebooks were run on Google Colab.

## Notes and possible improvements

- **Classification:**
  - The Week 2 and Week 3 prediction cells evaluate on the whole EuroSAT dataset, including training images. Their 99.5% and 99.8% figures are therefore optimistic; the held-out split is the fair test.
  - The Week 3 fine-tuning cell saves `Build1.pth`, while its prediction cell loads `Build 2 Week 3.pth`.
- **Tracking:** the biggest gain would come from better detection recall, for example a detector fine-tuned on crowded pedestrian scenes. After that:
  - camera-motion compensation, as in full BoT-SORT
  - a Re-ID network trained on person re-identification data instead of ImageNet features
  - evaluating every tracker with the same `motmetrics` script

## What I learned

- Why convolutions beat fully connected layers on images, and how to spot and fight overfitting.
- How batch normalisation, residual connections, augmentation and learning-rate schedules work together, and when to fine-tune a pretrained network rather than freeze it.
- How a tracker turns per-frame detections into identities over time:
  - Kalman-filter prediction
  - Hungarian matching
  - two-stage use of low-confidence detections
  - appearance features for re-identification
- How tracking is measured (MOTA, IDF1, ID switches), and why a tracker can only be as good as its detector.
