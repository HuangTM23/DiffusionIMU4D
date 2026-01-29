# DiffusionIMU4D

This project implements an **IMU-based Trajectory Estimation** system using **Diffusion Models**, specifically designed to refine or generate 4D trajectories (Velocity + Time) from raw inertial data.

It provides two main architectural variants:
- **Variant A (Residual/Refinement)**: Uses a ResNet1D prior to estimate a coarse trajectory, then uses a Diffusion model to predict the residual errors.
- **Variant B (End-to-End Conditional)**: Uses a ResNet1D purely as a feature encoder to condition a Diffusion model, which generates the trajectory from noise.

## 📂 Project Structure

```
├── configs/             # Configuration files for training variants
├── data/                # Data loading logic and dataset wrappers
├── models/              # PyTorch model definitions (ResNet1D, DiffUNet1D, DiffusionSystem)
├── utils/               # Utility functions (Geometry, Logging, Metrics)
├── train_diff.py        # Main training script
├── test_diff.py         # Inference and evaluation script
├── train_ronin.py       # Baseline RoNIN training script
└── requirements.txt     # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Ensure your RoNIN dataset is placed in `data/RoNIN`. The directory structure should look like:
```
data/RoNIN/
  ├── extracted/       # Processed .hdf5 files
  └── lists/          # list_train.txt, list_val.txt, list_test.txt
```

### 3. Training

**Variant A (Residual Scheme):**
```bash
python train_diff.py --config configs/diffusion_variant_a.yaml
```

**Variant B (End-to-End Scheme):**
```bash
python train_diff.py --config configs/diffusion_variant_b.yaml
```

### 4. Evaluation / Inference

```bash
# Example: Evaluate using the Variant A config and a trained checkpoint
python test_diff.py --config configs/diffusion_variant_a.yaml --checkpoint experiments/checkpoints/diff_residual_epoch_99.pth
```

## 📊 Configuration (WandB)

Weights & Biases logging is configured via `configs/wandb.yaml`.
You can switch to offline mode or change the project name there.

```yaml
project_name: "Diffusion4d-Diff"
mode: "online" # or "offline"
```

## 🧠 Architectures

Visual architecture diagrams are available in the root directory:
- `architecture_variant_a.canvas`
- `architecture_variant_b.canvas`

## 📄 License

MIT