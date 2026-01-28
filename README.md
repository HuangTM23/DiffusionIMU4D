# DiffusionIMU4D

[English](./README.md) | [中文](./README_zh.md)

**DiffusionIMU4D** is a research project exploring the application of **Conditional Diffusion Models** to Inertial Navigation. By remodeling the IMU-to-Velocity task as a conditional sequence generation problem, we aim to overcome the long-term drift and smoothing effects common in traditional regression-based methods (e.g., RoNIN, TLIO).

## 🚀 Project Goal

The primary goal is to verify whether generative models can recover high-fidelity 3D velocity sequences from noisy IMU data, providing a robust alternative to end-to-end regression.

**Key Deliverables:**
*   **Diffusion-based Core**: A 1D UNet-based diffusion model conditioned on IMU features.
*   **Hybrid Architecture**: Supports both **Residual Refinement** (correcting a coarse guess) and **End-to-End Generation**.
*   **Evaluation System**: Full trajectory integration and error analysis metrics (ATE, RTE).

## 🏗️ Architecture

The system consists of three main components:

1.  **Condition Encoder (`ResNet1D`)**:
    *   Takes raw IMU data (Accel + Gyro) as input.
    *   Extracts deep temporal features to guide the generation process.
2.  **Denoising Network (`DiffUNet1D`)**:
    *   A 1D U-Net that iteratively denoises a random Gaussian noise sequence into a clean velocity curve.
    *   Uses **Cross-Attention / FiLM** to incorporate IMU features.
3.  **Physics Integration**:
    *   Integrates the generated velocity sequence into a 3D trajectory.

### Two Modes
*   **Variant A (Residual)**: A lightweight PriorNet (RoNIN-like) predicts a coarse velocity, and the Diffusion Model generates the *residual* to refine details.
*   **Variant B (End-to-End)**: The Diffusion Model generates the full velocity sequence from pure noise.

## 🛠️ Setup & Usage

### 1. Environment
```bash
conda create -n DiffM python=3.9
conda activate DiffM
pip install -r requirements.txt
```

### 2. Data Preparation
This project uses the **RoNIN Dataset**.

1.  **Download Data**:
    *   Visit the [RoNIN Project Website](http://ronin.cs.sfu.ca/) or [Data Repository](https://doi.org/10.20383/102.0543) to download the dataset.
2.  **Extract & Place**:
    *   Extract the downloaded data into `data/RoNIN/extracted/`.
    *   **Expected Directory Structure** (each sequence is a folder containing `data.hdf5` and `info.json`):
        ```text
        data/RoNIN/extracted/
        ├── a001_1/
        │   ├── data.hdf5
        │   └── info.json
        ├── a001_2/
        ...
        ```
3.  **Verify Lists**:
    *   Ensure the sequence names in `data/RoNIN/lists/list_train.txt` (e.g., `a001_1`) match the folders you actually have.
    *   *Note: The public RoNIN dataset might not contain all sequences listed in the default full lists. If you encounter a FileNotFoundError, please remove the missing sequence names from the txt lists.*

### 3. Training
**Train Residual Scheme (Recommended First):**
```bash
python train_diff.py --config configs/diffusion_variant_a.yaml
```

**Train End-to-End Scheme:**
```bash
python train_diff.py --config configs/diffusion_variant_b.yaml
```

### 4. Evaluation
```bash
python test_diff.py --config configs/diffusion_variant_a.yaml --checkpoint experiments/checkpoints/your_ckpt.pth
```

## 📄 License
MIT License
