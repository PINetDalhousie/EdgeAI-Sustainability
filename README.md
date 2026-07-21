# Sustainable Edge AI Inference: Measuring Performance and Energy Trade-offs

[![Python 3.9](https://img.shields.io/badge/Python-3.9.2-green.svg)](https://www.python.org/)
[![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Replication package** for the paper *"Characterizing Performance–Power Trade-offs of AI Inference on Heterogeneous Edge Devices"* — a unified benchmarking framework for evaluating AI model inference performance and energy consumption across heterogeneous edge devices.

> **Note:** This repository is the artifact accompanying the above paper (currently under
> review). Everything the paper reports — the measurement scheme, measurement window,
> models, datasets, frameworks, metrics, and results — is described here to match the
> paper. Results the paper **defers to this repository for space** (MobileSSD input-size
> impact and the LLM parameter-tuning tables) live in [`results/`](results/README.md). The
> repository also ships some **supplementary material beyond the paper's scope** (e.g.
> Traditional ML models/results), which is clearly marked where it appears.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Hardware Requirements](#hardware-requirements)
- [Software Dependencies](#software-dependencies)
- [Device Setup](#device-setup)
- [Model Categories](#model-categories)
- [Hyperparameters](#hyperparameters)
- [Model Conversion Pipeline](#model-conversion-pipeline)
- [Inference and Measurement](#inference-and-measurement)
- [Power Measurement Setup](#power-measurement-setup)
- [Experimental Protocol](#experimental-protocol)
- [Reproducibility](#reproducibility)
- [Running Experiments](#running-experiments)
- [Key Results Summary](#key-results-summary)
<!-- - [Citation](#citation) -->

---

## Overview

This project benchmarks AI model inference across heterogeneous edge devices, measuring four key metrics — **F1-score**, **inference time**, **memory utilization**, and **power consumption** — under a unified, hardware-based measurement scheme. It covers the full model spectrum from traditional ML classifiers through neural networks and deep learning to large language models.

The framework addresses a key challenge in edge AI: no single software-based power measurement tool works across all platforms (see [Power Measurement Setup](#power-measurement-setup)). Our unified approach uses a hardware USB power meter at 16 Hz with baseline subtraction, ensuring directly comparable measurements across all devices.

**What's included:**
- Training scripts for 12 AI models across 4 categories
- Model conversion pipelines for LiteRT (`.tflite`), OpenVINO IR, TensorRT, and EdgeTPU
- Per-device inference scripts following a unified 5-phase measurement protocol
- Post-processing scripts for phase identification and energy computation from hardware power-meter logs

---

## Repository Structure

The repository is organized by **model family** (`ML-NN/`, `DL/`, `LLM/`), and within
each family by **target device** (`train/`, `raspberry/`, `stick/`, `coral/`, `jetson/`).
Each device directory holds the inference scripts run on that platform.

```
EdgeAI-Sustainability/
├── ML-NN/                          # Traditional ML + Neural Networks
│   │                               #   (KNN, SVM, DT, Linear, ANN, CNN, FFNN, R-CNN)
│   ├── train/                      # Training + FP16/16-bit quantization scripts
│   ├── raspberry/                  # Raspberry Pi (LiteRT) inference scripts
│   │   └── lite_images/            # Saved trained models (.h5, .tflite, .pkl, .joblib)
│   ├── stick/                      # Intel NCS 2 (OpenVINO) inference scripts
│   ├── coral/                      # Google Coral USB (Edge TPU) inference scripts
│   └── jetson/                     # NVIDIA Jetson Nano inference scripts
├── DL/                             # Deep Learning (ResNet-50, MobileSSD)
│   ├── train/                      # Training scripts (ResNet-50, MobileSSD, TinyBERT)
│   ├── raspberry/                  # Raspberry Pi inference scripts
│   ├── stick/                      # Intel NCS 2 inference scripts
│   ├── coral/                      # Google Coral USB inference scripts
│   ├── jetson/                     # NVIDIA Jetson Nano inference scripts
│   └── dl_models.py                # Consolidated training/conversion reference (Colab export)
├── LLM/                            # Large Language Models (TinyBERT, Phi-2-Orange)
│   ├── train/                      # Training scripts (Phi-2-Orange)
│   ├── raspberry/                  # Raspberry Pi inference scripts
│   ├── stick/                      # Intel NCS 2 inference scripts
│   ├── coral/                      # Google Coral USB inference scripts
│   └── jetson/                     # NVIDIA Jetson Nano inference scripts
├── results/                        # Supplementary results referenced in the paper
│   ├── README.md                   #   LLM parameter-tuning tables (token length / window)
│   └── Input_Size_Impact_on_MobileSSD_combined.pdf   # MobileSSD input-resolution figure
├── docs/
│   └── power-measurement.md        # USB power meter (FNB48S) + measurement procedure
├── measurments.py                  # Energy computation from USB power-meter CSV logs
├── phase_identification.py         # Splits power-meter logs into the 5 inference phases
└── README.md
```

> **Note on measurement:** there is no single `measure_inference.py`. Instead, each
> per-device inference script prints phase-boundary timestamps and computes F1/accuracy
> and memory (via `psutil`); the two top-level scripts (`phase_identification.py` and
> `measurments.py`) then post-process the hardware power-meter log against those
> timestamps. See [Inference and Measurement](#inference-and-measurement).

---

## Hardware Requirements

| Device | Specifications | Architecture | Role |
|--------|---------------|-------------|------|
| **Raspberry Pi 4 Model B** | 4 GB LPDDR4, 1.5 GHz quad-core ARM Cortex-A72 ‡, RPi OS Bullseye 64-bit | CPU (ARM) | Baseline edge device |
| **Intel Neural Compute Stick 2** | Intel Movidius Myriad X VPU | VPU | USB accelerator (via RPi USB 3.0) |
| **Google Coral USB Accelerator** | Google Edge TPU ASIC | TPU | USB accelerator (via RPi USB) |
| **NVIDIA Jetson Nano Dev Kit** | 128-core Maxwell GPU, quad-core ARM Cortex-A57, 4 GB LPDDR4 | GPU | GPU-accelerated edge device |
| **USB Power Meter** | 4–24 V, 0–6.5 A, **16 Hz** sampling, ±1% voltage / ±2% current | — | Hardware power measurement |

> ‡ The Raspberry Pi 4 Model B ships with a quad-core ARM **Cortex-A72**; the paper prints
> "Cortex-A57" for the RPi, which is a typo (the Cortex-A57 is the Jetson Nano's CPU).

These devices were selected for: (1) widespread adoption in edge AI research, (2) architectural diversity (CPU / VPU / TPU / GPU), and (3) accessibility for researchers. See the paper for discussion of excluded platforms (Qualcomm Snapdragon NPU, Apple Neural Engine).

---

## Software Dependencies

All experiments used these **exact versions** for reproducibility:

| Package | Version | Devices |
|---------|---------|---------|
| Python | 3.9.2 | All |
| TensorFlow | 2.12.0 | Training server |
| tflite-runtime | 2.12.0 | RPi, RPi+GCU, NJn (LiteRT) |
| OpenVINO | 2023.0 | RPi + INCS |
| TensorRT | 8.5.1 | NJn (TensorRT) |
| CUDA | 11.4.3 | NJn |
| cuDNN | 8.6.0 | NJn |
| JetPack | 4.6.1 | NJn |
| psutil | 5.9.4 | All (memory measurement) |
| numpy | 1.23.5 | All |
| libedgetpu1-std | 16.0 | RPi + GCU |
| edgetpu-compiler | 16.0 | GCU model compilation |
| **OS** | RPi OS Bullseye 64-bit (kernel 6.1) | RPi-based configs |

**Additional Python packages** imported by the scripts (install per configuration; versions
follow each platform's availability):

| Package | Needed for |
|---------|-----------|
| scikit-learn, joblib | Traditional ML models (KNN, SVM, Decision Tree) — training and inference |
| transformers, datasets, torch | LLMs (TinyBERT, Phi-2-Orange) — training and Phi-2 inference |
| tensorflow-datasets | ImageNet (`imagenet_v2/matched-frequency`) for ResNet-50 / MobileSSD training |
| opencv-python (`cv2`), Pillow (`PIL`) | Image pre-processing in some inference scripts |
| pandas | `measurments.py` power-log post-processing |
| pycoral / edgetpu | GCU (Edge TPU) runtime bindings |
| pycuda, tensorrt, torch2trt | NJn TensorRT inference |

### Install Dependencies

```bash
# On each device — create virtual environment first
python3 -m venv edgeai_env
source edgeai_env/bin/activate

# Core (all configs)
pip install tflite-runtime==2.12.0 psutil==5.9.4 numpy==1.23.5

# Traditional ML + Neural Networks
pip install scikit-learn joblib

# Large Language Models (RPi/INCS host and NJn)
pip install transformers datasets torch

# Post-processing on the analysis machine
pip install pandas
```

> LiteRT inference uses `tflite-runtime`; the training/conversion scripts use full
> `tensorflow==2.12.0`. Accelerator runtimes (OpenVINO, Edge TPU, TensorRT) are installed
> as described in [Device Setup](#device-setup), not via `pip`.

---

## Device Setup

### Raspberry Pi 4 (RPi)

```bash
# Flash Raspberry Pi OS Bullseye 64-bit
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv
python3 -m venv edgeai_env
source edgeai_env/bin/activate
pip install tflite-runtime==2.12.0 psutil==5.9.4 numpy==1.23.5
```

### RPi + Intel Neural Compute Stick 2 (INCS)

Complete the RPi setup above, then install OpenVINO:

```bash
# Follow: https://docs.openvino.ai/2023.0/openvino_docs_install_guides_installing_openvino_raspbian.html
source /opt/intel/openvino/bin/setupvars.sh
```

### RPi + Google Coral USB Accelerator (GCU)

Complete the RPi setup above, then install EdgeTPU runtime:

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
    sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update
sudo apt install -y libedgetpu1-std
```

> **Note:** EdgeTPU requires fully quantized INT8 TFLite models. See [Model Conversion Pipeline](#model-conversion-pipeline).

### NVIDIA Jetson Nano (NJn)

```bash
# Flash JetPack 4.6.1 to SD card (includes CUDA 11.4.3, cuDNN 8.6.0, TensorRT 8.5.1)
sudo apt update && sudo apt upgrade -y
python3 -m venv edgeai_env
source edgeai_env/bin/activate
pip install tflite-runtime==2.12.0 psutil==5.9.4 numpy==1.23.5

# Verify TensorRT:
python3 -c "import tensorrt; print(tensorrt.__version__)"  # → 8.5.1
```

---

## Model Categories

| Category | Models | Dataset | Parameters |
|----------|--------|---------|------------|
| **Traditional ML** † | KNN, SVM, Decision Tree, Linear Regression | MNIST | Lightweight |
| **Neural Networks** | ANN, CNN, FFNN, R-CNN | MNIST | ~10K–500K |
| **Deep Learning** | ResNet-50, MobileSSD | ImageNet | ~3.5M–25.6M |
| **Large Language Models** | TinyBERT, Phi-2-Orange | GLUE / OASST1 | 14.5M / 2.7B |

> † **Supplementary material.** The paper evaluates **Neural Networks, Deep Learning, and
> LLMs** (8 models). The **Traditional ML** category (KNN, SVM, Decision Tree, Linear) and
> its results are included in this repository as extra material and are **not reported in
> the paper** (omitted due to page constraints). The scripts are kept here for
> completeness and reproducibility.

### Training

Training scripts live in the `train/` subfolder of each model family. Run them on a
server with a GPU.

```bash
# Traditional ML + Neural Networks (ML-NN/train/)
cd ML-NN/train/
python3 knn_train.py && python3 train_svm_16.py && python3 train_dt_16.py && python3 train_linear_16.py
python3 cnn_train.py && python3 train_ffnn_16.py && python3 ann_train_16.py && python3 train_r_cnn_16.py

# Deep Learning (DL/train/)
cd ../../DL/train/
python3 ResNet50_imagenet.py && python3 ResNet50_mnist.py
python3 mobileSSD_imagenet.py && python3 mobileSSD_mnist.py
python3 tinyBERT.py

# Large Language Models (LLM/train/)
cd ../../LLM/train/
python3 phi_2_orange_train.py
```

> `DL/dl_models.py` is a consolidated Colab export containing reference training and
> TFLite-conversion snippets for the deep-learning models.

Traditional-ML models are saved as `.pkl`/`.joblib` and neural/deep models in **H5**
(and their converted `.tflite`) format. Copies of the saved model images used on the
Raspberry Pi are checked in under `ML-NN/raspberry/lite_images/`.

---

## Hyperparameters

### Traditional ML Models

| Model | Hyperparameters |
|-------|----------------|
| **KNN** | Neighbors: 3, Data Precision: 16-bit |
| **Decision Tree** | Data Precision: 16-bit, Random State: Fixed |
| **SVM** | Gamma: 0.001, Data Precision: 16-bit |
| **Linear Classifier** | Optimizer: SGD, Loss: Categorical Cross-Entropy, Epochs: 5, Batch Size: 64, Validation Split: 20%, Quantization: FP16 |

### Neural Network Models

| Model | Hyperparameters |
|-------|----------------|
| **FFNN** | 2 Dense Layers (128, 64 units) + ReLU, Output: 10 + Softmax, Optimizer: Adam, Loss: Categorical Cross-Entropy, Epochs: 5, Batch Size: 64, Validation Split: 20%, Quantization: FP16 |
| **CNN** | 2 Conv Layers (32, 64 filters, 3×3), 2 MaxPool (2×2), Dense: 128 + ReLU, Output: 10 + Softmax, Optimizer: Adam, Epochs: 5, Batch Size: 64, Quantization: FP16 |
| **R-CNN** | 3 Conv Layers (32, 64, 64 filters, 3×3), 3 MaxPool (2×2), Dense: 256 + ReLU, Output: 10 + Softmax, Optimizer: Adam, Epochs: 5, Batch Size: 32, Quantization: FP16 |
| **ANN** | Custom Layer Config, Optimizer: Adam, Loss: Categorical Cross-Entropy, Epochs: 10, Batch Size: 64, Validation Split: 20% |

### Deep Learning Models

| Model | Hyperparameters |
|-------|----------------|
| **ResNet-50** | 50 layers + GlobalAveragePooling2D, Dense: 1024 + ReLU, Input: 224×224, Optimizer: Adam, Loss: Sparse Categorical Cross-Entropy, Batch Size: 32, Transfer Learning: Pre-trained ImageNet (frozen layers) |
| **MobileSSD** | Depthwise Separable Convolutions, Input: 224×224 (300×300 for detection), Optimizer: Adam, Loss: Sparse Categorical Cross-Entropy, Batch Size: 32, Transfer Learning: Pre-trained ImageNet (frozen layers) |

### Large Language Models

| Model | Hyperparameters |
|-------|----------------|
| **TinyBERT** | Architecture: 4 transformer layers, hidden dim 312 (distilled from BERT-base), 14.5M params. Learning Rate: 2×10⁻⁵, Epochs: 3, Optimizer: Adam, Loss: Sparse Categorical Cross-Entropy, Dataset: GLUE, Tokenizer: BertTokenizer, Deployment: TFLite |
| **Phi-2-Orange** | Architecture: 2.7B params (Phi-2 base). Training Batch Size: 16, Eval Batch Size: 64, Epochs: 3, Warm-up Steps: Configured, Weight Decay: Configured, Dataset: OpenAssistant, Managed by: Hugging Face Trainer |

---

## Model Conversion Pipeline

All models follow a standardized conversion pipeline from training format to device-optimized format:

```
H5 (trained model)
 │
 ├──→ .tflite (FP16) ──────────────→ RPi (LiteRT interpreter)
 │         │
 │         ├──→ OpenVINO IR ────────→ RPi + INCS (Inference Engine)
 │         │    (.xml + .bin)          via mo_tf.py --data_type FP16
 │         │
 │         └──→ EdgeTPU .tflite ────→ RPi + GCU (EdgeTPU runtime)
 │              (INT8 quantized)       via edgetpu_compiler
 │
 └──→ TensorRT engine ─────────────→ NJn (TF-TRT, FP16/INT8)
      (.plan)                          via TF-TRT integration
```

> **Note:** conversion is performed **inline inside the training scripts** rather than by
> standalone converter scripts. The FP16 training/quantization scripts in `ML-NN/train/`
> (the `*_16.py` files) emit the `.tflite` artifacts directly, and `DL/dl_models.py`
> contains the reference `TFLiteConverter` snippets for the deep-learning models. The
> commands below show the underlying operations each script performs.

### Step 1: H5 → TensorFlow Lite

The training scripts convert with `tf.lite.TFLiteConverter.from_keras_model(...)` after
saving the H5 model. Equivalent standalone conversion:

```python
import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]   # FP16
tflite_model = converter.convert()
open('model_fp16.tflite', 'wb').write(tflite_model)
```

### Step 2: TFLite → OpenVINO IR (for INCS)

Uses the OpenVINO Model Optimizer CLI (`mo_tf.py`) shipped with OpenVINO 2023.0:

```bash
source /opt/intel/openvino/bin/setupvars.sh

mo_tf.py --input_model model.tflite \
    --output_dir ir/ \
    --data_type FP16 \
    --reverse_input_channels
```

- `--data_type FP16`: Reduces precision for Myriad X VPU acceleration
- `--reverse_input_channels`: Required for TFLite models with RGB input order
- **Output:** `.xml` (network architecture) + `.bin` (model weights)

### Step 3: TFLite → EdgeTPU (for GCU)

```bash
# First create a fully INT8-quantized .tflite (EdgeTPU requires full INT8) via the
# TFLiteConverter with a representative dataset, then compile with the Edge TPU CLI:
edgetpu_compiler model_int8.tflite -o edgetpu/
```

The Coral inference scripts (`*/coral/*.py`) load the resulting
`*_edgetpu.tflite` model via the `libedgetpu.so.1` delegate.

### Step 4: H5 → TensorRT (for NJn)

On the Jetson Nano, models are optimized through **TF-TRT** (TensorFlow-TensorRT
integration, bundled with JetPack 4.6.1). The Jetson inference scripts under
`*/jetson/` load the TF-TRT–optimized graph at FP16 precision:

```python
from tensorflow.python.compiler.tensorrt import trt_convert as trt
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='saved_model/',
    precision_mode=trt.TrtPrecisionMode.FP16)
converter.convert()
converter.save('tensorrt_model/')
```

**TensorRT optimizations applied:** layer fusion (Conv + BatchNorm + ReLU → single kernel), FP16 precision calibration, kernel auto-tuning for Maxwell GPU.

---

## Inference and Measurement

### Measurement Workflow

The measurement is not a single script — it is a **unified 5-phase protocol** that every
per-device inference script follows, combined with two top-level post-processing scripts
that align the results with the hardware power-meter log:

```
Phase 1: Initialize     → Baseline / idle window (3s ≈ 48 samples at 16 Hz)
Phase 2: Load Dataset   → Load test data into memory
Phase 3: Load Model     → Load model via the device's interpreter/runtime
Phase 4: Inference      → Run inference, compute F1/accuracy
Phase 5: Memory Util.   → Record memory (psutil) and finalize
```

**How it works in the code:**

1. **Per-device inference scripts** (e.g. `ML-NN/raspberry/knn_inference_16.py`,
   `ML-NN/coral/cnn_inference.py`) print a `Start`/`End` timestamp at each phase boundary
   via `time.strftime()`, and compute F1/accuracy plus memory usage with `psutil`.
2. The **USB power meter** logs time-stamped power readings continuously to a CSV/text file.
3. **`phase_identification.py`** splits that power log into the five phases using the
   timestamps printed by the inference script.
4. **`measurments.py`** adds a timestamp column to the power CSV (from a start time and the
   16 Hz sampling rate) and sums the energy within the inference time range.

> **Note:** Figure 2 in the paper shows the pseudocode of this protocol for clarity. Its
> executable form is distributed across the per-device inference scripts plus
> `phase_identification.py` and `measurments.py` in this repository.

### Evaluation Metrics

| Metric | Tool | Unit |
|--------|------|------|
| **F1-Score** | Ground-truth comparison | % |
| **Inference Time** | `time.time()` (start/end delta) | seconds |
| **Memory Utilization** | `psutil` (total RAM during inference) | MB |
| **Power Consumption** | USB power meter (16 Hz, hardware) | Watts |

---

## Power Measurement Setup

No single software tool supports power measurement across all our devices:

| Device | Software Tools Available | Limitation |
|--------|--------------------------|-----------|
| Raspberry Pi | PiJuice API, PowerAPI | Not compatible with USB accelerators |
| Intel NCS | Intel Power Gadget, psutil | No direct power reading |
| Google Coral USB | PowerAPI, psutil | No dedicated power API |
| Jetson Nano | tegrastats, Jetson Power Monitor | Not comparable to RPi tools |

**Our solution:** A hardware USB power meter (**FNB48S** USB-C multimeter, 156 W) connected inline between the power supply and the device under test, providing consistent 16 Hz measurements (±1% voltage, ±2% current) across all platforms. Its recorded sessions are read out and exported to CSV with the **Power-Z (KM001)** Windows application (<https://sourceforge.net/projects/power-z-km001/>).

> **Details:** see [`docs/power-measurement.md`](docs/power-measurement.md) for the meter,
> wiring, software readout, and the full step-by-step measurement procedure.

### Baseline Power Subtraction

The first 3 seconds of each run capture **48 power samples** (at 16 Hz) while the device is near-idle. The mean of these samples is the baseline power, which is subtracted from inference-phase readings to isolate compute-specific power consumption.

### Thermal Management

- **NJn:** Active cooling (fan) enabled to prevent throttling
- **RPi:** Passive heatsinks installed
- **All devices:** 5-minute warm-up after boot before measurements begin
- No thermal throttling was observed during experiments

### Synchronization

Each per-device inference script logs timestamps at every phase boundary via
`time.strftime()`. The USB power meter records time-stamped readings continuously.
Post-experiment, `phase_identification.py` and `measurments.py` align the power-meter
timestamps with the script's phase timestamps to extract inference-specific power data.

---

## Experimental Protocol

| Parameter | Value |
|-----------|-------|
| **Repetitions** | **Each test run 10×** |
| **Reporting** | Mean ± **95% confidence interval** |
| **Pre-measurement** | Terminate background processes, reactivate venv, verify no caching |
| **Warm-up** | Stable thermal conditions with active cooling where necessary |

Following the paper's evaluation protocol, **each test is run ten times and we report the
average and 95% confidence interval**.

> **Note on batch size / input parameters:** the default measurements use the models'
> standard input settings. Inference-time parameters — input resolution (e.g. ResNet-50
> 224×224 → 512×512), batch size (the RPi effectively supports only batch size 1, while
> the Jetson Nano sustains 16–32), and, for LLMs, input token length and token window
> size — are studied separately as tunable parameters in the paper (RQ3). The detailed
> per-device results the paper defers for space (MobileSSD input-resolution impact, and
> LLM input-token-length / token-window-size tables) are provided in
> [`results/`](results/README.md).

---

## Reproducibility

Reproducibility is a core contribution of this work, so the exact behaviour of the scripts
is documented here. Read this before [Running Experiments](#running-experiments).

### What the scripts assume

- **No command-line arguments.** Every training and inference script is run as
  `python3 <script>.py`. Model paths, dataset choices, and hyperparameters are hard-coded
  inside each script — reproduction depends on running the script from the correct working
  directory with the expected model file present, **not** on passing flags. *(Do not modify
  the scripts; adjust the working directory / file placement instead.)*
- **Datasets download automatically on first run:**
  - **MNIST** — via `keras.datasets.mnist.load_data()` (used by the NN and traditional-ML
    scripts, and the accuracy check in several DL scripts).
  - **ImageNet** (`imagenet_v2/matched-frequency`) — via `tensorflow-datasets`, for
    ResNet-50 / MobileSSD training (large; one-time TFDS setup).
  - **GLUE** (TinyBERT) and **OASST1 / OpenAssistant** (Phi-2-Orange) — via `transformers` /
    `datasets`; Phi-2 weights are pulled from the Hugging Face Hub.
- **Models are loaded by fixed filename** from the script's working directory (a few read
  from a `lite_images/` subfolder). The exact name is set at the top of each script
  (`model_path=` / `load_model(...)` / `joblib.load(...)`).

### Provided vs. generated artifacts

- **Provided in the repo:** the pre-trained / FP16-converted **RPi (LiteRT)** model images in
  [`ML-NN/raspberry/lite_images/`](ML-NN/raspberry/lite_images/) (`.h5`, FP16 `.tflite`,
  `.pkl`, `.joblib`). The RPi neural-network and traditional-ML experiments run directly
  from these — no retraining required.
- **Generated by you:** accelerator-specific artifacts are **not** checked in and must first
  be produced with the [Model Conversion Pipeline](#model-conversion-pipeline):
  - **INCS** → OpenVINO IR (`*.xml` + `*.bin`)
  - **GCU** → Edge TPU-compiled (`*_edgetpu.tflite`)
  - **NJn (TensorRT)** → TF-TRT SavedModel directory (e.g. `resnet50_tensorrt/`, `tinybert_tensorrt/`)
  - DL / LLM base models (ResNet-50, MobileSSD, TinyBERT, Phi-2) are trained and converted
    with the scripts in `DL/train/` and `LLM/train/`.

### Model artifact filenames (produced by the `*/train/` scripts)

| Model | RPi (LiteRT) | GCU (Edge TPU) | INCS (OpenVINO IR) |
|-------|--------------|----------------|--------------------|
| KNN | `knn_model_quantized.pkl` | `knn_model_quantized_edgetpu.tflite` | `knn_model_fp16.xml` + `.bin` |
| Decision Tree | `decision_tree_model_quantized.joblib` | *(same `.joblib`)* | `decision_tree_model_quantized.xml` + `.bin` |
| SVM | `svm_model_quantized.joblib` | `svm_model_quantized_edgetpu.tflite` | `svm_model_quantized.xml` + `.bin` |
| Linear | `linear_classifier_model_fp16.tflite` | `linear_classifier_model_edgetpu.tflite` | `linear_classifier_model_fp16.xml` + `.bin` |
| ANN | `neural_network_model_float16.h5` | `quantized_neural_network_model_fp16_edgetpu.tflite` | `quantized_neural_network_model_fp16.xml` + `.bin` |
| CNN | `quantized_cnn_model_fp16.tflite` | `quantized_cnn_model_edgetpu.tflite` | `cnn_lite.xml` + `.bin` |
| FFNN | `ffnn_model_fp16.tflite` | `model_edgetpu.tflite` | `ffnn_model_fp16.xml` + `.bin` |
| R-CNN | `r_cnn_model_fp16.tflite` | `r_cnn_model_quantized_edgetpu.tflite` | `r_cnn_model_fp16.xml` + `.bin` |

> Filenames above are exactly those the scripts open. Place (or symlink) the required file
> into the script's working directory before running.

### End-to-end reproduction of one experiment

1. Set up the device and virtual environment ([Device Setup](#device-setup),
   [Install Dependencies](#install-dependencies)).
2. Obtain the model artifact — use the provided `lite_images/` file (RPi), or run the
   [training](#training) + [conversion](#model-conversion-pipeline) steps for the target device.
3. Place the artifact where the script expects it (same directory, exact filename above).
4. Start the USB power-meter capture (16 Hz, time-stamped log).
5. Run the inference script (`python3 <model>_inference.py`); it prints per-phase
   `Start`/`End` timestamps and the F1/accuracy + memory results.
6. Post-process: `phase_identification.py` splits the power log by those timestamps, then
   `measurments.py` integrates energy over the inference window after baseline subtraction.
7. Repeat **10×** per configuration and report **mean ± 95% confidence interval**.

---

## Running Experiments

Each experiment runs the inference script that matches the **model family** and **target
device**. The script prints per-phase timestamps and the accuracy/memory results; run it
alongside the USB power-meter capture, then post-process with `phase_identification.py`
and `measurments.py`. The CNN model is used as an example below.

> The **RPi** example runs directly from the checked-in `lite_images/` artifacts. The
> **INCS / GCU / NJn** commands additionally require you to first generate and place the
> device-specific artifact — see [Reproducibility](#reproducibility).

```bash
# Activate virtual environment on the target device
source edgeai_env/bin/activate

# --- Raspberry Pi (LiteRT) ---
cd ML-NN/raspberry/
python3 cnn_inference_lite.py

# --- RPi + Intel Neural Compute Stick 2 (OpenVINO) ---
cd ML-NN/stick/
python3 cnn_inference.py

# --- RPi + Google Coral USB (Edge TPU) ---
cd ML-NN/coral/
python3 cnn_inference.py

# --- Jetson Nano ---
cd ML-NN/jetson/
python3 cnn_inference.py
```

Every device folder uses the same `<model>_inference.py` naming, so only the directory
changes between devices. Deep-learning and LLM experiments follow the same pattern under
`DL/<device>/` and `LLM/<device>/` (e.g. `DL/jetson/resnet50_inference.py`,
`LLM/raspberry/tinyBERT_inference.py`). On the Raspberry Pi, some models additionally
carry a variant suffix — `_lite` (LiteRT/TFLite), `_16` (16-bit), or a dataset name
(`_mnist`, `_imagenet`).

**Post-processing the power-meter log:**

```bash
# 1. Split the raw power log into the 5 phases using the timestamps the inference
#    script printed
python3 phase_identification.py        # edit input_file / phases at the bottom of the script

# 2. Compute energy over the inference time range from the power-meter CSV
python3 measurments.py                 # prompts for CSV path, start timestamp, sampling rate
```

---

## Key Results Summary

| Finding | Detail |
|---------|--------|
| **Fastest inference** | NJn + TensorRT — fastest for deep-learning and LLM workloads; GCU (EdgeTPU) is competitive on neural networks |
| **Lowest power** | RPi consumes the lowest inference power, as expected (though it is the slowest and most memory-hungry) |
| **Performance–energy sweet spot** | NJn delivers superior performance at competitive energy, unlike the common assumption that faster edge platforms are more power-hungry |
| **Accelerator gains** | TensorRT (NJn) gives the largest speed/efficiency gains; OpenVINO (INCS) and EdgeTPU (GCU) help but less so; EdgeTPU notably improves LLM inference time and power on the RPi over INCS |
| **LLM comparison** | TinyBERT is far lighter than Phi-2-Orange — ≈16× less memory at default settings (up to 40% less when tuned) and lower inference power on all devices |

Full evaluation results (Figure 2) with 95% confidence intervals are reported in the
paper. Additional results deferred from the paper for space — the MobileSSD input-size
figure and the LLM parameter-tuning tables (input token length and token window size) —
are in [`results/`](results/README.md).

<!-- ---

## Citation

```bibtex
@article{sobhani2025sustainability,
  title={On the Sustainability of AI Inferences in the Edge},
  author={Sobhani, Ghazal and Ifath, Md Monzurul Amin and Sharma, Tushar and Haque, Israat},
  journal={arXiv preprint arXiv:2507.23093},
  year={2025}
}
```

--- -->

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

**Contact:** [PINet Lab, Dalhousie University](https://github.com/PINetDalhousie) · Issues: [GitLab Issues](https://gitlab.com/sobhanii/edgeai/-/issues)
