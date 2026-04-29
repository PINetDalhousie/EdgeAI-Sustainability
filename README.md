# Sustainable Edge AI Inference: Measuring Performance and Energy Trade-offs

[![Python 3.9](https://img.shields.io/badge/Python-3.9.2-green.svg)](https://www.python.org/)
[![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Replication package** for the paper *"Sustainable Edge AI Inference: Measuring Performance and Energy Trade-offs"* — a unified benchmarking framework for evaluating AI model inference performance and energy consumption across heterogeneous edge devices.

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
- [Running Experiments](#running-experiments)
- [Key Results Summary](#key-results-summary)
- [Citation](#citation)

---

## Overview

This project benchmarks AI model inference across heterogeneous edge devices, measuring four key metrics — **F1-score**, **inference time**, **memory utilization**, and **power consumption** — under a unified, hardware-based measurement scheme. It covers the full model spectrum from traditional ML classifiers through neural networks and deep learning to large language models.

The framework addresses a key challenge in edge AI: no single software-based power measurement tool works across all platforms (see [Power Measurement Setup](#power-measurement-setup)). Our unified approach uses a hardware USB power meter at 16 Hz with baseline subtraction, ensuring directly comparable measurements across all devices.

**What's included:**
- Training scripts for 12 AI models across 4 categories
- Model conversion pipelines for LiteRT (`.tflite`), OpenVINO IR, TensorRT, and EdgeTPU
- Unified measurement script with 5-step inference protocol
- Raw results with 95% confidence intervals

---

## Repository Structure

```
edgeai/
├── train/                          # Model training scripts
│   ├── traditional_ml/             # KNN, SVM, DT, Linear Regression
│   ├── neural_networks/            # ANN, CNN, FFNN, R-CNN
│   ├── deep_learning/              # ResNet-50, MobileSSD
│   └── llm/                        # TinyBERT, Phi-2-Orange
├── models/                         # Saved trained models (.h5)
├── inference/                      # Inference and measurement scripts
│   ├── measure_inference.py        # Unified 5-step measurement script
│   ├── tflite_transform.py         # H5 → TFLite conversion
│   └── utils/
│       ├── metrics.py              # F1-score, accuracy computation
│       └── power_parser.py         # USB power meter log parser
├── stick/                          # Intel NCS / OpenVINO scripts
│   └── openvino_transform.py       # TFLite → OpenVINO IR conversion
├── conversion/                     # Device-specific model conversions
│   ├── tensorrt_transform.py       # TF → TensorRT (for Jetson Nano)
│   └── edgetpu_compile.sh          # TFLite → EdgeTPU (for Coral USB)
├── data/                           # Test datasets
├── results/                        # Raw measurement results (with CIs)
└── README.md
```

---

## Hardware Requirements

| Device | Specifications | Architecture | Role |
|--------|---------------|-------------|------|
| **Raspberry Pi 4 Model B** | 4 GB LPDDR4, 1.5 GHz quad-core ARM Cortex-A72, RPi OS Bullseye 64-bit | CPU (ARM) | Baseline edge device |
| **Intel Neural Compute Stick 2** | Intel Movidius Myriad X VPU | VPU | USB accelerator (via RPi USB 3.0) |
| **Google Coral USB Accelerator** | Google Edge TPU ASIC | TPU | USB accelerator (via RPi USB) |
| **NVIDIA Jetson Nano Dev Kit** | 128-core Maxwell GPU, quad-core ARM Cortex-A57, 4 GB LPDDR4 | GPU | GPU-accelerated edge device |
| **USB Power Meter** | 4–24 V, 0–6.5 A, **16 Hz** sampling, ±1% voltage / ±2% current | — | Hardware power measurement |

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

### Install Dependencies

```bash
# On each device — create virtual environment first
python3 -m venv edgeai_env
source edgeai_env/bin/activate
pip install tflite-runtime==2.12.0 psutil==5.9.4 numpy==1.23.5
```

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
| **Traditional ML** | KNN, SVM, Decision Tree, Linear Regression | MNIST | Lightweight |
| **Neural Networks** | ANN, CNN, FFNN, R-CNN | MNIST | ~10K–500K |
| **Deep Learning** | ResNet-50, MobileSSD | ImageNet | ~3.5M–25.6M |
| **Large Language Models** | TinyBERT, Phi-2-Orange | GLUE / OpenAssistant | 14.5M / 2.7B |

### Training

```bash
# Train all models (run on server with GPU)
cd train/traditional_ml/
python3 train_knn.py && python3 train_svm.py && python3 train_dt.py && python3 train_linear.py

cd ../neural_networks/
python3 train_cnn.py && python3 train_ffnn.py && python3 train_ann.py && python3 train_rcnn.py

cd ../deep_learning/
python3 train_resnet50.py && python3 train_mobilessd.py

cd ../llm/
python3 train_tinybert.py && python3 train_phi2orange.py
```

All trained models are saved in **H5 format** in the `models/` directory.

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

### Step 1: H5 → TensorFlow Lite

```bash
python3 inference/tflite_transform.py --input models/model.h5 \
    --output models/tflite/model.tflite --quantize fp16
```

### Step 2: TFLite → OpenVINO IR (for INCS)

```bash
source /opt/intel/openvino/bin/setupvars.sh

mo_tf.py --input_model models/tflite/model.tflite \
    --output_dir models/ir/ \
    --data_type FP16 \
    --reverse_input_channels
```

- `--data_type FP16`: Reduces precision for Myriad X VPU acceleration
- `--reverse_input_channels`: Required for TFLite models with RGB input order
- **Output:** `.xml` (network architecture) + `.bin` (model weights)

### Step 3: TFLite → EdgeTPU (for GCU)

```bash
# First create INT8 quantized model (EdgeTPU requires full INT8)
python3 inference/tflite_transform.py --input models/model.h5 \
    --output models/tflite/model_int8.tflite --quantize int8

# Compile for EdgeTPU
edgetpu_compiler models/tflite/model_int8.tflite -o models/edgetpu/
```

### Step 4: H5 → TensorRT (for NJn)

```bash
python3 conversion/tensorrt_transform.py --input_model models/model.h5 \
    --output_dir models/tensorrt/ --precision FP16
```

**TensorRT optimizations applied:** layer fusion (Conv + BatchNorm + ReLU → single kernel), FP16 precision calibration, kernel auto-tuning for Maxwell GPU.

---

## Inference and Measurement

### Measurement Script

The unified measurement script (`inference/measure_inference.py`) executes a **5-step process** on every device:

```
Step 1: Initialize     → Record baseline power (3s idle = 48 samples at 16 Hz)
Step 2: Load Dataset   → Load test data into memory
Step 3: Load Model     → Load model via appropriate interpreter/runtime
Step 4: Execute        → Run inference, log predictions with timestamps
Step 5: Finalize       → Stop power monitoring, compute and save all metrics
```

> **Note:** Figure 2 in the paper shows the pseudocode of this script for clarity. The full executable implementation is in this repository at `inference/measure_inference.py`.

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

**Our solution:** A hardware USB power meter connected between the power supply and the device under test, providing consistent 16 Hz measurements (±1% voltage, ±2% current) across all platforms.

### Baseline Power Subtraction

The first 3 seconds of each run capture **48 power samples** (at 16 Hz) while the device is near-idle. The mean of these samples is the baseline power, which is subtracted from inference-phase readings to isolate compute-specific power consumption.

### Thermal Management

- **NJn:** Active cooling (fan) enabled to prevent throttling
- **RPi:** Passive heatsinks installed
- **All devices:** 5-minute warm-up after boot before measurements begin
- No thermal throttling was observed during experiments

### Synchronization

The measurement script logs timestamps at each phase boundary via `time.strftime()`. The USB power meter records time-stamped readings continuously. Post-experiment, we align power meter timestamps with script phase timestamps to extract inference-specific power data.

---

## Experimental Protocol

| Parameter | Value |
|-----------|-------|
| **Repetitions (Traditional ML, Neural Networks)** | **10×** per experiment |
| **Repetitions (ResNet-50, MobileSSD)** | **5×** per experiment |
| **Repetitions (TinyBERT, Phi-2-Orange)** | **1×** (extended runtime) |
| **Test batch size** | 25 images (vision models) |
| **Reporting** | Mean ± **95% confidence interval** |
| **Pre-measurement** | Terminate background processes, reactivate venv, verify no caching |
| **Warm-up** | 5 minutes after boot |

All reported device differences are statistically significant (non-overlapping 95% confidence intervals).

---

## Running Experiments

```bash
# Activate virtual environment on the target device
source edgeai_env/bin/activate

# --- Raspberry Pi (LiteRT) ---
python3 inference/measure_inference.py \
    --model models/tflite/cnn.tflite \
    --device rpi \
    --dataset data/mnist/ \
    --output results/rpi/cnn_results.json

# --- RPi + Intel Neural Compute Stick (OpenVINO) ---
python3 inference/measure_inference.py \
    --model models/ir/cnn.xml \
    --device incs \
    --dataset data/mnist/ \
    --output results/rpi_incs/cnn_results.json

# --- RPi + Google Coral USB (EdgeTPU) ---
python3 inference/measure_inference.py \
    --model models/edgetpu/cnn_edgetpu.tflite \
    --device gcu \
    --dataset data/mnist/ \
    --output results/rpi_gcu/cnn_results.json

# --- Jetson Nano (LiteRT) ---
python3 inference/measure_inference.py \
    --model models/tflite/cnn.tflite \
    --device njn_litert \
    --dataset data/mnist/ \
    --output results/njn_litert/cnn_results.json

# --- Jetson Nano (TensorRT) ---
python3 inference/measure_inference.py \
    --model models/tensorrt/cnn.plan \
    --device njn_tensorrt \
    --dataset data/mnist/ \
    --output results/njn_tensorrt/cnn_results.json
```

---

## Key Results Summary

| Finding | Detail |
|---------|--------|
| **Fastest inference** | NJn + TensorRT across all model categories |
| **Most power-efficient (traditional ML)** | RPi + LiteRT (~2–3 W) |
| **Best speed-power balance (vision)** | RPi + GCU (EdgeTPU) |
| **LLM comparison** | TinyBERT outperforms Phi-2 in power (3×) and memory (16×) on all devices |
| **GPU acceleration** | NJn 1.7–2.1× faster than RPi for neural networks (128-core Maxwell parallelism) |

Full results with confidence intervals are in the `results/` directory and in the paper.

---

## Citation

```bibtex
@article{sobhani2025sustainability,
  title={On the Sustainability of AI Inferences in the Edge},
  author={Sobhani, Ghazal and Ifath, Md Monzurul Amin and Sharma, Tushar and Haque, Israat},
  journal={arXiv preprint arXiv:2507.23093},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

**Contact:** [PINet Lab, Dalhousie University](https://github.com/PINetDalhousie) · Issues: [GitLab Issues](https://gitlab.com/sobhanii/edgeai/-/issues)
