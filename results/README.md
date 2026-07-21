# Supplementary Results

This directory holds additional experimental results that are **referenced in the paper**
*"Characterizing Performance–Power Trade-offs of AI Inference on Heterogeneous Edge
Devices"* but could not be included inline due to page constraints. They support the
inference-time parameter-tuning analysis (**RQ3**).

All metrics were collected with the same measurement scheme as the main experiments —
inference time via Python's `time` module, memory via `psutil`, and power from a hardware
USB power meter sampled at **16 Hz** with idle-baseline subtraction. Each test was run
**10 times**; values are reported as **mean [lower, upper] 95% confidence interval**
(where shown). See the [top-level README](../README.md) for the full methodology.

## Contents

| File | Description |
|------|-------------|
| [`Input_Size_Impact_on_MobileSSD_combined.pdf`](Input_Size_Impact_on_MobileSSD_combined.pdf) | Impact of input resolution on **MobileSSD** across edge devices |
| Tables below | Impact of **input token length** and **token window size** on LLM performance |

---

## 1. Input-size impact on MobileSSD

Figure: [`Input_Size_Impact_on_MobileSSD_combined.pdf`](Input_Size_Impact_on_MobileSSD_combined.pdf)

Complements Figure 3 in the paper (which shows ResNet-50). Scaling MobileSSD's input
resolution from **300×300 to 512×512** consistently improves F1-score across all devices,
while inference time and power rise most sharply on the RPi and remain comparatively
stable on the NJn with TensorRT — confirming that the resolution–cost trade-off is
governed primarily by device capability rather than model architecture.

---

## 2. LLM inference-time parameter tuning

For LLMs, the dominant inference-time parameters are **input token length** and **token
window size** (the model processes tokenized text sequences). The tables below report how
each parameter affects F1-score, inference time, power, and memory for **TinyBERT** and
**Phi-2-Orange** on the RPi, RPi + INCS, and NJn.

### 2.1 Impact of input token length

*Metrics reported with their 95% confidence interval in square brackets.*

| Device | Model | Optimal Input Size (tokens) | F1-score (%) | Inference Time (s) | Inference Power (W) | Memory Utilization (MB) |
|--------|-------|-----------------------------|--------------|--------------------|---------------------|-------------------------|
| RPi | TinyBERT | 500–1000 | 85.50 [84.12, 86.88] | 9.86 [9.35, 10.37] | 46.5 [34.05, 51.45] | 332.24 |
| RPi | Phi-2-Orange | 500–1000 | 83.90 [82.78, 84.83] | 32.76 [29.48, 36.00] | 62.4 [48.96, 75.84] | 5,250 |
| RPi + INCS | TinyBERT | 1024 | 87.60 [86.32, 88.88] | 6.97 [6.46, 7.48] | 43.2 [29.76, 56.64] | 257.2 |
| RPi + INCS | Phi-2-Orange | 1024 | 85.90 [84.77, 86.98] | 14.56 [11.64, 17.47] | 55.5 [42.45, 58.65] | 4,620 |
| NJn | TinyBERT | 2048 | 87.90 [86.70, 89.10] | 6.29 [5.78, 6.80] | 52.2 [40.05, 64.35] | 280 |
| NJn | Phi-2-Orange | 2048 | 85.90 [84.75, 87.06] | 9.64 [8.37, 10.92] | 93.12 [79.52, 106.72] | 4,930 |

Increasing input token length causes super-linear growth in inference time and memory
(expanded attention and KV-cache). Both models operate reliably only up to **500–1000
tokens on the RPi**, whereas the **NJn sustains up to 2048 tokens** thanks to its larger
compute and memory capacity.

### 2.2 Impact of token window size

*Metrics reported with their 95% confidence interval in square brackets.*

| Device | Model | Token Window Size | F1-score (%) | Inference Time (s) | Inference Power (W) | Memory Utilization (MB) |
|--------|-------|-------------------|--------------|--------------------|---------------------|-------------------------|
| RPi | TinyBERT | 512 | 85.10 [83.66, 86.54] | 9.42 [8.90, 9.94] | 45.3 [32.78, 49.82] | 320.14 |
| RPi | Phi-2-Orange | 512 | 83.40 [82.22, 84.58] | 30.24 [27.96, 32.52] | 60.8 [46.78, 74.82] | 5,100 |
| RPi + INCS | TinyBERT | 1024 | 87.00 [85.64, 88.36] | 6.32 [5.84, 6.80] | 41.0 [27.72, 54.28] | 249.8 |
| RPi + INCS | Phi-2-Orange | 1024 | 85.50 [84.32, 86.68] | 13.26 [10.90, 15.62] | 53.1 [40.22, 56.98] | 4,550 |
| NJn | TinyBERT | 2048 | 87.50 [86.20, 88.80] | 5.95 [5.45, 6.45] | 51.0 [38.55, 63.45] | 275 |
| NJn | Phi-2-Orange | 2048 | 85.70 [84.55, 86.85] | 9.10 [7.98, 10.22] | 91.2 [77.45, 104.95] | 4,890 |

Larger token windows improve language coherence but amplify inference time and power. On
the **RPi and RPi + INCS**, window sizes above **512 tokens** cause sharp increases in
power and memory, whereas the **NJn efficiently supports 2048-token windows**. Across all
configurations, **TinyBERT consistently uses less memory and power than Phi-2-Orange**,
confirming its suitability for resource-constrained edge deployment.
