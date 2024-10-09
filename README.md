# EdgeAi
# AI on Edge Testing Project

## Overview
This project is aimed at testing the performance of various AI models on edge devices. The edge devices used in this project are Raspberry Pi 4, Intel Neural Compute Stick with Raspberry Pi, and Nvidia Jetson Nano. The project involves training different categories of models on a server, saving the trained models, and then deploying them onto the edge devices for inference. Performance metrics such as energy consumption, memory utilization, and inference time will be evaluated for each model.

## Device Setup
Before running the project, ensure the following devices are set up and configured properly:
- Raspberry Pi 4
- Intel Neural Compute Stick with Raspberry Pi
- Nvidia Jetson Nano

## Model Categories
The project involves testing three categories of models:
1. Traditional Machine Learning Models:
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Linear Model
   - Decision Tree (DT)
   
2. Neural Network Models:
   - Artificial Neural Network (ANN)
   - Convolutional Neural Network (CNN)
   - Feedforward Neural Network (FFNN)
   - Region-based Convolutional Neural Network (R-CNN)
   
3. Deep Learning Models:
   - ResNet-50
   - BERT
   - Single Shot MultiBox Detector (SSD)


## Training Models
To train the models, navigate to the `train` directory and run the respective scripts for each model category. Trained models will be saved in the corresponding directories.

## Transforming Models to TensorFlow Lite
To transform the trained models to TensorFlow Lite format for deployment on edge devices, use the `tflite_transform.py` script located in the `inference` directory. The command to run the script is as follows:


## Running Models on Intel Neural Compute Stick
To run the models on the Intel Neural Compute Stick using OpenVINO, first, ensure that OpenVINO is installed on your system. Then, use the `openvino_transform.py` script located in the `stick` directory to transform the TensorFlow Lite models to the format compatible with OpenVINO. The command to run the script is as follows:


## Running Models on Intel Neural Compute Stick
To run the models on the Intel Neural Compute Stick using OpenVINO, follow these steps:

1. Activate the OpenVINO environment:
   ```bash
   source /opt/intel/openvino/bin/setupvars.sh

Convert the TensorFlow Lite model to OpenVINO IR format using the Model Optimizer:
```
mo_tf.py --input_model model.tflite --output_dir output_directory --data_type FP16 --reverse_input_channels
```
Replace model.tflite with the path to your TensorFlow Lite model and output_directory with the directory where you want the IR files to be saved.

--mo_tf.py: This is the Model Optimizer script for TensorFlow models.
--input_model model.tflite: Specifies the input TensorFlow Lite model.
--output_dir output_directory: Specifies the directory where the IR files will be saved.
--data_type FP16: Specifies the precision of the IR files. You can choose FP16, FP32, or INT8 based on your requirements.
--reverse_input_channels: This option is used if the input image has channels in the RGB order. It's often required for TensorFlow Lite models.

After executing the command, you will find the generated XML and BIN files in the specified output directory.

## Prerequisite Packages Installation
Ensure that the following prerequisite packages are installed on your system:
- TensorFlow Lite
- OpenVINO

You can install TensorFlow Lite using pip:
```
pip install tensorflow-lite
```
on the raspberry remember that you have to create a virtual environment.


For installing OpenVINO, please refer to the official documentation provided by Intel.

## Conclusion
This project provides a comprehensive framework for testing AI models on edge devices. By following the instructions outlined in this README, you can train, transform, and deploy models on various edge devices and evaluate their performance metrics effectively.



### Table 1: Hyperparameters of KNN, Decision Tree (DT), SVM, and Linear Classifier

| **Model** | **Hyperparameters**                                                                                                                                         |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **KNN**   | - Neighbors: 3<br>- Data Precision: 16-bit                                                                                                               |
| **DT**    | - Data Precision: 16-bit<br>- Random State: Fixed                                                                                                        |
| **SVM**   | - Gamma: 0.001<br>- Data Precision: 16-bit                                                                                                               |
| **Linear**| - Optimizer: SGD<br>- Loss Function: Categorical Cross-Entropy<br>- Epochs: 5<br>- Batch Size: 64<br>- Validation Split: 20%<br>- Quantization: FP16 for TensorFlow Lite |

### Table 2: Hyperparameters of FFNN, CNN, R-CNN, and ANN

| **Model** | **Hyperparameters**                                                                                                                                         |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **FFNN**  | - Architecture: 2 Dense Layers (128, 64 units) with ReLU<br>- Output Layer: 10 units with Softmax<br>- Optimizer: Adam<br>- Loss: Categorical Cross-Entropy<br>- Epochs: 5<br>- Batch Size: 64<br>- Validation Split: 20%<br>- Quantization: FP16 for TensorFlow Lite |
| **CNN**   | - Layers: 2 Conv Layers (32, 64 filters), 3x3 kernels<br>- Pooling: 2 Max-Pooling layers (2x2)<br>- Dense Layer: 128 units with ReLU<br>- Output Layer: 10 units with Softmax<br>- Optimizer: Adam<br>- Loss: Categorical Cross-Entropy<br>- Epochs: 5<br>- Batch Size: 64<br>- Quantization: FP16 for TensorFlow Lite |
| **R-CNN** | - Conv Layers: 3 Layers (32, 64, 64 filters), 3x3 kernels<br>- Pooling: 3 Max-Pooling layers (2x2)<br>- Dense Layer: 256 units with ReLU<br>- Output Layer: 10 units with Softmax<br>- Optimizer: Adam<br>- Loss: Categorical Cross-Entropy<br>- Epochs: 5<br>- Batch Size: 32<br>- Quantization: FP16 for TensorFlow Lite |
| **ANN**   | - Architecture: Custom Layer Configuration<br>- Optimizer: Adam<br>- Loss: Categorical Cross-Entropy<br>- Epochs: 10<br>- Batch Size: 64<br>- Validation Split: 20%  |

### Table 3: Hyperparameters of ResNet50 and MobileSSD

| **Model**     | **Hyperparameters**                                                                                                                                         |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **ResNet-50** | - Architecture: 50 layers with GlobalAveragePooling2D<br>- Dense Layers: 1024 units with ReLU<br>- Image Size: 224x224 pixels<br>- Optimizer: Adam<br>- Loss Function: Sparse Categorical Cross-Entropy<br>- Batch Size: 32<br>- Transfer Learning: Pre-trained ImageNet weights with frozen layers |
| **MobileSSD** | - Architecture: Depthwise Separable Convolutions<br>- Image Size: 224x224 pixels<br>- Optimizer: Adam<br>- Loss Function: Sparse Categorical Cross-Entropy<br>- Batch Size: 32<br>- Transfer Learning: Pre-trained ImageNet weights with frozen layers |

### Table 4: Hyperparameters of TinyBERT and Phi-2-Orange

| **Model**     | **Hyperparameters**                                                                                                                                         |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **TinyBERT**  | - Learning Rate: \(2 \times 10^{-5}\)<br>- Epochs: 3<br>- Optimizer: Adam<br>- Loss Function: Sparse Categorical Cross-Entropy<br>- Dataset: GLUE<br>- Tokenizer: BertTokenizer<br>- Deployment: TensorFlow Lite |
| **Phi-2-Orange** | - Batch Size (Training): 16<br>- Batch Size (Evaluation): 64<br>- Epochs: 3<br>- Warm-up Steps: Configured<br>- Weight Decay: Configured<br>- Dataset: OpenAssistant<br>- Managed by: Hugging Face Trainer |
