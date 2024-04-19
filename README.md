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

