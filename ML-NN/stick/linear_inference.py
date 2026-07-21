import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import time
from openvino.inference_engine import IECore

phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test.reshape(-1, 784).astype(np.float32)  # Reshape and cast to float32
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)  # One-hot encode labels
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Initialize the OpenVINO Inference Engine
ie = IECore()

# Load the TensorFlow Lite model onto the Intel Neural Compute Stick
net = ie.read_network(model='linear_classifier_model_fp16.xml', weights='linear_classifier_model_fp16.bin')
exec_net = ie.load_network(network=net, device_name='MYRIAD')

# Get input and output details
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize accuracy counter
tflite_accuracy = 0.0

phase_name = 'Inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Perform inference on the test data
for i in range(len(x_test)):
    res = exec_net.infer(inputs={input_blob: x_test[i:i+1]})
    tflite_predictions = res[output_blob]
    tflite_accuracy += np.argmax(y_test[i:i+1], axis=1) == np.argmax(tflite_predictions, axis=1)

tflite_accuracy /= len(x_test)
print(f'TFLite Model Accuracy: {tflite_accuracy.item() * 100:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")