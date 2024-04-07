import numpy as np
from openvino.inference_engine import IECore
from tensorflow.keras.datasets import mnist
import time

phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test.reshape(-1, 28, 28).astype(np.float32)  # Reshape and cast to float32
y_test = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to categorical
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load the OpenVINO Inference Engine
ie = IECore()

# Load the OpenVINO IR model
openvino_model_xml = 'ffnn_model_fp16.xml'
openvino_model_bin = 'ffnn_model_fp16.bin'
net = ie.read_network(model=openvino_model_xml, weights=openvino_model_bin)

# Load the network onto the Intel Neural Compute Stick
exec_net = ie.load_network(network=net, device_name='MYRIAD')

print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Initialize accuracy counter
tflite_accuracy = 0.0

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Run inference on the test data
for i in range(len(x_test)):
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    input_data = {input_blob: x_test[i:i+1]}
    res = exec_net.infer(input_data)
    tflite_predictions = res[output_blob]
    tflite_accuracy += y_test[i] == np.argmax(tflite_predictions, axis=1)

tflite_accuracy /= len(x_test)
accuracy_percentage = float(tflite_accuracy) * 100
print(f'TFLite Model Accuracy: {accuracy_percentage:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")