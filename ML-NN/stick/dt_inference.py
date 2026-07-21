import cv2
import numpy as np
import time

# Load the OpenVINO Inference Engine
from openvino.inference_engine import IECore

phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], -1)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the saved Decision Tree model
# Modify this part to load the OpenVINO IR model
ie = IECore()
net = ie.read_network(model='decision_tree_model_quantized.xml', weights='decision_tree_model_quantized.bin')
exec_net = ie.load_network(network=net, device_name='MYRIAD')

print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Run inference on the test data using the loaded model
correct_predictions = 0
for i in range(len(x_test)):
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    input_data = {input_blob: x_test[i].reshape(1, -1)}
    res = exec_net.infer(input_data)
    predictions = res[output_blob]
    predicted_label = np.argmax(predictions)
    if predicted_label == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(x_test)
print(f'Accuracy (Loaded Model): {accuracy * 100:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
