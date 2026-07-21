import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from tensorflow.keras.datasets import mnist
import time
from openvino.inference_engine import IECore

phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], -1)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Quantize the data to 16-bit precision
x_test_quantized = np.float16(x_test)

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the saved KNN model
loaded_knn = joblib.load('knn_model_quantized.pkl')  # Load the quantized model
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize the OpenVINO Inference Engine
ie = IECore()

# Load the network onto the Intel Neural Compute Stick
net = ie.read_network(model='knn_model_fp16.xml', weights='knn_model_fp16.bin')
exec_net = ie.load_network(network=net, device_name='MYRIAD')

# Make predictions on the quantized test data
y_pred = []
for i in range(len(x_test_quantized)):
    input_data = {'input': x_test_quantized[i:i+1]}
    res = exec_net.infer(inputs=input_data)
    y_pred.append(np.argmax(res['output']))

# Calculate the accuracy of the KNN model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")