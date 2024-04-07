import numpy as np
import time
from openvino.inference_engine import IECore

# Initialize the OpenVINO Inference Engine
ie = IECore()

# Load the IR model onto the Intel Neural Compute Stick
net = ie.read_network(model='svm_model_quantized.xml', weights='svm_model_quantized.bin')
exec_net = ie.load_network(network=net, device_name='MYRIAD')

# Load the MNIST dataset
from sklearn import datasets
digits = datasets.load_digits()

# Split the data into training and testing sets
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_test, y_test = data[n_samples // 2:], digits.target[n_samples // 2:]

# Quantize the test data to 16-bit precision
X_test_quantized = np.float16(X_test)

# Initialize variables for accuracy calculation
correct_predictions = 0
total_samples = len(X_test_quantized)

# Perform inference on the quantized test data
for i in range(total_samples):
    res = exec_net.infer(inputs={'input': X_test_quantized[i:i+1]})
    output = res['output']
    predicted_label = np.argmax(output)
    true_label = y_test[i]

    # Check if the prediction matches the true label
    if predicted_label == true_label:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_samples * 100
print(f'Accuracy: {accuracy:.2f}%')