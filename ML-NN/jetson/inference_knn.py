import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from tensorflow.keras.datasets import mnist
import time
import psutil

def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb

def batch_predict(knn_model, x, batch_size=1000):
    y_pred = []
    num_batches = len(x) // batch_size + (len(x) % batch_size != 0)
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(x))
        y_pred.extend(knn_model.predict(x[start:end]))
    return np.array(y_pred)

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
# Make predictions on the quantized test data in batches
y_pred = batch_predict(loaded_knn, x_test_quantized)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Calculate the accuracy of the KNN model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

phase_name = 'Memory Utilization'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")