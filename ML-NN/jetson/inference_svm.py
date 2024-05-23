from joblib import load
from sklearn import datasets
from sklearn import metrics
import numpy as np
import time
import psutil

def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb

# Load the MNIST dataset
digits = datasets.load_digits()

# Split the data into training and testing sets
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_test, y_test = data[n_samples // 2:], digits.target[n_samples // 2:]

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the saved SVM classifier
loaded_classifier = load('svm_model_quantized.joblib')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Quantize the test data to 16-bit precision
X_test_quantized = np.float16(X_test)

# Make predictions on the quantized test set using the loaded model
y_pred_loaded = loaded_classifier.predict(X_test_quantized)

# Calculate the accuracy
accuracy_loaded = metrics.accuracy_score(y_test, y_pred_loaded)
print(f'Accuracy (Loaded Model with Quantized Data): {accuracy_loaded * 100:.2f}%')

phase_name = 'Memory Utilization'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
